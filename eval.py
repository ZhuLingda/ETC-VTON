import argparse
import os
import torch
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import DDIMScheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPVisionModelWithProjection
from src.models.Embeddings import  FrozenDinoV2Encoder
from src.dataset.vvt_dataset import VVTListDataset
from diffusers.models import AutoencoderKL as VAE_vanila
from src.models.AutoencoderKL import AutoencoderKL
from src.utils.image_from_pipe import visual_dataset_from_pipe_frames
from src.models.unet_3d_condition import UNet3DConditionModel, InflatedConv3d
from src.pipelines.tryon_pipe_frames import StableDiffusionTryOnePipelineNoTextFrames
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WANDB_START_METHOD"] = "thread"


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset", type=str, required=False, choices=["dresscode", "vitonhd", "deepfashion", "vvton"], help="dataset to use")
    parser.add_argument('--dresscode_dataroot', type=str, help='DressCode dataroot')
    parser.add_argument('--vitonhd_dataroot', type=str, help='VitonHD dataroot')
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        help="Path to the output directory",
    )

    parser.add_argument("--save_name", type=str, required=False, help="Name of the saving folder inside output_dir")

    parser.add_argument("--vae_dir", required=False, type=str, help="Directory where to load the trained vae from")

    parser.add_argument("--unet_dir", required=False, type=str, help="Directory where to load the trained unet from")
    parser.add_argument("--unet_name", type=str, default="latest",
                        help="Name of the unet to load from the directory specified by `--unet_dir`. "
                             "To load the latest checkpoint, use `latest`.")

    parser.add_argument(
        "--inversion_adapter_dir", type=str, default=None,
        help="Directory where to load the trained inversion adapter from. Required when using --text_usage=inversion_adapter",
    )
    parser.add_argument("--inversion_adapter_name", type=str, default="latest",
                        help="Name of the inversion adapter to load from the directory specified by `--inversion_adapter_dir`. "
                             "To load the latest checkpoint, use `latest`.")

    # parser.add_argument("--emasc_dir", type=str, default=None,
    #                     help="Directory where to load the trained EMASC from. Required when --emasc_type!=none")
    # parser.add_argument("--emasc_name", type=str, default="latest",
    #                     help="Name of the EMASC to load from the directory specified by `--projector_dir`. "
    #                          "To load the latest checkpoint, use `latest`.")
    parser.add_argument("--projector_dir", type=str, default=None,
                        help="Directory where to load the trained projector_dir from. Required when --emasc_type!=none")
    parser.add_argument("--projector", type=str, default="latest",
                        help="Name of the projector_dir to load from the directory specified by `--projector_dir`. "
                             "To load the latest checkpoint, use `latest`.")

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2-inpainting",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )


    parser.add_argument("--seed", type=int, default=1234, help="A seed for reproducible training.")

    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size (per device) for the testing dataloader."
    )

    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )

    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )


    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for the dataloader")

    parser.add_argument("--category", type=str, choices=['all', 'lower_body', 'upper_body', 'dresses'], default='all')

    # parser.add_argument("--emasc_type", type=str, default='nonlinear', choices=["none", "linear", "nonlinear"],
    #                     help="Whether to use linear or nonlinear EMASC.")
    # parser.add_argument("--emasc_kernel", type=int, default=3, help="EMASC kernel size.")
    # parser.add_argument("--emasc_padding", type=int, default=1, help="EMASC padding size.")

    parser.add_argument("--cloth_input_type", type=str, choices=["warped", "none"], default='warped',
                        help="cloth input type. If 'warped' use the warped cloth, if none do not use the cloth as input of the unet")
    parser.add_argument("--num_vstar", default=16, type=int, help="Number of predicted v* images to use")
    parser.add_argument("--num_encoder_layers", default=1, type=int,
                        help="Number of ViT layer to use in inversion adapter")

    parser.add_argument("--use_png", default=False, action="store_true", help="Use png instead of jpg")
    parser.add_argument("--num_inference_steps", default=50, type=int, help="Number of diffusion steps")
    parser.add_argument("--guidance_scale", default=1, type=float, help="Guidance scale for the diffusion")
    parser.add_argument("--use_clip_cloth_features", action="store_true",
                        help="Whether to use precomputed clip cloth features")
    parser.add_argument("--compute_metrics", default=False, action="store_true",
                        help="Compute metrics after generation")
    parser.add_argument("--add_warped_cloth_agnostic", action="store_true",
                        help="")
    parser.add_argument("--frames_num", default=3, type=int)
    parser.add_argument("--embedding_type", type=str, default='dinov2',
                        choices=["dinov2", "clip",],)
    parser.add_argument("--pair", action="store_true",
                        help="")
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


@torch.inference_mode()
def main():
    args = parse_args()
    # args.dataset = "vvton"
    # args.output_dir = "/home/user/zld/VideoVTON/LaDI-VTON/output"
    # args.save_name = "test_unpaired_dino-v2_projector_skip-attn"
    # args.vae_dir = "/home/user/zld/VideoVTON/LaDI-VTON/src/models/weights/vae.pth"
    # args.unet_dir = "/home/user/zld/VideoVTON/LaDI-VTON/src/models/weights/unet.pth"
    # args.projector_dir = "/home/user/zld/VideoVTON/LaDI-VTON/src/models/weights/vision_encoder_projector.pth"

    # Enable TF32 for faster inference on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Setup accelerator and device.
    accelerator = Accelerator()
    device = accelerator.device

    # If passed along, set the training seed now.
    # if args.seed is not None:
    #     set_seed(args.seed)
    
    # Load scheduler, tokenizer and models.
    val_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    val_scheduler.set_timesteps(args.num_inference_steps, device=device)
    vanila_vae = VAE_vanila.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    vae = AutoencoderKL(**vanila_vae.config)
    vae.load_state_dict(torch.load(args.vae_dir))
    unet = UNet3DConditionModel.from_pretrained_2d('/home/user/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-inpainting/snapshots/81a84f49b15956b60b4272a405ad3daef3da4590', subfolder="unet")

    # Define the extended unet
    new_in_channels = 34 if args.cloth_input_type == "none" else 38
    # the posemap has 18 channels, the (encoded) cloth has 4 channels, the standard SD inpaining has 9 channels
    with torch.no_grad():
        # Replace the first conv layer of the unet with a new one with the correct number of input channels
        conv_new = InflatedConv3d(
            in_channels=new_in_channels,
            out_channels=unet.conv_in.out_channels,
            kernel_size=3,
            padding=1,
        )

        torch.nn.init.kaiming_normal_(conv_new.weight)  # Initialize new conv layer
        conv_new.weight.data = conv_new.weight.data * 0.  # Zero-initialize new conv layer

        conv_new.weight.data[:, :9] = unet.conv_in.weight.data  # Copy weights from old conv layer
        conv_new.bias.data = unet.conv_in.bias.data  # Copy bias from old conv layer

        unet.conv_in = conv_new  # replace conv layer in unet
        unet.config['in_channels'] = new_in_channels  # update config

    # Load the trained unet
    # if args.unet_name != "latest":
    #     path = args.unet_name
    # else:
    #     # Get the most recent checkpoint
    #     dirs = os.listdir(args.unet_dir)
    #     # dirs = [d for d in dirs if d.startswith("unet")]
    #     # dirs = sorted(dirs, key=lambda x: int(os.path.splitext(x.split("_")[-1])[0]))
    #     path = dirs[-1]
        
    accelerator.print(f"Loading Unet checkpoint {args.unet_dir}")
    unet.load_state_dict(torch.load(args.unet_dir))
    print(f"Unet loaded from {args.unet_dir}")

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.embedding_type == 'dinov2':
        vision_encoder = FrozenDinoV2Encoder()
        if args.projector_dir is not None:
            vision_encoder.projector.load_state_dict(torch.load(args.projector_dir))
    elif args.embedding_type == 'clip':
        vision_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
        # encoder_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
        if args.projector_dir is not None:
            vision_encoder.visual_projection.load_state_dict(torch.load(args.projector_dir))
    
    print(f"Vision encoder {args.projector_dir} loaded")

    test_dataset = VVTListDataset(
            dataroot="/home/user/zld/VideoVTON/VVT",
            image_size=256,
            mode="test",
            frames_num=args.frames_num,
            is_pair=args.pair,
        )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    test_dataloader = accelerator.prepare(test_dataloader)

    weight_dtype = torch.float16

    # Move models to device and eval mode
    vae.to(device, dtype=weight_dtype)
    vae.eval()
    unet.to(device, dtype=weight_dtype)
    unet.eval()
    # if emasc is not None:
    #     emasc.to(device, dtype=weight_dtype)
    #     emasc.eval()
    if vision_encoder is not None:
        vision_encoder.to(device, dtype=weight_dtype)
        vision_encoder.eval()

    # Define the pipeline
    val_pipe = StableDiffusionTryOnePipelineNoTextFrames(
        vae=vae,
        unet=unet,
        scheduler=val_scheduler,
        enhance=True
    ).to(device)

    # Generate images
    with torch.cuda.amp.autocast():
        visual_dataset_from_pipe_frames(pipe=val_pipe,
                                test_dataloader=test_dataloader, 
                                output_dir=args.output_dir,
                                save_name=args.save_name, 
                                vision_encoder=vision_encoder,
                                cloth_input_type=args.cloth_input_type, 
                                cloth_cond_rate=1,
                                seed=args.seed,
                                num_inference_steps=args.num_inference_steps, 
                                guidance_scale=args.guidance_scale,
                                add_warped_cloth_agnostic=args.add_warped_cloth_agnostic,)


if __name__ == "__main__":
    main()
