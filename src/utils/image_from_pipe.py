import os
import sys
from pathlib import Path
from typing import List
from utils.colorvisual import cat2rgb, getshow
from torchvision import utils
import torch
import torchvision.transforms as transforms
from .data_utils import mask_features
import numpy as np
import cv2
PROJECT_ROOT = Path(__file__).absolute().parents[2].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from diffusers import StableDiffusionInpaintPipeline
from tqdm import tqdm
from src.vto_pipelines.tryon_pipe import StableDiffusionTryOnPipeline, StableDiffusionTryOnPipelineNoText
from src.models.AutoencoderKL import AutoencoderKL
from src.models.emasc import EMASC
from src.models.inversion_adapter import InversionAdapter
from transformers import CLIPVisionModelWithProjection, CLIPProcessor
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange, repeat

import torchvision
from src.utils.encode_text_word_embedding import encode_text_word_embedding


@torch.no_grad()
def generate_images_from_tryon_pipe(pipe: StableDiffusionTryOnPipeline, inversion_adapter: InversionAdapter,
                                    test_dataloader: torch.utils.data.DataLoader, output_dir: str, order: str,
                                    save_name: str, text_usage: str, vision_encoder: CLIPVisionModelWithProjection,
                                    processor: CLIPProcessor, cloth_input_type: str, cloth_cond_rate: int = 1,
                                    num_vstar: int = 1, seed: int = 1234, num_inference_steps: int = 50,
                                    guidance_scale: int = 7.5, use_png: bool = False):
    # Create output directory
    save_path = os.path.join(output_dir, f"{save_name}_{order}")
    os.makedirs(save_path, exist_ok=True)

    # Set seed
    generator = torch.Generator("cuda").manual_seed(seed)
    num_samples = 1

    # Generate images
    for idx, batch in enumerate(tqdm(test_dataloader)):
        model_img = batch.get("image")
        mask_img = batch.get("inpaint_mask")
        if mask_img is not None:
            mask_img = mask_img.type(torch.float32)
        pose_map = batch.get("pose_map")
        warped_cloth = batch.get('warped_cloth')
        category = batch.get("category")
        cloth = batch.get("cloth")

        # Generate text prompts
        if text_usage == "noun_chunks":
            prompts = batch["captions"]
        elif text_usage == "none":
            prompts = [""] * len(batch["captions"])
        elif text_usage == 'inversion_adapter':
            category_text = {
                'dresses': 'a dress',
                'upper_body': 'an upper body garment',
                'lower_body': 'a lower body garment',

            }
            text = [f'a photo of a model wearing {category_text[category]} {" $ " * num_vstar}' for
                    category in batch['category']]

            clip_cloth_features = batch.get('clip_cloth_features')
            if clip_cloth_features is None:
                with torch.no_grad():
                    # Get the visual features of the in-shop cloths
                    input_image = torchvision.transforms.functional.resize((batch["cloth"] + 1) / 2, (224, 224),
                                                                           antialias=True).clamp(0, 1)
                    processed_images = processor(images=input_image, return_tensors="pt")
                    clip_cloth_features = vision_encoder(
                        processed_images.pixel_values.to(model_img.device)).last_hidden_state

            # Compute the predicted PTEs
            word_embeddings = inversion_adapter(clip_cloth_features.to(model_img.device))
            word_embeddings = word_embeddings.reshape((word_embeddings.shape[0], num_vstar, -1))

            # Tokenize text
            tokenized_text = pipe.tokenizer(text, max_length=pipe.tokenizer.model_max_length, padding="max_length",
                                            truncation=True, return_tensors="pt").input_ids
            tokenized_text = tokenized_text.to(word_embeddings.device)

            # Encode the text using the PTEs extracted from the in-shop cloths
            encoder_hidden_states = encode_text_word_embedding(pipe.text_encoder, tokenized_text,
                                                               word_embeddings, num_vstar).last_hidden_state
        else:
            raise ValueError(f"Unknown text usage {text_usage}")

        # Generate images
        if text_usage == 'inversion_adapter':
            generated_images = pipe(
                image=model_img,
                mask_image=mask_img,
                pose_map=pose_map,
                warped_cloth=warped_cloth,
                prompt_embeds=encoder_hidden_states,
                height=512,
                width=384,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_samples,
                generator=generator,
                cloth_input_type=cloth_input_type,
                cloth_cond_rate=cloth_cond_rate,
                num_inference_steps=num_inference_steps
            ).images
        else:
            generated_images = pipe(
                prompt=prompts,
                image=model_img,
                mask_image=mask_img,
                pose_map=pose_map,
                warped_cloth=warped_cloth,
                height=512,
                width=384,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_samples,
                generator=generator,
                cloth_input_type=cloth_input_type,
                cloth_cond_rate=cloth_cond_rate,
                num_inference_steps=num_inference_steps
            ).images

        # Save images
        for gen_image, cat, name in zip(generated_images, category, batch["im_name"]):
            if not os.path.exists(os.path.join(save_path, cat)):
                os.makedirs(os.path.join(save_path, cat))

            if use_png:
                name = name.replace(".jpg", ".png")
                gen_image.save(
                    os.path.join(save_path, cat, name))
            else:
                gen_image.save(
                    os.path.join(save_path, cat, name), quality=95)


def generate_images_inversion_adapter(pipe: StableDiffusionInpaintPipeline, inversion_adapter: InversionAdapter,
                                      vision_encoder: CLIPVisionModelWithProjection, processor: CLIPProcessor,
                                      test_dataloader: torch.utils.data.DataLoader, output_dir, order: str,
                                      save_name: str, num_vstar=1, seed=1234, num_inference_steps=50,
                                      guidance_scale=7.5, use_png=False) -> None:
    """
    Extract and save images using the SD inpainting pipeline using the PTEs from the inversion adapter.
    """
    # Create output directory
    save_path = os.path.join(output_dir, f"{save_name}_{order}")
    os.makedirs(save_path, exist_ok=True)

    # Set seed
    generator = torch.Generator("cuda").manual_seed(seed)
    num_samples = 1

    # Generate images
    for idx, batch in enumerate(tqdm(test_dataloader)):
        model_img = batch["image"]
        mask_img = batch["inpaint_mask"]
        mask_img = mask_img.type(torch.float32)
        category = batch["category"]
        # Generate images
        cloth = batch.get("cloth")
        clip_cloth_features = batch.get('clip_cloth_features')

        if clip_cloth_features is None:
            # Get the visual features of the in-shop cloths
            input_image = torchvision.transforms.functional.resize(
                (cloth + 1) / 2, (224, 224), antialias=True).clamp(0, 1)
            processed_images = processor(images=input_image, return_tensors="pt")
            clip_cloth_features = vision_encoder(processed_images.pixel_values.to(model_img.device)).last_hidden_state

        word_embeddings = inversion_adapter(clip_cloth_features.to(model_img.device))
        word_embeddings = word_embeddings.reshape((word_embeddings.shape[0], num_vstar, -1))

        category_text = {
            'dresses': 'a dress',
            'upper_body': 'an upper body garment',
            'lower_body': 'a lower body garment',

        }
        text = [f'a photo of a model wearing {category_text[category]} {" $ " * num_vstar}' for category in
                batch['category']]

        # Tokenize text
        tokenized_text = pipe.tokenizer(text, max_length=pipe.tokenizer.model_max_length, padding="max_length",
                                        truncation=True, return_tensors="pt").input_ids
        tokenized_text = tokenized_text.to(model_img.device)

        # Encode the text using the PTEs extracted from the in-shop cloths
        encoder_hidden_states = encode_text_word_embedding(pipe.text_encoder, tokenized_text,
                                                           word_embeddings,
                                                           num_vstar=num_vstar).last_hidden_state

        # Generate images
        generated_images = pipe(
            image=model_img,
            mask_image=mask_img,
            prompt_embeds=encoder_hidden_states,
            height=512,
            width=384,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_samples,
            generator=generator,
            num_inference_steps=num_inference_steps
        ).images

        # Save images
        for gen_image, cat, name in zip(generated_images, category, batch["im_name"]):
            if not os.path.exists(os.path.join(save_path, cat)):
                os.makedirs(os.path.join(save_path, cat))

            if use_png:
                name = name.replace(".jpg", ".png")
                gen_image.save(
                    os.path.join(save_path, cat, name))
            else:
                gen_image.save(
                    os.path.join(save_path, cat, name), quality=95)


@torch.inference_mode()
def extract_save_vae_images(vae: AutoencoderKL, emasc: EMASC, test_dataloader: torch.utils.data.DataLoader,
                            int_layers: List[int], output_dir: str, order: str, save_name: str,
                            emasc_type: str) -> None:
    """
    Extract and save image using only VAE or VAE + EMASC
    """
    # Create output directory
    save_path = os.path.join(output_dir, f"{save_name}_{order}")
    os.makedirs(save_path, exist_ok=True)

    for idx, batch in enumerate(tqdm(test_dataloader)):
        category = batch["category"]

        if emasc_type != "none":
            # Extract intermediate features from 'im_mask' and encode image
            posterior_im, _ = vae.encode(batch["image"])
            _, intermediate_features = vae.encode(batch["im_mask"])
            intermediate_features = [intermediate_features[i] for i in int_layers]

            # Use EMASC
            processed_intermediate_features = emasc(intermediate_features)

            processed_intermediate_features = mask_features(processed_intermediate_features, batch["inpaint_mask"])
            latents = posterior_im.latent_dist.sample()
            generated_images = vae.decode(latents, processed_intermediate_features, int_layers).sample
        else:
            # Encode and decode image without EMASC
            posterior_im = vae.encode(batch["image"])
            latents = posterior_im.latent_dist.sample()
            generated_images = vae.decode(latents).sample

        # Save images
        for gen_image, cat, name in zip(generated_images, category, batch["im_name"]):
            gen_image = (gen_image + 1) / 2  # [-1, 1] -> [0, 1]
            if not os.path.exists(os.path.join(save_path, cat)):
                os.makedirs(os.path.join(save_path, cat))
            torchvision.utils.save_image(gen_image, os.path.join(save_path, cat, name), quality=95)

@torch.no_grad()
def visual_batch_from_pipe(pipe,
                                    batch, step, output_dir: str,
                                    vision_encoder,
                                    processor: CLIPProcessor, cloth_input_type: str = 'warped', cloth_cond_rate: int = 1,
                                    num_vstar: int = 1, seed: int = 1234, num_inference_steps: int = 50,
                                    guidance_scale: int = 1, use_png: bool = False, add_warped_cloth_agnostic = False):
    # Create output directory

    

    # Set seed
    generator = torch.Generator("cuda").manual_seed(seed)
    num_samples = 1

    # Generate images
    if add_warped_cloth_agnostic:
        mask_input = batch.get("source_mask_inpaint_warped").to(pipe.device, torch.float16)
        unet_cloth_input = batch.get("target_cloth_img").to(pipe.device, torch.float16)
        model_img = batch.get("source_img_inpaint_warped").to(pipe.device, torch.float16)

    else:
        model_img = batch.get("source_image").to(pipe.device, torch.float16)
        mask_input = batch.get("inpaint_mask").to(pipe.device, torch.float16)
        unet_cloth_input = batch.get('warp_feat').to(pipe.device, torch.float16)

    pose_map = batch.get("source_densepose").to(pipe.device, torch.float16)
    cloth = batch.get("target_cloth_img").to(pipe.device, torch.float16)


    input_image = torchvision.transforms.functional.resize((cloth + 1) / 2, (224, 224),
                                                            antialias=True).clamp(0, 1)
    # processed_images = processor(images=input_image, return_tensors="pt")
    # clip_cloth_features = vision_encoder(
    #     processed_images.pixel_values.to(model_img.device)).last_hidden_state
    # encoder_hidden_states = clip_cloth_features.to(pipe.device, torch.float16)
    #dino

    encoder_hidden_states = vision_encoder(input_image).to(pipe.device, torch.float16)
    # Generate images

    generated_images = pipe(

        image=model_img,
        mask_image=mask_input,
        pose_map=pose_map,
        warped_cloth=unet_cloth_input,
        prompt_embeds=encoder_hidden_states,
        height=256,
        width=256,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_samples,
        generator=generator,
        cloth_input_type=cloth_input_type,
        cloth_cond_rate=cloth_cond_rate,
        num_inference_steps=num_inference_steps
    ).images


    source_image = batch['source_image'].to(pipe.device)
    source_parsing = batch['source_parsing'].to(pipe.device)
    source_densepose = batch['source_densepose'].to(pipe.device)
    source_pose_forshow = batch['source_pose_forshow'].to(pipe.device)
    source_cloth = batch['source_cloth_img'].to(pipe.device)
    source_cloth_mask = batch['source_cloth_mask'].to(pipe.device)
    source_preserve_mask_forshow = batch['source_preserve_mask_forshow'].to(pipe.device)
    target_cloth = batch['target_cloth_img'].to(pipe.device)
    target_cloth_mask = batch['target_cloth_mask'].to(pipe.device)
    warped_cloth = batch['warp_feat'].to(pipe.device)
    inpaint_image = batch['source_img_inpaint_warped'].to(pipe.device)
    source_img_agnostic = batch['source_img_agnostic'].to(pipe.device)
    to_tensor = transforms.Compose([ 
	    transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
	])
    simple_image_pil = generated_images[0]
    simple_image = to_tensor(simple_image_pil)

    simple_image = simple_image.unsqueeze(0)
    # to tensor
    combine = torch.cat([
        source_image[0:1, :, :, :],
        getshow(source_parsing, 20),
        getshow(source_densepose, 25),
        cat2rgb(mask_input[0:1, :, :, :]),
        inpaint_image[0:1, :, :, :],
        cat2rgb(source_cloth_mask[0:1, :, :, :]),
        cat2rgb(target_cloth_mask[0:1, :, :, :]),
        warped_cloth[0:1, :, :, :],
        unet_cloth_input[0:1, :, :, :],
        source_img_agnostic[0:1, :, :, :],
        simple_image.cuda(),
        ], 0).squeeze()
  
    target_path = f"{output_dir}/{int(step)}.jpg"
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    utils.save_image(combine, target_path, nrow=int(4), normalize=True, range=(-1, 1), )
    print(f"saved at {target_path}")

@torch.no_grad()
def visual_batch_from_pipe_frames(pipe,
                                    batch, step, output_dir: str,
                                    vision_encoder,
                                    processor: CLIPProcessor, cloth_input_type: str = 'warped', cloth_cond_rate: int = 1,
                                    num_vstar: int = 1, seed: int = 1234, num_inference_steps: int = 50,
                                    guidance_scale: int = 1, use_png: bool = False, add_warped_cloth_agnostic = True):
    # Create output directory

    

    # Set seed
    generator = torch.Generator("cuda").manual_seed(seed)
    num_samples = 1

    # Generate images
    if add_warped_cloth_agnostic:
        mask_input = batch.get("source_cloth_agnostic_mask_frames").to(pipe.device, torch.float16)
        unet_cloth_input = batch.get("target_cloth_frames").to(pipe.device, torch.float16)
        model_img = batch.get("source_warped_cloth_agnostic_image_frames").to(pipe.device, torch.float16)

    else:
        model_img = batch.get("source_cloth_agnostic_image_frames").to(pipe.device, torch.float16)
        mask_input = batch.get("source_cloth_agnostic_mask_frames").to(pipe.device, torch.float16)
        unet_cloth_input = batch.get('target_cloth_frames').to(pipe.device, torch.float16)

    pose_map = batch.get("source_densepose_one_hot_frames").to(pipe.device, torch.float16)
    cloth = batch.get("target_cloth_img").to(pipe.device, torch.float16)

    source_frames = batch.get("source_image_frames").to(pipe.device, torch.float16)

    input_image = torchvision.transforms.functional.resize((cloth + 1) / 2, (224, 224),
                                                            antialias=True).clamp(0, 1)
    
    if isinstance(vision_encoder, CLIPVisionModelWithProjection):
        # clip
        # processed_images = processor(images=input_image, return_tensors="pt")
        # clip_cloth_features = vision_encoder(
        #     processed_images.pixel_values.to(model_img.device)).last_hidden_state
        encoder_hidden_states = vision_encoder(input_image).last_hidden_state.to(pipe.device, torch.float16)
    else:
        # dino
        encoder_hidden_states = vision_encoder(input_image).to(pipe.device, torch.float16)

    # Generate images
    generated_images = pipe(
        image=model_img,
        mask_image=mask_input,
        pose_map=pose_map,
        warped_cloth=unet_cloth_input,
        prompt_embeds=encoder_hidden_states,
        height=256,
        width=256,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_samples,
        generator=generator,
        cloth_input_type=cloth_input_type,
        cloth_cond_rate=cloth_cond_rate,
        num_inference_steps=num_inference_steps
    ).images


    source_image = batch['source_image'].to(pipe.device)
    source_parsing = batch['source_parsing'].to(pipe.device)
    source_densepose = batch['source_densepose'].to(pipe.device)
    source_pose_forshow = batch['source_pose_forshow'].to(pipe.device)
    source_cloth = batch['source_cloth_img'].to(pipe.device)
    source_cloth_mask = batch['source_cloth_mask'].to(pipe.device)
    source_preserve_mask_forshow = batch['source_preserve_mask_forshow'].to(pipe.device)
    target_cloth = batch['target_cloth_img'].to(pipe.device)
    target_cloth_mask = batch['target_cloth_mask'].to(pipe.device)
    warped_cloth = batch['warp_feat'].to(pipe.device)
    inpaint_image = batch['source_img_inpaint_warped'].to(pipe.device)
    source_img_agnostic = batch['source_img_agnostic'].to(pipe.device)
    source_mask_agnostic = batch['source_mask_agnostic'].to(pipe.device)

    generated_images = generated_images.clamp(-1, 1)
    frames_num = source_frames.shape[1]
    # to tensor
    combine = torch.cat([
        getshow(source_parsing, 20),
        getshow(source_densepose, 25),
        cat2rgb(mask_input[0:1,frames_num//2 ,:, :, :]),
        model_img[0:1,frames_num//2 ,:, :, :],
        cat2rgb(source_cloth_mask[0:1, :, :, :]),
        cat2rgb(target_cloth_mask[0:1, :, :, :]),
        unet_cloth_input[0:1 ,0,:, :, :],
        warped_cloth[0:1, :, :, :],
        generated_images[0:1,0 ,:, :, :],
        generated_images[0:1,frames_num//2 ,:, :, :],
        generated_images[0:1,-1 ,:, :, :],
        torch.zeros_like(generated_images[0:1,0 ,:, :, :]),
        source_frames[0:1,0 ,:, :, :],
        source_frames[0:1,frames_num//2 ,:, :, :],
        source_frames[0:1,-1 ,:, :, :],
        ], 0).squeeze()
  
    target_path = f"{output_dir}/{int(step)}.jpg"
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    utils.save_image(combine, target_path, nrow=int(4), normalize=True, range=(-1, 1), )
    print(f"saved at {target_path}")

@torch.no_grad()
def visual_dataset_from_pipe_frames(pipe,
                            test_dataloader: torch.utils.data.DataLoader, output_dir: str,
                            save_name: str, vision_encoder,
                            cloth_input_type: str, cloth_cond_rate: int = 1,
                            seed: int = 1234, num_inference_steps: int = 50,
                            guidance_scale: int = 1,
                            add_warped_cloth_agnostic: bool = True,):

    # Set seed
    generator = torch.Generator("cuda").manual_seed(seed)
    num_samples = 1
    b, f, _, h, w = next(iter(test_dataloader))['target_cloth_frames'].shape
    latent_shape = (b, 4, f, h//8, w//8)
    default_latents = randn_tensor(latent_shape, generator, pipe.device, torch.float16)
    # default_latents = None
    # Generate images
    for idx, batch in enumerate(tqdm(test_dataloader)):
        one_batch_frames_name = [frame_name[0] for frame_name in batch['source_file_name_frames']]
        cur_frame_name = batch['source_image_name'][0]
        frame_num = len(one_batch_frames_name)
        is_last_frame = batch['is_last_frame'][0]
        next_pose_path = batch['next_pose_path'][0]
        tqdm.write(f"checking {cur_frame_name=}, adj={one_batch_frames_name}, {is_last_frame=}, {next_pose_path=}")
        is_all_new_flag = True
        target_path_list = []
        cur_target_path = os.path.join(f"{output_dir}/{save_name}", cur_frame_name[:-4] + ".jpg")
        for frame_name in one_batch_frames_name:
            target_path = os.path.join(f"{output_dir}/{save_name}", frame_name[:-4] + ".jpg")
            target_path_list.append(target_path)
            # 滑动窗口有重合且当前帧没到最后一帧
            if os.path.exists(target_path):
                if is_last_frame:
                    if not os.path.exists(cur_target_path):
                        tqdm.write(f"!!!!!!!NOTE: this is last but {cur_target_path=} not exists, so still sample")
                        continue
                    else:
                        tqdm.write(f"-------NOTE: this is last but {cur_target_path=} exists, skip this batch")
                tqdm.write(f"in this batch {target_path=} exists")
                is_all_new_flag = False
                break
        if not is_all_new_flag:
            continue
        else:
            pass
            # print(f"this batch will simple {target_path_list}")
        if add_warped_cloth_agnostic:
            model_img = batch.get("source_warped_cloth_agnostic_image_frames").to(pipe.device, torch.float16)
            mask_input = batch.get("source_cloth_agnostic_mask_frames").to(pipe.device, torch.float16)
            unet_cloth_input = batch.get("target_cloth_frames").to(pipe.device, torch.float16)
        else:
            mask_input = batch.get("source_cloth_agnostic_mask_frames").to(pipe.device, torch.float16)
            unet_cloth_input = batch.get('target_cloth_frames').to(pipe.device, torch.float16)
            model_img = batch.get("source_cloth_agnostic_image_frames").to(pipe.device, torch.float16)

        # if mask_input is not None:
        #     mask_input = mask_input.type(torch.float32)
        pose_map = batch.get("source_densepose_one_hot_frames").to(pipe.device, torch.float16)
        
        cloth = batch.get("target_cloth_img").to(pipe.device, torch.float16)

        source_frames = batch.get("source_image_frames").to(pipe.device, torch.float16)

        input_image = torchvision.transforms.functional.resize((cloth + 1) / 2, (224, 224),
                                                                antialias=True).clamp(0, 1)
        # processed_images = processor(images=input_image, return_tensors="pt")
        # clip_cloth_features = vision_encoder(
        #     processed_images.pixel_values.to(model_img.device)).last_hidden_state
        # encoder_hidden_states = clip_cloth_features.to(pipe.device, torch.float16)
        if isinstance(vision_encoder, CLIPVisionModelWithProjection):
            # clip
            # processed_images = processor(images=input_image, return_tensors="pt")
            # clip_cloth_features = vision_encoder(
            #     processed_images.pixel_values.to(model_img.device)).last_hidden_state
            encoder_hidden_states = vision_encoder(input_image).last_hidden_state.to(pipe.device, torch.float16)
        else:
            # dino
            encoder_hidden_states = vision_encoder(input_image).to(pipe.device, torch.float16)
        # Generate images

        generated_images = pipe(
            image=model_img,
            mask_image=mask_input,
            pose_map=pose_map,
            warped_cloth=unet_cloth_input,
            prompt_embeds=encoder_hidden_states,
            height=256,
            width=256,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_samples,
            generator=generator,
            cloth_input_type=cloth_input_type,
            cloth_cond_rate=cloth_cond_rate,
            num_inference_steps=num_inference_steps,
            latents=default_latents
        ).images

        for i in range(len(batch['source_image'])):
            source_image = batch['source_image'].to(pipe.device)
            # target_imgage = batch['target_image'].to(pipe.device)
            source_parsing = batch['source_parsing'].to(pipe.device)
            source_densepose = batch['source_densepose'].to(pipe.device)
            source_pose_forshow = batch['source_pose_forshow'].to(pipe.device)
            source_cloth = batch['source_cloth_img'].to(pipe.device)
            source_cloth_mask = batch['source_cloth_mask'].to(pipe.device)
            source_preserve_mask_forshow = batch['source_preserve_mask_forshow'].to(pipe.device)
            target_cloth = batch['target_cloth_img'].to(pipe.device)
            target_cloth_mask = batch['target_cloth_mask'].to(pipe.device)
            warped_cloth = batch['warp_feat'].to(pipe.device)
            # inpaint_image = batch['inpaint_image'].to(pipe.device)
            inpaint_image = batch['inpaint_image_with_warped'].to(pipe.device)
            source_img_agnostic = batch['source_img_agnostic'].to(pipe.device)
            source_mask_agnostic = batch['source_mask_agnostic'].to(pipe.device)

            generated_images = generated_images.clamp(-1, 1) 


            for j, target_path in enumerate(target_path_list):
                # combine = torch.cat([
                # getshow(source_parsing, 20),
                # # getshow(source_densepose, 25),
                # getshow(pose_map[i:i+1,j , :, :, :],25),
                # cat2rgb(mask_input[i:i+1,j , :, :, :]),
                # inpaint_image[i:i+1, :, :, :],
                # cat2rgb(source_cloth_mask[i:i+1, :, :, :]),
                # cat2rgb(target_cloth_mask[i:i+1, :, :, :]),
                # warped_cloth[i:i+1, :, :, :],
                # model_img[i:i+1,j , :, :, :],
                # generated_images[i:i+1,j ,:, :, :],
                # source_frames[i:i+1,j ,:, :, :],
                # unet_cloth_input[i:i+1,j, :, :, :],
                # ], 0).squeeze()
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                # utils.save_image(combine, target_path, nrow=int(4), normalize=True, range=(-1, 1), )
                frame_img = generated_images[i,j ,:, :, :]
                # frame_img = frame_img * 0.5 + 0.5
                # frame_img = frame_img.squeeze()
                # frame_img = frame_img.permute(1, 2, 0).cpu().numpy()
                # frame_img = (frame_img * 255).astype('uint8')
 
                cv_img = (frame_img.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
                rgb = (cv_img * 255).astype(np.uint8)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(target_path, bgr)
                # cv2.imwrite(target_path, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

                tqdm.write(f"saved at {target_path}")


@torch.no_grad()
def visual_dataset_from_pipe(pipe,
                            test_dataloader: torch.utils.data.DataLoader, output_dir: str,
                            save_name: str, vision_encoder,
                            cloth_input_type: str, cloth_cond_rate: int = 1,
                            seed: int = 1234, num_inference_steps: int = 50,
                            guidance_scale: int = 1,
                            add_warped_cloth_agnostic: bool = False,):

    # Set seed
    generator = torch.Generator("cuda").manual_seed(seed)
    num_samples = 1

    # Generate images
    for idx, batch in enumerate(tqdm(test_dataloader)):
        if add_warped_cloth_agnostic:
            model_img = batch.get("source_img_inpaint_warped").to(pipe.device, torch.float16)
            mask_input = batch.get("source_mask_agnostic").to(pipe.device, torch.float16)
            unet_cloth_input = batch.get("target_cloth_img").to(pipe.device, torch.float16)
        else:
            mask_input = batch.get("inpaint_mask").to(pipe.device, torch.float16)
            unet_cloth_input = batch.get('warp_feat').to(pipe.device, torch.float16)
            model_img = batch.get("source_image").to(pipe.device, torch.float16)

        if mask_input is not None:
            mask_input = mask_input.type(torch.float32)
        pose_map = batch.get("source_densepose").to(pipe.device, torch.float16)
        
        cloth = batch.get("target_cloth_img").to(pipe.device, torch.float16)


        input_image = torchvision.transforms.functional.resize((cloth + 1) / 2, (224, 224),
                                                                antialias=True).clamp(0, 1)
        # processed_images = processor(images=input_image, return_tensors="pt")
        # clip_cloth_features = vision_encoder(
        #     processed_images.pixel_values.to(model_img.device)).last_hidden_state
        # encoder_hidden_states = clip_cloth_features.to(pipe.device, torch.float16)
        #dino

        encoder_hidden_states = vision_encoder(input_image).to(pipe.device, torch.float16)
        # Generate images

        generated_images = pipe(

            image=model_img,
            mask_image=mask_input,
            pose_map=pose_map,
            warped_cloth=unet_cloth_input,
            prompt_embeds=encoder_hidden_states,
            height=256,
            width=256,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_samples,
            generator=generator,
            cloth_input_type=cloth_input_type,
            cloth_cond_rate=cloth_cond_rate,
            num_inference_steps=num_inference_steps
        ).images

        for i in range(len(batch['source_image'])):
            source_image = batch['source_image'].to(pipe.device)
            # target_imgage = batch['target_image'].to(pipe.device)
            source_parsing = batch['source_parsing'].to(pipe.device)
            source_densepose = batch['source_densepose'].to(pipe.device)
            source_pose_forshow = batch['source_pose_forshow'].to(pipe.device)
            source_cloth = batch['source_cloth_img'].to(pipe.device)
            source_cloth_mask = batch['source_cloth_mask'].to(pipe.device)
            source_preserve_mask_forshow = batch['source_preserve_mask_forshow'].to(pipe.device)
            target_cloth = batch['target_cloth_img'].to(pipe.device)
            target_cloth_mask = batch['target_cloth_mask'].to(pipe.device)
            warped_cloth = batch['warp_feat'].to(pipe.device)
            # inpaint_image = batch['inpaint_image'].to(pipe.device)
            inpaint_image = batch['inpaint_image_with_warped'].to(pipe.device)
            source_img_agnostic = batch['source_img_agnostic'].to(pipe.device)
            to_tensor = transforms.Compose([ 
                transforms.ToTensor(), 
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            generated_images = generated_images.clamp(-1, 1)
            cur_generated_image = generated_images[i]
            # simple_image_pil = generated_images[i]
            # simple_image = to_tensor(simple_image_pil)

            # simple_image = simple_image.unsqueeze(0)
            # to tensor
            # combine = torch.cat([
            #     source_image[i:i+1, :, :, :],
            #     # target_imgage[0:1, :, :, :],
            #     getshow(source_parsing, 20, index=i),
            #     getshow(source_densepose, 25, index=i),
            #     cat2rgb(mask_input[i:i+1, :, :, :]),
            #     inpaint_image[i:i+1, :, :, :],
            #     cat2rgb(source_cloth_mask[i:i+1, :, :, :]),
            #     cloth[i:i+1, :, :, :],
            #     cat2rgb(target_cloth_mask[i:i+1, :, :, :]),
            #     warped_cloth[i:i+1, :, :, :],
            #     source_img_agnostic[i:i+1, :, :, :],
            #     simple_image.cuda(),
            #     ], 0).squeeze()
            filename = batch['file_name'][i]
            target_path = os.path.join(f"{output_dir}/{save_name}", filename[:-4] + ".jpg")  
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            # utils.save_image(combine, target_path, nrow=int(4), normalize=True, range=(-1, 1), )
            # print(f"saved at {target_path}")
            cv_img = (cur_generated_image.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
            rgb = (cv_img * 255).astype(np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(target_path, bgr)
            tqdm.write(f"saved at {target_path}")    
@torch.inference_mode()
def visual_vae_images(vae: AutoencoderKL, batch, int_layers: List[int], output_dir: str, save_name: str) -> None:
    """
    Extract and save image using only VAE 
    """

    # Extract intermediate features from 'im_mask' and encode image
    posterior_im, _ = vae.encode(batch["source_image"])
    _, intermediate_features = vae.encode(batch["source_img_agnostic"])
    intermediate_features = [intermediate_features[i] for i in int_layers]

    latents = posterior_im.latent_dist.sample()
    generated_images = vae.decode(latents, intermediate_features, int_layers).sample

    # Encode and decode image without EMASC
    generated_images_wo_esmac = vae.decode(latents).sample

    source_image = batch['source_image']
    source_parsing = batch['source_parsing']
    source_densepose = batch['source_densepose']
    source_pose_forshow = batch['source_pose_forshow']
    source_cloth = batch['source_cloth_img']
    source_cloth_mask = batch['source_cloth_mask']
    source_preserve_mask_forshow = batch['source_preserve_mask_forshow']
    target_cloth = batch['target_cloth_img']
    target_cloth_mask = batch['target_cloth_mask']
    warped_cloth = batch['warp_feat']
    inpaint_image = batch['source_img_inpaint_warped']
    mask_img = batch["inpaint_mask"]
    source_img_agnostic = batch['source_img_agnostic']

    simple_image = generated_images[0].unsqueeze(0)

    simple_image = simple_image.clamp(-1, 1)

    simple_image_wo_esmac = generated_images_wo_esmac[0].unsqueeze(0)

    simple_image_wo_esmac = simple_image_wo_esmac.clamp(-1, 1)
    # to tensor
    combine = torch.cat([
        source_image[0:1, :, :, :],
        getshow(source_parsing, 20),
        getshow(source_densepose, 25),
        cat2rgb(mask_img[0:1, :, :, :]),
        cat2rgb(source_cloth_mask[0:1, :, :, :]),
        target_cloth[0:1, :, :, :],
        cat2rgb(target_cloth_mask[0:1, :, :, :]),
        warped_cloth[0:1, :, :, :],
        inpaint_image[0:1, :, :, :],
        source_img_agnostic[0:1, :, :, :],
        simple_image_wo_esmac.cuda(),
        simple_image.cuda(),
        ], 0).squeeze()

    # Create output directory
    output_dir = os.path.join(output_dir, "show1")
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    target_path = os.path.join(output_dir, f"{save_name}.jpg")
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    utils.save_image(combine, target_path, nrow=int(4), normalize=True, range=(-1, 1), )
    print(f"saved at {target_path}")

def visual_inversion_adapter_batch(pipe: StableDiffusionInpaintPipeline, inversion_adapter: InversionAdapter,
                                      vision_encoder: CLIPVisionModelWithProjection, processor: CLIPProcessor,
                                      batch, output_dir, order: str,
                                      save_name: str, num_vstar=1, seed=1234, num_inference_steps=50,
                                      guidance_scale=7.5, use_png=False) -> None:
    """
    Extract and save images using the SD inpainting pipeline using the PTEs from the inversion adapter.
    """

    # Set seed
    generator = torch.Generator("cuda").manual_seed(seed)
    num_samples = 1

    # Generate images

    model_img = batch["source_image"]
    mask_img = batch["inpaint_mask"]
    mask_img = mask_img.type(torch.float32)
    # Generate images
    cloth = batch.get("target_cloth_img")
    clip_cloth_features = batch.get('clip_cloth_features')

    if clip_cloth_features is None:
        # Get the visual features of the in-shop cloths
        input_image = torchvision.transforms.functional.resize(
            (cloth + 1) / 2, (224, 224), antialias=True).clamp(0, 1)
        processed_images = processor(images=input_image, return_tensors="pt")
        clip_cloth_features = vision_encoder(processed_images.pixel_values.to(model_img.device)).last_hidden_state

    word_embeddings = inversion_adapter(clip_cloth_features.to(model_img.device))
    word_embeddings = word_embeddings.reshape((word_embeddings.shape[0], num_vstar, -1))

    text = [f'a photo of a model wearing {" $ " * num_vstar}'] * model_img.shape[0]

    # Tokenize text
    tokenized_text = pipe.tokenizer(text, max_length=pipe.tokenizer.model_max_length, padding="max_length",
                                    truncation=True, return_tensors="pt").input_ids
    tokenized_text = tokenized_text.to(model_img.device)

    # Encode the text using the PTEs extracted from the in-shop cloths
    encoder_hidden_states = encode_text_word_embedding(pipe.text_encoder, tokenized_text,
                                                        word_embeddings,
                                                        num_vstar=num_vstar).last_hidden_state

    # Generate images
    generated_images = pipe(
        image=model_img,
        mask_image=mask_img,
        prompt_embeds=encoder_hidden_states,
        height=256,
        width=256,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_samples,
        generator=generator,
        num_inference_steps=num_inference_steps
    ).images

    # Save images
    source_image = batch['source_image'].to(pipe.device)
    source_parsing = batch['source_parsing'].to(pipe.device)
    source_densepose = batch['source_densepose'].to(pipe.device)
    source_pose_forshow = batch['source_pose_forshow'].to(pipe.device)
    source_cloth = batch['source_cloth_img'].to(pipe.device)
    source_cloth_mask = batch['source_cloth_mask'].to(pipe.device)
    source_preserve_mask_forshow = batch['source_preserve_mask_forshow'].to(pipe.device)
    target_cloth = batch['target_cloth_img'].to(pipe.device)
    target_cloth_mask = batch['target_cloth_mask'].to(pipe.device)
    warped_cloth = batch['warp_feat'].to(pipe.device)
    inpaint_image = batch['inpaint_image'].to(pipe.device)
    source_img_agnostic = batch['source_img_agnostic'].to(pipe.device)
    to_tensor = transforms.Compose([ 
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    simple_image_pil = generated_images[0]
    simple_image = to_tensor(simple_image_pil)

    simple_image = simple_image.unsqueeze(0)
    # to tensor
    combine = torch.cat([
        source_image[0:1, :, :, :],
        getshow(source_parsing, 20),
        getshow(source_densepose, 25),
        cat2rgb(mask_img[0:1, :, :, :]),
        inpaint_image[0:1, :, :, :],
        cat2rgb(source_cloth_mask[0:1, :, :, :]),
        target_cloth[0:1, :, :, :],
        cat2rgb(target_cloth_mask[0:1, :, :, :]),
        warped_cloth[0:1, :, :, :],
        source_img_agnostic[0:1, :, :, :],
        simple_image.cuda(),
        ], 0).squeeze()

    # Create output directory
    target_path = os.path.join(output_dir, f"{save_name}.jpg")
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    utils.save_image(combine, target_path, nrow=int(4), normalize=True, range=(-1, 1), )
    print(f"saved at {target_path}")