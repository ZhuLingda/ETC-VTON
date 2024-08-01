# ETC-TON:Enhancing Temporal Consistency of video-based virtual Try-ON with latent diffusion models

![pipeline](assets/pipeline.png)
NOTE: Thanks to [ladi-vton](https://github.com/miccunifi/ladi-vton), [DisCo](https://github.com/Wangt-CN/DisCo) for the inspiration, and any related discussions are welcome.

## Result Visualization
![result visualization](assets/visualization.gif)


## Instructions
- Improved 3D denoising UNet, improved face reconstruction performance and temporal consistency, using [VVT](https://competitions.codalab.org/competitions/23472) dataset, fine-tuning based on [stable-diffusion-2-inpainting](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting)
- Preprocessing dataset based on [Densepose](https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose), [Human-Parsing](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing). The processed files will be uploaded soon
- Based on a pre-trained self-developed clothing deformation model, images of the warped clothing will also be released. The deformation model can be replaced by the existing graph virtual try-on deformation model

## Release Plan
- [x] Test results visualization
- [x] Code for model
- [x] Trained checkpoint
- [x] Code for evaluation on VVT dataset
- [ ] Warping model
- [ ] Code for training
- [ ] Code for general inference

## Model checkpoint
- [VAE](https://drive.google.com/file/d/1AFEIZAtiSvwbdcJxuBAxLfxexY104tpQ/view?usp=drive_link)
- [UNet](https://drive.google.com/file/d/1-q8B2pe9sWEst4859paJjJyXuPmyh03y/view?usp=drive_link)
- [Vision Encoder Projector](https://drive.google.com/file/d/1lC7XyK9DJw7gt6-66BK1cgv_dX5Pb0-z/view?usp=drive_link)

