
























# AI resizing for Vapoursynth using [LIIF](https://github.com/yinboc/liif) 
Up- or downscaling to arbitrary resolutions and aspect ratios. Can for example be used to go from 720x480 to 720x540, or to downscale without resizing artifacts like ringing or blurriness.
For upscaling likely not as good as dedicated upscaling models.

### Requirements
* pip install numpy
* pip install opencv-python
* [pytorch](https://pytorch.org/) 

## Setup
Drop the entire "vs_liif" folder to where you typically load scripts from.

## Usage

    import vs_liif
    clip = vs_liif.resize(clip, width=720, height=540, device="cuda")

__*clip*__  
Input clip must be in RGBS format.

__*width*__  
Output width in pixel. Artifacts appear when resize factor is around 0.25 or lower.

__*height*__  
Output height in pixel. Artifacts appear when resize factor is around 0.25 or lower.

__*device*__  
Possible values are "cuda" to use with an Nvidia GPU, or "cpu". This will be extremely slow on CPU.

## Tips
With large differences between input and output resolution, the liif model sometimes exhibits a small color shift. If you would like to undo this shift, try this: https://github.com/pifroggi/vs_colorfix

## Benchmarks

| Hardware | Resolution  | Resize Factor   | Average FPS
| -------- | ----------- | --------------- | -----------
|          |             |                 |           
| RTX 4090 | 720x480     | 0.25x (180x120) | ~30 fps
| RTX 4090 | 720x480     | 0.5x (360x240)  | ~20 fps
| RTX 4090 | 720x480     | 1.5x (1080x720) | ~5 fps
| RTX 4090 | 720x480     | 2x (1440x960)   | ~3 fps
| RTX 4090 | 720x480     | to 720x540      | ~10 fps
|          |             |                 |           
| RTX 4090 | 1440x1080   | 0.25x (360x270) | ~5 fps
| RTX 4090 | 1440x1080   | 0.5x (720x540)  | ~3 fps
| RTX 4090 | 1440x1080   | 1.5x (2160x1620)| ~0.8 fps
| RTX 4090 | 1440x1080   | 2x (2880x2160)  | ~0.5 fps

## Acknowledgements 
Orignal code from "Learning Continuous Image Representation with Local Implicit Image Function" or [LIIF](https://github.com/yinboc/liif).  
Vapoursynth functions created with the help of [ViktorThink](https://github.com/ViktorThink). 
