
























# AI resizing for Vapoursynth using [LIIF](https://github.com/yinboc/liif) 
Up- or downscaling to arbitrary resolutions and aspect ratios without ringing or blurriness. For example to go from 720x480 to 720x540, or to remove small black borders and stretch, or to downscale without resizing artifacts.  
For large upscaling factors dedicated upscaling models are likely better.


## Requirements
* [pytorch](https://pytorch.org/) (with cuda)
* `pip install numpy`

## Setup
Put the entire `vs_liif` folder into your vapoursynth scripts folder.  
Or install via pip: `pip install git+https://github.com/pifroggi/vs_liif.git`

## Usage

    import vs_liif
    clip = vs_liif.resize(clip, width=720, height=540, src_left=0.0, src_top=0.0, src_width=None, src_height=None, batch_size=100000, device="cuda", fp16=True)

__*`clip`*__  
Input clip must be in RGBS format.

__*`width`, `height`*__  
Output width and height in pixel.

__*`src_width`, `src_height`* (optional)__  
Selects a window from the source frame to resize starting from top left.  
(Works identical to vapoursynths resizers.)

__*`src_left`, `src_top`* (optional)__  
Shifts the frame, or the window selected by src_width and src_height. Allows for subpixel and negative shift. Out of bound areas will be mirrored.  
(Works identical to vapoursynths resizers.)

__*`batch_size`* (optional)__  
The amount of pixels to process at once. Lower numbers need less VRAM but may be slower. There seems to be a goldilock zone, which can get around 10% extra speed. To find it go up/down in 50000 steps.

__*`device`, `fp16`* (optional)__  
Device values are "cuda" to use with an Nvidia GPU, or "cpu". This will be extremely slow on CPU.  
Fp16 up to doubles speed and lowers VRAM usage if the GPU supports it. Does not work on CPU.

<br />

## Tips & Troubleshooting
> [!TIP]
> With large differences between input and output resolution, the liif model sometimes exhibits a small color shift. If you would like to undo this shift, try this: https://github.com/pifroggi/vs_colorfix

## Benchmarks

| Hardware | Resolution  | Resize Factor   | Average FPS
| -------- | ----------- | --------------- | -----------
|          |             |                 |           
| RTX 4090 | 720x480     | 0.25x (180x120) | ~45 fps
| RTX 4090 | 720x480     | 0.5x (360x240)  | ~34 fps
| RTX 4090 | 720x480     | to 720x540      | ~14 fps
| RTX 4090 | 720x480     | 1.5x (1080x720) | ~8 fps
| RTX 4090 | 720x480     | 2x (1440x960)   | ~5 fps
|          |             |                 |           
| RTX 4090 | 1440x1080   | 0.25x (360x270) | ~8 fps
| RTX 4090 | 1440x1080   | 0.5x (720x540)  | ~6 fps
| RTX 4090 | 1440x1080   | 1.5x (2160x1620)| ~1.5 fps
| RTX 4090 | 1440x1080   | 2x (2880x2160)  | ~1 fps

<br />

## Acknowledgements 
Orignal code from "Learning Continuous Image Representation with Local Implicit Image Function" or [LIIF](https://github.com/yinboc/liif).  
Vapoursynth functions created with the help of [ViktorThink](https://github.com/ViktorThink). 
