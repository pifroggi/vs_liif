
























# AI resizing for VapourSynth using [LIIF](https://github.com/yinboc/liif) 
Up- or downscaling to arbitrary resolutions and aspect ratios. For example to go from 720x480 to 720x540, or to remove small black borders and stretch, or to downscale with less detail loss. For large upscaling factors dedicated upscaling models are usually better and faster. 

<br />

<p align="center">
  <img src="https://raw.githubusercontent.com/pifroggi/vs_liif/refs/heads/main/README_img.png" width="600" />
</p>

## Installation

```
pip install -U vs_liif
```
* This package requires [PyTorch with CUDA](https://pytorch.org/get-started/locally/).

<br />

## Usage

```python
import vs_liif
clip = vs_liif.resize(clip, width=None, height=None, src_left=0.0, src_top=0.0, src_width=None, src_height=None, batch_size=100000, backend="cuda")
```

__*`clip`*__  
Input clip must be in RGB format.

__*`width`, `height`* (optional)__  
Output width and height in pixel.

__*`src_width`, `src_height`* (optional)__  
Selects a window from the source frame to resize starting from top left. Works like vapoursynths built-in resizers.

__*`src_left`, `src_top`* (optional)__  
Shifts the entire frame, or the window selected by src_width and src_height.  
Allows for subpixel and negative shift. Works like vapoursynths built-in resizers.

__*`batch_size`* (optional)__  
The amount of pixels to process at once. Lower reduces VRAM usage, but may be slower.  
There seems to be a goldilock zone for speed. To find it go up/down in 50000 steps.

__*`backend`* (optional)__  
The backend used to run the LIIF model:
* `cpu` CPU mode *(very slow)*.
* `cuda` GPU mode. Requires an Nvidia GPU *(fast)*.

> [!TIP]
> With large differences between input and output resolution, the model sometimes exhibits a small color shift. If you would like to fix this shift, try [vs_colorfix](https://github.com/pifroggi/vs_colorfix).

<br />

## Benchmarks
Benchmarks were done on a RTX 4090 GPU.

<table>
  <tr>
    <td valign="top">

<table>
  <thead>
    <tr align="center">
      <th colspan="2">720x480</th>
    </tr>
    <tr align="center">
      <th>Resize Factor</th>
      <th>Average FPS</th>
    </tr>
  </thead>
  <tbody>
    <tr align="center">
      <td>0.25x (180x120)</td>
      <td>~45 fps</td>
    </tr>
    <tr align="center">
      <td>0.5x (360x240)</td>
      <td>~34 fps</td>
    </tr>
    <tr align="center">
      <td>to 720x540</td>
      <td>~14 fps</td>
    </tr>
    <tr align="center">
      <td>1.5x (1080x720)</td>
      <td>~8 fps</td>
    </tr>
    <tr align="center">
      <td>2x (1440x960)</td>
      <td>~5 fps</td>
    </tr>
  </tbody>
</table>

</td>
<td valign="top">

<table>
  <thead>
    <tr align="center">
      <th colspan="2">1440x1080</th>
    </tr>
    <tr align="center">
      <th>Resize Factor</th>
      <th>Average FPS</th>
    </tr>
  </thead>
  <tbody>
    <tr align="center">
      <td>0.25x (360x270)</td>
      <td>~8 fps</td>
    </tr>
    <tr align="center">
      <td>0.5x (720x540)</td>
      <td>~6 fps</td>
    </tr>
    <tr align="center">
      <td>1.5x (2160x1620)</td>
      <td>~1.5 fps</td>
    </tr>
    <tr align="center">
      <td>2x (2880x2160)</td>
      <td>~1 fps</td>
    </tr>
  </tbody>
</table>

</td>
  </tr>
</table>

<br />

## Acknowledgements 
Orignal code from "Learning Continuous Image Representation with Local Implicit Image Function" or [LIIF](https://github.com/yinboc/liif).  
VapourSynth functions created with the help of [ViktorThink](https://github.com/ViktorThink). 
