import os.path as osp
import vapoursynth as vs
import numpy as np
import math

def resize(
    clip: vs.VideoNode,
    width: int = 720,
    height: int = 540,
    src_left: float = 0.0,
    src_top: float = 0.0,
    src_width: float = None,
    src_height: float = None,
    batch_size: int = 100000,
    backend: str = 'cuda',
) -> vs.VideoNode:
    """Resizes a clip to arbitrary resolutions and aspect ratios using LIIF.

    Args:
        clip: Input clip. Must be in RGB format.
        width: Output width in pixels.
        height: Output height in pixels.
        src_left: Shifts the entire frame horizontally, or the window selected by src_width and src_height. Allows subpixel and negative shifts.
        src_top: Shifts the entire frame vertically, or the window selected by src_width and src_height. Allows subpixel and negative shifts.
        src_width: Width of the source window to resize. Defaults to the input clip width.
        src_height: Height of the source window to resize. Defaults to the input clip height.
        batch_size: Amount of pixels to process at once. Lower values reduce VRAM usage but may be slower.
        backend: Backend used to run the LIIF model.
            - `cpu` = CPU mode (very slow).
            - `cuda` = GPU mode. Requires an Nvidia GPU (fast).
    """
    
    # checks
    width = int(width)
    height = int(height)

    if not isinstance(backend, str):
        raise TypeError('vs_liif.resize: Backend must be a string.')
    device = backend.lower()
    if device not in ['cuda', 'cpu']:
        raise ValueError('vs_liif.resize: Backend must be either "cuda" or "cpu".')

    # checks for torch
    if device == 'cpu':
        try:
            import torch
        except ImportError:
            raise RuntimeError('vs_liif.resize: PyTorch not found. Please install it from https://pytorch.org/. For the CUDA backend specifically, install it with CUDA support.') from None

    if device == 'cuda':
        try:
            import torch
        except ImportError:
            raise RuntimeError('vs_liif.resize: PyTorch not found. Please install a version with CUDA support from: https://pytorch.org/') from None
        if not torch.cuda.is_available():
            raise RuntimeError('vs_liif.resize: The CUDA backend requires PyTorch with CUDA, but the installed version has no CUDA support. Please upgrade: https://pytorch.org/')

    from . import utils

    if not isinstance(clip, vs.VideoNode):
        raise TypeError('vs_liif.resize: Clip must be a vapoursynth clip.')
    if clip.format.id == vs.PresetVideoFormat.NONE or clip.width == 0 or clip.height == 0:
        raise TypeError('vs_liif.resize: Clip must have constant format and dimensions.')
    if clip.format.color_family != vs.RGB:
        raise ValueError('vs_liif.resize: Clip must be in RGB format.')
    if type(batch_size) is not int:
        raise TypeError('vs_liif.resize: Batch size must be an integer.')
    if batch_size < 1:
        raise ValueError('vs_liif.resize: Batch size must be at least 1.')
    if src_width is None:
        src_width = clip.width
    if src_height is None:
        src_height = clip.height
    if src_width <= 0:
        raise ValueError('vs_liif.resize: Active window must be positive and greater than 0. Check src_width.')
    if src_height <= 0:
        raise ValueError('vs_liif.resize: Active window must be positive and greater than 0. Check src_height.')
    if width <= 0:
        raise ValueError('vs_liif.resize: Resize width must be positive and greater than 0.')
    if height <= 0:
        raise ValueError('vs_liif.resize: Resize height must be positive and greater than 0.')

    # defaults
    fp16 = device == 'cuda' and torch.cuda.get_device_capability()[0] >= 7
    dtype = torch.float16 if fp16 else torch.float32

    # convert input to half if fp16/full if not fp16
    format_id = vs.RGBH if fp16 else vs.RGBS
    orig_format = clip.format
    if clip.format.id != format_id:
        clip = vs.core.resize.Point(clip, format=format_id)

    model = get_model(device=device, fp16=fp16)

    # equalize batch size to get maximum speed
    total_pixels = width * height
    num_batches = math.ceil(total_pixels / batch_size)
    bsize = math.ceil(total_pixels / num_batches)

    # compute scaling factors
    scale_x = src_width / clip.width
    scale_y = src_height / clip.height
    scales = [scale_y, scale_x]

    # compute shifts in normalized coordinates
    shift_x = -1 + 2 * (src_left + src_width / 2) / clip.width
    shift_y = -1 + 2 * (src_top + src_height / 2) / clip.height
    shifts = [shift_y, shift_x]

    # create coords, select active window, shift, mirror - needs fp32 to avoid pixelation
    coord = utils.make_coord((height, width), ranges=[(-1, 1), (-1, 1)], flatten=True).to(device, dtype=torch.float32)
    scales_tensor = torch.tensor(scales, device=device, dtype=torch.float32)
    shifts_tensor = torch.tensor(shifts, device=device, dtype=torch.float32)
    coord = coord * scales_tensor + shifts_tensor
    if torch.any((coord < -1) | (coord > 1)):
        coord = utils.mirror_coord(coord, vmin=-1.0, vmax=1.0)
    if fp16:
        coord = coord.half()

    # scale cells
    cell = torch.ones_like(coord, device=device, dtype=dtype)
    cell[:, 0] *= 2 / height
    cell[:, 1] *= 2 / width
    
    # unsqueeze
    cell  = cell.unsqueeze(0)
    coord = coord.unsqueeze(0)

    def _empty_cuda_cache():
        nonlocal model, coord, cell
        model = None
        coord = None
        cell = None
        try:
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # inference
    def liif_resize_frame(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        img = frame_to_array(f[0])
        img = torch.from_numpy(img).to(device, dtype=dtype)
        with torch.amp.autocast(device_type=device, enabled=fp16):
            pred = batched_predict(model, ((img - 0.5) / 0.5).unsqueeze(0), coord, cell, bsize, device)[0]
            pred = ((pred * 0.5 + 0.5).clamp(0, 1).view(height, width, 3).permute(2, 0, 1))
        pred = pred.cpu().numpy()
        return array_to_frame(pred, f[1].copy())

    new_clip = clip.std.BlankClip(width=width, height=height)
    new_clip = new_clip.std.ModifyFrame(clips=[clip, new_clip], selector=liif_resize_frame)
    new_clip = new_clip.std.CopyFrameProps(prop_src=clip)

    # free cache on destroy so reloading previewers doesn't cause issues
    if device == 'cuda':
        vs.register_on_destroy(_empty_cuda_cache)

    if new_clip.format.id != orig_format.id:
        return vs.core.resize.Point(new_clip, format=orig_format.id)

    return new_clip


def frame_to_array(frame: vs.VideoFrame) -> np.ndarray:
    array = np.stack([np.asarray(frame[plane]) for plane in range(frame.format.num_planes)])
    return array


def array_to_frame(array: np.ndarray, frame: vs.VideoFrame) -> vs.VideoFrame:
    for plane in range(frame.format.num_planes):
        np.copyto(np.asarray(frame[plane]), array[plane, :, :])
    return frame


def get_model(device='cpu', fp16=False):
    import torch
    from .models.liif import LIIF
    from .models import models as models

    model_name = 'models/edsr-baseline-liif.pth'
    current_location = osp.dirname(__file__)
    model_path = osp.join(current_location, model_name)
    loaded_model = torch.load(model_path, map_location=device, weights_only=True)
    model = models.make(loaded_model['model'], load_sd=True).to(device)
    if fp16:
        model = model.half()
    return model


def batched_predict(model, inp, coord, cell, bsize, device):
    import torch

    with torch.no_grad():
        model.gen_feat(inp, device=device)
        n = coord.shape[1]
        preds = []
        for ql in range(0, n, bsize):
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql:qr, :], cell[:, ql:qr, :], device=device)
            preds.append(pred)
        pred = torch.cat(preds, dim=1)
    return pred
