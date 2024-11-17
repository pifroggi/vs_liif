import os.path as osp
import vapoursynth as vs
import numpy as np
import torch
import math

from . import utils
from .models.liif import LIIF
from .models import models as models

def resize(
    clip: vs.VideoNode,
    width: int = 720,
    height: int = 540,
    src_left: float = 0.0,
    src_top: float = 0.0,
    src_width: float = None,
    src_height: float = None,
    batch_size: int = 100000,
    device: str = 'cuda',
    fp16: bool = True,
) -> vs.VideoNode:
    
    # checks
    width = int(width)
    height = int(height)

    if not isinstance(clip, vs.VideoNode):
        raise TypeError('This is not a clip.')
    if clip.format.id not in [vs.RGBS, vs.RGBH]:
        raise vs.Error('Input clip must be in RGBS or RGBH format.')
    if device not in ['cuda', 'cpu']:
        raise ValueError('Device must be either "cuda" or "cpu".')
    if fp16 and device == 'cpu':
        raise ValueError('CPU mode does not support fp16.')
    if src_width is None:
        src_width = clip.width
    if src_height is None:
        src_height = clip.height
    if src_width <= 0:
        raise ValueError('Active window must be positive and greater than 0. Check src_width.')
    if src_height <= 0:
        raise ValueError('Active window must be positive and greater than 0. Check src_height.')
    if width <= 0:
        raise ValueError('Resize width must be positive and greater than 0.')
    if height <= 0:
        raise ValueError('Resize height must be positive and greater than 0.')

    dtype = torch.float16 if fp16 else torch.float32
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
    return new_clip.std.CopyFrameProps(prop_src=clip)


def frame_to_array(frame: vs.VideoFrame) -> np.ndarray:
    array = np.stack([np.asarray(frame[plane]) for plane in range(frame.format.num_planes)])
    return array


def array_to_frame(array: np.ndarray, frame: vs.VideoFrame) -> vs.VideoFrame:
    for plane in range(frame.format.num_planes):
        np.copyto(np.asarray(frame[plane]), array[plane, :, :])
    return frame


def get_model(device='cpu', fp16=False):
    model_name = 'models/edsr-baseline-liif.pth'
    current_location = osp.dirname(__file__)
    model_path = osp.join(current_location, model_name)
    loaded_model = torch.load(model_path, map_location=device, weights_only=True)
    model = models.make(loaded_model['model'], load_sd=True).to(device)
    if fp16:
        model = model.half()
    return model


def batched_predict(model, inp, coord, cell, bsize, device):
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
