import numpy as np
import vapoursynth as vs
import torch
from . import process_image

def resize(
    clip: vs.VideoNode,
    width: int = 256,
    height: int = 256,
    device: str = 'cuda',
) -> vs.VideoNode:

    if clip.format.id != vs.RGBS:
        raise vs.Error('only RGBS format is supported')

    if device not in ['cuda', 'cpu']:
        raise ValueError('device must be either "cuda" or "cpu"')

    width = int(width)
    height = int(height)

    model = process_image.get_model(device=device)

    def liif_resize_frame(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        img = frame_to_array(f[0])
        img = torch.from_numpy(img[0]).to(device)

        with torch.no_grad():
            output = process_image.process_frame(model, img, (height, width), device=device)

        output = torch.unsqueeze(output, 0)
        output = output.cpu().detach().numpy()

        return array_to_frame(output, f[1].copy())

    new_clip = clip.std.BlankClip(width=width, height=height)
    return new_clip.std.ModifyFrame(clips=[clip, new_clip], selector=liif_resize_frame)

def frame_to_array(frame: vs.VideoFrame) -> np.ndarray:
    array = np.stack([np.asarray(frame[plane]) for plane in range(frame.format.num_planes)])
    return np.expand_dims(array, axis=0)

def array_to_frame(array: np.ndarray, frame: vs.VideoFrame) -> vs.VideoFrame:
    array = np.squeeze(array, axis=0)
    for plane in range(frame.format.num_planes):
        np.copyto(np.asarray(frame[plane]), array[plane, :, :])
    return frame
