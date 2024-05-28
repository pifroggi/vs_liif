import os.path as osp
import numpy as np
import cv2
import torch
from torchvision import transforms
from .models.liif import LIIF
from .models import models as models
from . import utils


def get_model(device='cpu'):
    model_name = 'models/edsr-baseline-liif.pth'
    current_location = osp.dirname(__file__)
    model_path = osp.join(current_location, model_name)
    model = torch.load(model_path, map_location=device)['model']
    model = models.make(model, load_sd=True).to(device)
    return model

def process_frame(model, img, resolution, device='cpu'):
    img = img.to(device)
    if isinstance(resolution, str):
        h, w = map(int, resolution.split(','))
    else:
        h, w = resolution
    coord = utils.make_coord((h, w)).to(device)
    cell = torch.ones_like(coord).to(device)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w
    pred = batched_predict(model, ((img - 0.5) / 0.5).unsqueeze(0),
                           coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000, device=device)[0]
    pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()
    return pred

def batched_predict(model, inp, coord, cell, bsize, device='cpu'):
    with torch.no_grad():
        model.gen_feat(inp, device=device)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql:qr, :], cell[:, ql:qr, :], device=device)
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred