import torch

def make_coord(shape, ranges=None, flatten=True, src_left=0.0, src_top=0.0):
    """ Make coordinates at grid centers with optional subpixel shifts.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        if i == 0:
            shift = src_top * (v1 - v0) / n
        else:
            shift = src_left * (v1 - v0) / n
        seq = v0 + r + shift + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret