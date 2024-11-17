import torch

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers """
    if ranges is None:
        ranges = [(-1, 1), (-1, 1)]
    coord_seqs = []
    for i, n in enumerate(shape):
        v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

def mirror_coord(coord, vmin=-1.0, vmax=1.0):
    """ Mirror out of bounds areas """
    range_size = vmax - vmin
    coord = (coord - vmin) % (2 * range_size)
    coord = torch.where(coord > range_size, 2 * range_size - coord, coord)
    coord = coord + vmin
    return coord
