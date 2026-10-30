"""3D models: (B,C,64,64,64) BL+FU patches -> (B,3) displacement in mm, (x,y,z) order.

LesionNet: residual CNN + linear head (Experiment 1).
SiameseNet: shared encoder, BL template correlated with every FU position, soft-argmax (Team 2 runs 2+).
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

GRID = 32  # SiameseNet feature cells per side: 2 voxels = 4 mm each


class ResBlock(nn.Module):
    """Two 3x3x3 convs with a skip; the first conv uses `stride` (2 halves the resolution)."""

    def __init__(self, cin, cout, stride=2, dilation=1):
        super().__init__()
        d = dilation
        # GroupNorm, not BatchNorm: batches are small (2-16 patches)
        self.body = nn.Sequential(
            nn.Conv3d(cin, cout, 3, stride=stride, padding=d, dilation=d), nn.GroupNorm(8, cout), nn.ReLU(inplace=True),
            nn.Conv3d(cout, cout, 3, padding=d, dilation=d), nn.GroupNorm(8, cout))
        self.skip = nn.Sequential(nn.Conv3d(cin, cout, 1, stride=stride), nn.GroupNorm(8, cout))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.body(x) + self.skip(x))


class LesionNet(nn.Module):
    def __init__(self, width=16, dropout=0.2):
        super().__init__()
        w = width
        self.features = nn.Sequential(ResBlock(2, w), ResBlock(w, 2 * w), ResBlock(2 * w, 4 * w),
                                      ResBlock(4 * w, 8 * w))  # 64 -> 4
        # ponytail: flatten the 4^3 map instead of global pooling; pooling discards *where* features are,
        # and the target is a position. Swap to AdaptiveAvgPool3d(1) + Linear(8w, 3) to compare.
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(8 * w * 4 ** 3, 3))
        nn.init.zeros_(self.head[-1].weight)  # start at the zero (registration) baseline
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        return self.head(self.features(x))  # mm


def cell_mm():
    """mm offset of each feature cell centre from the patch centre, one axis.
    The stem (kernel 4, stride 2, pad 1) centres cell i on voxel 2i+0.5; the patch centre is voxel 31.5."""
    return (torch.arange(GRID) * 2.0 - (GRID - 1)) * 2.0  # -62 .. 62 mm, symmetric


def xcorr(f, t):
    """Per-sample correlation: f (B,C,D,H,W) with template t (B,C,k,k,k) -> (B,D,H,W), same size (k odd)."""
    b = f.shape[0]
    return F.conv3d(f.reshape(1, -1, *f.shape[2:]), t, padding=t.shape[-1] // 2, groups=b)[0]


def soft_argmax(logits, coords):
    """logits (B,Z,Y,X) over cells -> (B,3) expected offset in mm, (x,y,z) order. fp32 for a stable softmax."""
    p = logits.float().flatten(1).softmax(1).view_as(logits)
    return torch.stack([(p.sum((1, 2)) * coords).sum(1),   # x: last axis
                        (p.sum((1, 3)) * coords).sum(1),   # y
                        (p.sum((2, 3)) * coords).sum(1)], 1)  # z


class SiameseNet(nn.Module):
    """x (B,4,64,64,64) = [bl, fu, bl_mask, fu_mask], (z,y,x). fu_mask is only for the loss, never an input.
    Returns (delta_mm (B,3), logits (B,32,32,32) over FU cells)."""

    def __init__(self, width=16, bl_mask=False, k=3):
        super().__init__()
        w = width
        self.bl_mask, self.k = bl_mask, k
        self.encoder = nn.Sequential(  # shared by BL and FU; input = [CT, mask or zeros]
            nn.Conv3d(2, w, 4, stride=2, padding=1), nn.GroupNorm(8, w), nn.ReLU(inplace=True),  # 64 -> 32
            ResBlock(w, w, 1, 1), ResBlock(w, 2 * w, 1, 2), ResBlock(2 * w, 2 * w, 1, 4))  # receptive field ~100 mm
        # cosine correlations lie in [-1, 1]; a softmax over 32^3 cells needs a sharper start than 1
        self.log_scale = nn.Parameter(torch.full((2,), math.log(10.0)))
        self.register_buffer("coords", cell_mm(), persistent=False)

    def forward(self, x):
        bl, fu, mbl = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        zero = torch.zeros_like(bl)
        f_bl = F.normalize(self.encoder(torch.cat([bl, mbl if self.bl_mask else zero], 1)), dim=1)
        f_fu = F.normalize(self.encoder(torch.cat([fu, zero], 1)), dim=1)
        # BL lesion sits between cells 15 and 16; a 2x2x2 mean puts a cell exactly on it (index 15 of 31)
        c, h = GRID // 2 - 1, self.k // 2
        t = F.avg_pool3d(f_bl, 2, stride=1)[:, :, c - h:c + h + 1, c - h:c + h + 1, c - h:c + h + 1]
        logits = self.log_scale[0].exp() * xcorr(f_fu, t) / self.k ** 3
        if self.bl_mask:  # size-invariant template: mean BL feature inside the lesion
            m = F.avg_pool3d(mbl, 4, stride=2, padding=1)  # same grid as the stem
            tm = F.normalize((f_bl * m).sum((2, 3, 4)) / m.sum((2, 3, 4)).clamp_min(1e-6), dim=1)
            logits = logits + self.log_scale[1].exp() * (f_fu * tm[:, :, None, None, None]).sum(1)
        return soft_argmax(logits, self.coords), logits
