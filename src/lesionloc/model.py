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


def resenc_encoder(in_ch, n_stages):
    """nnFoundationCNN / ResEnc-L encoder (nnssl architecture_registry.get_res_enc_l), cut after `n_stages` stages.
    At 2 mm input: stage 0 = 2 mm, stage 1 = 4 mm (64 ch, 32^3), stage 2 = 8 mm (128 ch, 16^3). Returns stage outputs."""
    from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
    enc = ResidualEncoderUNet(
        in_ch, 6, [32, 64, 128, 256, 320, 320], nn.Conv3d, 3, [1, 2, 2, 2, 2, 2], [1, 3, 4, 6, 6, 6], 1,
        [1, 1, 1, 1, 1], conv_bias=True, norm_op=nn.InstanceNorm3d, norm_op_kwargs={"eps": 1e-5, "affine": True},
        nonlin=nn.LeakyReLU, nonlin_kwargs={"inplace": True}).encoder
    enc.stages = enc.stages[:n_stages]
    return enc


def load_nnfoundation(enc, path):
    """Load nnFoundationCNN encoder weights (HF MIC-DKFZ/nnFoundationCNN checkpoint_final.pth) into a cut encoder.
    A 2-channel stem gets zero weights for the mask channel, so step 0 computes the pretrained function."""
    sd = torch.load(path, map_location="cpu", weights_only=True)["network_weights"]
    own = enc.state_dict()
    new = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.") and k[len("encoder."):] in own}
    for k, v in new.items():
        if v.shape != own[k].shape:  # stem conv: (32,1,3,3,3) -> (32,2,3,3,3)
            new[k] = torch.cat([v, v.new_zeros(v.shape[0], own[k].shape[1] - v.shape[1], *v.shape[2:])], 1)
    missing = set(own) - set(new)
    assert not missing, f"not in checkpoint: {sorted(missing)[:5]}"
    enc.load_state_dict(new)


def centre_template(f, k):
    """k^3 cells around the patch centre. The BL lesion sits between the two middle cells; a 2x2x2 mean puts a cell
    exactly on it. Encoder and readout share the grid for BL and FU, so the result is relative and needs no offset."""
    c, h = f.shape[-1] // 2 - 1, k // 2
    return F.avg_pool3d(f, 2, stride=1)[:, :, c - h:c + h + 1, c - h:c + h + 1, c - h:c + h + 1]


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


def local_soft_argmax(logits, coords, r_mm):
    """soft_argmax over only the cells within r_mm (per axis) of the arg-max cell: a second, weaker peak
    can no longer pull the estimate to a point between two candidate lesions."""
    i = logits.flatten(1).argmax(1)
    near = lambda j: (coords[None] - coords[j][:, None]).abs() <= r_mm  # (B,32)
    keep = near(i // GRID ** 2)[:, :, None, None] & near(i // GRID % GRID)[:, None, :, None] & near(i % GRID)[:, None, None, :]
    return soft_argmax(logits.float().masked_fill(~keep, -1e4), coords)


class SiameseNet(nn.Module):
    """x (B,4,64,64,64) = [bl, fu, bl_mask, fu_mask], (z,y,x). fu_mask is only for the loss, never an input.
    Returns (delta_mm (B,3), logits (B,32,32,32) over FU cells)."""

    def __init__(self, width=16, bl_mask=False, k=3, prior_sd=0.0, encoder="small", coarse=False, enc_mask=False):
        """encoder="resenc": nnFoundationCNN stages 0-1 + 1x1 projection (load weights with load_nnfoundation);
        coarse: also stage 2 (8 mm), its correlation map upsampled and added; enc_mask: BL mask as 2nd encoder channel
        (always on for "small" with bl_mask)."""
        super().__init__()
        w = width
        self.bl_mask, self.k, self.coarse = bl_mask, k, coarse
        if encoder == "small":
            self.mask_in, self.in_ch, self.proj = bl_mask, 2, None
            self.encoder = nn.Sequential(  # shared by BL and FU; input = [CT, mask or zeros]
                nn.Conv3d(2, w, 4, stride=2, padding=1), nn.GroupNorm(8, w), nn.ReLU(inplace=True),  # 64 -> 32
                ResBlock(w, w, 1, 1), ResBlock(w, 2 * w, 1, 2), ResBlock(2 * w, 2 * w, 1, 4))  # receptive field ~100 mm
        else:
            self.mask_in, self.in_ch = enc_mask, 2 if enc_mask else 1
            self.encoder = resenc_encoder(self.in_ch, 3 if coarse else 2)
            self.proj = nn.ModuleList([nn.Conv3d(64, 2 * w, 1)] + ([nn.Conv3d(128, 2 * w, 1)] if coarse else []))
        # cosine correlations lie in [-1, 1]; a softmax over 32^3 cells needs a sharper start than 1
        self.log_scale = nn.Parameter(torch.full((3 if coarse else 2,), math.log(10.0)))
        self.register_buffer("coords", cell_mm(), persistent=False)
        # prior_sd > 0: fixed log-Gaussian centre prior on the logits; far cells start penalised (the window is near the lesion)
        r2 = sum(torch.meshgrid(*[cell_mm() ** 2] * 3, indexing="ij"))
        self.register_buffer("log_prior", -r2 / (2 * prior_sd ** 2) if prior_sd else torch.zeros(()), persistent=False)

    def features(self, img, m):
        """L2-normalised feature maps: [4 mm] or, with coarse, [4 mm, 8 mm]."""
        x = torch.cat([img, m], 1) if self.in_ch == 2 else img
        if self.proj is None:
            return [F.normalize(self.encoder(x), dim=1)]
        return [F.normalize(p(s), dim=1) for p, s in zip(self.proj, self.encoder(x)[1:])]  # skip the 2 mm stage 0

    def forward(self, x):
        bl, fu, mbl = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        zero = torch.zeros_like(bl)
        fb, ff = self.features(bl, mbl if self.mask_in else zero), self.features(fu, zero)
        f_bl, f_fu = fb[0], ff[0]
        logits = self.log_scale[0].exp() * xcorr(f_fu, centre_template(f_bl, self.k)) / self.k ** 3
        if self.coarse:  # 8 mm cells, 3^3 template = 24 mm; trilinear x2 maps coarse cell centres onto the 4 mm grid
            lc = self.log_scale[2].exp() * xcorr(ff[1], centre_template(fb[1], 3)) / 27
            logits = logits + F.interpolate(lc[:, None], scale_factor=2, mode="trilinear")[:, 0]
        if self.bl_mask:  # size-invariant template: mean BL feature inside the lesion
            m = F.avg_pool3d(mbl, 4, stride=2, padding=1)  # same grid as the stem
            tm = F.normalize((f_bl * m).sum((2, 3, 4)) / m.sum((2, 3, 4)).clamp_min(1e-6), dim=1)
            logits = logits + self.log_scale[1].exp() * (f_fu * tm[:, :, None, None, None]).sum(1)
        logits = logits + self.log_prior
        return soft_argmax(logits, self.coords), logits
