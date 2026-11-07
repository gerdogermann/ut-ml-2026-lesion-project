"""PyTorch dataset over the preprocessed lesion npz files, filtered by patient split."""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def load_split(split_file):
    split = json.loads(Path(split_file).read_text())
    names = list(split)
    for i, a in enumerate(names):  # course requirement: no patient in two splits
        for b in names[i + 1:]:
            overlap = set(split[a]) & set(split[b])
            assert not overlap, f"patients in both {a} and {b}: {sorted(overlap)[:5]}"
    return split


class LesionDataset(Dataset):
    """Returns (x, target_mm, has_propagated): x float32 (2,64,64,64) = [bl, fu] in (z,y,x),
    or (4,64,64,64) = [bl, fu, bl_mask, fu_mask] with masks=True (Team 2 data),
    target_mm float32 (3,) in (x,y,z) mm, has_propagated bool.
    rejitter_sd > 0: ignore the stored FU jitter and cut fu / fu_mask from fu_big at a fresh offset,
    N(0, rejitter_sd) mm per axis, rounded to whole 2 mm voxels, clipped to what fu_big allows. Training only.
    With rejitter_sd, the offset is drawn per sample from a mixture with weights mix = (gauss, offsets, wide):
    N(0, rejitter_sd), a bootstrapped row of `offsets` ((N,3) real window residuals, true FU - centre, mm),
    or uniform in +-wide_mm. p_distractor: instead centre the window on, or between, a neighbouring FU lesion
    within 40 mm (needs nbr_mm). other: 5th channel = other FU lesions' mask (needs fu_other_big).
    p_growth: radially grow / shrink the FU lesion (radius x 0.67-2, log-uniform), target re-centred on the warped mask."""

    def __init__(self, root, split_file, split, augment=False, masks=False, rejitter_sd=0.0, offsets=None,
                 mix=(1.0, 0.0, 0.0), wide_mm=0.0, p_distractor=0.0, other=False, p_growth=0.0):
        patients = set(load_split(split_file)[split])
        roots = [root] if isinstance(root, (str, Path)) else root  # list: e.g. [lesions/, augmented lesions/]
        self.files = sorted(f for r in roots for f in Path(r).glob("*.npz") if f.name.split("_")[0] in patients)
        self.augment = augment
        self.keys = ["bl", "fu", "bl_mask", "fu_mask"] if masks else ["bl", "fu"]
        self.rejitter_sd, self.offsets, self.wide_mm = rejitter_sd, offsets, wide_mm
        self.mix = np.asarray(mix, float) / np.sum(mix)
        self.p_distractor, self.other, self.p_growth = p_distractor, other, p_growth
        assert offsets is not None or not self.mix[1], "offset mixture needs offsets"
        assert rejitter_sd or not (other or p_distractor or self.mix[0] < 1), "these options re-cut from fu_big"
        assert masks or not (other or p_growth)

    def window_offset(self, d):
        """Window centre minus true FU lesion, mm (x,y,z)."""
        nbr = d["nbr_mm"] if self.p_distractor else np.zeros((0, 3))
        nbr = nbr[np.linalg.norm(nbr, axis=1) <= 40]
        if len(nbr) and np.random.rand() < self.p_distractor:
            return nbr[np.random.randint(len(nbr))] * np.random.uniform(0.5, 1.0)
        k = np.random.choice(3, p=self.mix)
        if k == 1:
            return -self.offsets[np.random.randint(len(self.offsets))]
        return np.random.normal(0, self.rejitter_sd, 3) if k == 0 else np.random.uniform(-self.wide_mm, self.wide_mm, 3)

    def sample_weights(self, w_grown):
        """Per-file sampling weight: w_grown for FU/BL volume ratio > 4 or MERGING, else 1."""
        def grown(f):
            with np.load(f) as d:
                return bool(d["vol_ratio"] > 4 or d["merging"])
        return np.array([w_grown if grown(f) else 1.0 for f in self.files])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i])
        a = {k: d[k] for k in self.keys}
        t = d["target_mm"].astype(np.float32)
        if self.rejitter_sd:
            m = (d["fu_big"].shape[0] - 64) // 2  # max shift, voxels
            s = np.clip(np.round(self.window_offset(d) / 2), -m, m).astype(int)  # (x,y,z) voxels
            win = tuple(slice(m + s[j], m + s[j] + 64) for j in (2, 1, 0))  # arrays are (z,y,x)
            a["fu"] = d["fu_big"][win]
            if "fu_mask" in a:
                a["fu_mask"] = d["fu_mask_big"][win]
            if self.other:
                a["fu_other"] = d["fu_other_big"][win]
            t = (-2.0 * s).astype(np.float32)  # window centre = true FU + 2s mm
        x = np.stack(list(a.values())).astype(np.float32)
        if self.p_growth and np.random.rand() < self.p_growth and x[3].any():
            x, t = grow(x, t, np.exp(np.random.uniform(np.log(0.67), np.log(2.0))))
        if self.augment:
            # Random flips. Masks flip with the images. Patch centre is preserved (index i -> 63-i), so flipping an image axis
            # just negates the matching target component. Image axes (z,y,x) are 1,2,3; target is (x,y,z).
            for img_axis, t_idx in ((1, 2), (2, 1), (3, 0)):
                if np.random.rand() < 0.5:
                    x = np.flip(x, img_axis)
                    t[t_idx] = -t[t_idx]
            x = np.ascontiguousarray(x)
        return torch.from_numpy(x), torch.from_numpy(t), bool(d["has_propagated"])


def grow(x, t, g):
    """Radial growth (g > 1) or shrink (g < 1) of the FU lesion about its centre, LesionLocator-style.
    x (C,64,64,64) = [bl, fu, bl_mask, fu_mask, (fu_other)], t (3,) mm (x,y,z). Inside g * r_eq the lesion is scaled
    by g; a smoothstep over 2 r_eq + 2 voxels blends back to identity (monotone for g in [0.67, 2]). BL is untouched.
    The target moves with the warped mask's centre of mass; an emptied mask leaves x, t unchanged."""
    r = (3 * x[3].sum() / (4 * np.pi)) ** (1 / 3)  # equal-volume radius, voxels
    c = 31.5 + t[::-1] / 2  # lesion centre, voxel (z,y,x)
    p = np.stack(np.meshgrid(*[np.arange(64.0)] * 3, indexing="ij"), -1) - c
    u = np.clip((np.linalg.norm(p, axis=-1, keepdims=True) - g * r) / (2 * r + 2), 0, 1)
    src = c + p * (1 / g + (1 - 1 / g) * u * u * (3 - 2 * u))  # inverse map: output voxel -> input voxel
    grid = torch.from_numpy((src[..., ::-1] / 31.5 - 1)[None].astype(np.float32))  # (1,64,64,64,3), (x,y,z) in [-1,1]
    warp = lambda ch, mode: F.grid_sample(torch.from_numpy(np.ascontiguousarray(x[None, ch])), grid, mode=mode,
                                          padding_mode="border", align_corners=True)[0].numpy()
    fu_masks = list(range(3, len(x)))
    y = x.copy()
    y[[1]], y[fu_masks] = warp([1], "bilinear"), warp(fu_masks, "nearest")
    if not y[3].any():
        return x, t
    com = lambda m: np.argwhere(m).mean(0)[::-1] * 2.0  # mm (x,y,z)
    return y, (t + com(y[3]) - com(x[3])).astype(np.float32)
