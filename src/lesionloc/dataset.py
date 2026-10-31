"""PyTorch dataset over the preprocessed lesion npz files, filtered by patient split."""
import json
from pathlib import Path

import numpy as np
import torch
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
    N(0, rejitter_sd) mm per axis, rounded to whole 2 mm voxels, clipped to what fu_big allows. Training only."""

    def __init__(self, root, split_file, split, augment=False, masks=False, rejitter_sd=0.0):
        patients = set(load_split(split_file)[split])
        roots = [root] if isinstance(root, (str, Path)) else root  # list: e.g. [lesions/, augmented lesions/]
        self.files = sorted(f for r in roots for f in Path(r).glob("*.npz") if f.name.split("_")[0] in patients)
        self.augment = augment
        self.keys = ["bl", "fu", "bl_mask", "fu_mask"] if masks else ["bl", "fu"]
        self.rejitter_sd = rejitter_sd

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i])
        a = {k: d[k] for k in self.keys}
        t = d["target_mm"].astype(np.float32)
        if self.rejitter_sd:
            m = (d["fu_big"].shape[0] - 64) // 2  # max shift, voxels
            s = np.clip(np.round(np.random.normal(0, self.rejitter_sd / 2, 3)), -m, m).astype(int)  # (x,y,z) voxels
            win = tuple(slice(m + s[j], m + s[j] + 64) for j in (2, 1, 0))  # arrays are (z,y,x)
            a["fu"] = d["fu_big"][win]
            if "fu_mask" in a:
                a["fu_mask"] = d["fu_mask_big"][win]
            t = (-2.0 * s).astype(np.float32)  # window centre = true FU + 2s mm
        x = np.stack(list(a.values())).astype(np.float32)
        if self.augment:
            # Random flips. Masks flip with the images. Patch centre is preserved (index i -> 63-i), so flipping an image axis
            # just negates the matching target component. Image axes (z,y,x) are 1,2,3; target is (x,y,z).
            for img_axis, t_idx in ((1, 2), (2, 1), (3, 0)):
                if np.random.rand() < 0.5:
                    x = np.flip(x, img_axis)
                    t[t_idx] = -t[t_idx]
            x = np.ascontiguousarray(x)
        return torch.from_numpy(x), torch.from_numpy(t), bool(d["has_propagated"])
