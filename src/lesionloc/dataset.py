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
    target_mm float32 (3,) in (x,y,z) mm, has_propagated bool."""

    def __init__(self, root, split_file, split, augment=False, masks=False):
        patients = set(load_split(split_file)[split])
        self.files = sorted(f for f in Path(root).glob("*.npz") if f.name.split("_")[0] in patients)
        self.augment = augment
        self.keys = ["bl", "fu", "bl_mask", "fu_mask"] if masks else ["bl", "fu"]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i])
        x = np.stack([d[k] for k in self.keys]).astype(np.float32)
        t = d["target_mm"].astype(np.float32)
        if self.augment:
            # Random flips. Masks flip with the images. Patch centre is preserved (index i -> 63-i), so flipping an image axis
            # just negates the matching target component. Image axes (z,y,x) are 1,2,3; target is (x,y,z).
            for img_axis, t_idx in ((1, 2), (2, 1), (3, 0)):
                if np.random.rand() < 0.5:
                    x = np.flip(x, img_axis)
                    t[t_idx] = -t[t_idx]
            x = np.ascontiguousarray(x)
        return torch.from_numpy(x), torch.from_numpy(t), bool(d["has_propagated"])
