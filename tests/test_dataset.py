import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from lesionloc.dataset import LesionDataset, load_split

EXAMPLES = Path(__file__).parents[1] / "data" / "example_lesions"


def write_split(tmp_path, split):
    f = tmp_path / "split.json"
    f.write_text(json.dumps(split))
    return f


def test_loader_shapes(tmp_path):
    split = write_split(tmp_path, {"train": ["006f52e910", "01161aaa0b"], "val": [], "test": []})
    x, t, p = next(iter(DataLoader(LesionDataset(EXAMPLES, split, "train"), batch_size=2)))
    assert x.shape == (2, 2, 64, 64, 64) and x.dtype == torch.float32
    assert t.shape == (2, 3) and p.shape == (2,)


def test_split_filters_patients(tmp_path):
    split = write_split(tmp_path, {"train": ["006f52e910"], "val": ["01161aaa0b"], "test": []})
    assert [f.name for f in LesionDataset(EXAMPLES, split, "val").files] == ["01161aaa0b_1.npz"]


def test_patient_overlap_raises(tmp_path):
    split = write_split(tmp_path, {"train": ["006f52e910"], "val": [], "test": ["006f52e910"]})
    with pytest.raises(AssertionError):
        load_split(split)


def test_flip_negates_matching_target(tmp_path):
    split = write_split(tmp_path, {"train": ["006f52e910"]})
    plain = LesionDataset(EXAMPLES, split, "train")
    aug = LesionDataset(EXAMPLES, split, "train", augment=True)
    x0, t0, _ = plain[0]
    np.random.seed(0)
    for _ in range(10):
        x, t, _ = aug[0]
        # each target component is either unchanged or negated, and the image matches the same flips
        flips = [d for d, ti in ((1, 2), (2, 1), (3, 0)) if t[ti] == -t0[ti] and t0[ti] != 0]
        assert torch.equal(x, x0.flip(flips) if flips else x0)
