"""Offset mixture, distractor windows, growth warp, new loss terms and checkpoint ensembles (Team 2 next runs)."""
import json

import numpy as np
import pytest
import torch

from lesionloc.dataset import LesionDataset, grow
from lesionloc.losses import team2_loss
from lesionloc.model import SiameseNet, cell_mm
from lesionloc.prediction import PatchPredictor

M = 28  # 120^3 fu_big: room for +-56 mm


def sphere(n, c_zyx, r):
    z, y, x = np.ogrid[:n, :n, :n]
    return ((z - c_zyx[0]) ** 2 + (y - c_zyx[1]) ** 2 + (x - c_zyx[2]) ** 2 <= r * r).astype(np.uint8)


@pytest.fixture
def ds_dir(tmp_path):
    (tmp_path / "split.json").write_text('{"train": ["aaaaaaaaaa"]}')
    n, c = 64 + 2 * M, (64 + 2 * M - 1) / 2
    zero = np.zeros((64,) * 3, np.float16)
    other = sphere(n, (c, c, c + 10), 3)  # neighbour 20 mm along +x
    for i, ratio in ((1, 1.0), (2, 9.0)):
        np.savez(tmp_path / f"aaaaaaaaaa_{i}.npz", bl=zero, fu=zero, bl_mask=zero, fu_mask=zero,
                 fu_big=np.zeros((n,) * 3, np.float16), fu_mask_big=sphere(n, (c, c, c), 4), fu_other_big=other,
                 nbr_mm=np.float32([[20, 0, 0], [0, 60, 0]]), target_mm=np.float32([0, 0, 0]), has_propagated=True,
                 vol_ratio=np.float32(ratio), merging=False)
    return tmp_path


def test_offset_mixture_bootstraps_residual_vectors(ds_dir):
    ds = LesionDataset(ds_dir, ds_dir / "split.json", "train", masks=True, rejitter_sd=8,
                       offsets=np.float32([[6, -4, 30]]), mix=(0, 1, 0))
    _, t, _ = ds[0]
    assert torch.equal(t, torch.tensor([6.0, -4.0, 30.0]))  # target = true FU - window centre = the residual
    wide = LesionDataset(ds_dir, ds_dir / "split.json", "train", masks=True, rejitter_sd=8, mix=(0, 0, 1), wide_mm=56)
    ts = torch.stack([wide[0][1] for _ in range(200)])
    assert ts.abs().max() <= 56 and ts.abs().max() > 40


def test_distractor_window_and_other_channel(ds_dir):
    ds = LesionDataset(ds_dir, ds_dir / "split.json", "train", masks=True, rejitter_sd=8, p_distractor=1, other=True)
    for _ in range(20):
        x, t, _ = ds[0]
        assert x.shape[0] == 5
        assert -20 <= t[0] <= -10 and t[1] == t[2] == 0  # window on/between the 20 mm neighbour, not the 60 mm one
        assert x[4].any()  # the neighbour is in the window


def test_sample_weights(ds_dir):
    ds = LesionDataset(ds_dir, ds_dir / "split.json", "train", masks=True)
    assert ds.sample_weights(3).tolist() == [1.0, 3.0]


def test_grow_scales_volume_and_keeps_centre():
    c = np.array([31.5, 25.0, 40.0])  # off-centre lesion, voxel (z,y,x)
    t = ((c[::-1] - 31.5) * 2).astype(np.float32)  # its offset, mm (x,y,z)
    x = np.zeros((4, 64, 64, 64), np.float32)
    x[3] = x[1] = sphere(64, c, 5)
    y, t2 = grow(x, t, 2.0)
    assert 5 < y[3].sum() / x[3].sum() < 11  # radius x2 -> volume ~x8
    assert np.abs(t2 - t).max() < 1.0
    assert np.array_equal(y[0], x[0])  # BL untouched
    y, _ = grow(x, t, 0.67)
    assert 0.15 < y[3].sum() / x[3].sum() < 0.6


def test_uniform_target_and_other_penalty():
    coords = cell_mm()
    fu = torch.zeros(1, 1, 64, 64, 64)
    fu[..., 16:48, 16:48, 16:48] = 1  # r_FU ~ 40 mm
    t = torch.zeros(1, 3)
    peak = torch.full((1, 32, 32, 32), -10.0)
    peak[0, 10, 10, 10] = 10.0  # a sharp peak inside the mask but off its centre
    flat = torch.where(torch.nn.functional.max_pool3d(fu, 4, 2, 1)[:, 0] > 0, 0.0, -10.0)
    loss = lambda lg, **k: team2_loss(t, lg, t, fu, coords, lambda_mask=1, **k)[1]["mask"]
    assert loss(peak) < loss(flat)  # old term: any peak inside the mask is fine
    assert loss(flat, uniform_r=10) < loss(peak, uniform_r=10)  # new term: spread over the mask
    other = torch.zeros_like(fu)
    other[..., 18:22, 18:22, 18:22] = 1  # overlaps the own mask: must not be penalised
    other[..., 0:8, 0:8, 0:8] = 1
    on_other = torch.full((1, 32, 32, 32), -10.0)
    on_other[0, 1, 1, 1] = 10.0
    pen = lambda lg: team2_loss(t, lg, t, fu, coords, fu_other=other, lambda_other=1)[1]["other"]
    assert pen(on_other) > 5 and pen(peak) < 1e-3


def test_predictor_rebuilds_architecture_and_ensembles(tmp_path):
    torch.manual_seed(0)
    for d, args in (("a", dict(bl_mask=True, template_k=3)), ("b", dict(bl_mask=True, template_k=5))):
        (tmp_path / d).mkdir()
        (tmp_path / d / "args.json").write_text(json.dumps(args))
        torch.save(SiameseNet(bl_mask=True, k=args["template_k"]).state_dict(), tmp_path / d / "last.pt")
    p = PatchPredictor([tmp_path / "a" / "last.pt", tmp_path / "b" / "last.pt"])
    assert [m.k for m in p.models] == [3, 5]
    bl = np.random.rand(64, 64, 64).astype(np.float32)
    out = p.predict(bl, bl, (bl > 0.5).astype(np.float32))
    assert out["tta"].shape == out["plain"].shape == (3,) and out["spread"] >= 0
