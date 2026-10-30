import numpy as np
import torch
import torch.nn.functional as F

from lesionloc.dataset import LesionDataset
from lesionloc.losses import gaussian_target, team2_loss
from lesionloc.model import GRID, SiameseNet, cell_mm, soft_argmax


def test_output_shapes():
    delta, logits = SiameseNet(bl_mask=True)(torch.rand(2, 4, 64, 64, 64))
    assert delta.shape == (2, 3) and logits.shape == (2, GRID, GRID, GRID)
    assert torch.isfinite(delta).all()


def test_soft_argmax_axis_order():
    logits = torch.full((1, GRID, GRID, GRID), -1e4)
    logits[0, 5, 10, 20] = 0  # (z, y, x) cell
    c = cell_mm()
    assert torch.allclose(soft_argmax(logits, c)[0], torch.stack([c[20], c[10], c[5]]))


def test_fu_mask_is_not_an_input():
    torch.manual_seed(0)
    net, x = SiameseNet(bl_mask=True).eval(), torch.rand(1, 4, 64, 64, 64)
    y = x.clone()
    y[:, 3] = 1 - y[:, 3]
    assert torch.equal(net(x)[0], net(y)[0])


def blob(cx, cy, cz):
    """Gaussian blob (sd 3 voxels) at voxel (x,y,z) on smooth noise, (1,1,64,64,64)."""
    z, y, x = torch.meshgrid(*[torch.arange(64.0)] * 3, indexing="ij")
    v = torch.exp(-((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) / 18)
    return (v + 0.3 * F.avg_pool3d(torch.rand(1, 1, 64, 64, 64), 5, 1, 2)[0, 0])[None, None]


def test_untrained_correlation_peaks_at_shift():
    # FU lesion 7 voxels (14 mm) along +x, 6 voxels (12 mm) along -z: even random shared features match there
    torch.manual_seed(0)
    x = torch.cat([blob(31.5, 31.5, 31.5), blob(38.5, 31.5, 25.5), torch.zeros(1, 2, 64, 64, 64)], 1)
    _, logits = SiameseNet().eval()(x)
    z, y, xx = np.unravel_index(logits[0].argmax().item(), logits[0].shape)
    c = cell_mm()
    assert c[xx] == 14.0 and abs(c[y]) <= 2.0 and abs(c[z] + 12.0) <= 2.0  # within half a 4 mm cell


def test_losses_prefer_the_truth():
    c = cell_mm()
    t = torch.tensor([[14.0, -6.0, 2.0]])
    fu_mask = torch.zeros(1, 1, 64, 64, 64)
    fu_mask[0, 0, 30:36, 26:32, 36:44] = 1  # around voxel (x=39.5, y=28.5, z=32.5) ~ t
    good = torch.log(gaussian_target(t, torch.tensor([4.0]), c) + 1e-12)
    bad = good.roll(8, dims=3)
    lg, tg = team2_loss(soft_argmax(good, c), good, t, fu_mask, c, 1.0, 0.5)
    lb, tb = team2_loss(soft_argmax(bad, c), bad, t, fu_mask, c, 1.0, 0.5)
    assert lg < lb and all(tg[k] < tb[k] for k in tg)
    assert tg["mask"] < 0.1


def test_loss_backward():
    net = SiameseNet(bl_mask=True)
    x = torch.rand(2, 4, 64, 64, 64)
    x[:, 3] = 0
    x[:, 3, 28:36, 28:36, 28:36] = 1
    delta, logits = net(x)
    loss, _ = team2_loss(delta, logits, torch.zeros(2, 3), x[:, 3:4], net.coords, 1.0, 0.5)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


def test_dataset_masks_flip_with_images(tmp_path):
    (tmp_path / "split.json").write_text('{"train": ["aaaaaaaaaa"]}')
    rng = np.random.default_rng(0)
    img = rng.random((64, 64, 64)).astype(np.float16)
    np.savez(tmp_path / "aaaaaaaaaa_1.npz", bl=img, fu=img, bl_mask=(img > 0.5).astype(np.uint8),
             fu_mask=(img > 0.5).astype(np.uint8), target_mm=np.float32([1, 2, 3]), has_propagated=True)
    ds = LesionDataset(tmp_path, tmp_path / "split.json", "train", augment=True, masks=True)
    np.random.seed(1)
    for _ in range(5):
        x, _, _ = ds[0]
        assert x.shape == (4, 64, 64, 64)
        assert torch.equal(x[2], (x[0] > 0.5).float()) and torch.equal(x[3], x[2])


def test_rejitter_target_matches_window(tmp_path):
    (tmp_path / "split.json").write_text('{"train": ["aaaaaaaaaa"]}')
    z, y, x = np.meshgrid(*[np.arange(88)] * 3, indexing="ij")
    zero = np.zeros((64, 64, 64), np.float16)
    np.savez(tmp_path / "aaaaaaaaaa_1.npz", bl=zero, fu=zero, bl_mask=zero, fu_mask=zero,
             fu_big=(x + 100 * y).astype(np.float32), fu_mask_big=z.astype(np.uint8),  # index-coded: window is readable
             target_mm=np.float32([1, 2, 3]), has_propagated=True)
    ds = LesionDataset(tmp_path, tmp_path / "split.json", "train", masks=True, rejitter_sd=8)
    np.random.seed(0)
    seen = set()
    for _ in range(20):
        xx, t, _ = ds[0]
        corner = xx[1, 0, 0, 0].item()
        s = np.array([corner % 100, corner // 100, xx[3, 0, 0, 0].item()]) - 12  # window shift, voxels (x,y,z)
        assert np.all(np.abs(s) <= 12) and torch.equal(t, torch.tensor(-2.0 * s, dtype=torch.float32))
        seen.add(tuple(s))
    assert len(seen) > 10  # fresh offset per load
