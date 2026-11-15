import json

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")
from lesionloc.dataset import LesionDataset  # noqa: E402
from lesionloc.preprocess import com_offset_mm, mask_patch, patch, random_rotation  # noqa: E402


def scan():
    """Sphere (label 3) in a scan with 1 x 1 x 2.5 mm voxels and flipped x/y axes; returns img, mask, centre mm."""
    shape = (60, 120, 100)  # z, y, x
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    s = (x - 40) ** 2 + (y - 50) ** 2 + ((z - 30) * 2.5) ** 2 <= 100
    out = []
    for a in (np.where(s, 100, -800).astype(np.int16), (3 * s).astype(np.uint16)):
        im = sitk.GetImageFromArray(a)
        im.SetSpacing((1.0, 1.0, 2.5))
        im.SetDirection((-1, 0, 0, 0, -1, 0, 0, 0, 1))
        im.SetOrigin((50.0, 60.0, -100.0))
        out.append(im)
    return out[0], out[1], np.array(out[0].TransformContinuousIndexToPhysicalPoint((40, 50, 30)))


def test_random_rotation_is_a_rotation():
    rng = np.random.default_rng(0)
    for _ in range(20):
        r = random_rotation(rng, 15)
        assert np.allclose(r @ r.T, np.eye(3)) and np.isclose(np.linalg.det(r), 1)
        assert np.degrees(np.arccos((np.trace(r) - 1) / 2)) <= 15 + 1e-6


def test_identity_patch_unchanged():
    img, _, c = scan()
    old = sitk.Resample(img, [64] * 3, sitk.Transform(), sitk.sitkLinear, (c - 2.0 * 63 / 2).tolist(),
                        [2.0] * 3, (1, 0, 0, 0, 1, 0, 0, 0, 1), -1024.0, sitk.sitkFloat32)
    a = np.clip(sitk.GetArrayFromImage(old), -1000, 400)
    assert np.array_equal(patch(img, c), ((a + 1000) / 1400).astype(np.float16))


def test_rotated_scaled_patch_stays_centred():
    _, mask, c = scan()
    rng = np.random.default_rng(1)
    for _ in range(5):
        m = mask_patch(mask, c, 3, 64, random_rotation(rng, 15), 2.0 * rng.uniform(0.9, 1.1))
        assert m.any() and np.linalg.norm(com_offset_mm(m)) < 1.0


def test_dataset_reads_several_roots(tmp_path):
    for d, name in (("a", "aaaaaaaaaa_1"), ("b", "aaaaaaaaaa_1_a0"), ("b", "bbbbbbbbbb_1_a0")):
        (tmp_path / d).mkdir(exist_ok=True)
        np.savez(tmp_path / d / f"{name}.npz", x=0)
    (tmp_path / "split.json").write_text(json.dumps({"train": ["aaaaaaaaaa"], "val": ["bbbbbbbbbb"]}))
    ds = LesionDataset([tmp_path / "a", tmp_path / "b"], tmp_path / "split.json", "train")
    assert [f.name for f in ds.files] == ["aaaaaaaaaa_1.npz", "aaaaaaaaaa_1_a0.npz"]
