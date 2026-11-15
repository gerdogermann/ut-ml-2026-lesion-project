import numpy as np
import SimpleITK as sitk

from lesionloc import registration as R
from lesionloc.organs import local_affine, matched, predict_affine, predict_shift
from lesionloc.team1 import index_to_mm


def test_index_to_mm_matches_sitk():
    img = sitk.Image([20, 30, 40], sitk.sitkUInt8)
    img.SetSpacing([0.8, 0.9, 2.5])
    img.SetOrigin([-100.0, 50.0, 7.0])
    img.SetDirection([-1, 0, 0, 0, -1, 0, 0, 0, 1])  # typical LPS CT
    v = np.array([3.5, 12.25, 30.0])
    assert np.allclose(index_to_mm(img, v), img.TransformContinuousIndexToPhysicalPoint(v.tolist()))


def test_local_affine_recovers_affine_and_shift_limit():
    rng = np.random.default_rng(0)
    A = np.eye(3) + 0.05 * rng.normal(size=(3, 3))
    b = np.array([5.0, -12.0, 30.0])
    src = rng.normal(0, 80, (8, 3))
    dst = src @ A.T + b
    w = rng.uniform(0.2, 1, 8)
    M = local_affine(src, dst, w, lam=0.0)
    assert np.allclose(M[:3].T, A) and np.allclose(M[3], b)
    # a huge lam forces A = I: the answer is the weighted mean shift
    M = local_affine(src, dst, w, lam=1e12)
    assert np.allclose(M[3], (w[:, None] * (dst - src)).sum(0) / w.sum(), atol=1e-3)


def test_predict_affine_pure_shift_is_exact():
    c_bl = np.array([[0, 0, 0], [100, 0, 0], [0, 100, 0], [0, 0, 100], [50, 50, 50.0]])
    d = np.array([3.0, -4.0, 10.0])
    p = predict_affine(np.array([20.0, 30, 40]), c_bl, c_bl + d, k=4, sigma=100, lam=1e2)
    assert np.allclose(p, [23, 26, 50])


def test_organ_shift_host_and_fallback():
    org_bl = {1: (np.array([0.0, 0, 0]), 500, False), 2: (np.array([200.0, 0, 0]), 500, False),
              3: (np.array([10.0, 0, 0]), 500, True)}  # 3 is cut by the scan range: never used
    org_fu = {1: (np.array([0.0, 0, 5]), 500, False), 2: (np.array([200.0, 0, 20]), 500, False),
              3: (np.array([10.0, 0, 99]), 500, False)}
    labels, c_bl, c_fu = matched(org_bl, org_fu)
    assert labels.tolist() == [1, 2]
    bl = np.array([150.0, 0, 0])
    assert np.allclose(predict_shift(bl, 1, labels, c_bl, c_fu)[0], [150, 0, 5])   # host organ 1
    assert np.allclose(predict_shift(bl, 0, labels, c_bl, c_fu)[0], [150, 0, 20])  # nearest: organ 2
    assert np.allclose(predict_shift(bl, 3, labels, c_bl, c_fu)[0], [150, 0, 20])  # host cut: nearest
    assert predict_shift(bl, 1, labels[:0], c_bl[:0], c_fu[:0])[1] == -1


def _phantom(shift_mm):
    """Body (0 HU) with a lung (-800) and a bone (+700) block on 2 mm voxels; content moved by shift_mm."""
    n = 64
    z, y, x = np.mgrid[:n, :n, :n] * 2.0 - 63.0  # mm, centred
    x, y, z = x - shift_mm[0], y - shift_mm[1], z - shift_mm[2]
    a = np.full((n, n, n), -1000.0, np.float32)
    a[(x / 50) ** 2 + (y / 40) ** 2 + (z / 55) ** 2 < 1] = 0
    a[(np.abs(x + 15) < 12) & (np.abs(y) < 15) & (np.abs(z - 10) < 20)] = -800
    a[(np.abs(x - 20) < 6) & (np.abs(y + 10) < 8) & (np.abs(z + 15) < 10)] = 700
    img = sitk.GetImageFromArray(a)
    img.SetSpacing([2.0] * 3)
    img.SetOrigin([-63.0] * 3)
    return img


def test_rigid_maps_bl_points_to_fu():
    d = np.array([6.0, -4.0, 8.0])
    fixed, moving = _phantom(np.zeros(3)), _phantom(d)  # FU content sits d further along
    tx = R.rigid(fixed, moving, R.body_mask(fixed), R.body_mask(moving), sampling=0.5)  # tiny image
    p = np.array([20.0, -10.0, -15.0])  # the bone block in BL
    assert np.linalg.norm(R.move_points(tx, [p])[0] - (p + d)) < 2.0
