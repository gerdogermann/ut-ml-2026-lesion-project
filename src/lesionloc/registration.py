"""Whole-scan BL -> FU registration with SimpleITK (Team 1 steps 2 and 4).

Every transform returned here maps BL physical points (mm) to FU physical points, so the FU window centre for a
lesion is tx.TransformPoint(bl_mm). That holds because registration runs with fixed = BL and moving = FU: the
transform SimpleITK optimises maps fixed-space points into moving space (it is what resamples moving onto fixed).

Scans are smoothed and resampled to 2 mm first (whole-body CT at full resolution is slow and memory-hungry, and
a 128 mm window does not need sub-mm accuracy), clamped to [-1000, 1000] HU, and compared with Mattes mutual
information (contrast phase can differ between BL and FU) inside a body mask, so the air and the table are ignored.
"""
import numpy as np
import SimpleITK as sitk

HU_CLAMP = (-1000.0, 1000.0)
GRID_MM = 2.0


def prepare(path, spacing=GRID_MM):
    """Read a CT, smooth, resample to isotropic `spacing` mm, clamp HU. Returns (image, body mask)."""
    img = sitk.ReadImage(str(path), sitk.sitkFloat32)
    img = sitk.SmoothingRecursiveGaussian(img, spacing / 2)  # anti-aliasing before downsampling
    size = [int(round(n * s / spacing)) for n, s in zip(img.GetSize(), img.GetSpacing())]
    img = sitk.Resample(img, size, sitk.Transform(), sitk.sitkLinear, img.GetOrigin(), [spacing] * 3,
                        img.GetDirection(), HU_CLAMP[0], sitk.sitkFloat32)
    img = sitk.Clamp(img, sitk.sitkFloat32, *HU_CLAMP)
    return img, body_mask(img)


def body_mask(img):
    """Largest connected component above -500 HU after an opening (detaches the thin table)."""
    m = sitk.BinaryMorphologicalOpening(sitk.BinaryThreshold(img, -500.0, 1e4), [2] * 3)
    return sitk.RelabelComponent(sitk.ConnectedComponent(m), sortByObjectSize=True) == 1


def centre_mm(img):
    return np.array(img.TransformContinuousIndexToPhysicalPoint([(n - 1) / 2 for n in img.GetSize()]))


def mask_centroid(mask):
    f = sitk.LabelShapeStatisticsImageFilter()
    f.Execute(mask)
    return np.array(f.GetCentroid(1))


def cube_mask(c_mm, half_mm, spacing=4.0):
    """Mask image that is 1 inside the axis-aligned cube c +- half; ITK samples masks by physical point."""
    n = int(np.ceil(2 * half_mm / spacing)) + 1
    m = sitk.Image([n] * 3, sitk.sitkUInt8) + 1
    m.SetSpacing([spacing] * 3)
    m.SetOrigin((np.asarray(c_mm, dtype=float) - half_mm).tolist())
    return m


def _method(fixed_mask, shrink, smooth_mm, sampling, seed=42):
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(sampling, seed)
    if fixed_mask is not None:
        reg.SetMetricFixedMask(fixed_mask)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetShrinkFactorsPerLevel(shrink)
    reg.SetSmoothingSigmasPerLevel(smooth_mm)
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    return reg


def _linear_optimizer(reg, iters):
    # Regular-step descent halves its step whenever the direction flips, so it settles. Plain gradient descent keeps a
    # fixed learning rate: on the phantom in tests/test_team1.py it walked 60 mm away from an exact start.
    reg.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0, minStep=1e-3, numberOfIterations=iters,
                                                 relaxationFactor=0.5)
    reg.SetOptimizerScalesFromPhysicalShift()


def z_range(img):
    """Physical z extent of the scan (axial CT: the index z axis runs head to feet)."""
    zs = [img.TransformIndexToPhysicalPoint([0, 0, k])[2] for k in (0, img.GetSize()[2] - 1)]
    return min(zs), max(zs)


def initial_rigid(fixed, moving, fixed_mask, moving_mask, sweep_mm=400.0, step_mm=16.0):
    """Best start by the metric at 8 mm. BL and FU coordinates are unrelated (offsets of ~1 m) and the two scans
    often cover different body ranges (whole body vs chest), so body centroids line up in x and y but not
    necessarily in z. Candidates: identity, image centres, and body centroids with z shifted by up to
    +-sweep_mm. A shift that leaves the two scan ranges overlapping by less than half the shorter one is not
    tried: with little overlap the metric sees few points and can score a wrong shift well."""
    c = centre_mm(fixed)
    body = mask_centroid(moving_mask) - mask_centroid(fixed_mask)
    shifts = [np.zeros(3), centre_mm(moving) - c] + [body + [0, 0, dz] for dz in
                                                      np.arange(-sweep_mm, sweep_mm + step_mm / 2, step_mm)]
    (f0, f1), (m0, m1) = z_range(fixed), z_range(moving)
    need = 0.5 * min(f1 - f0, m1 - m0)
    ok = [s for s in shifts if min(f1 + s[2], m1) - max(f0 + s[2], m0) >= need] or shifts
    reg = _method(fixed_mask, [1], [0.0], 0.05)
    fs, ms = sitk.Shrink(fixed, [4] * 3), sitk.Shrink(moving, [4] * 3)
    best, best_v = None, np.inf
    for s in ok:
        t = sitk.Euler3DTransform()
        t.SetCenter(c.tolist())
        t.SetTranslation(np.asarray(s, dtype=float).tolist())
        reg.SetInitialTransform(t, inPlace=False)
        try:
            v = reg.MetricEvaluate(fs, ms)
        except RuntimeError:  # Mattes throws when no BL sample lands inside the FU scan: not a start at all
            continue
        if v < best_v:
            best, best_v = t, v
    if best is None:
        raise RuntimeError("no start candidate overlaps the FU scan")
    return best


def rigid(fixed, moving, fixed_mask, moving_mask, iters=200, sampling=0.05):
    """6 DoF (Euler angles + translation). sampling: share of masked BL voxels the metric uses per step."""
    t = sitk.Euler3DTransform(initial_rigid(fixed, moving, fixed_mask, moving_mask))
    reg = _method(fixed_mask, [4, 2, 1], [4.0, 2.0, 0.0], sampling)
    _linear_optimizer(reg, iters)
    reg.SetInitialTransform(t, inPlace=True)
    reg.Execute(fixed, moving)
    return t


def affine(fixed, moving, rigid_tx, fixed_mask, iters=200, sampling=0.05):
    """12 DoF, started from the rigid result."""
    a = sitk.AffineTransform(3)
    a.SetCenter(rigid_tx.GetCenter())
    a.SetMatrix(rigid_tx.GetMatrix())
    a.SetTranslation(rigid_tx.GetTranslation())
    reg = _method(fixed_mask, [4, 2, 1], [4.0, 2.0, 0.0], sampling)
    _linear_optimizer(reg, iters)
    reg.SetInitialTransform(a, inPlace=True)
    reg.Execute(fixed, moving)
    return a


def local_rigid(fixed, moving, init_tx, bl_mm, half_mm=80.0, iters=100):
    """Rigid correction on top of init_tx, with the metric only inside a 160 mm cube around one BL lesion:
    breathing and posture move body regions differently, which one global transform cannot follow."""
    t = sitk.Euler3DTransform()
    t.SetCenter(np.asarray(bl_mm, dtype=float).tolist())
    reg = _method(cube_mask(bl_mm, half_mm), [2, 1], [2.0, 0.0], 0.2)
    _linear_optimizer(reg, iters)
    reg.SetMovingInitialTransform(init_tx)
    reg.SetInitialTransform(t, inPlace=True)
    reg.Execute(fixed, moving)
    # ITK applies the optimised transform first, then the moving initial transform; a composite applies its
    # list from the back, so [init, t] maps a BL point through t and then init_tx, as the metric saw it
    return sitk.CompositeTransform([init_tx, t])


def bspline(fixed, moving, init_tx, fixed_mask, mesh=8, iters=100):
    """Deformable: cubic B-spline with a mesh x mesh x mesh control grid over the BL scan, on top of init_tx."""
    b = sitk.BSplineTransformInitializer(fixed, [mesh] * 3)
    reg = _method(fixed_mask, [4, 2], [4.0, 2.0], 0.02)
    reg.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=iters,
                             maximumNumberOfCorrections=5, maximumNumberOfFunctionEvaluations=1000,
                             costFunctionConvergenceFactor=1e7)
    reg.SetMovingInitialTransform(init_tx)
    reg.SetInitialTransform(b, inPlace=True)
    reg.Execute(fixed, moving)
    return sitk.CompositeTransform([init_tx, b])  # same order as in local_rigid


def move_points(tx, pts_mm):
    return np.array([tx.TransformPoint(np.asarray(p, dtype=float).tolist()) for p in pts_mm])
