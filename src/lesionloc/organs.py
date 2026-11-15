"""Organs as landmarks (Team 1 step 3), from TotalSegmentator multi-label masks of the BL and FU scans.

Each organ's centroid in BL and FU gives one matched point pair. Two ways to move a BL lesion with them:
  shift   add the BL -> FU centroid shift of the organ the lesion sits in, or of the nearest organ
  affine  fit a local affine to the K nearest organs, weighted by distance, pulled towards a pure shift
An organ that touches the scan border is not used: when the scan range cuts it off, its centroid moves with the
cut, not with the patient.
"""
import numpy as np
import SimpleITK as sitk

MIN_VOXELS = 50  # smaller organ masks give noisy centroids


def scan_organs(path, lesions_mm=()):
    """Centroids of one TotalSegmentator mask. Returns ({label: (centroid_mm, n_voxels, cut)}, host labels),
    one host label per lesion position: the organ at that point, else the most common organ within ~5 voxels, else 0."""
    img = sitk.ReadImage(str(path))
    f = sitk.LabelShapeStatisticsImageFilter()
    f.Execute(sitk.Cast(img, sitk.sitkUInt16))
    size = np.array(img.GetSize())
    organs = {}
    for l in f.GetLabels():
        bb = np.array(f.GetBoundingBox(l))  # index start (x,y,z), then size
        cut = bool((bb[:3] == 0).any() or (bb[:3] + bb[3:] >= size).any())
        organs[int(l)] = (np.array(f.GetCentroid(l)), int(f.GetNumberOfPixels(l)), cut)
    a = sitk.GetArrayViewFromImage(img)  # (z,y,x)
    hosts = []
    for p in lesions_mm:
        x, y, z = img.TransformPhysicalPointToIndex(np.asarray(p, dtype=float).tolist())
        if not (0 <= x < size[0] and 0 <= y < size[1] and 0 <= z < size[2]):
            hosts.append(0)
            continue
        h = int(a[z, y, x])
        if h == 0:  # a tumour is often left out of its organ's mask: look around it
            nb = a[max(z - 5, 0):z + 6, max(y - 5, 0):y + 6, max(x - 5, 0):x + 6].ravel()
            nb = nb[nb > 0]
            h = int(np.bincount(nb).argmax()) if nb.size else 0
        hosts.append(h)
    return organs, hosts


def matched(org_bl, org_fu):
    """Labels usable as landmarks: in both scans, big enough, not cut by either scan range.
    Returns (labels, BL centroids, FU centroids)."""
    ls = sorted(l for l in org_bl if l in org_fu
                and min(org_bl[l][1], org_fu[l][1]) >= MIN_VOXELS and not (org_bl[l][2] or org_fu[l][2]))
    return (np.array(ls, dtype=int), np.array([org_bl[l][0] for l in ls]).reshape(-1, 3),
            np.array([org_fu[l][0] for l in ls]).reshape(-1, 3))


def predict_shift(bl_mm, host, labels, c_bl, c_fu):
    """BL position + centroid shift of the host organ, or of the organ with the nearest BL centroid.
    Returns (FU position, label used); label -1 and the BL position when no organ is usable."""
    if not len(labels):
        return np.asarray(bl_mm, dtype=float), -1
    hit = np.nonzero(labels == host)[0]
    i = hit[0] if host and len(hit) else int(np.argmin(np.linalg.norm(c_bl - bl_mm, axis=1)))
    return bl_mm + c_fu[i] - c_bl[i], int(labels[i])


def local_affine(src, dst, w, lam):
    """Weighted least squares dst ~ A src + b with lam * ||A - I||^2 added; returns the (4, 3) matrix [A^T; b].
    lam -> 0: plain weighted affine (needs >= 4 points not in a plane); lam -> inf: weighted mean shift."""
    X = np.c_[src, np.ones(len(src))]
    D = np.diag([lam, lam, lam, 0.0])
    M0 = np.r_[np.eye(3), np.zeros((1, 3))]
    return np.linalg.solve(X.T @ (w[:, None] * X) + D, X.T @ (w[:, None] * dst) + D @ M0)


def predict_affine(bl_mm, c_bl, c_fu, k, sigma, lam):
    """Local affine from the k organs with the nearest BL centroids, weights exp(-distance / sigma).
    Coordinates are taken relative to the lesion, so A does not act on it and the prediction is bl + b."""
    if not len(c_bl):
        return np.asarray(bl_mm, dtype=float)
    d = np.linalg.norm(c_bl - bl_mm, axis=1)
    i = np.argsort(d)[:k]
    M = local_affine(c_bl[i] - bl_mm, c_fu[i] - bl_mm, np.exp(-d[i] / sigma), lam)
    return bl_mm + M[3]
