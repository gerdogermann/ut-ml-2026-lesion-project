"""Zero-shot LesionLocator (Rokuss et al., CVPR 2025) on our lesion cohort: prompts in, FU centres out.

LesionLocator runs on the raw scans, not on our patches. One `LesionLocator_track` call per scan pair:
  - mask mode:  -t prev_mask, the BL instance mask restricted to the cohort lesions, relabelled 1..K.
  - point mode: -t point, a JSON with one point per lesion at the BL centre (voxel index, Z,Y,X order).
                The tool first segments the BL lesions from the points, then tracks them.
Output: one mask per lesion, <FU basename>_lesion_<k>.nii.gz (point mode also <BL basename>_lesion_<k>.nii.gz).
The predicted FU centre is the mask centre of mass, as a SimpleITK physical point (same frame as lesions.csv).
"""
import json
import re
from pathlib import Path

import numpy as np
import SimpleITK as sitk

MODES = ("mask", "point")
NII = ".nii.gz"


def pair_name(patient, ib, ifu):
    return f"{patient}_BL{int(ib):02d}_FU{int(ifu):02d}"


def relabel(mask, lesion_ids):
    """Instance mask with only `lesion_ids`, relabelled to 1..K in the given order (other labels -> 0)."""
    a = sitk.GetArrayFromImage(mask)
    out = np.zeros(a.shape, np.uint16)
    for k, lid in enumerate(lesion_ids, 1):
        out[a == lid] = k
    img = sitk.GetImageFromArray(out)
    img.CopyInformation(mask)
    return img


def point_prompt(img, points_mm):
    """LesionLocator point JSON: {"k": {"point": [z, y, x]}} with continuous voxel indices of `img`, k = 1..K."""
    out = {}
    for k, p in enumerate(points_mm, 1):
        x, y, z = img.TransformPhysicalPointToContinuousIndex([float(v) for v in p])
        out[str(k)] = {"point": [f"{z:.2f}", f"{y:.2f}", f"{x:.2f}"]}  # strings, as in the project's prompting.md
    return out


def centre_mm(mask):
    """Centre of mass of all nonzero voxels as a physical point (x, y, z) mm; None if the mask is empty."""
    zyx = np.argwhere(sitk.GetArrayViewFromImage(mask) > 0)
    if not len(zyx):
        return None
    return np.array(mask.TransformContinuousIndexToPhysicalPoint(zyx.mean(0)[::-1].tolist()))


def outputs(folder, scan_stem):
    """{k: path} of LesionLocator outputs <scan_stem>_lesion_<k>.nii.gz in `folder`."""
    pat = re.compile(re.escape(scan_stem) + r"_lesion_(\d+)" + re.escape(NII) + "$")
    return {int(m.group(1)): f for f in Path(folder).glob(f"{scan_stem}_lesion_*{NII}") if (m := pat.match(f.name))}


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1))
