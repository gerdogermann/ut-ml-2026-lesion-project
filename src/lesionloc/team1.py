"""Team 1 shared pieces: the lesion table, centres.csv I/O and raw scan paths.

The lesion table (scripts/team1_lesions.py -> lesions.csv) is the ground truth for every Team 1 method: one row per
lesion that preprocess.py would consider (UNCHANGED/MERGING, linking clear, both centres known), with BL, FU and
propagated centres in patient mm. Unlike the npz patches it keeps the lesions that preprocess.py drops with
skip_outside_window, because placing the window well enough to keep them is exactly what Team 1 is scored on.

A method's output is a centres file (patient, lesion, x_mm, y_mm, z_mm): the predicted FU window centre per lesion,
in the same patient coordinates as preprocess.py (SimpleITK physical points).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

from lesionloc.preprocess import KEEP, parse_xyz

XYZ = ["x_mm", "y_mm", "z_mm"]
IDS = {"patient": str}  # patient ids are hex strings like 0640966322; read as numbers they lose the leading zero


def cols(prefix):
    """Column names of one point in the lesion table: bl / fu / prop."""
    return [f"{prefix}_{c}" for c in "xyz"]


def scan_name(pid, tag, i, kind="img"):
    return f"{pid}_{tag}_{kind}_{int(i):02d}.nii.gz"


def scan_path(raw, pid, tag, i, kind="img"):
    """inputsTr/<pid>_<BL|FU>_<img|mask>_<ii>.nii.gz; FU masks are the challenge targets and live in targetsTr/."""
    f = Path(raw) / "inputsTr" / scan_name(pid, tag, i, kind)
    return f if f.exists() else Path(raw) / "targetsTr" / scan_name(pid, tag, i, kind)


def header(path):
    """Image geometry (origin, spacing, direction, size) without reading the voxels."""
    r = sitk.ImageFileReader()
    r.SetFileName(str(path))
    r.ReadImageInformation()
    return r


def index_to_mm(h, voxel_xyz):
    """Image.TransformContinuousIndexToPhysicalPoint from a header (or image) alone."""
    d = np.array(h.GetDirection()).reshape(3, 3)
    return np.array(h.GetOrigin()) + d @ (np.array(h.GetSpacing()) * np.asarray(voxel_xyz, dtype=float))


def lesion_table(raw):
    """All eligible lesions of all patients, same filter as preprocess.py minus the 56 mm window check."""
    raw = Path(raw)
    official = json.loads((raw / "data_split.json").read_text())
    split_of = {p: k for k, v in official.items() for p in v}
    rows = []
    for csv in sorted((raw / "inputsTr").glob("??????????.csv")):
        pid, heads = csv.stem, {}
        for _, r in pd.read_csv(csv).iterrows():
            # bool(nan) is True, so compare as text: empty cell must not count as "unclear"
            if r["topology_class"] not in KEEP or str(r.get("linking_unclear", False)).lower() == "true":
                continue
            bl, fu, prop = parse_xyz(r["cog_bl"]), parse_xyz(r["cog_fu"]), parse_xyz(r.get("cog_propagated"))
            if bl is None or fu is None:
                continue
            ib, ifu = int(r["img_id_bl"]), int(r["img_id_fu"])
            for key in (("BL", ib), ("FU", ifu)):
                if key not in heads:
                    heads[key] = header(scan_path(raw, pid, *key))
            row = dict(patient=pid, lesion=int(r["lesion_id"]), split=split_of.get(pid, ""), img_id_bl=ib,
                       img_id_fu=ifu, lesion_type=r["lesion_type"], volume_bl=r["volume_bl"],
                       has_propagated=prop is not None)
            # cog_propagated is a voxel index in the FU scan, like cog_fu (see preprocess.py)
            for name, h, v in (("bl", heads["BL", ib], bl), ("fu", heads["FU", ifu], fu), ("prop", heads["FU", ifu], prop)):
                row.update(zip(cols(name), index_to_mm(h, v) if v is not None else [np.nan] * 3))
            rows.append(row)
    df = pd.DataFrame(rows)
    missing = sorted(set(df.loc[df["split"] == "", "patient"]))
    assert not missing, f"patients not in data_split.json: {missing[:5]}"
    return df


def read_lesions(path):
    return pd.read_csv(path, dtype=IDS)


def write_centres(lesions, xyz, path):
    """One predicted FU window centre per row of `lesions` (same order), mm, patient (x, y, z)."""
    out = lesions[["patient", "lesion"]].reset_index(drop=True).copy()
    out[XYZ] = np.asarray(xyz, dtype=float)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, float_format="%.3f")


def read_centres(path):
    return pd.read_csv(path, dtype=IDS)


def pairs(lesions):
    """(patient, img_id_bl, img_id_fu) -> that scan pair's lesion rows; methods register once per pair."""
    return lesions.groupby(["patient", "img_id_bl", "img_id_fu"], sort=True)
