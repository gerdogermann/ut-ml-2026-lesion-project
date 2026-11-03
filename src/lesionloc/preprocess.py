"""Raw Longitudinal-CT -> one small .npz per lesion pair + patient split.

Usage: python -m lesionloc.preprocess RAW_DIR OUT_DIR [--workers N] [--centre {propagated,csv}] [--centres-csv PATH]
RAW_DIR must contain inputsTr/ (images, csv) and data_split.json (official split).
"""
import argparse
import json
import re
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

SPACING = 2.0   # mm per voxel in patches
SIZE = 64       # voxels per side -> 128 mm cube
HU_MIN, HU_MAX = -1000.0, 400.0
KEEP = {"UNCHANGED", "MERGING"}  # lesions that exist in both scans


def parse_xyz(v):
    """'[1.0, 2.0, 3.0]' / '(1 2 3)' -> array of 3 floats, or None if empty."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    nums = re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", str(v))
    return np.array(nums[:3], dtype=float) if len(nums) >= 3 else None


def to_mm(img, voxel_xyz):
    return np.array(img.TransformContinuousIndexToPhysicalPoint(voxel_xyz.tolist()))


def patch(img, center_mm):
    """Cut SIZE^3 cube at SPACING, centred on center_mm, axes aligned to patient (not scanner)."""
    origin = center_mm - SPACING * (SIZE - 1) / 2
    out = sitk.Resample(img, [SIZE] * 3, sitk.Transform(), sitk.sitkLinear, origin.tolist(),
                        [SPACING] * 3, (1, 0, 0, 0, 1, 0, 0, 0, 1), -1024.0, sitk.sitkFloat32)
    a = np.clip(sitk.GetArrayFromImage(out), HU_MIN, HU_MAX)
    return ((a - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float16)  # [0,1], axes (z,y,x)


def load_centres_lookup(csv_path):
    """Read Team 1 centres.csv into {patient_id: {lesion_id: np.array([x_mm, y_mm, z_mm])}}."""
    df = pd.read_csv(csv_path, dtype={"patient": str})
    lookup = {}
    for r in df.itertuples():
        lookup.setdefault(str(r.patient), {})[int(r.lesion)] = np.array([r.x_mm, r.y_mm, r.z_mm], dtype=float)
    return lookup


def do_patient(args):
    pid, raw, out, custom_centres = args
    raw, out = Path(raw), Path(out)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)  # one process per core already; avoid oversubscription
    stats = Counter()
    df = pd.read_csv(raw / "inputsTr" / f"{pid}.csv")
    cache = {}

    def load(tag, i):
        if (tag, i) not in cache:
            cache[(tag, i)] = sitk.ReadImage(str(raw / "inputsTr" / f"{pid}_{tag}_img_{int(i):02d}.nii.gz"))
        return cache[(tag, i)]

    for _, r in df.iterrows():
        lid = int(r["lesion_id"])
        # bool(nan) is True, so compare as text: empty cell must not count as "unclear"
        if r["topology_class"] not in KEEP or str(r.get("linking_unclear", False)).lower() == "true":
            stats["skip_type"] += 1
            continue
        bl, fu = parse_xyz(r["cog_bl"]), parse_xyz(r["cog_fu"])
        if bl is None or fu is None:
            stats["skip_no_cog"] += 1
            continue
        ibl, ifu = load("BL", r["img_id_bl"]), load("FU", r["img_id_fu"])
        bl_mm, fu_mm = to_mm(ibl, bl), to_mm(ifu, fu)
        prop = parse_xyz(r.get("cog_propagated"))

        if custom_centres is not None:
            if lid not in custom_centres:
                stats["skip_no_centre_csv"] += 1
                continue
            centre_fu = custom_centres[lid]
            has_prop = True
            stats["from_centres_csv"] += 1
        else:
            # search-window centre in follow-up: registration guess if available, else same spot as baseline
            centre_fu = to_mm(ifu, prop) if prop is not None else bl_mm
            has_prop = prop is not None
            stats["has_propagated" if has_prop else "no_propagated"] += 1

        target = fu_mm - centre_fu  # mm, patient axes (x,y,z); this is what the model predicts
        if np.abs(target).max() > SPACING * SIZE / 2 - 8:  # true spot too near/outside window
            stats["skip_outside_window"] += 1
            continue
        np.savez_compressed(out / "lesions" / f"{pid}_{lid}.npz",
                            bl=patch(ibl, bl_mm), fu=patch(ifu, centre_fu),
                            target_mm=target.astype(np.float32), has_propagated=has_prop)
        stats["kept"] += 1
    return pid, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--centre", choices=["propagated", "csv"], default="propagated",
                    help="FU window centre source: dataset cog_propagated or Team 1 centres.csv")
    ap.add_argument("--centres-csv", default=None, help="Path to centres.csv when --centre csv is used")
    a = ap.parse_args()
    if a.centre == "csv" and not a.centres_csv:
        ap.error("--centres-csv is required when --centre csv is set")

    centres_by_pid = load_centres_lookup(a.centres_csv) if a.centre == "csv" else {}

    out = Path(a.out)
    (out / "lesions").mkdir(parents=True, exist_ok=True)
    pids = sorted(p.stem for p in (Path(a.raw) / "inputsTr").glob("??????????.csv"))

    total = Counter()
    jobs = [(p, a.raw, a.out, centres_by_pid.get(p) if a.centre == "csv" else None) for p in pids]
    with Pool(a.workers) as pool:
        for pid, s in pool.imap_unordered(do_patient, jobs):
            total.update(s)
            print(pid, dict(s), flush=True)
    print("TOTAL", dict(total))

    # official patient-level split, restricted to patients that produced samples
    have = {f.name.split("_")[0] for f in (out / "lesions").glob("*.npz")}
    official = json.loads((Path(a.raw) / "data_split.json").read_text())
    split = {k: [p for p in v if p in have] for k, v in official.items()}
    unassigned = have - {p for v in split.values() for p in v}
    assert not unassigned, f"patients not in data_split.json: {sorted(unassigned)[:5]}"
    print("SPLIT", {k: len(v) for k, v in split.items()})
    (out / "split.json").write_text(json.dumps(split, indent=1))


if __name__ == "__main__":
    main()
