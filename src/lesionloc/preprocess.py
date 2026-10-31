"""Raw Longitudinal-CT -> one small .npz per lesion pair + patient split.

Usage: python -m lesionloc.preprocess RAW_DIR OUT_DIR [--workers N] [--centre fu --jitter-sd 8]
RAW_DIR must contain inputsTr/ (images, BL masks, csv), targetsTr/ (FU masks) and data_split.json (official split).
--centre propagated: FU window at the registration guess (Experiment 1 data).
--centre fu: FU window at the true FU lesion plus a random offset (Team 2 data; simulates Team 1's error).
--augment K (with --centre fu): for train-split patients only, write K extra copies per lesion,
  <pid>_<lid>_a<k>.npz, with BL and FU each re-sampled from the raw scan at a random small rotation and scale.
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


def random_rotation(rng, max_deg):
    """3x3 rotation about a random axis by an angle uniform in [-max_deg, max_deg] (Rodrigues)."""
    k = rng.normal(size=3)
    k /= np.linalg.norm(k)
    a = np.deg2rad(rng.uniform(-max_deg, max_deg))
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def grid_origin(center_mm, size, direction, spacing):
    """Physical position of voxel 0 so that the grid centre lands on center_mm."""
    return center_mm - direction @ np.full(3, spacing * (size - 1) / 2)


def patch(img, center_mm, size=SIZE, direction=np.eye(3), spacing=SPACING):
    """Cut size^3 cube centred on center_mm; default axes aligned to patient (not scanner), 2 mm voxels.
    direction (columns = patch x,y,z axes in patient space) and spacing rotate / scale the grid for augmentation."""
    out = sitk.Resample(img, [size] * 3, sitk.Transform(), sitk.sitkLinear,
                        grid_origin(center_mm, size, direction, spacing).tolist(),
                        [spacing] * 3, direction.flatten().tolist(), -1024.0, sitk.sitkFloat32)
    a = np.clip(sitk.GetArrayFromImage(out), HU_MIN, HU_MAX)
    return ((a - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float16)  # [0,1], axes (z,y,x)


def mask_patch(mask, center_mm, label, size=SIZE, direction=np.eye(3), spacing=SPACING):
    """Same grid as patch(), nearest neighbour; 1 where the mask has this lesion's instance label."""
    out = sitk.Resample(mask, [size] * 3, sitk.Transform(), sitk.sitkNearestNeighbor,
                        grid_origin(center_mm, size, direction, spacing).tolist(),
                        [spacing] * 3, direction.flatten().tolist(), 0, sitk.sitkUInt16)
    return (sitk.GetArrayFromImage(out) == label).astype(np.uint8)  # axes (z,y,x)


def com_offset_mm(m):
    """Mask centre of mass relative to the patch centre, mm, (x,y,z) order."""
    zyx = np.argwhere(m).mean(0)
    return (zyx[::-1] - (SIZE - 1) / 2) * SPACING


def do_patient(args):
    pid, a, train = args
    raw, out, centre, jitter_sd, jitter_max, seed = Path(a.raw), Path(a.out), a.centre, a.jitter_sd, a.jitter_max, a.seed
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)  # one process per core already; avoid oversubscription
    stats = Counter()
    if a.augment and pid not in train:
        return pid, Counter(skip_not_train=1)
    df = pd.read_csv(raw / "inputsTr" / f"{pid}.csv")
    cache = {}

    def load(tag, i, kind="img"):
        if (tag, i, kind) not in cache:
            name = f"{pid}_{tag}_{kind}_{int(i):02d}.nii.gz"
            f = raw / "inputsTr" / name
            if not f.exists():  # FU masks are the challenge targets: they live in targetsTr/
                f = raw / "targetsTr" / name
            cache[(tag, i, kind)] = sitk.ReadImage(str(f))
        return cache[(tag, i, kind)]

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
        if centre == "fu":
            # true FU spot + offset, reproducible per lesion; target = -jitter
            rng = np.random.default_rng([seed, int(pid, 16), lid])
            jitter = np.clip(rng.normal(0, jitter_sd, 3), -jitter_max, jitter_max)
            centre_fu = fu_mm + jitter
        else:
            # search-window centre in follow-up: registration guess if available, else same spot as baseline
            jitter = np.zeros(3)
            centre_fu = to_mm(ifu, prop) if prop is not None else bl_mm
        stats["has_propagated" if prop is not None else "no_propagated"] += 1
        target = fu_mm - centre_fu  # mm, patient axes (x,y,z); this is what the model predicts
        if np.abs(target).max() > SPACING * SIZE / 2 - 8:  # true spot too near/outside window
            stats["skip_outside_window"] += 1
            continue
        try:
            mbl, mfu = load("BL", r["img_id_bl"], "mask"), load("FU", r["img_id_fu"], "mask")
        except RuntimeError:  # sitk.ReadImage on a missing file
            stats["skip_no_mask"] += 1
            continue
        # FU instance label: a merged lesion carries the label of the lesion it merged into
        m = r.get("merged_into")
        fu_label = int(float(m)) if r["topology_class"] == "MERGING" and pd.notna(m) else lid
        bl_mask, fu_mask = mask_patch(mbl, bl_mm, lid), mask_patch(mfu, centre_fu, fu_label)
        if not bl_mask.any() or not fu_mask.any():
            stats["skip_empty_mask"] += 1
            continue
        if np.linalg.norm(com_offset_mm(fu_mask) - target) > 4:  # wrong label or axis order would show here
            stats["warn_mask_cog_far"] += 1
        if a.augment:
            stats.update(augment(a, pid, lid, ibl, ifu, mbl, mfu, bl_mm, fu_mm, fu_label, jitter, out, prop is not None))
            continue
        big = {}
        if centre == "fu":  # FU cube centred on the true lesion, room for any jitter: lets training re-draw it
            n = SIZE + 2 * int(np.ceil(jitter_max / SPACING))
            big = dict(fu_big=patch(ifu, fu_mm, n), fu_mask_big=mask_patch(mfu, fu_mm, fu_label, n))
        np.savez_compressed(out / "lesions" / f"{pid}_{lid}.npz", **big,
                            bl=patch(ibl, bl_mm), fu=patch(ifu, centre_fu), bl_mask=bl_mask, fu_mask=fu_mask,
                            target_mm=target.astype(np.float32), jitter_mm=jitter.astype(np.float32),
                            has_propagated=prop is not None)
        stats["kept"] += 1
    return pid, stats


def augment(a, pid, lid, ibl, ifu, mbl, mfu, bl_mm, fu_mm, fu_label, jitter, out, has_prop):
    """K rotated + scaled copies of one lesion pair (BL and FU each get their own), same keys as the normal npz.
    Offsets are in patch mm (2 mm per patch voxel), as the model sees them; the fixed FU window sits at the
    lesion's own jitter rounded to whole voxels, cut from fu_big like LesionDataset's re-drawn jitter."""
    stats = Counter()
    m = int(np.ceil(a.jitter_max / SPACING))
    n = SIZE + 2 * m
    s = np.clip(np.round(jitter / SPACING), -m, m).astype(int)  # (x,y,z) voxels
    win = tuple(slice(m + s[j], m + s[j] + SIZE) for j in (2, 1, 0))  # arrays are (z,y,x)
    for k in range(a.augment):
        rng = np.random.default_rng([a.seed, int(pid, 16), lid, k + 1])
        rb, rf = random_rotation(rng, a.max_rot), random_rotation(rng, a.max_rot)
        sb, sf = SPACING * rng.uniform(1 - a.max_scale, 1 + a.max_scale, 2)
        bl_mask = mask_patch(mbl, bl_mm, lid, SIZE, rb, sb)
        fu_mask_big = mask_patch(mfu, fu_mm, fu_label, n, rf, sf)
        fu_mask = fu_mask_big[win]
        if not bl_mask.any() or not fu_mask.any():
            stats["skip_empty_mask"] += 1
            continue
        target = (-SPACING * s).astype(np.float32)
        if np.linalg.norm(com_offset_mm(fu_mask) - target) > 4:
            stats["warn_mask_cog_far"] += 1
        fu_big = patch(ifu, fu_mm, n, rf, sf)
        np.savez_compressed(out / "lesions" / f"{pid}_{lid}_a{k}.npz", fu_big=fu_big, fu_mask_big=fu_mask_big,
                            bl=patch(ibl, bl_mm, SIZE, rb, sb), fu=fu_big[win], bl_mask=bl_mask, fu_mask=fu_mask,
                            target_mm=target, jitter_mm=-target, has_propagated=has_prop)
        stats["kept"] += 1
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--centre", choices=["propagated", "fu"], default="propagated")
    ap.add_argument("--jitter-sd", type=float, default=8.0, help="mm per axis, --centre fu only")
    ap.add_argument("--jitter-max", type=float, default=24.0, help="mm per axis, --centre fu only")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--augment", type=int, default=0, help="K rotated/scaled copies per train lesion (--centre fu)")
    ap.add_argument("--max-rot", type=float, default=15.0, help="degrees, --augment only")
    ap.add_argument("--max-scale", type=float, default=0.1, help="scale in 1 +- this, --augment only")
    a = ap.parse_args()
    assert not a.augment or a.centre == "fu", "--augment needs --centre fu"
    out = Path(a.out)
    (out / "lesions").mkdir(parents=True, exist_ok=True)
    pids = sorted(p.stem for p in (Path(a.raw) / "inputsTr").glob("??????????.csv"))
    official = json.loads((Path(a.raw) / "data_split.json").read_text())

    total = Counter()
    with Pool(a.workers) as pool:
        for pid, s in pool.imap_unordered(
                do_patient, [(p, a, set(official["train"])) for p in pids]):
            total.update(s)
            print(pid, dict(s), flush=True)
    print("TOTAL", dict(total))

    # official patient-level split, restricted to patients that produced samples
    have = {f.name.split("_")[0] for f in (out / "lesions").glob("*.npz")}
    split = {k: [p for p in v if p in have] for k, v in official.items()}
    unassigned = have - {p for v in split.values() for p in v}
    assert not unassigned, f"patients not in data_split.json: {sorted(unassigned)[:5]}"
    print("SPLIT", {k: len(v) for k, v in split.items()})
    (out / "split.json").write_text(json.dumps(split, indent=1))


if __name__ == "__main__":
    main()
