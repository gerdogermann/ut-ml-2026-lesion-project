"""Zero-shot LesionLocator baseline on the joint-evaluation cohort (Team 1 lesions.csv, one split).

  prepare: per scan pair, write the prompts (BL mask relabelled 1..K, BL point JSON) and pairs.tsv for the job loop.
  score:   read LesionLocator's FU masks, take each mask's centre of mass, score against the FU truth.

Methods in predictions.csv / summary.csv:
  ll_mask, ll_point  LesionLocator FU mask centre (prev_mask prompt / point prompt).
  propagated         cog_propagated (the organisers' registration; BL centre if missing): the cheap bar.
  team1              Team 1 centres.csv (only with --centres): the alignment our pipeline starts from.
A lesion without a LesionLocator mask (failed run, empty mask) falls back to `propagated` and stays in the count,
with status "fallback_<reason>". Point mode also checks the BL segmentation: bl_check_mm = |BL mask centre - BL truth|;
large values mean the point prompt landed in the wrong place (axis order) or segmented the wrong thing.

Usage (see scripts/lesionlocator.sbatch):
  python scripts/lesionlocator_eval.py prepare --raw RAW --lesions lesions.csv --out OUT [--split val]
  python scripts/lesionlocator_eval.py score --raw RAW --lesions lesions.csv --out OUT [--centres centres.csv]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

from lesionloc import lesionlocator as L
from lesionloc.joint_eval import label_at, summarise
from lesionloc.team1 import cols, pairs, read_centres, read_lesions, scan_name, scan_path


def cohort(a):
    if a.split == "test" and not a.allow_test:
        raise SystemExit("Test is reserved for the final frozen experiment; pass --allow-test")
    lesions = read_lesions(a.lesions)
    lesions = lesions[lesions["split"] == a.split].sort_values(["patient", "lesion"])
    if a.limit_pairs:
        keep = list(pairs(lesions).groups)[: a.limit_pairs]
        lesions = lesions[[k in keep for k in zip(lesions.patient, lesions.img_id_bl, lesions.img_id_fu)]]
    if lesions.empty:
        raise SystemExit(f"no lesions in split {a.split}")
    return lesions


def prepare(a):
    lesions = cohort(a)
    out = Path(a.out)
    rows, empty = [], 0
    for (patient, ib, ifu), group in pairs(lesions):
        d = out / "pairs" / L.pair_name(patient, ib, ifu)
        d.mkdir(parents=True, exist_ok=True)
        ids = [int(v) for v in group.lesion]
        mask = sitk.ReadImage(str(scan_path(a.raw, patient, "BL", ib, "mask")))
        prompt = L.relabel(mask, ids)
        present = set(np.unique(sitk.GetArrayViewFromImage(prompt))) - {0}
        empty += len(ids) - len(present)
        sitk.WriteImage(prompt, str(d / "bl_mask_prompt.nii.gz"), useCompression=True)
        L.write_json(d / "bl_points.json", L.point_prompt(mask, group[cols("bl")].to_numpy(float)))
        L.write_json(d / "lesions.json", dict(patient=str(patient), img_id_bl=int(ib), img_id_fu=int(ifu),
                                              lesions=ids, missing_in_bl_mask=sorted(set(range(1, len(ids) + 1)) - present)))
        rows.append(dict(pair=d.name, bl=scan_path(a.raw, patient, "BL", ib), fu=scan_path(a.raw, patient, "FU", ifu)))
    pd.DataFrame(rows).to_csv(out / "pairs.tsv", sep="\t", index=False, header=False)
    L.write_json(out / "cohort.json", dict(split=a.split, lesions=len(lesions), patients=lesions.patient.nunique(),
                                           pairs=len(rows), lesions_missing_in_bl_mask=empty,
                                           lesions_csv=str(Path(a.lesions).resolve())))
    print(f"{a.split}: {len(lesions)} lesions, {lesions.patient.nunique()} patients, {len(rows)} scan pairs; "
          f"{empty} lesions have no voxels in the BL mask -> {out}/pairs.tsv")


def record(row, method, point, status, fu_mask, **extra):
    truth = row[cols("fu")].to_numpy(float)
    rec = dict(patient=str(row.patient), lesion=int(row.lesion), split=row["split"], method=method, status=status)
    rec.update(zip(["pred_x", "pred_y", "pred_z"], point))
    rec.update(zip(["true_x", "true_y", "true_z"], truth))
    rec.update(error_mm=float(np.linalg.norm(point - truth)), label_pred=label_at(fu_mask, point),
               label_true=label_at(fu_mask, truth), **extra)
    return rec


def score(a):
    lesions = cohort(a)
    out = Path(a.out)
    team1 = read_centres(a.centres).set_index(["patient", "lesion"]) if a.centres else None
    records = []
    for (patient, ib, ifu), group in pairs(lesions):
        d = out / "pairs" / L.pair_name(patient, ib, ifu)
        fu_mask_path = scan_path(a.raw, patient, "FU", ifu, "mask")
        fu_mask = sitk.ReadImage(str(fu_mask_path)) if fu_mask_path.exists() else None  # truth: scoring only
        ids = json.loads((d / "lesions.json").read_text())["lesions"]
        found = {m: L.outputs(d / m, scan_name(patient, "FU", ifu).removesuffix(L.NII)) for m in L.MODES}
        bl_seg = L.outputs(d / "point", scan_name(patient, "BL", ib).removesuffix(L.NII))
        for k, (_, row) in enumerate(group.set_index("lesion", drop=False).loc[ids].iterrows(), 1):
            prop = row[cols("prop")].to_numpy(float)
            fallback = prop if np.isfinite(prop).all() else row[cols("bl")].to_numpy(float)
            records.append(record(row, "propagated", fallback, "ok", fu_mask))
            if team1 is not None:
                records.append(record(row, "team1", team1.loc[(str(patient), int(row.lesion)), ["x_mm", "y_mm", "z_mm"]]
                                      .to_numpy(float), "ok", fu_mask))
            for m in L.MODES:
                done = (d / m / "DONE").exists()
                c = L.centre_mm(sitk.ReadImage(str(found[m][k]))) if k in found[m] else None
                status = "ok" if c is not None else "fallback_" + (
                    "not_run" if not done else "no_output" if k not in found[m] else "empty_mask")
                extra = {}
                if m == "point":
                    b = L.centre_mm(sitk.ReadImage(str(bl_seg[k]))) if k in bl_seg else None
                    extra["bl_check_mm"] = float(np.linalg.norm(b - row[cols("bl")].to_numpy(float))) \
                        if b is not None else np.nan
                records.append(record(row, f"ll_{m}", c if c is not None else fallback, status, fu_mask, **extra))
    p = pd.DataFrame(records)
    p.to_csv(out / "predictions.csv", index=False, float_format="%.3f")
    s = summarise(p)
    s.to_csv(out / "summary.csv", index=False, float_format="%.3f")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(s.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        print("\nstatus counts:\n", p.groupby(["method", "status"]).size().to_string())
        bc = p.loc[p.method == "ll_point", "bl_check_mm"]
        print(f"\npoint-mode BL check: median {bc.median():.2f} mm, >10 mm: {(bc > 10).sum()} of {bc.notna().sum()}")
        miss = p[(p.error_mm > 10) & (p.label_true > 0)]
        print("\nmisses > 10 mm landing in another FU lesion (swap):\n",
              miss.assign(swap=(miss.label_pred > 0) & (miss.label_pred != miss.label_true))
              .groupby("method").swap.agg(["sum", "size"]).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["prepare", "score"])
    ap.add_argument("--raw", required=True)
    ap.add_argument("--lesions", required=True, help="Team 1 lesions.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--limit-pairs", type=int, help="first N scan pairs only (smoke test)")
    ap.add_argument("--centres", help="Team 1 centres.csv, score only: adds the `team1` rows")
    a = ap.parse_args()
    prepare(a) if a.cmd == "prepare" else score(a)


if __name__ == "__main__":
    main()
