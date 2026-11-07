"""Frozen-checkpoint experiment: alignment alone vs alignment + lesion prediction."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch

from lesionloc.alignment import CentreTable
from lesionloc.dataset import load_split
from lesionloc.joint_eval import by_start, evaluate, summarise
from lesionloc.prediction import PatchPredictor
from lesionloc.team1 import cols, read_lesions


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--lesions", required=True, help="Team 1 lesions.csv, all eligible lesions")
    ap.add_argument("--centres", required=True, help="Independent alignment centres.csv")
    ap.add_argument("--ckpt", required=True, nargs="+",
                    help="Frozen Team 2 state_dict(s); several = ensemble. Architecture from args.json beside each")
    ap.add_argument("--out", required=True, help="New experiment directory; never overwritten")
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--allow-test", action="store_true", help="Explicit opt-in for final evaluation")
    ap.add_argument("--limit", type=int, help="First N lesions for a smoke test, marked in manifest")
    ap.add_argument("--template-k", type=int, default=5, help="Must match checkpoint (run 6b: 5); if no args.json")
    ap.add_argument("--two-pass", action="store_true", help="Also re-cut at the first estimate: hybrid_pass2(_gated)")
    ap.add_argument("--no-bl-mask", action="store_true")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    if a.split == "test" and not a.allow_test:
        ap.error("Test is reserved for the final frozen experiment; pass --allow-test")
    if a.limit is not None and a.limit <= 0:
        ap.error("--limit must be positive")
    splits = load_split(Path(a.raw) / "data_split.json")
    lesions = read_lesions(a.lesions)
    if lesions.duplicated(["patient", "lesion"]).any():
        ap.error("Duplicate patient/lesion rows in lesion table")
    expected = {p: name for name, patients in splits.items() for p in patients}
    if not lesions.patient.map(expected).eq(lesions["split"]).all():
        ap.error("Lesion table disagrees with official patient split")
    lesions = lesions[lesions["split"] == a.split].copy()
    full_n = len(lesions)
    if a.limit:
        lesions = lesions.sort_values(["patient", "lesion"]).head(a.limit)
    if lesions.empty or not np.isfinite(lesions[cols("bl") + cols("fu")].to_numpy(float)).all():
        ap.error("Need a nonempty lesion cohort with finite BL and FU truth")
    alignment = CentreTable.read(a.centres)
    alignment.require(lesions)
    predictor = PatchPredictor(a.ckpt, bl_mask=not a.no_bl_mask, template_k=a.template_k,
                               tta=not a.no_tta, device=a.device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    manifest = dict(args=vars(a), git_commit=git.stdout.strip(), git_dirty=bool(dirty.stdout.strip()),
                    checkpoint_sha256=[sha256(c) for c in a.ckpt], centres_sha256=sha256(a.centres),
                    lesions_sha256=sha256(a.lesions), split_sha256=sha256(Path(a.raw) / "data_split.json"),
                    torch_version=torch.__version__, full_split_lesions=full_n,
                    evaluated_lesions=len(lesions), smoke_test=a.limit is not None,
                    status="running")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    predictions = evaluate(a.raw, lesions, alignment, predictor,
                           use_bl_mask=not a.no_bl_mask, tta=not a.no_tta, two_pass=a.two_pass)
    predictions["method"] = predictions.method.replace({"alignment": "hybrid", "plain": "hybrid_plain", "tta": "hybrid_tta",
                                                        "pass2": "hybrid_pass2", "pass2_gated": "hybrid_pass2_gated"})
    # Same frozen predictor with the dataset's alignment estimate: a paired
    # comparison isolates the value of replacing the registration stage.
    prop = lesions[cols("prop")].to_numpy(float)
    prop = np.where(np.isfinite(prop).all(axis=1)[:, None], prop, lesions[cols("bl")].to_numpy(float))
    propagated = lesions[["patient", "lesion"]].copy()
    propagated[["x_mm", "y_mm", "z_mm"]] = prop
    prop_predictions = evaluate(a.raw, lesions, CentreTable(propagated), predictor,
                                use_bl_mask=not a.no_bl_mask, tta=not a.no_tta)
    prop_predictions = prop_predictions[prop_predictions.method.isin(["plain", "tta"])].copy()
    prop_predictions["method"] = "propagated_" + prop_predictions.method
    predictions = pd.concat([predictions, prop_predictions], ignore_index=True)
    predictions.to_csv(out / "predictions.csv", index=False)
    summary = summarise(predictions)
    summary.to_csv(out / "summary.csv", index=False)
    refined = predictions.method.str.endswith(("_plain", "_tta"))
    manifest["prediction_fallbacks"] = int(predictions.loc[refined, "status"].ne("ok").sum())
    manifest["status"] = "complete_with_fallbacks" if manifest["prediction_fallbacks"] else "complete"
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(summary.to_string(index=False), flush=True)
    start, where = by_start(predictions[predictions.method.str.startswith("hybrid")])
    start.to_csv(out / "by_start.csv")
    where.to_csv(out / "misses.csv")
    print("\nMean error (count) by start error |centre - truth|, mm:")
    print(start.to_string(float_format="%.2f"))
    print("\nMisses > 10 mm: share inside the own FU lesion / inside another FU lesion (swap):")
    print(where.to_string(float_format="%.2f"), flush=True)
    for method in ("hybrid", "propagated_plain"):
        print(f"{method} in window@56: {predictions.loc[predictions.method == method, 'in_window'].mean():.1%}", flush=True)
    print(f"Results: {out}", flush=True)
    if manifest["prediction_fallbacks"]:
        raise SystemExit("Experiment contains prediction failures; inspect predictions.csv before interpreting results")


if __name__ == "__main__":
    main()
