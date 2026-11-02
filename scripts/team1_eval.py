"""Score Team 1 centres files against the true FU lesion centres.

Usage: PYTHONPATH=src python scripts/team1_eval.py TEAM1_DIR/lesions.csv TEAM1_DIR/centres/*.csv [--test] [--by-type] [--covered]
Prints one markdown row per (split, method), ready for docs/wiki/Experiments.md. Error = ||window centre - true FU||.
Tune a method's settings on train, report val; --test only for final numbers.
A lesion missing from a centres file (registration failed, no organ found) is scored at its BL position and
counted in the Missing column, so every method is scored on the same lesions. BL coordinates are unrelated to FU
coordinates (errors of ~1 m), so a missing lesion is a miss. --covered instead scores only the lesions present in
every given file: a paired comparison, and the readable view of a run on a few patients (PATIENTS=...).
"""
import argparse
from pathlib import Path

import numpy as np

from lesionloc.metrics import window_metrics
from lesionloc.team1 import XYZ, cols, read_centres, read_lesions

SHOW = ["Mean error", "Median error", "CPM@10", "CPM@20", "In window@56"]


def predictions(lesions, path):
    c = read_centres(path).drop_duplicates(["patient", "lesion"])
    m = lesions.merge(c, on=["patient", "lesion"], how="left")
    miss = m["x_mm"].isna().to_numpy()
    pred = np.where(miss[:, None], m[cols("bl")].to_numpy(), m[XYZ].to_numpy())
    return pred, miss


def row(split, name, n, miss, m):
    vals = [f"{m[k]:.2f}" if "error" in k else f"{100 * m[k]:.1f}%" for k in SHOW]
    return f"| {split} | {name} | {n} | {miss} | " + " | ".join(vals) + " |"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lesions")
    ap.add_argument("centres", nargs="+")
    ap.add_argument("--test", action="store_true", help="also report the test split (final numbers only)")
    ap.add_argument("--by-type", action="store_true", help="also split val by lesion_type")
    ap.add_argument("--covered", action="store_true", help="only lesions present in every centres file")
    a = ap.parse_args()
    les = read_lesions(a.lesions)
    true = les[cols("fu")].to_numpy()
    splits = ["train", "val"] + (["test"] if a.test else [])
    preds = {Path(p).stem: predictions(les, p) for p in a.centres}
    if a.covered:
        keep = ~np.any([m for _, m in preds.values()], axis=0)
        print(f"--covered: {int(keep.sum())} of {len(les)} lesions")
        les, true = les[keep].reset_index(drop=True), true[keep]
        preds = {n: (p[keep], m[keep]) for n, (p, m) in preds.items()}

    print("| Split | Method | N | Missing | Mean (mm) | Median (mm) | CPM@10 | CPM@20 | In window@56 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, (pred, miss) in preds.items():
        for s in splits:
            k = (les["split"] == s).to_numpy()
            print(row(s, name, int(k.sum()), int(miss[k].sum()), window_metrics(pred[k], true[k])))
        if a.by_type:
            for t, g in les[les["split"] == "val"].groupby("lesion_type"):
                k = g.index.to_numpy()
                print(row(f"val {t}", name, len(k), int(miss[k].sum()), window_metrics(pred[k], true[k])))


if __name__ == "__main__":
    main()
