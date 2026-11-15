"""Team 1 step 5c: combine ugi_io50 (sharpest median) with local T1-2c (100% in-window safety net).

Usage: PYTHONPATH=src python scripts/team1_combine.py TEAM1_DIR [--tau 25.0]
Uses ugi_io50 where it agrees with local within --tau mm (Euclidean distance), else falls back to local.
Writes TEAM1_DIR/centres/hybrid_ugi_local.csv and copies it to TEAM1_DIR/centres.csv.
Also prints the per-axis residual distribution (true_fu - window_centre) for Team 2's training jitter comparison.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from lesionloc.alignment import gated_centres
from lesionloc.metrics import window_metrics
from lesionloc.team1 import XYZ, cols, read_centres, read_lesions, write_centres

GRID_TAU = [10.0, 15.0, 20.0, 25.0, 30.0, 40.0]


def load_aligned(lesions: pd.DataFrame, path: Path) -> tuple[np.ndarray, np.ndarray]:
    c = read_centres(path).drop_duplicates(["patient", "lesion"])
    m = lesions[["patient", "lesion"]].merge(c, on=["patient", "lesion"], how="left")
    miss = m["x_mm"].isna().to_numpy()
    return m[XYZ].to_numpy(dtype=float), miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", help="TEAM1_DIR containing lesions.csv and centres/")
    ap.add_argument("--tau", type=float, default=25.0, help="agreement threshold in mm (chosen on train)")
    a = ap.parse_args()
    out = Path(a.out)

    les = read_lesions(out / "lesions.csv")
    true_fu = les[cols("fu")].to_numpy(dtype=float)
    is_train = (les["split"] == "train").to_numpy()
    is_val = (les["split"] == "val").to_numpy()

    local_pts, _ = load_aligned(les, out / "centres" / "local.csv")
    ugi_pts, _ = load_aligned(les, out / "centres" / "ugi_io50.csv")

    print("| Tau (mm) | Train Mean | Train Median | Train CPM@10 | Train InWin@56 | Val Mean | Val Median | Val CPM@10 | Val CPM@20 | Val InWin@56 | UGI Used (Val) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for tau in GRID_TAU:
        comb, use_ugi = gated_centres(local_pts, ugi_pts, tau)
        mt = window_metrics(comb[is_train], true_fu[is_train])
        mv = window_metrics(comb[is_val], true_fu[is_val])
        print(
            f"| {tau:4.0f} | {mt['Mean error']:.2f} | {mt['Median error']:.2f} | "
            f"{100*mt['CPM@10']:.1f}% | {100*mt['In window@56']:.1f}% | "
            f"{mv['Mean error']:.2f} | {mv['Median error']:.2f} | "
            f"{100*mv['CPM@10']:.1f}% | {100*mv['CPM@20']:.1f}% | "
            f"{100*mv['In window@56']:.1f}% | {100*use_ugi[is_val].mean():.1f}% |"
        )

    best_comb, use_ugi = gated_centres(local_pts, ugi_pts, a.tau)
    write_centres(les, best_comb, out / "centres" / "hybrid_ugi_local.csv")
    write_centres(les, best_comb, out / "centres.csv")
    print(f"\nWrote hybrid_ugi_local.csv and centres.csv (tau={a.tau:g} mm)")

    print("\n### Residual distribution (true_fu - window_centre) in mm (inside 56 mm window)")
    print("| Split | Method | Mean (x, y, z) | Std (x, y, z) | 90th pct |x|,|y|,|z| |")
    print("|---|---|---|---|---|")
    for name, pts in [("local", local_pts), ("hybrid_ugi_local", best_comb)]:
        for s, mask in [("train", is_train), ("val", is_val)]:
            res = true_fu[mask] - pts[mask]
            res_in = res[np.abs(res).max(axis=1) <= 56.0]
            mu, sd = res_in.mean(axis=0), res_in.std(axis=0)
            p90 = np.percentile(np.abs(res_in), 90, axis=0)
            print(f"| {s} | {name} | ({mu[0]:+.1f}, {mu[1]:+.1f}, {mu[2]:+.1f}) | "
                  f"({sd[0]:.1f}, {sd[1]:.1f}, {sd[2]:.1f}) | ({p90[0]:.1f}, {p90[1]:.1f}, {p90[2]:.1f}) |")


if __name__ == "__main__":
    main()
