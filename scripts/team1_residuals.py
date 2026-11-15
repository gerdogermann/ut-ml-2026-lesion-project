"""Team 1 window residuals (true FU - centre, mm, patient x,y,z) for Team 2's training offset sampler.

Usage: PYTHONPATH=src python scripts/team1_residuals.py TEAM1_DIR OUT.npy
Saves the train-split residual vectors within the 128 mm window (|axis| <= 56 mm) as an (N,3) float32 array, for
scripts/train.py --offsets. Prints train vs val statistics: if train is much narrower than val, Team 1 was fitted
on train and its train residuals would teach the model too small offsets.
"""
import argparse
from pathlib import Path

import numpy as np

from lesionloc.team1 import XYZ, cols, read_centres, read_lesions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("team1", help="dir with lesions.csv and centres.csv")
    ap.add_argument("out")
    a = ap.parse_args()
    les = read_lesions(Path(a.team1) / "lesions.csv")
    m = les.merge(read_centres(Path(a.team1) / "centres.csv"), on=["patient", "lesion"], how="left", validate="1:1")
    assert m[XYZ].notna().all().all(), "centres.csv misses lesions"
    res = m[cols("fu")].to_numpy(float) - m[XYZ].to_numpy(float)
    inside = np.abs(res).max(1) <= 56
    print("| split | n | in window | sd x, y, z (mm) | > 20 mm | > 30 mm |\n|---|---|---|---|---|---|")
    for split in ("train", "val"):
        r, i = res[m.split == split], inside[m.split == split]
        d = np.linalg.norm(r, axis=1)
        print(f"| {split} | {len(r)} | {i.mean():.1%} | {', '.join(f'{v:.1f}' for v in r[i].std(0))} | "
              f"{(d > 20).mean():.1%} | {(d > 30).mean():.1%} |")
    train = res[(m.split == "train").to_numpy() & inside].astype(np.float32)
    np.save(a.out, train)
    print(f"wrote {len(train)} train residuals to {a.out}")


if __name__ == "__main__":
    main()
