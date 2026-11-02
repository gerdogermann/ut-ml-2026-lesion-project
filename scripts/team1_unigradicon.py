"""Team 1 step 5: pretrained deformable registration with uniGradICON (Tian et al., MICCAI 2024), no training.

Usage: PYTHONPATH=src python scripts/team1_unigradicon.py RAW TEAM1_DIR [--io-iterations None] [--patients P1,P2]
Per scan pair runs `unigradicon-register` with fixed = BL and moving = FU, caches the transform in
TEAM1_DIR/transforms/<pid>_<bl>_<fu>_ugi_io<N>.hdf5, then moves every BL lesion centre with it and writes
TEAM1_DIR/centres/ugi_io<N>.csv. The transform maps fixed (BL) points to moving (FU) points, as in SimpleITK.
--io-iterations: instance optimisation steps per pair on top of the network (None = network only; try 50 as the
second run, slower but usually more accurate). Check the flag names once with `unigradicon-register --help`.
Check once that this beats affine on train; if it is far worse, the direction is flipped.
Needs a GPU and its own venv (see scripts/unigradicon.sbatch).
"""
import argparse
import subprocess
from pathlib import Path

import itk
import numpy as np
import pandas as pd

from lesionloc.team1 import XYZ, cols, pairs, read_lesions, scan_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("out", help="TEAM1_DIR with lesions.csv")
    ap.add_argument("--io-iterations", default="None")
    ap.add_argument("--patients", default="", help="comma-separated subset, for a quick check")
    a = ap.parse_args()
    out = Path(a.out)
    name = f"ugi_io{a.io_iterations}"
    (out / "transforms").mkdir(parents=True, exist_ok=True)
    (out / "centres").mkdir(parents=True, exist_ok=True)

    les = read_lesions(out / "lesions.csv")
    if a.patients:
        les = les[les["patient"].isin(a.patients.split(","))]
    res, failed = [], 0
    for (pid, ib, ifu), g in pairs(les):
        tf = out / "transforms" / f"{pid}_{ib:02d}_{ifu:02d}_{name}.hdf5"
        if not tf.exists():
            cmd = ["unigradicon-register",
                   f"--fixed={scan_path(a.raw, pid, 'BL', ib)}", "--fixed_modality=ct",
                   f"--moving={scan_path(a.raw, pid, 'FU', ifu)}", "--moving_modality=ct",
                   f"--transform_out={tf}", f"--io_iterations={a.io_iterations}"]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode or not tf.exists():
                failed += 1
                print(pid, "FAILED", r.stderr[-2000:], flush=True)
                continue
        txs = itk.transformread(str(tf))
        assert len(txs) == 1, f"{tf}: expected one (composite) transform, got {len(txs)}"
        pts = [np.array(txs[0].TransformPoint([float(v) for v in p])) for p in g[cols("bl")].to_numpy()]
        res.append(pd.DataFrame(pts, columns=XYZ).assign(patient=pid, lesion=g["lesion"].to_numpy()))
        print(pid, "ok", flush=True)
    print("pairs", len(pairs(les)), "failed", failed)
    pd.concat(res)[["patient", "lesion", *XYZ]].to_csv(out / "centres" / f"{name}.csv", index=False,
                                                        float_format="%.3f")


if __name__ == "__main__":
    main()
