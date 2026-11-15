"""Team 1 steps 2 and 4: register each BL scan to its FU scan and move every BL lesion centre with the transform.

Usage: PYTHONPATH=src python scripts/team1_register.py RAW TEAM1_DIR [--methods rigid,affine,local,bspline]
           [--workers N] [--mesh 8] [--patients P1,P2]
Reads TEAM1_DIR/lesions.csv (scripts/team1_lesions.py). Writes TEAM1_DIR/centres/<method>.csv, and caches each
pair's global transforms in TEAM1_DIR/transforms/<pid>_<bl>_<fu>_<method>.h5 so a rerun only registers new pairs.
  rigid    6 DoF whole scan
  affine   12 DoF whole scan, from rigid
  local    affine, then a rigid correction inside a 160 mm cube around each lesion
  bspline  B-spline (--mesh control points per axis) on top of affine
A pair that fails is logged and its lesions are left out; team1_eval.py then scores them at the BL position.
A lesion whose 160 mm cube leaves the FU scan keeps its affine point in `local` (counted in the log).
--patients reruns only those patients and replaces just their rows in the existing centres files.
"""
import argparse
import os
import traceback
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import SimpleITK as sitk

from lesionloc import registration as R
from lesionloc.team1 import XYZ, cols, pairs, read_centres, read_lesions, scan_path

NEEDS = {"rigid": [], "affine": ["rigid"], "local": ["rigid", "affine"], "bspline": ["rigid", "affine"]}


def cached(path, make):
    if path.exists():
        try:  # Downcast: affine() needs the rigid transform's GetMatrix()
            return sitk.ReadTransform(str(path)).Downcast()
        except RuntimeError:  # unreadable, e.g. half-written when an earlier job hit its time limit: redo it
            path.unlink()
    tx = make()
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)  # renamed only once complete
    sitk.WriteTransform(tx, str(tmp))
    tmp.replace(path)
    return tx


def local_point(fixed, moving, init_tx, p):
    """FU point from the per-lesion rigid correction, or None when the lesion's cube leaves the FU scan
    (Mattes throws once no BL sample lands inside it), e.g. a lesion at the edge of the FU scan range."""
    try:
        return R.move_points(R.local_rigid(fixed, moving, init_tx, p), [p])[0]
    except RuntimeError:
        return None


def do_pair(args):
    (pid, ib, ifu), rows, a = args
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(a.threads)
    tdir = Path(a.out) / "transforms"
    tf = lambda m: tdir / f"{pid}_{ib:02d}_{ifu:02d}_{m}.h5"
    try:
        fixed, fmask = R.prepare(scan_path(a.raw, pid, "BL", ib))
        moving, mmask = R.prepare(scan_path(a.raw, pid, "FU", ifu))
        want = set(a.methods) | {n for m in a.methods for n in NEEDS[m]}
        tx = {}
        if "rigid" in want:
            tx["rigid"] = cached(tf("rigid"), lambda: R.rigid(fixed, moving, fmask, mmask))
        if "affine" in want:
            tx["affine"] = cached(tf("affine"), lambda: R.affine(fixed, moving, tx["rigid"], fmask))
        if "bspline" in want:
            tx["bspline"] = cached(tf(f"bspline{a.mesh}"),
                                   lambda: R.bspline(fixed, moving, tx["affine"], fmask, mesh=a.mesh))
        bl = rows[cols("bl")].to_numpy()
        out = {}
        for m in a.methods:
            if m == "local":
                pts = [local_point(fixed, moving, tx["affine"], p) for p in bl]
                nfb = sum(p is None for p in pts)
                pts = [R.move_points(tx["affine"], [b])[0] if p is None else p for p, b in zip(pts, bl)]
                if nfb:
                    print(pid, "local: affine point kept for", nfb, "of", len(bl), "lesions", flush=True)
            else:
                pts = R.move_points(tx[m], bl)
            out[m] = pd.DataFrame(pts, columns=XYZ).assign(patient=pid, lesion=rows["lesion"].to_numpy())
        return pid, out, None
    except Exception:  # noqa: BLE001 - one bad scan pair must not stop the other 300
        return pid, {}, traceback.format_exc(limit=3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("out", help="TEAM1_DIR with lesions.csv")
    ap.add_argument("--methods", default="rigid,affine,local,bspline")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=0, help="SimpleITK threads per worker; 0 = cpus / workers")
    ap.add_argument("--mesh", type=int, default=8, help="B-spline control points per axis")
    ap.add_argument("--patients", default="", help="comma-separated subset, for a quick check")
    a = ap.parse_args()
    a.methods = a.methods.split(",")
    assert set(a.methods) <= set(NEEDS), f"unknown method in {a.methods}"
    a.threads = a.threads or max(1, len(os.sched_getaffinity(0)) // a.workers)
    out = Path(a.out)
    for d in ("transforms", "centres"):
        (out / d).mkdir(parents=True, exist_ok=True)

    les = read_lesions(out / "lesions.csv")
    if a.patients:
        les = les[les["patient"].isin(a.patients.split(","))]
    jobs = [(key, g, a) for key, g in pairs(les)]
    res = {m: [] for m in a.methods}
    failed = 0
    with Pool(a.workers) as pool:
        for pid, o, err in pool.imap_unordered(do_pair, jobs):
            if err:
                failed += 1
                print(pid, "FAILED", err, flush=True)
                continue
            for m, d in o.items():
                res[m].append(d)
            print(pid, "ok", flush=True)
    print("pairs", len(jobs), "failed", failed)
    for m, ds in res.items():
        name = f"bspline{a.mesh}" if m == "bspline" else m
        f = out / "centres" / f"{name}.csv"
        new = pd.concat(ds)[["patient", "lesion", *XYZ]] if ds else pd.DataFrame(columns=["patient", "lesion", *XYZ])
        if f.exists():  # keep the rows of patients this run did not touch (a --patients rerun)
            old = read_centres(f)
            new = pd.concat([old[~old["patient"].isin(les["patient"])], new])
        if len(new):
            new.sort_values(["patient", "lesion"]).to_csv(f, index=False, float_format="%.3f")


if __name__ == "__main__":
    main()
