"""Team 1 step 3: organ landmarks from TotalSegmentator masks.

Usage: PYTHONPATH=src python scripts/team1_organs.py TEAM1_DIR [--workers N]
Needs TEAM1_DIR/lesions.csv (scripts/team1_lesions.py) and TEAM1_DIR/totalseg/<pid>_<BL|FU>_img_<ii>.nii.gz
(scripts/totalseg.sbatch). Caches per-scan organ centroids in TEAM1_DIR/organs.csv and per-lesion host organs in
TEAM1_DIR/organ_hosts.csv (delete them to recompute), then writes
  centres/organ_shift.csv                       step 3a: shift of the host organ, else the nearest organ
  centres/organ_affine_k<K>_s<sigma>_l<lam>.csv  step 3b: local affine, one file per grid point
Pick K, sigma and lam on train with team1_eval.py, then report that one file on val.
"""
import argparse
import itertools
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from lesionloc.organs import matched, predict_affine, predict_shift, scan_organs
from lesionloc.team1 import cols, read_lesions, scan_name, write_centres

GRID = dict(k=[4, 8], sigma=[50, 100, 200], lam=[1e2, 1e4])


def do_scan(args):
    path, scan, lesion_ids, pts = args
    organs, hosts = scan_organs(path, pts)
    rows = [dict(scan=scan, label=l, x=c[0], y=c[1], z=c[2], n_voxels=n, cut=cut) for l, (c, n, cut) in organs.items()]
    return rows, [dict(scan=scan, lesion=i, host=h) for i, h in zip(lesion_ids, hosts)]


def build_cache(out, les, workers):
    seg = out / "totalseg"
    jobs = []
    for tag in ("BL", "FU"):
        col = "img_id_bl" if tag == "BL" else "img_id_fu"
        for (pid, i), g in les.groupby(["patient", col]):
            scan = scan_name(pid, tag, i)
            # hosts are only needed in BL, where the lesion position is known
            pts = g[cols("bl")].to_numpy() if tag == "BL" else []
            ids = g["lesion"].tolist() if tag == "BL" else []
            jobs.append((seg / scan, scan, ids, pts))
    missing = [j[1] for j in jobs if not j[0].exists()]
    if missing:
        print("missing TotalSegmentator masks:", len(missing), missing[:5])
    organs, hosts = [], []
    with Pool(workers) as pool:
        for o, h in pool.imap_unordered(do_scan, [j for j in jobs if j[0].exists()]):
            organs += o
            hosts += h
    pd.DataFrame(organs).to_csv(out / "organs.csv", index=False, float_format="%.3f")
    pd.DataFrame(hosts).to_csv(out / "organ_hosts.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", help="TEAM1_DIR with lesions.csv and totalseg/")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out)
    les = read_lesions(out / "lesions.csv")
    if not (out / "organs.csv").exists() or not (out / "organ_hosts.csv").exists():
        build_cache(out, les, a.workers)

    org = {s: {int(r.label): (np.array([r.x, r.y, r.z]), int(r.n_voxels), bool(r.cut)) for r in g.itertuples()}
           for s, g in pd.read_csv(out / "organs.csv").groupby("scan")}
    hosts = {(r.scan, r.lesion): r.host for r in pd.read_csv(out / "organ_hosts.csv").itertuples()}

    shift, used, on_host, aff = [], [], 0, {p: [] for p in itertools.product(*GRID.values())}
    for r in les.itertuples():
        sb, sf = scan_name(r.patient, "BL", r.img_id_bl), scan_name(r.patient, "FU", r.img_id_fu)
        bl = np.array([r.bl_x, r.bl_y, r.bl_z])
        labels, c_bl, c_fu = matched(org.get(sb, {}), org.get(sf, {}))
        host = hosts.get((sb, r.lesion), 0)
        p, l = predict_shift(bl, host, labels, c_bl, c_fu)
        shift.append(p)
        used.append(l)
        on_host += bool(host) and l == host
        for k, s, lam in aff:
            aff[k, s, lam].append(predict_affine(bl, c_bl, c_fu, k, s, lam))
    print("lesions", len(les), "no usable organ", int((np.array(used) < 0).sum()), "shifted by host organ", on_host)
    write_centres(les, shift, out / "centres" / "organ_shift.csv")
    for (k, s, lam), pts in aff.items():
        write_centres(les, pts, out / "centres" / f"organ_affine_k{k}_s{s}_l{lam:g}.csv")


if __name__ == "__main__":
    main()
