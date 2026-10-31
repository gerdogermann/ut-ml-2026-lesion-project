"""Dump per-lesion errors and the worst heatmaps of a trained SiameseNet, for notebooks/worst_cases.ipynb.

Usage (from repo root):
  PYTHONPATH=src python scripts/worst_cases.py --ckpt runs/<job>/best.pt --root DATA/lesions \
      --split-file DATA/split.json [--bl-mask] [--top 20] [--split val]
Writes next to the checkpoint, in worst/: errors.csv (every lesion, worst first) and heatmaps.npz
(32^3 probability maps of the top N). CT is not copied: the notebook reads it from --root.
Unknown flags are ignored, so train_siamese.sbatch can pass its train.py flags through.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from lesionloc.dataset import LesionDataset
from lesionloc.losses import fu_radius_mm
from lesionloc.metrics import compute_metrics
from lesionloc.model import SiameseNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--bl-mask", action="store_true", help="must match the checkpoint's training run")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    a, _ = ap.parse_known_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseNet(bl_mask=a.bl_mask).to(dev)
    model.load_state_dict(torch.load(a.ckpt, map_location=dev))
    model.eval()
    ds = LesionDataset(a.root, a.split_file, a.split, masks=True)
    rows, maps = [], {}
    with torch.no_grad():
        for b, (x, t, p) in enumerate(DataLoader(ds, a.batch_size)):  # no shuffle: item i is ds.files[i]
            delta, logits = model(x.to(dev))
            prob = logits.float().flatten(1).softmax(1).view_as(logits).cpu()
            delta, r = delta.cpu().numpy(), fu_radius_mm(x[:, 3:4]).numpy()
            for j in range(len(t)):
                name = ds.files[b * a.batch_size + j].stem
                rows.append(dict(lesion=name, err_mm=float(np.linalg.norm(delta[j] - t[j].numpy())),
                                 tx=t[j, 0].item(), ty=t[j, 1].item(), tz=t[j, 2].item(),
                                 px=delta[j, 0], py=delta[j, 1], pz=delta[j, 2], fu_radius_mm=r[j],
                                 peak_prob=prob[j].max().item(), has_propagated=bool(p[j])))
                maps[name] = prob[j].numpy().astype(np.float16)

    df = pd.DataFrame(rows).sort_values("err_mm", ascending=False)
    m = compute_metrics(df[["px", "py", "pz"]].to_numpy(), df[["tx", "ty", "tz"]].to_numpy())
    assert np.isclose(m["Mean error"], df.err_mm.mean(), atol=1e-3)
    out = Path(a.ckpt).parent / "worst"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "errors.csv", index=False, float_format="%.3f")
    np.savez_compressed(out / "heatmaps.npz", **{n: maps[n] for n in df.lesion[:a.top]})
    print(f"{a.split}: {len(df)} lesions, mean {m['Mean error']:.2f} mm, median {m['Median error']:.2f} mm; "
          f"wrote {out}/errors.csv, heatmaps.npz (top {a.top})")
    print(df.head(a.top).to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
