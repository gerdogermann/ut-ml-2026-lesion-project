"""Train LesionNet on npz patches (overfit check) or benchmark GPU memory/time on random tensors.

Usage (from repo root):
  PYTHONPATH=src python scripts/train.py --bench --batch-size 8
  PYTHONPATH=src python scripts/train.py data/example_lesions/*.npz --steps 300
  PYTHONPATH=src python scripts/train.py --root DATA/lesions --split-file DATA/split.json --n 100 --steps 2000
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lesionloc.model import SCALE, LesionNet


def load(path):
    # ponytail: inline loader until Gerdo's lesionloc.dataset.LesionDataset is merged, then use that
    d = np.load(path)
    return np.stack([d["bl"], d["fu"]]).astype(np.float32), d["target_mm"].astype(np.float32)


def pick_files(a):
    if a.files:
        return a.files
    train = set(json.loads(Path(a.split_file).read_text())["train"])
    files = sorted(f for f in Path(a.root).glob("*.npz") if f.name.split("_")[0] in train)
    return files[:a.n]


def bench(model, opt, a, dev):
    x = torch.rand(a.batch_size, 2, 64, 64, 64, device=dev)
    y = torch.zeros(a.batch_size, 3, device=dev)
    for i in range(a.steps + 3):
        if i == 3:  # skip warm-up steps (cuDNN autotune, allocator)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
        loss = F.smooth_l1_loss(model(x) / SCALE, y / SCALE)
        opt.zero_grad()
        loss.backward()
        opt.step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t) / a.steps * 1000
    mem = torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == "cuda" else float("nan")
    print(f"bench batch={a.batch_size}: {ms:.1f} ms/step, peak mem {mem:.2f} GiB")


def train(model, opt, a, dev):
    files = pick_files(a)
    xs, ys = zip(*(load(f) for f in files))
    x, y = torch.from_numpy(np.stack(xs)).to(dev), torch.from_numpy(np.stack(ys)).to(dev)
    zero_err = y.norm(dim=1).mean().item()
    print(f"{len(files)} files, zero-prediction mean error {zero_err:.2f} mm")
    for step in range(1, a.steps + 1):
        idx = torch.randperm(len(x), device=dev)[:a.batch_size]
        loss = F.smooth_l1_loss(model(x[idx]) / SCALE, y[idx] / SCALE)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == a.steps:
            print(f"step {step} loss {loss.item():.6f}", flush=True)

    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(x[i:i + a.batch_size]) for i in range(0, len(x), a.batch_size)])
    print(f"train mean error {(pred - y).norm(dim=1).mean().item():.2f} mm (zero baseline {zero_err:.2f} mm)")
    for f, p, t in list(zip(files, pred.tolist(), y.tolist()))[:3]:
        print(Path(f).name, "pred", np.round(p, 1), "true", np.round(t, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--root", help="dir of npz files (used with --split-file when no files given)")
    ap.add_argument("--split-file")
    ap.add_argument("--n", type=int, default=100, help="number of train files taken from --root")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bench", action="store_true", help="random tensors, report time and memory per step")
    a = ap.parse_args()

    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", dev, torch.cuda.get_device_name() if dev.type == "cuda" else "")
    model = LesionNet().to(dev)
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    (bench if a.bench else train)(model, opt, a, dev)


if __name__ == "__main__":
    main()
