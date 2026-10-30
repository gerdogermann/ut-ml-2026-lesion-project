"""Train LesionNet or SiameseNet on the patient split, overfit a few npz files, or benchmark GPU memory/time.

Usage (from repo root):
  PYTHONPATH=src python scripts/train.py --root DATA/lesions --split-file DATA/split.json --epochs 30
  PYTHONPATH=src python scripts/train.py data/example_lesions/*.npz --epochs 150      # overfit check
  PYTHONPATH=src python scripts/train.py --bench --batch-size 8
Team 2 (needs Longitudinal-CT-processed-team2, which has masks):
  ... --model siamese [--lambda-hm 1 --dist-warmup 5] [--bl-mask] [--lambda-mask 0.5]
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from lesionloc.dataset import LesionDataset
from lesionloc.losses import team2_loss
from lesionloc.metrics import compute_metrics
from lesionloc.model import LesionNet, SiameseNet


def load_files(files, masks=False):
    keys = ["bl", "fu", "bl_mask", "fu_mask"] if masks else ["bl", "fu"]
    xs, ts, ps = [], [], []
    for f in files:
        d = np.load(f)
        xs.append(np.stack([d[k] for k in keys]).astype(np.float32))
        ts.append(d["target_mm"].astype(np.float32))
        ps.append(bool(d["has_propagated"]))
    return TensorDataset(torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ts)), torch.tensor(ps))


def step_loss(model, x, t, a, epoch):
    """Returns (loss, dict of loss terms). Siamese: Team 2 loss, distance term ramped in after --dist-warmup."""
    if not isinstance(model, SiameseNet):
        return F.smooth_l1_loss(model(x), t), {}  # mm units, quadratic below 1 mm
    n = a.dist_warmup
    w = min(1.0, max(0.0, (epoch - n) / n)) if n else 1.0  # epoch is 0-based: 0 for n epochs, ramp over n, then 1
    delta, logits = model(x)
    return team2_loss(delta, logits, t, x[:, 3:4], model.coords, a.lambda_hm, a.lambda_mask, w)


def bench(model, opt, a, dev):
    x = torch.rand(a.batch_size, 4 if a.model == "siamese" else 2, 64, 64, 64, device=dev)
    y = torch.zeros(a.batch_size, 3, device=dev)
    for i in range(a.steps + 3):
        if i == 3:  # skip warm-up steps (cuDNN autotune, allocator)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
        loss, _ = step_loss(model, x, y, a, a.epochs)
        opt.zero_grad()
        loss.backward()
        opt.step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t) / a.steps * 1000
    mem = torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == "cuda" else float("nan")
    print(f"bench batch={a.batch_size}: {ms:.1f} ms/step, peak mem {mem:.2f} GiB")


@torch.no_grad()
def predict(model, loader, dev):
    model.eval()
    preds, trues, props = [], [], []
    for x, t, p in loader:
        y = model(x.to(dev))
        preds.append((y[0] if isinstance(y, tuple) else y).cpu())
        trues.append(t)
        props.append(p)
    model.train()
    return torch.cat(preds).numpy(), torch.cat(trues).numpy(), torch.cat(props).numpy()


def report(name, pred, true, prop):
    m, z = compute_metrics(pred, true), compute_metrics(np.zeros_like(true), true)
    line = f"{name}: mean {m['Mean error']:.2f} mm, median {m['Median error']:.2f} mm, CPM@10 {m['CPM@10']:.1%}"
    if prop.any() and not prop.all():
        line += f", has_propagated mean {compute_metrics(pred[prop], true[prop])['Mean error']:.2f} mm"
    print(f"{line}  (zero baseline mean {z['Mean error']:.2f} mm)", flush=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="npz files to overfit (skips the split)")
    ap.add_argument("--root", help="dir of lesion npz files")
    ap.add_argument("--split-file")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="runs", help="best checkpoint goes to OUT/best.pt")
    ap.add_argument("--test", action="store_true", help="also report the test split with the best checkpoint")
    ap.add_argument("--model", choices=["cnn", "siamese"], default="cnn")
    ap.add_argument("--bl-mask", action="store_true", help="siamese: BL mask as input + mask-pooled template")
    ap.add_argument("--lambda-hm", type=float, default=0.0, help="siamese: heatmap CE weight")
    ap.add_argument("--lambda-mask", type=float, default=0.0, help="siamese: FU mask-mass weight")
    ap.add_argument("--dist-warmup", type=int, default=0,
                    help="siamese: distance loss off for N epochs, then ramped in over N; 0 = always on")
    ap.add_argument("--bench", action="store_true", help="random tensors, report time and memory per step")
    ap.add_argument("--steps", type=int, default=20, help="bench steps")
    a = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", dev, torch.cuda.get_device_name() if dev.type == "cuda" else "")
    print("args", vars(a))
    masks = a.model == "siamese"
    model = (SiameseNet(bl_mask=a.bl_mask) if masks else LesionNet()).to(dev)
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    if a.bench:
        return bench(model, opt, a, dev)

    if a.files:  # overfit check: train and "val" are the same files, no augmentation
        train_ds = val_ds = load_files(a.files, masks)
    else:
        train_ds = LesionDataset(a.root, a.split_file, "train", augment=True, masks=masks)
        val_ds = LesionDataset(a.root, a.split_file, "val", masks=masks)
    print(f"train {len(train_ds)} lesions, val {len(val_ds)} lesions")
    loader = lambda ds, shuffle: DataLoader(ds, a.batch_size, shuffle=shuffle, num_workers=0 if a.files else a.workers,
                                            pin_memory=dev.type == "cuda")
    train_dl, val_dl = loader(train_ds, True), loader(val_ds, False)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * len(train_dl))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for epoch in range(1, a.epochs + 1):
        t0, losses, terms = time.perf_counter(), [], {}
        for x, t, _ in train_dl:
            loss, parts = step_loss(model, x.to(dev), t.to(dev), a, epoch - 1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
            for k, v in parts.items():
                terms.setdefault(k, []).append(v.item())
        parts = "".join(f" {k} {np.mean(v):.3f}" for k, v in terms.items())
        print(f"epoch {epoch} loss {np.mean(losses):.3f}{parts} ({time.perf_counter() - t0:.0f} s)", end="  ")
        m = report("val", *predict(model, val_dl, dev))
        if m["Mean error"] < best:
            best = m["Mean error"]
            torch.save(model.state_dict(), out / "best.pt")
    print(f"best val mean error {best:.2f} mm, checkpoint {out / 'best.pt'}")

    if a.test:
        model.load_state_dict(torch.load(out / "best.pt", map_location=dev))
        test_dl = loader(LesionDataset(a.root, a.split_file, "test", masks=masks), False)
        report("test", *predict(model, test_dl, dev))


if __name__ == "__main__":
    main()
