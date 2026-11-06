"""Train LesionNet or SiameseNet on the patient split, overfit a few npz files, or benchmark GPU memory/time.

Usage (from repo root):
  PYTHONPATH=src python scripts/train.py --root DATA/lesions --split-file DATA/split.json --epochs 30
  PYTHONPATH=src python scripts/train.py data/example_lesions/*.npz --epochs 150      # overfit check
  PYTHONPATH=src python scripts/train.py --bench --batch-size 8
Team 2 (needs Longitudinal-CT-processed-team2, which has masks):
  ... --model siamese [--lambda-hm 1 --dist-warmup 5] [--bl-mask] [--lambda-mask 0.5] [--template-k 5] [--prior-sd 12]
      [--ema 0.999]
  pretrained encoder (pip install dynamic-network-architectures): --encoder resenc --pretrained checkpoint_final.pth
      [--coarse] [--enc-mask] [--freeze none|early|all] [--enc-lr 1e-4] [--warmup 0.1] [--amp]
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from lesionloc.dataset import LesionDataset
from lesionloc.losses import team2_loss
from lesionloc.metrics import compute_metrics
from lesionloc.model import LesionNet, SiameseNet, load_nnfoundation


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


def train_step(model, opt, scaler, x, t, a, epoch):
    with torch.autocast(x.device.type, dtype=torch.float16, enabled=a.amp):
        loss, parts = step_loss(model, x, t, a, epoch)
    opt.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    return loss, parts


def bench(model, opt, scaler, a, dev):
    x = torch.rand(a.batch_size, 4 if a.model == "siamese" else 2, 64, 64, 64, device=dev)
    y = torch.zeros(a.batch_size, 3, device=dev)
    for i in range(a.steps + 3):
        if i == 3:  # skip warm-up steps (cuDNN autotune, allocator)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
        train_step(model, opt, scaler, x, y, a, a.epochs)
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
    ap.add_argument("--template-k", type=int, default=3, help="siamese: BL template side in 4 mm cells (odd)")
    ap.add_argument("--prior-sd", type=float, default=0.0, help="siamese: centre prior sd in mm; 0 = off")
    ap.add_argument("--ema", type=float, default=0.0,
                    help="validate and save an exponential moving average of the weights, e.g. 0.999; 0 = off")
    ap.add_argument("--aug-root", help="extra train-split npz dir (augment_team2.sbatch output)")
    ap.add_argument("--no-flips", action="store_true",
                    help="disable random training flips; does not disable rotations stored in --aug-root")
    ap.add_argument("--rejitter-sd", type=float, default=0.0,
                    help="train split only: fresh FU offset each load, mm sd per axis (needs fu_big); 0 = stored jitter")
    ap.add_argument("--dist-warmup", type=int, default=0,
                    help="siamese: distance loss off for N epochs, then ramped in over N; 0 = always on")
    ap.add_argument("--encoder", choices=["small", "resenc"], default="small",
                    help="siamese: resenc = nnFoundationCNN stages 0-1 (+ stage 2 with --coarse)")
    ap.add_argument("--pretrained", help="resenc: nnFoundationCNN checkpoint_final.pth; omit for random init")
    ap.add_argument("--coarse", action="store_true", help="resenc: add the 8 mm stage-2 correlation map")
    ap.add_argument("--enc-mask", action="store_true", help="resenc: BL mask as zero-initialised 2nd stem channel")
    ap.add_argument("--freeze", choices=["none", "early", "all"], default="none",
                    help="resenc: early = stem + stage 0, all = whole encoder")
    ap.add_argument("--enc-lr", type=float, help="lr of the trainable encoder layers; default --lr")
    ap.add_argument("--warmup", type=float, default=0.0, help="linear lr warmup, fraction of all steps")
    ap.add_argument("--amp", action="store_true", help="fp16 autocast + grad scaling (CUDA)")
    ap.add_argument("--bench", action="store_true", help="random tensors, report time and memory per step")
    ap.add_argument("--steps", type=int, default=20, help="bench steps")
    a = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", dev, torch.cuda.get_device_name() if dev.type == "cuda" else "")
    print("args", vars(a))
    masks = a.model == "siamese"
    model = (SiameseNet(bl_mask=a.bl_mask, k=a.template_k, prior_sd=a.prior_sd, encoder=a.encoder, coarse=a.coarse,
                        enc_mask=a.enc_mask) if masks else LesionNet())
    if a.pretrained:
        load_nnfoundation(model.encoder, a.pretrained)
    if a.freeze != "none":
        enc = model.encoder
        (enc if a.freeze == "all" else nn.ModuleList([enc.stem, enc.stages[0]])).requires_grad_(False)
    model = model.to(dev)
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M, "
          f"trainable {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f} M")
    is_enc = lambda n: n.startswith("encoder.")
    groups = [{"params": [p for n, p in model.named_parameters() if p.requires_grad and not is_enc(n)]},
              {"params": [p for n, p in model.named_parameters() if p.requires_grad and is_enc(n)], "lr": a.enc_lr or a.lr}]
    opt = torch.optim.AdamW([g for g in groups if g["params"]], lr=a.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(enabled=a.amp)
    if a.bench:
        return bench(model, opt, scaler, a, dev)

    if a.files:  # overfit check: train and "val" are the same files, no augmentation
        train_ds = val_ds = load_files(a.files, masks)
    else:
        roots = [a.root, a.aug_root] if a.aug_root else a.root
        train_ds = LesionDataset(roots, a.split_file, "train", augment=not a.no_flips,
                                 masks=masks, rejitter_sd=a.rejitter_sd)
        val_ds = LesionDataset(a.root, a.split_file, "val", masks=masks)
    print(f"train {len(train_ds)} lesions, val {len(val_ds)} lesions")
    loader = lambda ds, shuffle: DataLoader(ds, a.batch_size, shuffle=shuffle, num_workers=0 if a.files else a.workers,
                                            pin_memory=dev.type == "cuda")
    train_dl, val_dl = loader(train_ds, True), loader(val_ds, False)
    total = a.epochs * len(train_dl)
    warm = int(a.warmup * total)
    # linear warmup, then cosine to 0 (warm = 0: same as CosineAnnealingLR(opt, total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (total - warm))))
    # overfit mode takes 1 step per epoch: an EMA would lag far behind, so it is off there
    ema = (torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(a.ema))
           if a.ema and not a.files else None)
    evaluated = ema.module if ema else model

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for epoch in range(1, a.epochs + 1):
        t0, losses, terms = time.perf_counter(), [], {}
        for x, t, _ in train_dl:
            loss, parts = train_step(model, opt, scaler, x.to(dev), t.to(dev), a, epoch - 1)
            sched.step()
            if ema:
                ema.update_parameters(model)
            losses.append(loss.item())
            for k, v in parts.items():
                terms.setdefault(k, []).append(v.item())
        parts = "".join(f" {k} {np.mean(v):.3f}" for k, v in terms.items())
        print(f"epoch {epoch} loss {np.mean(losses):.3f}{parts} ({time.perf_counter() - t0:.0f} s)", end="  ")
        m = report("val", *predict(evaluated, val_dl, dev))
        if m["Mean error"] < best:
            best = m["Mean error"]
            torch.save(evaluated.state_dict(), out / "best.pt")
    print(f"best val mean error {best:.2f} mm, checkpoint {out / 'best.pt'}")

    if a.test:
        evaluated.load_state_dict(torch.load(out / "best.pt", map_location=dev))
        test_dl = loader(LesionDataset(a.root, a.split_file, "test", masks=masks), False)
        report("test", *predict(evaluated, test_dl, dev))


if __name__ == "__main__":
    main()
