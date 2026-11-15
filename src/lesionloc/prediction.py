"""Frozen Team 2 model inference; consumes patches and returns offsets in mm."""
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from lesionloc.model import SiameseNet, soft_argmax

FLIPS = [f for n in range(4) for f in itertools.combinations(((4, 0), (3, 1), (2, 2)), n)]


def flip_preds(model, x, readout):
    """(8,B,3): one prediction per flip combination, each flip undone in (x,y,z) offset space."""
    out = []
    for f in FLIPS:
        delta = readout(model(x.flip([i for i, _ in f]) if f else x)[1]).clone()
        for _, j in f:
            delta[:, j] = -delta[:, j]
        out.append(delta)
    return torch.stack(out)


def flip_tta(model, x, readout):
    """Average 8 flip predictions, undoing each flip in (x,y,z) offset space."""
    return flip_preds(model, x, readout).mean(0)


def load_siamese(checkpoint, device="cpu", bl_mask=True, template_k=5):
    """SiameseNet with the architecture in args.json beside the checkpoint (scripts/train.py writes it).
    Without one (older runs), the small encoder with bl_mask / template_k."""
    f = Path(checkpoint).parent / "args.json"
    a = json.loads(f.read_text()) if f.exists() else dict(bl_mask=bl_mask, template_k=template_k)
    model = SiameseNet(bl_mask=a["bl_mask"], k=a["template_k"], prior_sd=a.get("prior_sd", 0.0),
                       encoder=a.get("encoder", "small"), coarse=a.get("coarse", False), enc_mask=a.get("enc_mask", False))
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    return model.to(device).eval()


class PatchPredictor:
    """Only BL CT, FU CT and optional BL lesion mask enter the network.

    The unused fourth channel preserves the existing checkpoint interface.
    It is filled with zeros; no FU mask or FU centre is accepted by this API.
    Several checkpoints form an ensemble: offsets are averaged, and with TTA `spread` is the RMS distance (mm)
    of all checkpoint x flip predictions from their mean, a failure score.
    """

    def __init__(self, checkpoint, *, bl_mask=True, template_k=5, tta=True, device="cpu"):
        self.device, self.tta = torch.device(device), tta
        ckpts = [checkpoint] if isinstance(checkpoint, (str, Path)) else checkpoint
        self.models = [load_siamese(c, self.device, bl_mask, template_k) for c in ckpts]

    @torch.inference_mode()
    def predict(self, bl, fu, bl_mask):
        images = np.stack([bl, fu, bl_mask, np.zeros_like(bl)]).astype(np.float32)
        if images.shape != (4, 64, 64, 64) or not np.isfinite(images).all():
            raise ValueError("Expected finite 64^3 patches")
        x = torch.from_numpy(images[None]).to(self.device)
        out = {"plain": torch.stack([m(x)[0][0] for m in self.models]).mean(0).cpu().numpy()}
        if self.tta:
            p = torch.cat([flip_preds(m, x, lambda lg, m=m: soft_argmax(lg, m.coords))[:, 0] for m in self.models])
            out["tta"] = p.mean(0).cpu().numpy()
            out["spread"] = float((p - p.mean(0)).norm(dim=1).pow(2).mean().sqrt())
        return out
