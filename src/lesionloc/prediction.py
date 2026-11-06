"""Frozen Team 2 model inference; consumes patches and returns offsets in mm."""
import itertools

import numpy as np
import torch

from lesionloc.model import SiameseNet, soft_argmax

FLIPS = [f for n in range(4) for f in itertools.combinations(((4, 0), (3, 1), (2, 2)), n)]


def flip_tta(model, x, readout):
    """Average 8 flip predictions, undoing each flip in (x,y,z) offset space."""
    out = 0
    for f in FLIPS:
        delta = readout(model(x.flip([i for i, _ in f]) if f else x)[1]).clone()
        for _, j in f:
            delta[:, j] = -delta[:, j]
        out = out + delta
    return out / len(FLIPS)


class PatchPredictor:
    """Only BL CT, FU CT and optional BL lesion mask enter the network.

    The unused fourth channel preserves the existing checkpoint interface.
    It is filled with zeros; no FU mask or FU centre is accepted by this API.
    """

    def __init__(self, checkpoint, *, bl_mask=True, template_k=5, tta=True, device="cpu"):
        self.device, self.tta = torch.device(device), tta
        self.model = SiameseNet(bl_mask=bl_mask, k=template_k).to(self.device)
        self.model.load_state_dict(torch.load(checkpoint, map_location=self.device, weights_only=True))
        self.model.eval()

    @torch.inference_mode()
    def predict(self, bl, fu, bl_mask):
        images = np.stack([bl, fu, bl_mask, np.zeros_like(bl)]).astype(np.float32)
        if images.shape != (4, 64, 64, 64) or not np.isfinite(images).all():
            raise ValueError("Expected finite 64^3 patches")
        x = torch.from_numpy(images[None]).to(self.device)
        plain = self.model(x)[0]
        if not self.tta:
            return {"plain": plain[0].cpu().numpy()}
        readout = lambda logits: soft_argmax(logits, self.model.coords)
        return {"plain": plain[0].cpu().numpy(),
                "tta": flip_tta(self.model, x, readout)[0].cpu().numpy()}
