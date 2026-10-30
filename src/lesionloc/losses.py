"""Team 2 loss: L = w_dist * L_dist + lambda_hm * L_hm + lambda_mask * L_mask (see reports/, Teams.md)."""
import math

import torch
import torch.nn.functional as F

JITTER_SD = 8.0  # mm; scales L_dist to O(1)


def fu_radius_mm(fu_mask):
    """(B,1,64,64,64) mask at 2 mm -> (B,) radius of a sphere with the same volume."""
    vol = fu_mask.flatten(1).sum(1) * 8.0
    return (3 * vol / (4 * math.pi)) ** (1 / 3)


def gaussian_target(t, sigma, coords):
    """Normalised 3D Gaussian over cells, (B,Z,Y,X): centre t (B,3) mm (x,y,z), sigma (B,) mm."""
    g = torch.exp(-(coords[None, None] - t[:, :, None]) ** 2 / (2 * sigma[:, None, None] ** 2))  # (B,3,32)
    q = g[:, 2, :, None, None] * g[:, 1, None, :, None] * g[:, 0, None, None, :]
    return q / q.flatten(1).sum(1)[:, None, None, None]


def team2_loss(delta, logits, t, fu_mask, coords, lambda_hm=0.0, lambda_mask=0.0, w_dist=1.0):
    """Returns (total, dict of per-term means). delta, t (B,3) mm; logits (B,32,32,32); fu_mask (B,1,64,64,64)."""
    l_dist = torch.sqrt(((delta - t) ** 2).sum(1) + 0.1 ** 2).mean() / JITTER_SD
    logp = logits.float().flatten(1).log_softmax(1)
    terms = {"dist": l_dist}
    total = w_dist * l_dist
    r = fu_radius_mm(fu_mask)
    if lambda_hm:
        q = gaussian_target(t, (0.5 * r).clamp(4.0, 12.0), coords)
        terms["hm"] = -(q.flatten(1) * logp).sum(1).mean()
        total = total + lambda_hm * terms["hm"]
    if lambda_mask:
        m = F.max_pool3d(fu_mask, 4, stride=2, padding=1)  # same grid as SiameseNet's stem
        small = (r < 5.0)[:, None, None, None, None]
        m = torch.where(small, F.max_pool3d(m, 3, stride=1, padding=1), m)  # tiny lesions: dilate 1 cell
        terms["mask"] = -torch.logsumexp(logp.masked_fill(m.flatten(1) == 0, -1e4), 1).mean()
        total = total + lambda_mask * terms["mask"]
    return total, terms
