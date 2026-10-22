"""Small 3D CNN: (B,2,64,64,64) BL+FU patches -> (B,3) displacement in mm, (x,y,z) order."""
import torch
from torch import nn

SCALE = 56.0  # mm; preprocessing drops targets with |t| > 64*2/2 - 8, so outputs stay in about [-1,1] * SCALE


def block(cin, cout):
    # GroupNorm, not BatchNorm: batches are tiny (2-8 patches)
    return nn.Sequential(nn.Conv3d(cin, cout, 3, stride=2, padding=1), nn.GroupNorm(8, cout), nn.ReLU(inplace=True))


class LesionNet(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        w = width
        self.features = nn.Sequential(block(2, w), block(w, 2 * w), block(2 * w, 4 * w), block(4 * w, 8 * w))  # 64 -> 4
        # ponytail: flatten the 4^3 map instead of global pooling; pooling discards *where* features are,
        # and the target is a position. Swap to AdaptiveAvgPool3d(1) + Linear(8w, 3) to compare.
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(8 * w * 4 ** 3, 3))

    def forward(self, x):
        return self.head(self.features(x)) * SCALE  # mm
