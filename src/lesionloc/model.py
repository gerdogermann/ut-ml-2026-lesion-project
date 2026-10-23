"""Small residual 3D CNN: (B,2,64,64,64) BL+FU patches -> (B,3) displacement in mm, (x,y,z) order."""
from torch import nn


class ResBlock(nn.Module):
    """Two 3x3x3 convs with a skip; the first conv halves the resolution (stride 2)."""

    def __init__(self, cin, cout):
        super().__init__()
        # GroupNorm, not BatchNorm: batches are small (2-16 patches)
        self.body = nn.Sequential(
            nn.Conv3d(cin, cout, 3, stride=2, padding=1), nn.GroupNorm(8, cout), nn.ReLU(inplace=True),
            nn.Conv3d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout))
        self.skip = nn.Sequential(nn.Conv3d(cin, cout, 1, stride=2), nn.GroupNorm(8, cout))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.body(x) + self.skip(x))


class LesionNet(nn.Module):
    def __init__(self, width=16, dropout=0.2):
        super().__init__()
        w = width
        self.features = nn.Sequential(ResBlock(2, w), ResBlock(w, 2 * w), ResBlock(2 * w, 4 * w),
                                      ResBlock(4 * w, 8 * w))  # 64 -> 4
        # ponytail: flatten the 4^3 map instead of global pooling; pooling discards *where* features are,
        # and the target is a position. Swap to AdaptiveAvgPool3d(1) + Linear(8w, 3) to compare.
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(8 * w * 4 ** 3, 3))
        nn.init.zeros_(self.head[-1].weight)  # start at the zero (registration) baseline
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        return self.head(self.features(x))  # mm
