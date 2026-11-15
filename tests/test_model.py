import torch

from lesionloc.model import LesionNet


def test_output_shape():
    y = LesionNet()(torch.zeros(2, 2, 64, 64, 64))
    assert y.shape == (2, 3)
    assert torch.isfinite(y).all()
