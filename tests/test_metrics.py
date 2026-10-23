import numpy as np

from lesionloc.metrics import _test_hand_computed, compute_metrics


def test_compute_metrics_hand_computed():
    _test_hand_computed()


def test_example_lesions_zero_baseline():
    # Matches the two example lesions in logs/train-2941892.out (mean error 14.04 mm)
    true = np.array([[18.5, 8.1, -17.1], [0.5, 0.9, 1.2]], dtype=np.float64)
    pred = np.zeros_like(true)
    m = compute_metrics(pred, true)
    assert np.isclose(m["Mean error"], 14.04, atol=0.05)
