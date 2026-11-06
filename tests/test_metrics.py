import numpy as np

from lesionloc.metrics import _test_hand_computed, compute_metrics, window_metrics


def test_compute_metrics_hand_computed():
    _test_hand_computed()


def test_example_lesions_zero_baseline():
    # Matches the two example lesions in logs/train-2941892.out (mean error 14.04 mm)
    true = np.array([[18.5, 8.1, -17.1], [0.5, 0.9, 1.2]], dtype=np.float64)
    pred = np.zeros_like(true)
    m = compute_metrics(pred, true)
    assert np.isclose(m["Mean error"], 14.04, atol=0.05)


def test_window_metrics_linf():
    # Euclidean 57.4 mm but every axis <= 56 mm: still inside the window; 57 mm on one axis: outside
    true = np.array([[40.0, 40.0, 10.0], [57.0, 0.0, 0.0], [0.0, 0.0, 56.0]])
    m = window_metrics(np.zeros_like(true), true)
    assert np.isclose(m["In window@56"], 2 / 3)
    assert np.isclose(m["Mean error"], (np.sqrt(3300) + 57 + 56) / 3)
