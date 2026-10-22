"""Evaluation metrics for 3D lesion localisation in millimeters."""
from __future__ import annotations

import numpy as np


def compute_metrics(pred_mm: np.ndarray, true_mm: np.ndarray) -> dict[str, float]:
    """Compute 3D localisation metrics from (N, 3) arrays in (x, y, z) mm order."""
    pred = np.asarray(pred_mm, dtype=np.float64)
    true = np.asarray(true_mm, dtype=np.float64)

    if pred.shape != true.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {true.shape}")
    if pred.ndim != 2 or pred.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) array, got {pred.shape}")
    if len(pred) == 0:
        return {
            k: float("nan")
            for k in (
                "Median error",
                "Mean error",
                "MEDx",
                "MEDy",
                "MEDz",
                "CPM@5",
                "CPM@10",
                "CPM@20",
            )
        }

    diff = pred - true
    abs_diff = np.abs(diff)
    med_x, med_y, med_z = np.mean(abs_diff, axis=0)
    dist = np.linalg.norm(diff, axis=1)

    return {
        "Median error": float(np.median(dist)),
        "Mean error": float(np.mean(dist)),
        "MEDx": float(med_x),
        "MEDy": float(med_y),
        "MEDz": float(med_z),
        "CPM@5": float(np.mean(dist <= 5.0)),
        "CPM@10": float(np.mean(dist <= 10.0)),
        "CPM@20": float(np.mean(dist <= 20.0)),
    }


def _test_hand_computed() -> None:
    pred = np.zeros((4, 3), dtype=np.float64)
    true = np.array(
        [
            [0.0, 0.0, 0.0],     # dist = 0.0
            [3.0, 4.0, 0.0],     # dist = 5.0
            [0.0, 6.0, 8.0],     # dist = 10.0
            [12.0, 16.0, 15.0],  # dist = 25.0
        ],
        dtype=np.float64,
    )
    m = compute_metrics(pred, true)
    expected = {
        "Median error": 7.5,
        "Mean error": 10.0,
        "MEDx": 3.75,
        "MEDy": 6.5,
        "MEDz": 5.75,
        "CPM@5": 0.5,
        "CPM@10": 0.75,
        "CPM@20": 0.75,
    }
    for k, v in expected.items():
        assert np.isclose(m[k], v), f"{k}: got {m[k]}, expected {v}"


if __name__ == "__main__":
    _test_hand_computed()
    print("metrics.py hand-computed test passed.")
