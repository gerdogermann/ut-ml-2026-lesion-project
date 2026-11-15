"""Evaluate localisation baselines (Zero and Train Mean) on target_mm only.

Usage (from repo root):
  # 1. Test with mock data (no external data access needed):
  PYTHONPATH=src python scripts/baselines.py --mock

  # 2. Test on the 2 real example files inside the repo:
  PYTHONPATH=src python scripts/baselines.py data/example_lesions/*.npz

  # 3. Run on full dataset once group folder permissions are granted:
  PYTHONPATH=src python scripts/baselines.py --root $DATA/lesions --split-file $DATA/split.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from lesionloc.metrics import compute_metrics

DEFAULT_DATA = Path(
    os.environ.get(
        "DATA",
        "/gpfs/helios/projects/ml_project_lesion_tracking/data/Longitudinal-CT-processed",
    )
)


def load_meta(path: Path) -> tuple[np.ndarray, bool]:
    """Load only target_mm and has_propagated without decompressing bl/fu volumes."""
    with np.load(path) as d:
        target_mm = d["target_mm"].astype(np.float64)
        has_prop = bool(d["has_propagated"]) if "has_propagated" in d.files else True
    return target_mm, has_prop


def load_split(root: Path, patient_ids: set[str]) -> tuple[np.ndarray, np.ndarray]:
    files = sorted(
        f for f in Path(root).glob("*.npz") if f.name.split("_")[0] in patient_ids
    )
    if not files:
        raise FileNotFoundError(f"No .npz files found in {root} for given split.")
    targets, props = zip(*(load_meta(f) for f in files))
    return np.stack(targets, axis=0), np.asarray(props, dtype=bool)


def make_mock_split(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic (target_mm, has_propagated) to test pipeline without data access."""
    has_prop = rng.random(n) < 0.85  # ~85% propagated via registration
    target_mm = np.empty((n, 3), dtype=np.float64)
    n_prop = int(np.sum(has_prop))
    # Propagated lesions have smaller residual error (~3-5 mm per axis)
    target_mm[has_prop] = rng.normal(loc=[0.2, -0.4, 0.8], scale=3.5, size=(n_prop, 3))
    # Unpropagated lesions have larger residual error (~12 mm per axis)
    target_mm[~has_prop] = rng.normal(
        loc=[1.0, -1.5, 2.5], scale=12.0, size=(n - n_prop, 3)
    )
    return target_mm, has_prop


def evaluate_all_subsets(
    pred_mm: np.ndarray, true_mm: np.ndarray, has_prop: np.ndarray
) -> list[tuple[str, int, dict[str, float]]]:
    subsets = [
        ("Overall", np.ones(len(true_mm), dtype=bool)),
        ("has_propagated=True", has_prop),
        ("has_propagated=False", ~has_prop),
    ]
    out = []
    for name, mask in subsets:
        n = int(np.sum(mask))
        if n > 0:
            out.append((name, n, compute_metrics(pred_mm[mask], true_mm[mask])))
    return out


def format_table(rows: list[dict]) -> str:
    headers = [
        "Split",
        "Baseline",
        "Subset",
        "N",
        "Median error (mm)",
        "Mean error (mm)",
        "MEDx (mm)",
        "MEDy (mm)",
        "MEDz (mm)",
        "CPM@5 (%)",
        "CPM@10 (%)",
        "CPM@20 (%)",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in rows:
        lines.append(
            f"| {r['Split']} | {r['Baseline']} | {r['Subset']} | {r['N']} "
            f"| {r['Median error']:.2f} | {r['Mean error']:.2f} "
            f"| {r['MEDx']:.2f} | {r['MEDy']:.2f} | {r['MEDz']:.2f} "
            f"| {r['CPM@5'] * 100:.1f}% | {r['CPM@10'] * 100:.1f}% | {r['CPM@20'] * 100:.1f}% |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="Optional .npz files for quick check")
    ap.add_argument("--root", type=Path, default=DEFAULT_DATA / "lesions")
    ap.add_argument("--split-file", type=Path, default=DEFAULT_DATA / "split.json")
    ap.add_argument(
        "--mock",
        action="store_true",
        help="Run on synthetic target_mm data when cluster data is inaccessible",
    )
    a = ap.parse_args()

    if a.mock:
        rng = np.random.default_rng(42)
        train_true, _ = make_mock_split(rng, 400)
        train_mean = np.mean(train_true, axis=0, keepdims=True)
        splits = {
            "val (mock)": make_mock_split(rng, 60),
            "test (mock)": make_mock_split(rng, 60),
        }
    elif a.files:
        targets, props = zip(*(load_meta(Path(f)) for f in a.files))
        true_mm = np.stack(targets, axis=0)
        has_prop = np.asarray(props, dtype=bool)
        splits = {"example_lesions": (true_mm, has_prop)}
        train_mean = np.mean(true_mm, axis=0, keepdims=True)
    else:
        split_dict = json.loads(a.split_file.read_text())
        train_true, _ = load_split(a.root, set(split_dict["train"]))
        train_mean = np.mean(train_true, axis=0, keepdims=True)
        splits = {
            "val": load_split(a.root, set(split_dict["val"])),
            "test": load_split(a.root, set(split_dict["test"])),
        }

    print(f"Train mean target_mm (x, y, z): {np.round(train_mean.ravel(), 3)} mm\n")

    rows = []
    for split_name, (true_mm, has_prop) in splits.items():
        pred_zero = np.zeros_like(true_mm)
        pred_mean = np.broadcast_to(train_mean, true_mm.shape)

        for b_name, pred in [
            ("Baseline A (Zero)", pred_zero),
            ("Baseline B (Train Mean)", pred_mean),
        ]:
            for sub_name, n, m in evaluate_all_subsets(pred, true_mm, has_prop):
                rows.append(
                    {
                        "Split": split_name,
                        "Baseline": b_name,
                        "Subset": sub_name,
                        "N": n,
                        **m,
                    }
                )

    print(format_table(rows))


if __name__ == "__main__":
    main()
