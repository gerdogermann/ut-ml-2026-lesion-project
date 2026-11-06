"""Alignment-stage outputs, independent of the patch localisation model.

Registration producers (SimpleITK / uniGradICON) write physical FU centres.
This boundary can also consume the dataset's propagated centres. No FU labels
are used to select a centre or combine registration predictions.
"""
from pathlib import Path

import numpy as np
import pandas as pd

XYZ = ["x_mm", "y_mm", "z_mm"]


class CentreTable:
    """Strict (patient, lesion) -> FU centre in patient (x,y,z) millimetres."""

    def __init__(self, frame):
        frame = frame.copy()
        required = ["patient", "lesion", *XYZ]
        if not set(required) <= set(frame.columns):
            raise ValueError(f"Centres require columns {required}")
        if frame[required].isna().any().any():
            raise ValueError("Centres contain missing identifiers or coordinates")
        frame["patient"] = frame.patient.astype(str)
        lesions = pd.to_numeric(frame.lesion, errors="raise").to_numpy(float)
        if not np.isfinite(lesions).all() or (lesions != np.floor(lesions)).any():
            raise ValueError("Lesion IDs must be finite integers")
        frame["lesion"] = lesions.astype(int)
        if frame.duplicated(["patient", "lesion"]).any():
            raise ValueError("Duplicate patient/lesion centres")
        points = frame[XYZ].to_numpy(dtype=float)
        if not np.isfinite(points).all():
            raise ValueError("Centres must be finite physical coordinates")
        self._points = dict(zip(zip(frame.patient, frame.lesion), points))

    @classmethod
    def read(cls, path: str | Path):
        return cls(pd.read_csv(path, dtype={"patient": str}))

    def centre(self, patient, lesion):
        key = (str(patient), int(lesion))
        if key not in self._points:
            raise ValueError(f"Missing alignment centre for {key}; no implicit fallback")
        return self._points[key].copy()

    def require(self, lesions):
        """Validate coverage before starting an expensive preprocessing/eval run."""
        for row in lesions.itertuples():
            self.centre(row.patient, row.lesion)


def gated_centres(local, candidate, tau_mm=25.0):
    """Use candidate within tau of local; otherwise local. Missing candidates fall back.

    Local is the required safety net. Candidate rows may be NaN when its
    registration failed; non-finite local centres are always an error.
    """
    local, candidate = np.asarray(local, float), np.asarray(candidate, float)
    if local.ndim != 2 or local.shape[1] != 3 or candidate.shape != local.shape:
        raise ValueError("Expected matching (N,3) centre arrays")
    if not np.isfinite(local).all() or not np.isfinite(tau_mm) or tau_mm < 0:
        raise ValueError("Local centres and non-negative threshold must be finite")
    use = np.isfinite(candidate).all(axis=1) & (np.linalg.norm(candidate - local, axis=1) <= tau_mm)
    return np.where(use[:, None], candidate, local), use
