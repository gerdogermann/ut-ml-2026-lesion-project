"""Compose alignment centres and lesion offsets without access to FU labels."""
from dataclasses import dataclass

import numpy as np

from lesionloc.preprocess import mask_patch, patch


@dataclass(frozen=True)
class LesionQuery:
    patient: str
    lesion: int
    bl_mm: tuple[float, float, float]


def locate(query, bl_scan, fu_scan, bl_mask_scan, alignment, predictor):
    """Alignment places the FU crop; the independent predictor refines it.

    Returns physical centres and predictions. Deliberately accepts neither
    FU ground-truth coordinates nor FU lesion masks.
    """
    centre = alignment.centre(query.patient, query.lesion)
    bl_mm = np.asarray(query.bl_mm, dtype=float)
    mask = mask_patch(bl_mask_scan, bl_mm, query.lesion) if bl_mask_scan is not None else np.zeros((64,) * 3)
    if bl_mask_scan is not None and not mask.any():
        raise ValueError("BL lesion mask is empty in the baseline crop")
    offsets = predictor.predict(patch(bl_scan, bl_mm), patch(fu_scan, centre), mask)
    result = {"centre": centre}
    for name, delta in offsets.items():
        delta = np.asarray(delta, dtype=float)
        if delta.shape != (3,) or not np.isfinite(delta).all():
            raise ValueError(f"Invalid {name} offset")
        result[name] = centre + delta
    return result
