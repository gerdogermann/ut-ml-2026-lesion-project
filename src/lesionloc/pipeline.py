"""Compose alignment centres and lesion offsets without access to FU labels."""
from dataclasses import dataclass

import numpy as np

from lesionloc.preprocess import mask_patch, patch


@dataclass(frozen=True)
class LesionQuery:
    patient: str
    lesion: int
    bl_mm: tuple[float, float, float]


def _points(offsets, centre):
    """Predictor output -> physical points (centre + offset); scalar scores such as `spread` pass through."""
    result = {}
    for name, delta in offsets.items():
        if np.ndim(delta) == 0:
            result[name] = float(delta)
            continue
        delta = np.asarray(delta, dtype=float)
        if delta.shape != (3,) or not np.isfinite(delta).all():
            raise ValueError(f"Invalid {name} offset")
        result[name] = centre + delta
    return result


def locate(query, bl_scan, fu_scan, bl_mask_scan, alignment, predictor, two_pass=False):
    """Alignment places the FU crop; the independent predictor refines it.

    two_pass: re-cut the FU window at the first estimate and predict again (`pass2`), which brings a far start back
    into the offset range the model was trained on; `pass2_gated` keeps pass 2 only if its `spread` is lower.
    Returns physical centres and predictions. Deliberately accepts neither
    FU ground-truth coordinates nor FU lesion masks.
    """
    centre = alignment.centre(query.patient, query.lesion)
    bl_mm = np.asarray(query.bl_mm, dtype=float)
    mask = mask_patch(bl_mask_scan, bl_mm, query.lesion) if bl_mask_scan is not None else np.zeros((64,) * 3)
    if bl_mask_scan is not None and not mask.any():
        raise ValueError("BL lesion mask is empty in the baseline crop")
    bl = patch(bl_scan, bl_mm)
    result = {"centre": centre, **_points(predictor.predict(bl, patch(fu_scan, centre), mask), centre)}
    if two_pass:
        c2 = result.get("tta", result["plain"])
        second = _points(predictor.predict(bl, patch(fu_scan, c2), mask), c2)
        result["pass2"] = second.get("tta", second["plain"])
        result["spread2"] = second.get("spread", np.nan)
        better = second.get("spread", np.inf) < result.get("spread", np.inf)
        result["pass2_gated"] = result["pass2"] if better else c2
    return result
