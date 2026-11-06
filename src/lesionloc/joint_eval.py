"""Evaluate the composed pipeline. FU truth is confined to this scoring layer."""
import numpy as np
import pandas as pd
import SimpleITK as sitk

from lesionloc.metrics import compute_metrics
from lesionloc.pipeline import LesionQuery, locate
from lesionloc.team1 import cols, pairs, scan_path


def evaluate(raw, lesions, alignment, predictor, *, use_bl_mask=True, tta=True):
    """Score every supplied lesion, including out-of-window centres.

    A scan/mask/inference failure explicitly falls back to the alignment
    centre. It stays in the denominator and is recorded in predictions.csv.
    Missing alignment centres are a contract error and stop the run.
    """
    alignment.require(lesions)
    records = []
    readouts = ["plain", "tta"] if tta else ["plain"]
    for (patient, ib, ifu), group in pairs(lesions):
        scans, load_error = None, ""
        try:
            scans = [sitk.ReadImage(str(scan_path(raw, patient, tag, idx)))
                     for tag, idx in (("BL", ib), ("FU", ifu))]
            scans.append(sitk.ReadImage(str(scan_path(raw, patient, "BL", ib, "mask"))) if use_bl_mask else None)
        except RuntimeError as exc:
            load_error = str(exc)
        for _, row in group.iterrows():
            centre = alignment.centre(patient, row.lesion)
            truth = row[cols("fu")].to_numpy(float)
            bl = row[cols("bl")].to_numpy(float)
            prop = row[cols("prop")].to_numpy(float)
            has_prop = np.isfinite(prop).all()
            points = {"propagated": prop if has_prop else bl, "alignment": centre}
            status = {"propagated": "ok" if has_prop else "missing_propagation_bl_fallback", "alignment": "ok"}
            error = load_error
            try:
                if error:
                    raise ValueError(error)
                query = LesionQuery(str(patient), int(row.lesion), tuple(bl))
                predicted = locate(query, *scans, alignment, predictor)
                points.update({name: predicted[name] for name in readouts})
                status.update({name: "ok" for name in readouts})
            except (RuntimeError, ValueError) as exc:
                error = str(exc)
                # Never turn a CUDA failure or bad prediction into an omitted lesion.
                points.update({name: centre for name in readouts})
                status.update({name: "alignment_fallback" for name in readouts})
            for method, point in points.items():
                window_centre = point if method == "propagated" else centre
                rec = dict(patient=str(patient), lesion=int(row.lesion), split=row["split"],
                           method=method, status=status[method],
                           message=error if status[method] == "alignment_fallback" else "",
                           has_propagated=bool(has_prop),
                           in_window=bool(np.abs(truth - window_centre).max() <= 56))
                rec.update(zip(["pred_x", "pred_y", "pred_z"], point))
                rec.update(zip(["true_x", "true_y", "true_z"], truth))
                rec.update(zip(["centre_x", "centre_y", "centre_z"], window_centre))
                rec["error_mm"] = float(np.linalg.norm(point - truth))
                records.append(rec)
    return pd.DataFrame(records)


def summarise(predictions):
    rows = []
    for method, group in predictions.groupby("method", sort=False):
        metrics = compute_metrics(group[["pred_x", "pred_y", "pred_z"]].to_numpy(),
                                  group[["true_x", "true_y", "true_z"]].to_numpy())
        rows.append(dict(method=method, n=len(group), patients=group.patient.nunique(),
                         fallbacks=int((group.status != "ok").sum()),
                         patient_mean_mm=float(group.groupby("patient").error_mm.mean().mean()),
                         **metrics))
    return pd.DataFrame(rows)
