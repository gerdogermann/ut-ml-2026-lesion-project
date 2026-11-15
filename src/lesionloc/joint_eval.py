"""Evaluate the composed pipeline. FU truth is confined to this scoring layer."""
import numpy as np
import pandas as pd
import SimpleITK as sitk

from lesionloc.metrics import compute_metrics
from lesionloc.pipeline import LesionQuery, locate
from lesionloc.team1 import cols, pairs, scan_path


def label_at(mask, point):
    """FU instance label at a physical point (scoring only); -1 outside the scan or without a FU mask."""
    if mask is None:
        return -1
    i = mask.TransformPhysicalPointToIndex([float(v) for v in point])
    return int(mask[i]) if all(0 <= j < n for j, n in zip(i, mask.GetSize())) else -1


def evaluate(raw, lesions, alignment, predictor, *, use_bl_mask=True, tta=True, two_pass=False):
    """Score every supplied lesion, including out-of-window centres.

    two_pass adds the `pass2` and `pass2_gated` readouts (see pipeline.locate). Each row also records the FU mask
    label at the prediction and at the truth (-1 = no FU mask): a miss whose label_pred is another lesion is a swap.

    A scan/mask/inference failure explicitly falls back to the alignment
    centre. It stays in the denominator and is recorded in predictions.csv.
    Missing alignment centres are a contract error and stop the run.
    """
    alignment.require(lesions)
    records = []
    readouts = (["plain", "tta"] if tta else ["plain"]) + (["pass2", "pass2_gated"] if tta and two_pass else [])
    for (patient, ib, ifu), group in pairs(lesions):
        scans, load_error = None, ""
        try:
            scans = [sitk.ReadImage(str(scan_path(raw, patient, tag, idx)))
                     for tag, idx in (("BL", ib), ("FU", ifu))]
            scans.append(sitk.ReadImage(str(scan_path(raw, patient, "BL", ib, "mask"))) if use_bl_mask else None)
        except RuntimeError as exc:
            load_error = str(exc)
        fu_mask_path = scan_path(raw, patient, "FU", ifu, "mask")
        fu_mask = sitk.ReadImage(str(fu_mask_path)) if fu_mask_path.exists() else None  # truth: scoring only
        for _, row in group.iterrows():
            centre = alignment.centre(patient, row.lesion)
            truth = row[cols("fu")].to_numpy(float)
            bl = row[cols("bl")].to_numpy(float)
            prop = row[cols("prop")].to_numpy(float)
            has_prop = np.isfinite(prop).all()
            points = {"propagated": prop if has_prop else bl, "alignment": centre}
            status = {"propagated": "ok" if has_prop else "missing_propagation_bl_fallback", "alignment": "ok"}
            error, predicted = load_error, {}
            try:
                if error:
                    raise ValueError(error)
                query = LesionQuery(str(patient), int(row.lesion), tuple(bl))
                predicted = locate(query, *scans, alignment, predictor, two_pass=two_pass and tta)
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
                rec.update(spread=predicted.get("spread", np.nan), spread2=predicted.get("spread2", np.nan),
                           label_pred=label_at(fu_mask, point), label_true=label_at(fu_mask, truth))
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


def by_start(predictions, bins=(0, 10, 20, 30, 40, np.inf)):
    """For methods that share one alignment: mean error per method and lesion count n, binned by the start error
    |window centre - truth| (mm); plus, per method, where misses > 10 mm land: inside the true FU lesion (own),
    inside another FU lesion (swap), else background. Rows without a FU mask (label -1) are left out of the latter."""
    p = predictions.copy()
    start = np.linalg.norm(p[["centre_x", "centre_y", "centre_z"]].to_numpy() - p[["true_x", "true_y", "true_z"]].to_numpy(), axis=1)
    p["start"] = pd.cut(start, bins, right=False)
    table = p.pivot_table(index="start", columns="method", values="error_mm", aggfunc="mean", observed=False, dropna=False)
    table.insert(0, "n", p[p.method == p.method.iloc[0]].groupby("start", observed=False).size())
    miss = p[(p.error_mm > 10) & (p.label_pred >= 0)]
    by = miss.groupby("method", sort=False)
    where = pd.DataFrame(dict(
        misses=by.size(),
        own=((miss.label_pred == miss.label_true) & (miss.label_true > 0)).groupby(miss.method, sort=False).mean(),
        swap=((miss.label_pred > 0) & (miss.label_pred != miss.label_true)).groupby(miss.method, sort=False).mean()))
    return table, where
