"""LesionLocator wrapper: prompts and scoring on a synthetic scan pair; LesionLocator itself is faked."""
import argparse
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

from lesionloc import lesionlocator as L
from lesionloc.team1 import index_to_mm

spec = importlib.util.spec_from_file_location("ll_eval", Path(__file__).parents[1] / "scripts" / "lesionlocator_eval.py")
ll_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ll_eval)

PID = "0000000001"


def image(a, spacing=(0.8, 0.9, 2.5), origin=(100, 80, -50)):
    img = sitk.GetImageFromArray(a)
    img.SetSpacing(spacing)
    img.SetDirection([-1, 0, 0, 0, -1, 0, 0, 0, 1])  # LPS-like, flips x and y
    img.SetOrigin(origin)
    return img


def box(a, label, z, y, x, r=2):
    a[z - r:z + r + 1, y - r:y + r + 1, x - r:x + r + 1] = label


def test_relabel_keeps_only_requested_ids_in_order():
    a = np.zeros((10, 10, 10), np.uint16)
    a[1, 1, 1], a[5, 5, 5], a[8, 8, 8] = 3, 7, 9
    out = sitk.GetArrayFromImage(L.relabel(image(a), [7, 3]))
    assert out[5, 5, 5] == 1 and out[1, 1, 1] == 2 and out[8, 8, 8] == 0


def test_point_prompt_is_zyx_voxel_index():
    img = image(np.zeros((40, 30, 20), np.uint16))
    xyz = np.array([3.5, 12.25, 30.0])
    p = L.point_prompt(img, [index_to_mm(img, xyz)])
    assert [float(v) for v in p["1"]["point"]] == [30.0, 12.25, 3.5]


def test_centre_mm_and_outputs(tmp_path):
    a = np.zeros((40, 30, 20), np.uint16)
    box(a, 1, 30, 12, 4)
    img = image(a)
    assert np.allclose(L.centre_mm(img), index_to_mm(img, [4, 12, 30]))
    assert L.centre_mm(image(np.zeros_like(a))) is None
    for name in ["s_FU_img_00_lesion_1.nii.gz", "s_FU_img_00_lesion_12.nii.gz", "s_BL_img_00_lesion_1.nii.gz"]:
        (tmp_path / name).touch()
    assert sorted(L.outputs(tmp_path, "s_FU_img_00")) == [1, 12]


def test_prepare_and_score_end_to_end(tmp_path):
    raw = tmp_path / "raw"
    (raw / "inputsTr").mkdir(parents=True)
    (raw / "targetsTr").mkdir()
    bl_c = {4: (30, 10, 5), 9: (10, 20, 14)}   # lesion id -> (z, y, x) voxel
    fu_c = {4: (28, 12, 6), 9: (12, 18, 12)}
    bl, fu = np.zeros((40, 30, 20), np.uint16), np.zeros((40, 30, 20), np.uint16)
    for lid in bl_c:
        box(bl, lid, *bl_c[lid])
        box(fu, lid, *fu_c[lid])
    bl_img, fu_img = image(bl), image(fu, origin=(90, 85, -40))
    sitk.WriteImage(bl_img, str(raw / "inputsTr" / f"{PID}_BL_mask_00.nii.gz"))
    sitk.WriteImage(fu_img, str(raw / "targetsTr" / f"{PID}_FU_mask_00.nii.gz"))
    for tag in ("BL", "FU"):
        (raw / "inputsTr" / f"{PID}_{tag}_img_00.nii.gz").touch()  # only the path is used before LesionLocator
    rows = []
    for lid in bl_c:
        r = dict(patient=PID, lesion=lid, split="val", img_id_bl=0, img_id_fu=0)
        r.update(zip(["bl_x", "bl_y", "bl_z"], index_to_mm(bl_img, bl_c[lid][::-1])))
        r.update(zip(["fu_x", "fu_y", "fu_z"], index_to_mm(fu_img, fu_c[lid][::-1])))
        r.update(prop_x=np.nan, prop_y=np.nan, prop_z=np.nan)
        rows.append(r)
    pd.DataFrame(rows).to_csv(tmp_path / "lesions.csv", index=False)
    out = tmp_path / "out"
    a = argparse.Namespace(raw=str(raw), lesions=str(tmp_path / "lesions.csv"), out=str(out), split="val",
                           allow_test=False, limit_pairs=None, centres=None)
    ll_eval.prepare(a)

    d = out / "pairs" / L.pair_name(PID, 0, 0)
    assert json.loads((d / "lesions.json").read_text())["lesions"] == [4, 9]
    prompt = sitk.GetArrayFromImage(sitk.ReadImage(str(d / "bl_mask_prompt.nii.gz")))
    assert prompt[bl_c[4]] == 1 and prompt[bl_c[9]] == 2
    pts = json.loads((d / "bl_points.json").read_text())
    assert [float(v) for v in pts["2"]["point"]] == list(bl_c[9])
    assert (out / "pairs.tsv").read_text().count("\n") == 1

    # fake LesionLocator: mask mode finds both FU lesions exactly; point mode finds lesion 4 only, and its BL seg
    (d / "mask").mkdir()
    (d / "point").mkdir()
    for k, lid in enumerate([4, 9], 1):
        sitk.WriteImage(L.relabel(fu_img, [lid]), str(d / "mask" / f"{PID}_FU_img_00_lesion_{k}.nii.gz"))
    sitk.WriteImage(L.relabel(fu_img, [4]), str(d / "point" / f"{PID}_FU_img_00_lesion_1.nii.gz"))
    sitk.WriteImage(L.relabel(bl_img, [4]), str(d / "point" / f"{PID}_BL_img_00_lesion_1.nii.gz"))
    (d / "mask" / "DONE").touch()
    (d / "point" / "DONE").touch()
    ll_eval.score(a)

    p = pd.read_csv(out / "predictions.csv", dtype={"patient": str}).set_index(["method", "lesion"])
    assert np.allclose(p.loc["ll_mask", "error_mm"], 0, atol=1e-6)
    assert (p.loc["ll_mask", "status"] == "ok").all()
    assert p.loc[("ll_point", 4), "status"] == "ok" and p.loc[("ll_point", 4), "bl_check_mm"] < 1e-6
    assert p.loc[("ll_point", 9), "status"] == "fallback_no_output"
    # no propagated centre -> the fallback is the BL centre, same as the `propagated` row
    assert np.isclose(p.loc[("ll_point", 9), "error_mm"], p.loc[("propagated", 9), "error_mm"])
    assert p.loc[("ll_mask", 9), "label_pred"] == 9
    s = pd.read_csv(out / "summary.csv").set_index("method")
    assert s.loc["ll_point", "fallbacks"] == 1 and s.loc["ll_mask", "n"] == 2
    shutil.rmtree(d / "mask")  # a pair that never ran is a fallback too, not a crash
    ll_eval.score(a)
    p = pd.read_csv(out / "predictions.csv").set_index(["method", "lesion"])
    assert (p.loc["ll_mask", "status"] == "fallback_not_run").all()
