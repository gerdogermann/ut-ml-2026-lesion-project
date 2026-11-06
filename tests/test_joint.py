"""Tests of the boundary between registration, cropping, inference and scoring."""
import argparse
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk
import torch

from lesionloc.alignment import CentreTable, gated_centres
from lesionloc.joint_eval import evaluate, summarise
from lesionloc.model import SiameseNet
from lesionloc.pipeline import LesionQuery, locate
from lesionloc.prediction import flip_tta
from lesionloc.preprocess import do_patient


@pytest.fixture
def case(tmp_path):
    raw = tmp_path / "raw"
    (raw / "inputsTr").mkdir(parents=True)
    pid = "0000000001"
    image = sitk.GetImageFromArray(np.zeros((64,) * 3, np.int16))
    image.SetSpacing([1.0, 1.5, 2.5])
    image.SetDirection([-1, 0, 0, 0, -1, 0, 0, 0, 1])
    image.SetOrigin([100, 80, -50])
    a = np.zeros((64,) * 3, np.uint16)
    a[28:36, 28:36, 28:36] = 1
    mask = sitk.GetImageFromArray(a)
    mask.CopyInformation(image)
    for tag, kind, im in [("BL", "img", image), ("FU", "img", image), ("BL", "mask", mask)]:
        sitk.WriteImage(im, str(raw / "inputsTr" / f"{pid}_{tag}_{kind}_00.nii.gz"))
    centre = np.array(image.TransformContinuousIndexToPhysicalPoint([31.5] * 3))
    row = dict(patient=pid, lesion=1, split="val", img_id_bl=0, img_id_fu=0)
    for name in ("bl", "fu", "prop"):
        row.update(zip([f"{name}_{axis}" for axis in "xyz"], centre))
    lesions = pd.DataFrame([row])
    (raw / "data_split.json").write_text(json.dumps(dict(train=[], val=[pid], test=[])))
    return raw, pid, image, mask, centre, lesions


def table(pid, centre):
    return CentreTable(pd.DataFrame([dict(patient=pid, lesion=1, **dict(zip(["x_mm", "y_mm", "z_mm"], centre)))]))


def test_csv_contract(tmp_path):
    path = tmp_path / "centres.csv"
    path.write_text("patient,lesion,x_mm,y_mm,z_mm\n0000000001,1,10,20,30\n")
    centres = CentreTable.read(path)
    assert np.array_equal(centres.centre("0000000001", 1), [10, 20, 30])
    with pytest.raises(ValueError, match="Missing alignment"):
        centres.centre("0000000002", 1)
    frame = pd.read_csv(path, dtype={"patient": str})
    with pytest.raises(ValueError, match="Duplicate"):
        CentreTable(pd.concat([frame, frame]))
    frame["z_mm"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        CentreTable(frame)


def test_hybrid_gate_has_no_truth_input():
    local = np.zeros((3, 3))
    chosen, used = gated_centres(local, [[25, 0, 0], [26, 0, 0], [np.nan] * 3])
    assert used.tolist() == [True, False, False]
    assert np.array_equal(chosen, [[25, 0, 0], [0, 0, 0], [0, 0, 0]])
    with pytest.raises(ValueError):
        gated_centres([[np.nan] * 3], [[0] * 3])


class OffsetPredictor:
    def __init__(self, offset):
        self.offset = np.array(offset)

    def predict(self, bl, fu, bl_mask):
        assert bl.shape == fu.shape == bl_mask.shape == (64,) * 3
        assert bl_mask.any()
        return dict(plain=self.offset, tta=self.offset)


def test_pipeline_physical_coordinates_and_scoring_without_fu_masks(case):
    raw, pid, image, mask, centre, lesions = case
    delta = np.array([6, -4, 8])
    alignment = table(pid, centre + delta)
    predictor = OffsetPredictor(-delta)
    result = locate(LesionQuery(pid, 1, tuple(centre)), image, image, mask, alignment, predictor)
    assert np.allclose(result["plain"], centre)
    predictions = evaluate(raw, lesions, alignment, predictor)
    summary = summarise(predictions).set_index("method")
    assert summary.loc["alignment", "Mean error"] == pytest.approx(np.linalg.norm(delta))
    assert summary.loc["plain", "Mean error"] == 0
    assert summary.n.eq(1).all() and summary.fallbacks.eq(0).all()
    # FU labels change scoring only, never the network input or prediction.
    lesions.loc[:, ["fu_x", "fu_y", "fu_z"]] += 20
    altered = evaluate(raw, lesions, alignment, predictor)
    assert np.array_equal(predictions[["pred_x", "pred_y", "pred_z"]], altered[["pred_x", "pred_y", "pred_z"]])


def test_outside_window_and_failed_prediction_stay_in_denominator(case):
    raw, pid, _, _, centre, lesions = case
    alignment = table(pid, centre + [80, 0, 0])
    (raw / "inputsTr" / f"{pid}_BL_mask_00.nii.gz").unlink()
    lesions.loc[:, ["prop_x", "prop_y", "prop_z"]] = np.nan
    predictions = evaluate(raw, lesions, alignment, OffsetPredictor([0, 0, 0]))
    assert len(predictions) == 4
    assert not predictions.loc[predictions.method != "propagated", "in_window"].any()
    summary = summarise(predictions).set_index("method")
    assert summary.loc["plain", "Mean error"] == 80
    assert summary.loc["plain", "fallbacks"] == 1
    assert summary.loc["propagated", "fallbacks"] == 1


def test_csv_preprocessing_preserves_masks_and_physical_target(case, tmp_path):
    raw, pid, _, mask, centre, _ = case
    (raw / "targetsTr").mkdir()
    sitk.WriteImage(mask, str(raw / "targetsTr" / f"{pid}_FU_mask_00.nii.gz"))
    pd.DataFrame([dict(lesion_id=1, topology_class="UNCHANGED", linking_unclear=False,
                       img_id_bl=0, img_id_fu=0, cog_bl="[31.5,31.5,31.5]",
                       cog_fu="[31.5,31.5,31.5]", cog_propagated="")]).to_csv(raw / "inputsTr" / f"{pid}.csv", index=False)
    out = tmp_path / "patches"
    (out / "lesions").mkdir(parents=True)
    a = argparse.Namespace(raw=raw, out=out, centre="csv", jitter_sd=8, jitter_max=24, seed=42,
                           augment=0, centres=table(pid, centre + [6, -4, 8]))
    _, stats = do_patient((pid, a, set()))
    assert stats["kept"] == stats["from_centres_csv"] == 1
    with np.load(out / "lesions" / f"{pid}_1.npz") as data:
        assert np.array_equal(data["target_mm"], [-6, 4, -8])
        assert data["bl_mask"].any() and data["fu_mask"].any()
        assert not data["has_propagated"]  # CSV source does not change this flag's meaning
        assert np.allclose(data["centre_fu_mm"] + data["target_mm"], data["true_fu_mm"])
    a.centres = table("0000000002", centre)
    with pytest.raises(ValueError, match="Missing alignment"):
        do_patient((pid, a, set()))


def test_tta_unflips_xyz_offsets():
    # Toy network locates one peak. An asymmetric point detects swapped axes/signs.
    class Peak:
        def __call__(self, x):
            return None, x[:, 0]

    def readout(logits):
        i = logits.flatten(1).argmax(1)
        return torch.stack([i % 8, i // 8 % 8, i // 64], 1).float() - 3.5

    x = torch.zeros(1, 1, 8, 8, 8)
    x[0, 0, 1, 3, 6] = 1
    assert torch.equal(flip_tta(Peak(), x, readout), torch.tensor([[2.5, -0.5, -2.5]]))


def test_joint_cli_runs_with_real_checkpoint_interface(case, tmp_path):
    raw, pid, _, _, centre, lesions = case
    les_path, centres_path = tmp_path / "lesions.csv", tmp_path / "centres.csv"
    lesions.to_csv(les_path, index=False)
    pd.DataFrame([dict(patient=pid, lesion=1, x_mm=centre[0], y_mm=centre[1], z_mm=centre[2])]).to_csv(centres_path, index=False)
    ckpt = tmp_path / "random-test-only.pt"
    torch.manual_seed(0)
    torch.save(SiameseNet(bl_mask=True, k=5).state_dict(), ckpt)
    out = tmp_path / "result"
    command = [sys.executable, "scripts/evaluate_joint.py", "--raw", str(raw), "--lesions", str(les_path),
               "--centres", str(centres_path), "--ckpt", str(ckpt), "--out", str(out),
               "--device", "cpu", "--limit", "1"]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["smoke_test"]
    assert pd.read_csv(out / "summary.csv").n.tolist() == [1] * 6
    assert subprocess.run(command, capture_output=True).returncode != 0  # no overwrite
