# Experiments

Joint experiment (branch `karl-joint-alignment-localisation`): implementation ready;
cluster run pending. Compare dataset propagation and Team 1's hybrid centres,
each alone and with frozen Team 2 run 6b (plain / flip TTA), on the same val cohort.
See `docs/Joint-experiment.md` and `scripts/evaluate_joint.sbatch`. No joint accuracy
result has been measured yet.

| # | Date | Issue | Model | Config | Split | Metric(s) | Result | Notes |
|---|------|-------|-------|--------|-------|-----------|--------|-------|
| 0 | 2026-10-22 | | LesionNet (0.32 M params) | scripts/train.sbatch | train only | mean error (mm) | 2 files: 0.00 mm (zero baseline 14.04). 100 train files, 2000 steps, bs 8: 2.41 mm (zero baseline 4.78) | Overfit sanity check, not a real result. 100-file run not yet fully overfit; try smaller smooth-L1 beta / LR decay. Log: logs/train-2941892.out |
| 1 | 2026-10-23 | | LesionNet v2: residual 3D CNN, BL+FU as 2 channels, flatten + linear head (0.92 M params) | `scripts/train.sbatch` (job 2961900): 30 epochs, batch 8, AdamW lr 1e-3 + cosine, smooth-L1 in mm, flip augmentation; branch `first_e2e_pipeline` | train 2039 / val 186 lesions (224 / 28 patients); test not used | val mean / median error, CPM@10 | Best val mean 14.37 mm (epoch 20) vs zero baseline 14.51 mm; median 8.90 mm, CPM@10 51.6%. Epoch 30: mean 14.41, median 9.76 mm | No better than the registration (zero) baseline. Train loss only 3.74 → 3.47, val median got worse: the model does not learn to locate the lesion. Overfit check on 2 example files in the same job: 0.08 mm. Next: Siamese + heatmap model. Log: `logs/train-2961900.out`, checkpoint `runs/2961900/best.pt` |
| 2 | 2026-10-29 | | Baselines: zero and train mean (Team 2 run 0) | `scripts/baselines.py` on `Longitudinal-CT-processed-team2` (FU window = true FU lesion + jitter N(0, 8²) mm per axis, clipped ±24, seed 42) | val 207 / test 286 lesions | val mean / median error, CPM@5/10/20 | Zero: val mean 12.47 mm, median 12.06 mm, CPM@5 7.2%, CPM@10 33.3%, CPM@20 89.9%. Train mean ≈ 0, so Baseline B = zero | Bar for all Team 2 runs. Matches the expected value for N(0, 8²) jitter (≈12.75 mean). Test zero mean 12.52 mm |
| 3 | 2026-10-29 | | LesionNet v2 direct regression (Team 2 run 1), same as row 1 | `scripts/train.sbatch` (job 3193583) with `DATA=.../Longitudinal-CT-processed-team2`: 30 epochs, batch 8, AdamW lr 1e-3 + cosine, smooth-L1 in mm, flip augmentation | train 2159 / val 207 lesions | val mean / median error, CPM@10 | Best val mean 9.95 mm (epoch 27) vs zero 12.47 mm (−20%); median 9.50 mm, CPM@10 53.1% (zero 33.3%). Epoch 30: mean 9.96, median 9.32 mm | Unlike row 1, the CNN learns once the window is centred on the true lesion + jitter. Train loss 5.91 → 3.35, val still falling slowly at epoch 30. Overfit check on 2 example files: < 1 mm. Bar for the Siamese runs. Log: `logs/train-3193583.out` |
| 4 | 2026-10-30 | | SiameseNet (Team 2 run 2): shared 3D encoder to 32³ cells of 4 mm, 3³ BL centre template correlated with FU, softmax + soft-argmax (0.12 M params) | `scripts/train_siamese.sbatch` (job 3276914): 150 epochs, batch 16, AdamW lr 3e-4 + cosine, L = L_dist only (Euclidean / 8 mm), flip augmentation, no BL mask | train 2159 / val 207 lesions | val mean / median error, CPM@10 | Best val mean 7.52 mm (epoch 92) vs run 1 9.95 mm and zero 12.47 mm (−40%); median 5.83 mm, CPM@10 71.5%. Epoch 150: mean 7.71, median 5.77 mm, CPM@10 73.4% | Spatial output beats the FC head. Val plateaus from epoch ~70 while train L_dist keeps falling to 0.41 (≈ 3.3 mm): overfitting. Mean ≫ median: a tail of large misses. Overfit check on 10 files: 0.00 mm. Bench: 278 ms/step, 2.2 GiB at batch 16; epochs ~115 s (data loading bound). Log: `logs/train-3276914.out` |
| 5 | 2026-10-31 | | SiameseNet + re-drawn FU jitter (Team 2 run 2b) | `scripts/train_siamese.sbatch --rejitter-sd 8` (job 3279130), else same as row 4; data re-cut by job 3279118 (adds `fu_big` 88³, same `fu` / `target_mm`) | train 2159 / val 207 lesions | val mean / median error, CPM@10 | Best val mean 6.78 mm (epoch 109) vs 7.52 mm (row 4) and zero 12.47 mm (−46%); median 4.74 mm, CPM@10 76.8%. Epoch 150: mean 6.85, median 5.19 mm, CPM@10 77.8% | Fresh offset each epoch cuts overfitting: train L_dist 0.52 (≈ 4.1 mm) vs 0.41 in row 4, train/val gap 4.4 → 2.7 mm. Val flat after ~epoch 100; mean − median still ~2 mm (tail of large misses). New default for later runs. Overfit check 0.00 mm. Log: `logs/train-3279130.out` |
| 6 | 2026-10-31 | | SiameseNet + re-drawn jitter + heatmap loss (Team 2 run 3) | `train_siamese.sbatch --rejitter-sd 8 --lambda-hm 1 --dist-warmup 5` (job 3287287), else same as row 5. L_hm = CE to a Gaussian, σ = clip(0.5·r_FU, 4, 12) mm | train 2159 / val 207 lesions | val mean / median error, CPM@10 | Best val mean 6.92 mm (epoch 109) vs 6.78 (row 5); median 5.80 mm (row 5: 4.74), CPM@10 77.8%. Epoch 150: mean 6.99, median 5.67 mm, CPM@10 80.7% | Not better: mean within noise, median ~1 mm worse, but more lesions within 10 mm (80.7% vs 77.8% at epoch 150) and faster early learning (epoch 11: 8.57 mm vs ~10.7). Reading: the wide Gaussian target makes the map robust but blurry, so the soft-argmax is less precise. Flag dropped for run 5; retry later with smaller σ or λ_hm 0.3. CE floor ≈ target entropy, so hm ≈ 5.4 is not a fit measure. Overfit check 0.02 mm. Log: `logs/train-3287287.out` |
| 7 | 2026-10-31 | | SiameseNet + re-drawn jitter + BL mask input (Team 2 run 4) | `train_siamese.sbatch --rejitter-sd 8 --bl-mask` (job 3287288), else same as row 5: BL mask as 2nd encoder channel + mask-pooled template | train 2159 / val 207 lesions | val mean / median error, CPM@10 | Best val mean 6.61 mm (epoch 105) vs 6.78 (row 5), median 4.51 mm (row 5: 4.74), CPM@10 76.8%. Epoch 150: mean 6.79, median 4.28 mm, CPM@10 78.3% | Small gain, best so far: mean −0.17 mm (near noise for 207 lesions), median −0.2 to −0.5 mm. Train L_dist 0.48 (≈ 3.8 mm): gap to val still ~3 mm. Kept. Epochs 37 s. Overfit check 0.00 mm. Log: `logs/train-3287288.out` |
| 8 | 2026-10-31 | | SiameseNet + re-drawn jitter + BL mask + FU-mask loss (Team 2 run 5) | `train_siamese.sbatch --rejitter-sd 8 --bl-mask --lambda-mask 0.5` (job 3287946), else same as row 7. No heatmap loss (dropped in row 6). L_mask = −log(softmax mass inside the FU mask, max-pooled to the 32³ grid) | train 2159 / val 207 lesions | val mean / median error, CPM@10 | Best val mean 6.55 mm (epoch 90) vs 6.61 (row 7), median 4.57 mm (row 7: 4.51). Epoch 150: mean 6.83, median 4.94 mm, CPM@10 75.8% | Tie with row 7: −0.06 mm is noise. Faster early learning (epoch 25: 8.17 vs 8.54 mm; epoch 50: 7.19 vs 7.61) but the same end point. Train L_dist 0.46 (≈ 3.7 mm): the gap to val is still ~3 mm. The logged mask term (~18.5) is dominated by ~4 train lesions per epoch whose re-drawn window holds no FU mask: the `-1e4` fill in `team2_loss` gives each of them a loss of 10⁴. These add no gradient, but the true term is ≈ 0.03 by the end. Worst cases (`runs/3287946/worst/`): top 20 again led by patients e2c0be2456 (6) and 3988c7f88e (4); 8 of 20 are tiny lesions (r < 5 mm) and 3 are huge (r > 20 mm); all have peak prob < 0.43. Log: `logs/train-3287946.out` |
| 9 | 2026-10-31 | | Row 8 checkpoint, readouts only (no retraining) | `worst_cases.py --ckpt runs/3287946/best.pt --bl-mask` (srun). plain = soft-argmax; local = soft-argmax within ±8 mm of the peak; tta = mean over the 8 flip combinations; per-patient = mean of the 28 patient means | val 207 lesions | val mean / median error, CPM@10, per-patient mean | plain 6.55 / 4.57 mm, CPM@10 76.8%, per-patient 5.86. local 6.81 / 4.58, 77.3%, 6.13. **tta 5.97 / 4.25, 80.2%, 5.31.** tta_local 6.02 / 4.25, 82.1%, 5.45 | TTA: −0.58 mm mean, the biggest gain since re-drawn jitter. It is a paired comparison on the same lesions, so it is less noisy than comparing two runs, and it has no tuned parameter. Use it for Team 2's final predictions. Local soft-argmax hurts the mean (+0.26 mm), because it sometimes locks onto the wrong peak, so it is dropped. Per-patient mean < lesion mean: the patients with many lesions are harder. e2c0be2456 (50 lesions, 24% of val) has a mean of 7.58 mm and 3988c7f88e (26) has 8.21; together they are 37% of val. The worst patient is 7ef605fc8d (1 lesion, 20.4 mm) |
| 10 | 2026-11-01 | | Row 8 + offline-augmented train data + weight EMA (Team 2 run 6) | `train_siamese.sbatch --rejitter-sd 8 --bl-mask --lambda-mask 0.5 --aug-root $DATA-aug/lesions --ema 0.999 --epochs 50` (job 3288500). Aug data from job 3287945: 4 rotated (≤ 15°) / scaled (0.9–1.1) copies per train lesion | train 2159 + 8610 aug / val 207 lesions | val mean / median error, CPM@10, per-patient mean; plain and flip-TTA readout | Best epoch 32: plain 6.12 / 3.92 mm, CPM@10 83.6%, per-patient 5.79. **TTA 5.63 / 3.84 mm**, 82.1%, per-patient 5.00 (row 9: 5.97). Epoch 50: 6.16 / 3.98 | Augmentation works: plain −0.43 mm mean and −0.65 mm median vs row 8; TTA −0.34 mm vs row 9. Train L_dist 0.49 (≈ 3.95 mm): gap to val ~2.2 mm (row 8: ~2.9). The mask term now logs 0.08, so the empty-window fix works. Epochs 182 s on a V100 (5× the files), about 2.6 h in total. Log: `logs/train-3288500.out` |
| 11 | 2026-11-01 | | Row 10 + 5³ BL template, 20 mm (Team 2 run 6b) | row 10 + `--template-k 5` (job 3288501) | as row 10 | as row 10 | Best epoch 34: plain 5.81 / 3.79 mm, CPM@10 84.1%, per-patient 5.47. **TTA 5.58 / 3.49 mm**, 83.6%, per-patient 5.05. tta_local 5.54 / 3.40, 84.5%. Epoch 50: 6.06 / 3.76 | Best so far. Plain −0.31 mm vs row 10, but under TTA only −0.05 mm mean; the median gains 0.35 mm. The remaining mean comes from a few gross misses that the template size does not fix. Val rises after epoch ~35, so 50 epochs is enough. Steps cost 20% more (310 vs 258 ms). Kept. Log: `logs/train-3288501.out` |
| 12 | 2026-11-01 | | Row 10 + centre prior, σ 12 mm (Team 2 run 6c) | row 10 + `--prior-sd 12` (job 3288502) | as row 10 | as row 10 | Best epoch 35: plain 6.16 / 3.92 mm, CPM@10 82.6%, per-patient 5.47. TTA 5.63 / 3.59 mm, 83.1%, per-patient 5.05 | Tie with row 10 on the mean (plain +0.04, TTA 0.00); the median is slightly better (TTA 3.59 vs 3.84). It doesn't pay for the risk that Team 1's window errors don't match our N(0, 8 mm) jitter. Dropped. Log: `logs/train-3288502.out` |


## Planned: run 6b without training flips

Prepared 2026-11-05; not submitted and no results yet.

- Question: does preserving anatomical orientation during training improve localisation?
- Change from run 6b (job 3288501): disable random flips with `--no-flips`.
- Keep the same original and offline rotated/scaled train data, fresh 8 mm jitter,
  BL mask, mask-loss weight 0.5, EMA 0.999, template size 5, batch size 16,
  learning rate 3e-4, 50 epochs and existing fixed patient split. Training seed remains 0.
- The small offline rotations (up to 15 degrees) remain enabled; no mirror flips are used.
- Primary comparison: best **unflipped validation** mean/median error and CPM@10,
  against run 6b's **5.81 / 3.79 mm and 84.1%**. Do not compare plain predictions
  directly with run 6b's TTA result. Test split is not used.
- Launch from the cluster repo root:

  ```bash
  mkdir -p logs
  sbatch scripts/train_siamese_no_flips.sbatch
  ```

  Save the returned job ID here with the result. The job logs the Git commit,
  working-tree status and full training arguments; checkpoint: `runs/<jobid>/best.pt`.

# Imre's test stuff (some mock data with some real data features) (2026-10-22; needs formatting):

$ # 1. Ran unit tests (hand-computed values + example lesion check)
PYTHONPATH=src pytest

$ # 2. Ran baselines on the 2 real example .npz files in the repo (Mean error for Baseline A will be 14.04 mm!)
PYTHONPATH=src python scripts/baselines.py data/example_lesions/*.npz

$ # 3. Ran baselines in --mock mode to verify the full Val & Test table formatting
PYTHONPATH=src python scripts/baselines.py --mock

==================================================================================== 

test session starts 

=====================================================================================

platform linux -- Python 3.14.5, pytest-9.1.1, pluggy-1.7.0

rootdir: /gpfs/helios/home/saksimre/ut-ml-2026-lesion-project

configfile: pyproject.toml

collected 2 items


tests/test_metrics.py ..                                                                                                                                                               [100%]

===================================================================================== 

2 passed in 0.44s 

======================================================================================

Train mean target_mm (x, y, z): [ 9.485  4.491 -7.952] mm


| Split | Baseline | Subset | N | Median error (mm) | Mean error (mm) | MEDx (mm) | MEDy (mm) | MEDz (mm) | CPM@5 (%) | CPM@10 (%) | CPM@20 (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| example_lesions | Baseline A (Zero) | Overall | 2 | 14.04 | 14.04 | 9.48 | 4.49 | 9.19 | 50.0% | 50.0% | 50.0% |
| example_lesions | Baseline A (Zero) | has_propagated=True | 2 | 14.04 | 14.04 | 9.48 | 4.49 | 9.19 | 50.0% | 50.0% | 50.0% |
| example_lesions | Baseline B (Train Mean) | Overall | 2 | 13.39 | 13.39 | 9.03 | 3.62 | 9.19 | 0.0% | 0.0% | 100.0% |
| example_lesions | Baseline B (Train Mean) | has_propagated=True | 2 | 13.39 | 13.39 | 9.03 | 3.62 | 9.19 | 0.0% | 0.0% | 100.0% |
Train mean target_mm (x, y, z): [ 0.117 -0.945  0.661] mm

| Split | Baseline | Subset | N | Median error (mm) | Mean error (mm) | MEDx (mm) | MEDy (mm) | MEDz (mm) | CPM@5 (%) | CPM@10 (%) | CPM@20 (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val (mock) | Baseline A (Zero) | Overall | 60 | 6.25 | 7.62 | 3.51 | 4.56 | 3.19 | 33.3% | 88.3% | 95.0% |
| val (mock) | Baseline A (Zero) | has_propagated=True | 50 | 5.85 | 5.84 | 2.84 | 3.24 | 2.54 | 38.0% | 98.0% | 100.0% |
| val (mock) | Baseline A (Zero) | has_propagated=False | 10 | 14.85 | 16.50 | 6.82 | 11.14 | 6.43 | 10.0% | 40.0% | 70.0% |
| val (mock) | Baseline B (Train Mean) | Overall | 60 | 6.04 | 7.45 | 3.49 | 4.34 | 3.20 | 33.3% | 85.0% | 95.0% |
| val (mock) | Baseline B (Train Mean) | has_propagated=True | 50 | 5.62 | 5.72 | 2.83 | 3.06 | 2.61 | 38.0% | 94.0% | 100.0% |
| val (mock) | Baseline B (Train Mean) | has_propagated=False | 10 | 14.60 | 16.10 | 6.80 | 10.76 | 6.17 | 10.0% | 40.0% | 70.0% |
| test (mock) | Baseline A (Zero) | Overall | 60 | 5.65 | 7.60 | 3.88 | 4.16 | 3.44 | 38.3% | 86.7% | 91.7% |
| test (mock) | Baseline A (Zero) | has_propagated=True | 53 | 5.43 | 5.59 | 2.71 | 3.24 | 2.46 | 43.4% | 96.2% | 100.0% |
| test (mock) | Baseline A (Zero) | has_propagated=False | 7 | 24.20 | 22.82 | 12.76 | 11.13 | 10.92 | 0.0% | 14.3% | 28.6% |
| test (mock) | Baseline B (Train Mean) | Overall | 60 | 5.71 | 7.43 | 3.87 | 3.91 | 3.44 | 38.3% | 86.7% | 91.7% |
| test (mock) | Baseline B (Train Mean) | has_propagated=True | 53 | 5.50 | 5.45 | 2.70 | 2.98 | 2.47 | 43.4% | 96.2% | 100.0% |
| test (mock) | Baseline B (Train Mean) | has_propagated=False | 7 | 23.47 | 22.46 | 12.74 | 10.92 | 10.82 | 0.0% | 14.3% | 28.6% |
$ 


## Team 1 results


All rows: every eligible lesion (`lesions.csv`, 2669 = the 2502 npz lesions + 167 that `skip_outside_window` dropped),
error = ‖window centre − true FU centre‖ in mm, `scripts/team1_eval.py`. Val = 208 lesions / 28 patients; train =
2173 / 227. Nothing is trained; settings are chosen on train, methods compared on val, test not used yet.
In window@56 = every axis within 56 mm, i.e. the lesion survives `preprocess.py`. Full table:
`/gpfs/helios/projects/ml_project_lesion_tracking/data/team1/eval_all.md`.

| # | Date | Method | Code / job | Val mean / median (mm) | Val CPM@10 / CPM@20 | Val in window | Train median, in window | Notes |
|---|------|--------|------------|------------------------|---------------------|---------------|-------------------------|-------|
| T1-0 | 2026-11-03 | Same mm coordinates as BL | `team1_lesions.py` | 1094.64 / 984.11 | 0.0% / 0.0% | 6.7% | 575.69, 4.9% | BL and FU patient coordinates are unrelated (origins ~1 m apart). Not a usable baseline: every method needs a registration step |
| T1-1 | 2026-11-03 | `cog_propagated` from the dataset (**the bar**), BL position when missing | `team1_lesions.py` | 142.35 / 11.44 | 46.2% / 63.5% | 89.4% | 4.82, 93.8% | Precise when it works (train median 4.8 mm), but 101 lesions have no propagated point (BL fallback, ~1 m) and some are far off: ~11% of val lesions fall outside the window |
| T1-2a | 2026-11-03 | Rigid whole-scan registration | `team1_register.py`, job 3330170 | 17.02 / 12.97 | 40.9% / 68.3% | 99.5% | 9.08, 98.2% | SimpleITK, Mattes MI in a body mask, regular-step GD, 8→4→2 mm; start = best of identity, image centres, body centroids ± z sweep of 400 mm. 2 train pairs failed (46ba9f2a69, 6cdd60ea00): their 2 lesions scored at BL in 2a–2c and 4 |
| T1-2b | 2026-11-03 | Affine whole-scan registration | same | 15.42 / 11.58 | 45.2% / 75.5% | 99.5% | 9.33, 98.0% | From rigid |
| T1-2c | 2026-11-03 | Affine + rigid correction in a 160 mm cube around each lesion | same | **12.28** / 9.36 | 53.8% / 85.1% | **100%** | 6.54, 98.1% | Best val mean and in-window together. **Hand-over candidate** for `centres.csv` |
| T1-3a | 2026-11-03 | Organ shift: BL→FU centroid shift of the host organ, else nearest | `team1_organs.py`, jobs 3330252 (TotalSegmentator `--fast`) + 3330695 | 12.39 / 10.59 | 45.7% / **85.6%** | **100%** | 7.65, 98.7% | 1912 / 2669 lesions moved by their host organ; 5 had no usable organ (BL kept). Organs cut by the scan range are never used |
| T1-3b | 2026-11-03 | Organ affine: K nearest organs, weights exp(−d/σ), ridge λ towards a pure shift | same | 13.40 / 10.20 | 48.1% / 80.3% | **100%** | 7.02, 98.6% | K=8, σ=50 mm, λ=10⁴, picked on train from 12 settings (best train median and CPM@10). λ=10⁴ beat λ=10² for every K and σ: a near-pure weighted shift beats a full local affine |
| T1-4 | 2026-11-03 | B-spline (8³ control grid) on top of affine | `team1_register.py`, job 3330170 | 13.56 / 10.71 | 47.6% / 82.2% | 99.5% | 6.53, 97.7% | Ties local on train, slightly behind on val |
| T1-5a | 2026-11-03 | uniGradICON, network only | `team1_unigradicon.py`, job 3330685 | 45.41 / 11.00 | 46.6% / 63.9% | 81.7% | 8.69, 86.5% | Pretrained, no training. Heavy tail: ~18% of val lesions outside the window |
| T1-5b | 2026-11-03 | uniGradICON + 50 instance-optimisation steps | jobs 3330731 (4-way array) + 3330738 | 38.77 / **7.73** | **56.7%** / 71.2% | 87.0% | 5.49, 91.0% | Best median and CPM@10 of all methods, but 13% of val lesions outside the window: either very close or badly off. Next: use it only where it agrees with T1-2c |
| T1-5c | 2026-11-03 | `ugi_io50` gated by `local` ($\tau=25\text{ mm}$), fallback to `local` | `team1_combine.py` | **10.75** / **6.71** | **63.0%** / **86.1%** | **100%** | 5.08, 97.7% | Uses `ugi_io50` when within 25 mm of `local` (75.5% of val lesions), else `local`. Best val mean, median, CPM@10, CPM@20, and 100% in-window together. **Selected for `centres.csv` hand-over** |
