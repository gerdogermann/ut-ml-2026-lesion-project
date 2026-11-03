# Experiments

| # | Date | Issue | Model | Config | Split | Metric(s) | Result | Notes |
|---|------|-------|-------|--------|-------|-----------|--------|-------|
| 1 | 2026-10-23 | | LesionNet v2: residual 3D CNN, BL+FU as 2 channels, flatten + linear head (0.92 M params) | `scripts/train.sbatch` (job 2961900): 30 epochs, batch 8, AdamW lr 1e-3 + cosine, smooth-L1 in mm, flip augmentation; branch `first_e2e_pipeline` | train 2039 / val 186 lesions (224 / 28 patients); test not used | val mean / median error, CPM@10 | Best val mean 14.37 mm (epoch 20) vs zero baseline 14.51 mm; median 8.90 mm, CPM@10 51.6%. Epoch 30: mean 14.41, median 9.76 mm | No better than the registration (zero) baseline. Train loss only 3.74 → 3.47, val median got worse: the model does not learn to locate the lesion. Overfit check on 2 example files in the same job: 0.08 mm. Next: Siamese + heatmap model. Log: `logs/train-2961900.out`, checkpoint `runs/2961900/best.pt` |

## Team 1: placing the FU window

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
