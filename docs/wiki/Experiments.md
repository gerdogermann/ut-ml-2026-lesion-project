# Experiments

Joint experiment (Karl's branch `karl-joint-alignment-localisation`, merged into `team2-training-data`): Team 2's
offset added to Team 1's hybrid centres, scored on the 208 val lesions (`scripts/evaluate_joint.py`). Team 1 alone:
10.75 mm mean; first joint run (retrained run 6b): 8.74 mm, see `reports/Joint experiment results.md`. Rows 18–25:
joint val of P3-rand and the runs planned in `reports/Team 2 model and data improvements.md`
(`scripts/submit_team2_next.sh`). Joint numbers are mean / median mm for the TTA readout unless noted; "start" bins
are the hybrid centre's own error. MDD on val ≈ 0.67 mm (unpaired); paired CIs pending.

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
| 13 | 2026-11-06 | | Row 11 with the encoder replaced by nnFoundationCNN (ResEnc-L, MIC-DKFZ/nnFoundationCNN) stem + stages 0–1, frozen; 1×1 projection 64 → 32 ch on the 4 mm stage (Team 2 run P1) | `scripts/submit_pretrained.sh` (job 3472218): row 11 flags + `--encoder resenc --pretrained checkpoint_final.pth --freeze all --amp`. CT-only encoder input; BL mask only in the pooled template | as row 10 | as row 10 | Best epoch 41: plain 13.43 / 10.02 mm, CPM@10 49.3%, per-patient 11.18. TTA 12.20 / 10.75 mm, 46.9%, per-patient 9.95 | Fails: worse than zero (12.47) on the plain readout. Overfit check only 3.28 mm. Frozen MAE features + a 1×1 projection do not carry matching signal (the OpenMind benchmark reports the same for frozen MAE encoders). 0.67 M params, ~2 k trainable. Bench 124 ms/step, 1.49 GiB; 92 s/epoch. Log: `logs/train-3472218.out` |
| 14 | 2026-11-06 | | Same architecture as row 13, random init, all trainable (Team 2 run P1-rand) | `submit_pretrained.sh` (job 3472219): row 11 flags + `--encoder resenc --amp` | as row 10 | as row 10 | Best epoch 29: plain 6.09 / 2.95 mm, CPM@10 82.1%, per-patient 5.09. **TTA 5.55 / 2.69 mm**, 82.6%, per-patient 4.91. Epoch 50: 6.41 / 2.89 | Tie with row 11 on the TTA mean: paired difference −0.03 mm, 95% patient-cluster bootstrap CI [−1.17, +0.96]. Median 0.8 mm better. The ResEnc 4 mm stage alone is not better than the small encoder. 0.67 M params. Bench 311 ms/step, 5.22 GiB; 218 s/epoch. Log: `logs/train-3472219.out` |
| 15 | 2026-11-06 | | Row 13 + stage 2 (8 mm, 128 → 32 ch projection, 3³ template = 24 mm), its correlation upsampled ×2 and added to the logits; stem + stage 0 frozen, stages 1–2 fine-tuned (Team 2 run P3) | `submit_pretrained.sh` (job 3472220): row 11 flags + `--encoder resenc --coarse --pretrained checkpoint_final.pth --freeze early --enc-lr 1e-4 --warmup 0.1 --amp` (head lr 3e-4) | as row 10 | as row 10 | Best epoch 28: plain 4.65 / 2.33 mm, CPM@10 85.5%, per-patient 4.07. **TTA 4.55 / 2.32 mm**, 87.0%, per-patient 3.87. Epoch 50: 5.02 / 2.22 | Beats row 11: TTA −1.02 mm, CI [−2.03, −0.20]; 86 lesions better by > 1 mm, 34 worse. Tie with its random-init twin (row 16): +0.12 mm, CI [−0.38, +0.55], so pretraining adds nothing. Val rises after epoch ~28. 4.00 M params, 3.95 M trainable. Bench 251 ms/step, 3.32 GiB; 177 s/epoch. Log: `logs/train-3472220.out` |
| 16 | 2026-11-06 | | Same architecture as row 15, random init, all trainable (Team 2 run P3-rand) | `submit_pretrained.sh` (job 3472221): row 11 flags + `--encoder resenc --coarse --warmup 0.1 --amp`, lr 3e-4 for all | as row 10 | as row 10 | Best epoch 18: plain 4.57 / 2.51 mm, CPM@10 88.9%, per-patient 4.00. **TTA 4.44 / 2.54 mm**, 88.4%, per-patient 3.88. Epoch 50: 5.38 / 2.41 | **Best so far.** vs row 11: TTA −1.14 mm, CI [−1.85, −0.56]; 74 lesions better by > 1 mm, 26 worse. vs row 14: −1.12 mm, CI [−1.68, −0.44], so the gain comes from the 8 mm map. Gains on small lesions (r < 5 mm: 4.29 → 2.94) and on patient 3988c7f88e (7.10 → 4.75); none on e2c0be2456 (6.74 → 6.75, 24% of val) or on lesions with r > 20 mm (11.7 → 11.6). 19 lesions are > 10 mm in both row 11 and this run. Peak prob < 0.2: 38 lesions, mean 9.40 mm vs 3.32. Overfits early: 25–30 epochs are enough. 4.00 M params. Bench 365 ms/step, 5.84 GiB; 255 s/epoch. Log: `logs/train-3472221.out`; per-lesion errors `runs/3472221/worst/errors.csv` |
| 17 | 2026-11-06 | | Row 15 + BL mask as zero-initialised 2nd stem channel (Team 2 run P4) | `submit_pretrained.sh` (job 3472222): row 15 flags + `--enc-mask` | as row 10 | as row 10 | Best epoch 13: plain 4.67 / 2.70 mm, CPM@10 86.0%, per-patient 4.24. TTA 4.45 / 2.51 mm, 89.9%, per-patient 4.00. Epoch 50: 5.15 / 2.45 | Not a test of the mask channel: `--freeze early` also freezes the stem, so the mask weights stay 0 and the model computes the same function as row 15. It is a second seed of row 15: −0.11 mm, CI [−0.60, +0.51], best epoch 13 vs 28. That is the run-to-run noise. Mean of the plain predictions of rows 15–17: 4.27 / 2.39 mm (seed ensemble, no TTA). Retest with a trainable stem. Log: `logs/train-3472222.out` |
| 18 | 2026-11-07 | | Row 16 (P3-rand, best.pt) on Team 1's hybrid centres (J0) | `submit_team2_next.sh joint0` → `evaluate_joint.sbatch --two-pass` (job 3528816), architecture from `runs/3472221/args.json` | Joint val, 208 lesions | joint mean / median, CPM@10; by start error; misses > 10 mm by landing place | plain 7.67 / 2.87, **TTA 7.67 / 2.96**, CPM@10 78.8%; two-pass 7.65, gated 7.64 / 2.67. Start 0–10 mm (n 131): 3.16; 10–20 (48): 7.78; 20–30 (18): 15.28; ≥ 40 (9): 54.72 | −1.07 mm vs the run 6b joint result (8.74), so most of P3-rand's synthetic gain carries over. Far starts are not fixed at all (≥ 40 mm: 55.3 → 54.7). Of 44 misses > 10 mm, 32% land inside the true FU lesion and 5% inside another lesion (swap); the rest on background. Log: `logs/joint-3528816.out`, results `runs/joint-P3rand/` |
| 19 | 2026-11-07 | | Ensemble of rows 15–17 (P3, P3-rand, P4) on hybrid centres (J0) | `CKPT='runs/3472220/best.pt runs/3472221/best.pt runs/3472222/best.pt'` `evaluate_joint.sbatch --two-pass` (job 3528817) | Joint val, 208 lesions | as row 18 | plain 7.37 / 3.05, **TTA 7.38 / 3.07**, CPM@10 79.8%; **two-pass 7.15 / 2.66**, gated 7.22. Start 0–10: 3.05; 10–20: 7.32; 20–30: 14.19; ≥ 40: 54.31 | Ensemble −0.29 mm vs row 18; the second pass adds −0.23 mm (mostly 20–40 mm starts: 14.2 → 11.1). Swaps 2–5% of misses. Log: `logs/joint-3528817.out`, results `runs/joint-ens3/` |
| 20 | 2026-11-07 | | Row 16 architecture, fixed 25 epochs, judged on last.pt (EMA) (N0, control) | `submit_team2_next.sh` → `train_eval_joint.sbatch` (job 3528821): row 16 flags + `--epochs 25`; data `Longitudinal-CT-processed-team2-big(-aug)` (prep job 3528819, aug job 3528820: `--big-max 56`, 120³ FU cube, same 2652 / 8610 pairs and val windows as before) | train 2159 + 8610 aug / val 207 lesions; Joint val, 208 lesions | synthetic val (last.pt); joint as row 18 | Synthetic: plain 4.53 / 2.38, **TTA 4.40 / 2.37**, CPM@10 89.4% (best epoch 4.50). Joint: **TTA 7.23 / 2.66**, CPM@10 81.3%; two-pass 7.22, gated 7.17. Start 0–10: 3.16; 20–30: 13.16; ≥ 40: 52.59 | Fixed 25-epoch schedule is safe: synthetic ties row 16 (4.44), joint −0.44 vs row 18 (seed / schedule noise or a real gain; unpaired). Baseline for rows 21–25. Epochs 425–560 s (slower node). Log: `logs/train-3528821.out` |
| 21 | 2026-11-07 | | Row 20 + training offset mixture (N1) | row 20 + `--offsets team1_residuals_train.npy --offset-mix 0.4 0.4 0.2 --wide-mm 56` (job 3528822): per sample N(0, 8 mm), a bootstrapped Team 1 train residual vector, or U(±56 mm) | as row 20 | as row 20 | Synthetic: plain 6.22 / 2.83, TTA 5.62 / 2.72, CPM@10 84.1%. Joint: **TTA 6.80 / 2.95**, CPM@10 80.8%; two-pass 7.12, gated 7.05. Start 0–10: 3.39; 10–20: 8.38; 20–30: 18.97; ≥ 40: **21.32** | Works where predicted: −0.43 mm joint vs row 20, almost all from the 9 lesions starting ≥ 40 mm off (52.6 → 21.3). Costs near starts (3.16 → 3.39) and synthetic val (+1.2 mm, more than expected). 20–30 mm bin worse (13.2 → 19.0, n 18): check per lesion. Second pass now hurts: the model already covers far starts. Log: `logs/train-3528822.out` |
| 22 | 2026-11-07 | | Row 21 + grown-lesion target and oversampling (N2) | row 21 + `--uniform-r 10 --oversample-grown 2` (job 3528823): mask term = CE to uniform over the FU mask for r_FU > 10 mm; pairs with volume ratio > 4 or MERGING sampled ×2 | as row 20 | as row 20 | Synthetic: plain 6.36 / 3.47, TTA 5.98 / 3.22. Joint: **TTA 6.88 / 3.94**, CPM@10 79.8%; gated 6.90. Start 0–10: 3.77; ≥ 40: 21.09 | No gain: +0.08 mm joint vs row 21, median 1 mm worse (2.95 → 3.94), synthetic +0.36. The uniform target blurs the map, as the wide Gaussian did in row 6. Logged mask term (~2.5) has a log(cells) floor, not comparable with other runs. Dropped. Log: `logs/train-3528823.out` |
| 23 | 2026-11-07 | | Row 22 + FU lesion growth / shrink warp (N2b) | row 22 + `--p-growth 0.3` (radius ×0.67–2, log-uniform) (job 3528824) | as row 20 | as row 20 | Synthetic: plain 6.20 / 3.33, TTA 5.82 / 3.30. Joint: **TTA 6.76 / 3.63**, CPM@10 78.4%; gated 6.77. Start 0–10: 3.83; ≥ 40: 22.53 | −0.12 mm vs row 22: noise. Grown-lesion branch (rows 22–23) does not beat row 21. Log: `logs/train-3528824.out` |
| 24 | 2026-11-07 | | Row 21 + distractor windows and other-lesion penalty (N3) | row 21 + `--p-distractor 0.2 --lambda-other 0.2` (job 3528825): 20% of windows centred on / between a neighbouring FU lesion ≤ 40 mm away; penalty −log(1 − softmax mass on other lesions' masks) | as row 20 | as row 20 | Synthetic: plain 5.88 / 3.05, TTA 5.40 / 2.61, CPM@10 82.6%. Joint: **TTA 6.67 / 2.93**, CPM@10 78.4%; two-pass 6.87, gated 6.73. Start 0–10: 3.62; 20–30: 17.11; ≥ 40: 20.85 | **Best joint so far**: −4.08 mm vs Team 1 alone, −0.56 vs row 20 (borderline vs MDD), −0.13 vs row 21 (noise). Not via fewer swaps: swap share of misses 11% (5 of 45) vs 5% in row 21, tiny counts. Swaps are rare in every run (2–11%), so distractors are a minor lever. Log: `logs/train-3528825.out` |
| 25 | 2026-11-07 | | Rows 22–24 combined (N4) | row 21 + `--uniform-r 10 --oversample-grown 2 --p-growth 0.3 --p-distractor 0.2 --lambda-other 0.2` (job 3528826) | as row 20 | as row 20 | Synthetic: plain 6.15 / 3.37, TTA 5.89 / 3.28. Joint: **TTA 6.88 / 3.73**, CPM@10 79.3%; gated 6.92. Start 0–10: 3.96; ≥ 40: 22.93 | No better than row 24 alone (+0.21); the grown-lesion changes cost the median again. Next: paired bootstrap on `predictions.csv` (rows 20–25); ensemble row 20 + row 24 (near starts from row 20, far from row 24) or more narrow weight (`--offset-mix 0.6 0.3 0.1`). Log: `logs/train-3528826.out` |


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
