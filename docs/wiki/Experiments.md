# Experiments

| # | Date | Issue | Model | Config | Split | Metric(s) | Result | Notes |
|---|------|-------|-------|--------|-------|-----------|--------|-------|
| 1 | 2026-10-23 | | LesionNet v2: residual 3D CNN, BL+FU as 2 channels, flatten + linear head (0.92 M params) | `scripts/train.sbatch` (job 2961900): 30 epochs, batch 8, AdamW lr 1e-3 + cosine, smooth-L1 in mm, flip augmentation; branch `first_e2e_pipeline` | train 2039 / val 186 lesions (224 / 28 patients); test not used | val mean / median error, CPM@10 | Best val mean 14.37 mm (epoch 20) vs zero baseline 14.51 mm; median 8.90 mm, CPM@10 51.6%. Epoch 30: mean 14.41, median 9.76 mm | No better than the registration (zero) baseline. Train loss only 3.74 → 3.47, val median got worse: the model does not learn to locate the lesion. Overfit check on 2 example files in the same job: 0.08 mm. Next: Siamese + heatmap model. Log: `logs/train-2961900.out`, checkpoint `runs/2961900/best.pt` |
