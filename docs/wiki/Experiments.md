# Experiments

| # | Date | Issue | Model | Config | Split | Metric(s) | Result | Notes |
|---|------|-------|-------|--------|-------|-----------|--------|-------|
| | | | | | | | | |
| 1 | 2026-10-22 | | LesionNet (0.32 M params) | scripts/train.sbatch | train only | mean error (mm) | 2 files: 0.00 mm (zero baseline 14.04). 100 train files, 2000 steps, bs 8: 2.41 mm (zero baseline 4.78) | Overfit sanity check, not a real result. 100-file run not yet fully overfit; try smaller smooth-L1 beta / LR decay. Log: logs/train-2941892.out |
