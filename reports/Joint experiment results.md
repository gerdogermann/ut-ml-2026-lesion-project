# Team 2 cuts Team 1's error by a fifth, but the 20 mm tail is still there

The first joint run gives a validation mean error of **8.74 mm** and a median of **4.21 mm**, with 76.4% of lesions within 10 mm. It places the window at Team 1's hybrid centres, then adds the offset from a retrained Team 2 SiameseNet with flip TTA. Team 1's centres alone give 10.75 mm, 6.71 mm and 63.0%. So the model lowers the mean by 2.0 mm (−19%) and the median by 2.5 mm (−37%), on the same 208 lesions with no fallbacks.

The joint mean is still 3.2 mm worse than Team 2's 5.58 mm on simulated jitter, and the shortfall is in the tail. Team 1 alone has 13.9% of lesions more than 20 mm off; the joint pipeline still has 11.5%. The model fixes most lesions that start close and very few that start far. That is the expected effect of training on N(0, 8 mm) jitter clipped at ±24 mm. The next step is to train on Team 1's real train residuals, not on more tuning of the model.

Source: Karl Väärtnõu, branch `karl-joint-alignment-localisation`, commits `14b5c67` and `4e8df07` (2026-11-06). Logs: `logs/joint-3467042.out`, `logs/joint-restart-3467988.out`. Method and wiring: `docs/Joint-experiment.md`.

## What was run

| Job | What | Outcome |
|---|---|---|
| 3467042 | `evaluate_joint.sbatch` with the old run 6b checkpoint | Failed at start: `FileNotFoundError: [Errno 2] No such file or directory: 'runs/3288501/best.pt'`. The checkpoint was not in Karl's checkout |
| 3467988 | `train_joint.sbatch`: retrain run 6b from scratch, then a 3-lesion smoke test, then full val | Completed. Checkpoint `runs/joint-restart-3467988/model/best.pt`, results `runs/joint-restart-3467988/validation/summary.csv` (cluster) |

The retrained model uses the run 6b configuration: BL mask input, mask loss 0.5, 5³ template, EMA 0.999, fresh 8 mm jitter, offline rotation/scale data, flips on, 50 epochs, batch 16, lr 3e-4. Training took 50 × 213 s ≈ 3 h on a V100.

## The retrained model reproduces run 6b

On Team 2's own validation data (207 lesions, simulated jitter, plain readout), the best epoch is 32: **5.58 / 3.72 mm, CPM@10 86.0%**. Run 6b (job 3288501) gave 5.81 / 3.79 mm and 84.1% plain. The 0.23 mm difference is within the run-to-run noise noted in [[Experiments]], so the restart is a fair stand-in for run 6b. As with run 6b, validation is flat to slightly rising after epoch ~32 (epoch 50: 5.73 mm).

## Joint validation results

Official val split, 208 lesions, 28 patients, every eligible lesion, no 56 mm filter. Errors in mm, CPM as share of lesions. MEDx/y/z are per-axis errors.

| Method | Mean | Median | Patient mean | CPM@5 | CPM@10 | CPM@20 | MEDx / MEDy / MEDz | Fallbacks |
|---|---|---|---|---|---|---|---|---|
| `propagated` (dataset alone) | 142.35 | 11.44 | 39.62 | 35.6% | 46.2% | 63.5% | 7.46 / 8.16 / 137.82 | 14 |
| `propagated_plain` | 138.91 | 5.23 | 37.07 | 47.6% | 62.0% | 75.0% | 6.20 / 7.07 / 135.15 | 0 |
| `propagated_tta` | 138.77 | 5.11 | 36.90 | 48.1% | 62.0% | 74.0% | 6.20 / 6.90 / 135.17 | 0 |
| `hybrid` (Team 1 alone) | 10.75 | 6.71 | 10.33 | 38.0% | 63.0% | 86.1% | 4.12 / 5.32 / 6.70 | 0 |
| `hybrid_plain` | 9.30 | 4.16 | 8.40 | 53.4% | 74.5% | 88.0% | 3.58 / 4.48 / 5.70 | 0 |
| **`hybrid_tta`** | **8.74** | 4.21 | **7.92** | **58.7%** | **76.4%** | **88.5%** | 3.20 / 4.33 / 5.37 | 0 |

In window@56: hybrid 100%, propagated 89.4%.

Three findings:

- **Team 1's hybrid centres are the right input.** Dataset propagation is missing for some lesions and falls back to the BL position about 1 m away, so its mean stays above 138 mm with or without Team 2. Its median after refinement (5.1 mm) shows the model works when the window is close, but 10.6% of val lesions are outside the window.
- **Flip TTA helps again:** −0.57 mm mean against plain, the same gain as on run 8 (−0.58 mm). The median does not move (4.16 vs 4.21), so TTA mostly trims larger errors. Keep it for final predictions.
- **The gain is concentrated in the easy cases.** Lesions within 5 mm rise from 38.0% to 58.7% (+20.7 pp), but lesions within 20 mm rise only from 86.1% to 88.5% (+2.4 pp). About 24 of 208 val lesions are still more than 20 mm off.

## Why the joint result is worse than 5.58 mm

Team 2's 5.58 mm was measured with the window placed at the true FU centre plus N(0, 8 mm) per axis, clipped at ±24 mm. Team 1's real errors differ in two ways (from [[Teams]]):

- They are wider in z: per-axis standard deviations of (7.1, 8.2, 11.7) mm on val, against 8 mm in training. The z error is also the largest after refinement (MEDz 5.37 mm).
- They have a heavy tail: 13.9% of val lesions are more than 20 mm off. Some have an axis beyond 24 mm, an offset the model never saw in training, even though the lesion is still inside the 128 mm window.

The CPM@20 numbers fit this. A model that had learned to search the whole window would rescue more of the lesions that start beyond 20 mm.

## Caveats

- **Validation is used twice.** The checkpoint was picked by best val mean on simulated jitter, and Team 1's hybrid gate (τ = 25 mm) was picked on val. Both are mild optimistic biases. The test split has not been used.
- **The sample is small and clustered.** With 208 lesions from 28 patients, differences under ~0.3 mm are noise. Patient e2c0be2456 alone has about a quarter of the val lesions. The patient mean (7.92 mm) is lower than the lesion mean, so patients with many lesions are harder.
- **No per-lesion inspection yet.** The paired per-lesion errors and the worst cases are in `runs/joint-restart-3467988/validation/predictions.csv` on the cluster, which this report does not cover. It is not yet known whether Team 2 ever makes a lesion worse than Team 1's centre.
- **The smoke test is not a result.** Its 3 lesions from 1 patient (1.3 mm) only show the wiring works.
- **The docs are out of date.** `docs/Joint-experiment.md` and the [[Experiments]] entry still say the run is pending.

## Next steps

1. **Train on real residuals.** Retrain Team 2 with the FU offset drawn from Team 1's hybrid residuals on the train split (2173 lesions), instead of N(0, 8 mm) clipped at ±24 mm, then rerun the joint val evaluation. `docs/Joint-experiment.md` already names this as the follow-up. It is the only change aimed at the tail.
2. **Break down the errors.** Split `predictions.csv` by the starting error of the hybrid centre (any axis over 24 mm or not), and list lesions where `hybrid_tta` is worse than `hybrid`. This shows how much step 1 can gain and whether a fallback rule is needed.
3. **Shrink the tail in Team 1.** Team 1's remaining value is the 13.9% of lesions more than 20 mm off, not the median.
4. **Update the docs.** Add the joint row to [[Experiments]] and fix the status line in `docs/Joint-experiment.md`.
5. **Evaluate on test once.** Run test one time only, after the configuration is frozen (`--split test --allow-test`).
