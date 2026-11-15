# Longitudinal CT Lesion Localization

University of Tartu, Machine Learning (Fall 2026), course project.

**Task.** The input is two CT scans of the same patient: a baseline (BL) scan and a follow-up (FU) scan.
The input also gives the centre of a lesion in the BL scan.
The output is the centre of the same lesion in the FU scan.
The project does localisation only, not segmentation.

**Status (2026-11-15).** The two-stage pipeline is complete.
On the validation split, the best pipeline puts the lesion centre **6.67 mm** from the truth on average (median 2.93 mm).
78.4% of the lesions are within 10 mm.

## Current approach

Currently we split the problem into two stages. Each stage has its own team.

```
BL + FU scans, BL lesion centre
        │
        ▼
Team 1: place the window ──► FU window centre c        (registration, no training)
        │
        ▼
Cut a 128 mm patch (64³ voxels, 2 mm) around c
        │
        ▼
Team 2: find the lesion ──► offset Δ                   (SiameseNet)
        │
        ▼
Prediction p = c + Δ
```

- **Team 1** (Gerdo, Imre). Image registration gives the FU window centre. The method is a hybrid of uniGradICON and a local registration.
- **Team 2** (Karl, Kristian). The model is a Siamese 3D network. A shared encoder (ResEnc, two scales) makes features for the BL and FU patches. The BL lesion region is the template. The network correlates the template with the FU patch and uses soft-argmax to get the offset. Inputs are BL CT, FU CT and the BL lesion mask.
- **Test-time augmentation (TTA).** The final prediction is the mean over the 8 flip combinations of the input.

## Results

All results are on the validation split (208 lesions, 28 patients). The Team 2 model is N3 (row 24, job 3528825).

| Method | Mean (mm) | Median (mm) | CPM@10 |
|---|---|---|---|
| Dataset `cog_propagated` (reference) | 142.35 | 11.44 | 46.2% |
| Team 1 hybrid centre only | 10.75 | 6.71 | 63.0% |
| **Team 1 + Team 2, TTA (best)** | **6.67** | **2.93** | **78.4%** |
| Team 2 only, synthetic window (true centre + 8 mm jitter) | 5.40 | 2.61 | 82.6% |

The table shows how the joint result improved:

| Step | Val mean / median (mm) |
|---|---|
| Team 1 hybrid centre only | 10.75 / 6.71 |
| Joint, run 6b model | 8.74 / 4.21 |
| Joint, P3-rand model (pretrained-architecture encoder, two scales) | 7.67 / 2.96 |
| Joint, N0: P3-rand with a fixed 25-epoch schedule | 7.23 / 2.66 |
| Joint, N1: N0 + training offsets from Team 1's real errors | 6.80 / 2.95 |
| **Joint, N3: N1 + distractor windows and other-lesion penalty** | **6.67 / 2.93** |

Rows 0–25 in [Experiments](docs/wiki/Experiments.md) have all the runs, configs and job ids.

### What we learned

1. **The registration step is necessary.** BL and FU patient coordinates are about 1 m apart. The dataset's `cog_propagated` is precise when it exists, but it is missing for some lesions. Then the fallback is about 1 m off.
2. **Team 2 decreases the error of Team 1's centres by about 40%,** from 10.75 mm to 6.67 mm mean. The median decreases from 6.71 mm to 2.93 mm.
3. **Training offsets must match Team 1's real errors.** Training on N(0, 8 mm) jitter only gave a joint error much larger than the synthetic error. Training on Team 1's real errors (N1) fixed most lesions that start 40 mm or more away (52.6 → 21.3 mm).
4. **Grown lesions are the largest error source.** Lesions with an FU/BL volume ratio above 4 have 44.5% of the val error. Two target changes for grown lesions (N2, N2b) made the median worse. We did not keep these changes.
5. **Swaps to a neighbouring lesion are rare.** 2–11% of misses above 10 mm land in another lesion.
6. **Flip TTA usually helps.** It decreases the mean by 0.2–0.6 mm on synthetic val and on the first joint run. In the joint runs 18 and 19, it gave no gain.

### Limits

- Val has 28 patients. One patient (e2c0be2456) has about a quarter of the val lesions. Val differences under about 0.5 mm are probably noise. We did not calculate paired confidence intervals for rows 20–25 yet.
- We used val for two selections: the checkpoint and Team 1's gate threshold (τ = 25 mm).

## Tried experiments

These approaches came before the current approach, or we tried them and did not keep them. Team 2 numbers are on synthetic val (TTA mean in mm, if not stated differently).

| Experiment | Rows | Result |
|---|---|---|
| End-to-end CNN on BL + FU patches, direct offset regression | 1, 3 | Did not beat the "predict zero offset" baseline (14.37 vs 14.51 mm). This result started the two-stage split. |
| First SiameseNet (small encoder, 3³ template) | 4–5 | 7.5 mm, then 6.8 mm. Better than direct regression. |
| Heatmap loss (Gaussian target) | 6 | No gain. The median became worse. |
| BL mask input and FU-mask loss | 7–8 | Small gain. Kept. |
| Offline rotation/scale augmentation and weight EMA | 10 | 5.63 mm. Kept. |
| 5³ template (run 6b) | 11 | 5.58 mm. Kept. |
| Centre prior | 12 | No gain. |
| Frozen pretrained encoder (nnFoundationCNN) | 13 | Failed: 12.2 mm, worse than zero offset. |
| ResEnc encoder, one or two scales (P1-rand, P3, P3-rand, P4) | 14–17 | Two scales gave about 4.5 mm. Pretrained and random start gave the same result. |
| Ensemble of three models, two-pass readout | 19 | Joint 7.15 mm vs 7.67 mm for one model. |
| Uniform target and growth warp for grown lesions (N2, N2b) | 22–23 | Joint median became worse. Not kept. |
| N2 + N2b + N3 combined (N4) | 25 | No better than N3 alone. |

Team 1 also tried rigid, affine, organ-based and B-spline registration, and uniGradICON alone. The hybrid centre gave the best val result. See "Team 1 results" in [Experiments](docs/wiki/Experiments.md).

## Next steps

1. **Calculate paired bootstrap confidence intervals** on the joint `predictions.csv` files of rows 20–25 (cluster, `experiments/02*/runs/*/joint/`). Do this before you choose between the runs.
2. **Ensemble N0 + N3.** N0 is better for near starts, and N3 is better for far starts. As an alternative, train with a narrower offset mix (`--offset-mix 0.6 0.3 0.1`).
3. **Decrease the Team 1 tail.** In N3, lesions that start 20–30 mm off still have a mean error of 17.1 mm. Lesions that start 40 mm or more off have 20.9 mm. A better start position helps these lesions most.
4. **Improve the result for grown lesions.**
5. **Check the 20–30 mm start bin.** It became worse in N1 (13.2 → 19.0 mm). Check each lesion in it.
6. **Run the planned no-flips training** (`scripts/train_siamese_no_flips.sbatch`). It is ready but not submitted.
7. **Fill [Decisions](docs/wiki/Decisions.md):** primary metric, lesions that are not visible at follow-up, and the baseline.
8. **Finish the experiments reorganisation:** move the Team 2 data folders to `processed/`, and test `scripts/run.sbatch` on the cluster.

## Metrics

All errors are Euclidean distances in mm between predicted and true FU centres, in patient (x, y, z) coordinates. The code is in [`src/lesionloc/metrics.py`](src/lesionloc/metrics.py).

| Metric | Meaning |
|---|---|
| Mean / Median error | over all lesions. Mean val error is the selection metric. |
| MEDx / MEDy / MEDz | mean absolute error on each axis |
| CPM@5 / @10 / @20 | share of lesions within 5, 10 or 20 mm |
| Patient mean | mean of the per-patient mean errors |
| In window@56 | Team 1 only: share of lesions with each axis within 56 mm, so the lesion is inside the 128 mm patch |
| Fallbacks | lesions where inference failed. The pipeline uses the Team 1 centre for them, and they stay in the results. |

The split is by patient (official `data_split.json`): train 227, val 28, test 29 patients.
A lesion is eligible when the dataset marks it UNCHANGED or MERGING, the BL–FU link is clear, and the dataset gives both centres.

## Repository layout

```
src/lesionloc/     package: preprocess, dataset, model, losses, registration, organs,
                   alignment (hybrid gate), prediction (TTA), pipeline, joint_eval, metrics
scripts/           entry points (*.py) and SLURM jobs (*.sbatch)
  run.sbatch       one SLURM job for Team 2 runs: train or eval
  team1_*.py       Team 1 registration, organs, uniGradICON, combine, eval
  train.py         Team 2 training (--experiment <row>_<name>)
  worst_cases.py   per-lesion errors on Team 2 patches
  evaluate_joint.py
docs/wiki/         project wiki: Experiments, Teams, Team2-Model, Compute, Decisions
docs/              Joint-experiment.md
reports/           written analyses (joint results, losses, pretrained encoders, improvements)
research_notes/    literature notes for the reports
notebooks/         example lesions, worst cases
tests/             pytest
data/              lesion metadata CSVs and 2 example patches (scans are not in git)
```

## Data and compute

- Dataset paper: <https://www.nature.com/articles/s41597-026-07466-y>
- Dataset download: <https://fdat.uni-tuebingen.de/records/qe950-g4h94>
- Cluster: UT HPC `rocket.hpc.ut.ee`. You can connect only from the university network or the UT VPN. See [Compute](docs/wiki/Compute.md).
- Shared project folder: `/gpfs/helios/projects/ml_project_lesion_tracking/`
  - `data/`: raw scans and `data_split.json`
  - `processed/`: derived data, for example Team 1 outputs (`team1/`). The old paths in `data/` are symlinks.
  - `experiments/<row>_<name>/runs/<job_id>/`: checkpoints (`best.pt`, `last.pt`), `args.json`, results and `slurm.log`. The row is the row number in [Experiments](docs/wiki/Experiments.md).
  - `scratch/`: temporary files
  - `pretrained/`: external weights (nnFoundationCNN, uniGradICON)
  - `totalseg_home/`: TotalSegmentator cache

Do not commit data, model weights or patient identifiers.

## Setup and run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
PYTHONPATH=src pytest
```

On the cluster, run from the repository root:

```bash
sbatch scripts/run.sbatch train <row>_<name> [train.py flags]
sbatch scripts/run.sbatch eval <row>_<name> <job_id> last.pt
```

Each run writes to `experiments/<row>_<name>/runs/<job_id>/` in the shared project folder. `args.json` there has all options and the git commit.

## Workflow

- Track work in GitHub Issues. Assign yourself to the issue.
- Make a branch from `main`.
- Branch names: `<issue-number>-short-description`, for example `12-baseline-displacement`.
- Open a pull request that references the issue (`Closes #12`). Get one review before you merge.
- Add one row in [Experiments](docs/wiki/Experiments.md) for each run. Use `<row>_<name>` as the experiment name.

## References

- Cai et al., *Deep Lesion Tracker: Monitoring Lesions in 4D Longitudinal Imaging Studies*, CVPR 2021.
- Rokuss et al., *LesionLocator: Zero-Shot Universal Tumor Segmentation and Tracking in 3D Whole-Body Imaging*, CVPR 2025.
- Dataset descriptor, *Scientific Data* (2026): <https://www.nature.com/articles/s41597-026-07466-y>
- [Research proposal](Lesion%20Localization%20Research%20Proposal.txt)

## Team

| Name | GitHub | Team |
|------|--------|------|
| Gerdo Germann | [@gerdogermann](https://github.com/gerdogermann) | 1 |
| Imre Saks | [@imresaks](https://github.com/imresaks) | 1 |
| Kristian Tamm | [@kristiantamm](https://github.com/kristiantamm) | 2 |
| Karl Väärtnõu | [@vaartnou](https://github.com/vaartnou) | 2 |
