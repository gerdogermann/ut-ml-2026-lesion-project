# Tasks before the mentor meeting

Goal: every member has visible progress to show. Tasks are independent: each one only depends on the
preprocessed data (already produced, 2502 lesions, split 224/28/29 patients) and on the shared interface below.
Each task should take roughly 2–4 hours. Open one GitHub issue per task (use the Task template), branch `<your name>`.

Larger picture: preprocessing (done) → shared data location (Karl) → data loading (Gerdo) → model training (Kristian)
→ evaluation against baselines (Imre). After the meeting these pieces join into one training and evaluation pipeline.

## Shared interface (so nobody blocks anybody)

- Data: currently `/gpfs/helios/home/kristiat/data/Longitudinal-CT-processed/lesions/<patient>_<lesion>.npz`,
  moving to the shared project directory (see Karl's task). Two samples are committed in `data/example_lesions/` for local work.
- Each npz: `bl`, `fu` float16 `(64,64,64)` in (z,y,x) order, values in [0,1]; `target_mm` float `(3,)` in **(x,y,z)** order;
  `has_propagated` bool.
- Split: `split.json` written by preprocessing (`train` / `val` / `test` patient lists).
- Target = true FU lesion position minus FU window centre (mm). The window centre is the registration guess
  (`cog_propagated`) when `has_propagated` is true, otherwise the BL position.
  So **predicting zero = the registration baseline** (or "same position" when there is no propagation).
- Model input: `(B, 2, 64, 64, 64)` (BL and FU stacked as channels); output `(B, 3)` in mm.
- Predictions: `(N,3)` arrays in mm, (x,y,z) order, same row order as the targets.

## Karl — Move data to the shared project directory

Why: the data now lives in Kristian's home directory, which has a small quota, is tied to one person's account,
and is not the place the cluster expects shared project data. Everyone's later training and evaluation jobs read from here,
so it needs one agreed path that the whole team can read and write.

- Copy `Longitudinal-CT/` (raw) and `Longitudinal-CT-processed/` (patches, `split.json`) to the shared project directory
  with `rsync -a`, and keep the folder structure.
- Check the copy: same file count and total size (`find ... | wc -l`, `du -sh`) as the source, and load a few npz files.
- Set group permissions so all four members can read and write (`chgrp -R`, `chmod -R g+rwX`, setgid bit on directories).
- Update the paths in `data/README.md`, `configs/default.yaml` and `scripts/preprocess.sbatch`.
- Fill the data part of `docs/wiki/Compute.md`: path, quota, who has access.
- Only after the team confirms the copy works, tell Kristian that the home-directory copy can be deleted.

Done when: every member can open a file from the new path, and the repo points to it.

## Gerdo — PyTorch Dataset and DataLoader

`LesionDataset` and its tests are written (branch `first_e2e_pipeline`) but not yet on `main`.
Remaining work: [#11](https://github.com/gerdogermann/ut-ml-2026-lesion-project/issues/11).

## Imre — Metrics and trivial baselines

Why: the course requires comparing against a simple localisation baseline, and every model result needs the same metrics
to be comparable. The baseline numbers are the bar the trained model must beat, and the first concrete results
to show the mentor. Predicting zero tells us how good the registration guess already is.

Goal: `src/lesionloc/metrics.py` and `scripts/baselines.py`, working on `target_mm` only (no images needed, so it is fast).

- `compute_metrics(pred_mm, true_mm) -> dict` with both arrays `(N,3)` in mm, (x,y,z) order:
  - Median error, Mean error: median / mean of the Euclidean distance `||pred - true||`.
  - MEDx, MEDy, MEDz: mean absolute error per axis.
  - CPM@5 / CPM@10 / CPM@20: share of lesions with Euclidean error ≤ 5 / 10 / 20 mm.
- Report every metric overall and split by `has_propagated`.
- Baseline A: predict zero (registration / same-position baseline).
- Baseline B: predict the mean `target_mm` of the train set.
- Report both on val and test, and write the numbers to `docs/wiki/Experiments.md`.

Done when: a table of baseline results on val and test is in the wiki, and `metrics.py` has a small test with hand-computed values.

## Kristian — 3D regression model skeleton and GPU run

Why: this is the core of the project, the model that predicts the displacement. Building the skeleton now and checking
that it fits in GPU memory and can learn (overfitting two samples) catches setup problems early, so after the meeting
training on the full dataset is just plugging in Gerdo's loader. The sbatch file and Compute page let everyone run jobs.

Goal: `src/lesionloc/model.py` and `scripts/train.sbatch`.

- Small 3D CNN: input `(B,2,64,64,64)`, a few conv + norm + stride-2 blocks, global pooling, linear head to `(B,3)`.
  Smooth-L1 or MSE loss on `target_mm` (optionally scaled by 56 mm).
- Run it on the GPU on random tensors to check memory and time per step, then overfit the two example files.
- Write the sbatch file and fill the GPU part of `docs/wiki/Compute.md` (partition, environment, command).

Done when: the sbatch job runs on a GPU, the loss on the example files goes near zero, and Compute.md lists exact steps.
Once Gerdo's dataset is merged, swap the example files for the real loader (a 15-minute follow-up).

## Questions to take to the mentor

1. Is it fine that lesions not present at follow-up are excluded (only `UNCHANGED`/`MERGING` kept)?
2. Is the 56 mm limit from the search window acceptable, or should the window be larger?
3. Regression versus heatmap: is regression enough for a first version?
