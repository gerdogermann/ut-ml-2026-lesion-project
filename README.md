# Longitudinal CT Lesion Localization

University of Tartu — Machine Learning (Fall 2026) course project.

## Task

Given two CT scans of the same patient and the known location of a lesion in the
first (baseline) scan, predict the location of that lesion in the follow-up scan.
The focus is **localisation, not segmentation**.

Candidate approaches:

- regress the 3D displacement (Δx, Δy, Δz) of the lesion centre
- predict a probability heatmap of the follow-up location

Project requirements (from the proposal):

- confirm how corresponding lesions across scans are identified in the dataset
- compare against a simple localisation baseline
- evaluate on **held-out patients** (patients never shared between train and test)
- agree on how to handle lesions that are not visible at follow-up

See [`Lesion Localization Research Proposal.txt`](Lesion%20Localization%20Research%20Proposal.txt)
for the original description and the [project wiki](../../wiki) for notes, decisions and experiment logs.

## Data

- Dataset paper: <https://www.nature.com/articles/s41597-026-07466-y>
- Dataset download: <https://fdat.uni-tuebingen.de/records/qe950-g4h94>

Raw and processed data are **not** committed. Put them under `data/` (git-ignored); see
[`data/README.md`](data/README.md).

## Repository layout

```
configs/      experiment configs (YAML)
data/         local data only — git-ignored
docs/wiki/    source of the GitHub wiki pages
notebooks/    exploration notebooks
papers/       reference PDFs — local only, git-ignored
scripts/      CLI entry points (preprocessing, training, evaluation)
src/lesionloc/ Python package with reusable code
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The instructors flag this project as needing GPU/HPC for 3D image work — see the wiki
page on compute for the agreed setup.

## Workflow

- Work is tracked in GitHub Issues; pick an issue, assign yourself, branch from `main`.
- Branch names: `<issue-number>-short-description`, e.g. `12-baseline-displacement`.
- Open a pull request that references the issue (`Closes #12`) and get one review before merging.
- Never commit data, model weights or patient identifiers.

## References

- Cai et al., *Deep Lesion Tracker: Monitoring Lesions in 4D Longitudinal Imaging Studies*, CVPR 2021.
- Rokuss et al., *LesionLocator: Zero-Shot Universal Tumor Segmentation and Tracking in 3D Whole-Body Imaging*, CVPR 2025.
- Dataset descriptor, *Scientific Data* (2026): <https://www.nature.com/articles/s41597-026-07466-y>

## Team

| Name | GitHub |
|------|--------|
| | |
| | |
| | |
| | |
