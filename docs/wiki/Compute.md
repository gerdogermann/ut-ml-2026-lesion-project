# Compute

The instructors expect GPU/HPC to be needed for 3D image preparation and training.

- [ ] Who has access to UT HPC (Rocket)? Request accounts early.
- [ ] Where does the data live on the cluster?

## GPU training

Cluster: UT HPC. Partition `gpu`, one GPU per job; a GPU subtype is required: `--gres=gpu:<subtype>:1`. Subtypes: `tesla` (falcon1-6), `a100-40g` (pegasus), `a100-80g` (pegasus2), `h200-141g` (firefly1-3), `b200-180g` (shepard2-3). `train.sbatch` uses `tesla`; override with `sbatch --gres=gpu:a100-40g:1 scripts/train.sbatch`.

### One-time setup (login node; compute nodes may lack internet)

```bash
module load python
python -m venv ~/venvs/lesion            # skip if it already exists (preprocessing uses the same venv)
source ~/venvs/lesion/bin/activate
pip install numpy pandas SimpleITK pytest
# V100 (`tesla`, CC 7.0) is not supported by the default CUDA 13 torch wheel; use the CUDA 12.6 build
# (also runs on A100/H200, but not on B200 which needs cu128+):
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
```

### Run a job

From the repo root on the cluster:

```bash
git pull
mkdir -p logs
sbatch scripts/train.sbatch                                   # data from the default path
DATA=/path/to/Longitudinal-CT-processed sbatch scripts/train.sbatch   # other data location
VENV=/path/to/venv sbatch scripts/train.sbatch                # other venv (e.g. the shared one in the project dir)
squeue -u $USER                                               # job state
tail -f logs/train-<jobid>.out                                # output
```

`scripts/train.sbatch` runs, in order:
1. An overfit check on the two example files in `data/example_lesions/` (error should go to ~0 mm).
2. Training on the `train` split of `$DATA` (`LesionDataset` with flip augmentation), 30 epochs, with val
   metrics printed every epoch and the best checkpoint saved to `runs/<jobid>/best.pt`.
   Add `--test` to also report the test split with the best checkpoint (only for final numbers).

GPU benchmark only: `python scripts/train.py --bench --batch-size 8`.

Locally (CPU) the same script works: `PYTHONPATH=src python scripts/train.py data/example_lesions/*.npz --epochs 150 --batch-size 2`.

### Measured

First model version (plain conv blocks, 0.32 M parameters), input `(B,2,64,64,64)`. The current residual
version has 0.92 M parameters; rerun `--bench` to update.

| Batch | ms/step | Peak GPU memory | GPU |
|-------|---------|-----------------|-----|
| 2 | 4.4 | 0.04 GiB | V100-PCIE-32GB |
| 8 | 11.1 | 0.10 GiB | V100-PCIE-32GB |
| 16 | 20.1 | 0.18 GiB | V100-PCIE-32GB |
