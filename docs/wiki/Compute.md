# Compute

The instructors expect GPU/HPC to be needed for 3D image preparation and training.

- [ ] Who has access to UT HPC (Rocket)? Request accounts early.
- [ ] Where does the data live on the cluster?

## GPU training

Cluster: UT HPC. Partition `gpu`, one GPU per job (`--gres=gpu:1`). Check what is available with `sinfo -s`.

### One-time setup (login node; compute nodes may lack internet)

```bash
module load python
python -m venv ~/venvs/lesion            # skip if it already exists (preprocessing uses the same venv)
source ~/venvs/lesion/bin/activate
pip install numpy pandas SimpleITK torch pytest
```

### Run a job

From the repo root on the cluster:

```bash
git pull
mkdir -p logs
sbatch scripts/train.sbatch                                   # data from the default path
DATA=/path/to/Longitudinal-CT-processed sbatch scripts/train.sbatch   # other data location
squeue -u $USER                                               # job state
tail -f logs/train-<jobid>.out                                # output
```

`scripts/train.sbatch` runs, in order:
1. A benchmark on random tensors at batch sizes 2, 8 and 16 (time per step, peak GPU memory).
2. An overfit run on the two example files in `data/example_lesions/`.
3. An overfit run on 100 training-set lesions taken from `$DATA/lesions` + `$DATA/split.json`.

Locally (CPU) the same script works: `PYTHONPATH=src python scripts/train.py data/example_lesions/*.npz --steps 150`.

### Measured

Model `LesionNet` (`src/lesionloc/model.py`), 0.32 M parameters, input `(B,2,64,64,64)`.

| Batch | ms/step | Peak GPU memory | GPU |
|-------|---------|-----------------|-----|
| 2 | _TBD_ | _TBD_ | _TBD_ |
| 8 | _TBD_ | _TBD_ | _TBD_ |
| 16 | _TBD_ | _TBD_ | _TBD_ |
