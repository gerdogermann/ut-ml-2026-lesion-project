#!/bin/bash
#SBATCH -J lesion_aug
#SBATCH --partition=main
#SBATCH -t 12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH -o logs/aug-%j.out

# One-time setup on the LOGIN node (compute nodes may lack internet):
#   module load python
#   python -m venv ~/venvs/lesion && source ~/venvs/lesion/bin/activate
#   pip install numpy pandas SimpleITK
# Team 2 augmented train data: 4 copies per train-split lesion, BL and FU each re-sampled from the raw scan
# at a random rotation (up to 15 deg) and scale (0.9-1.1). Use with train.py --aug-root OUT/lesions.
# Raw data is shared (read-only for the team); processed data goes next to it, also shared.
# Run ONCE. Teammates just read OUT.
# Submit from the repo root:  mkdir -p logs && sbatch scripts/augment_team2.sbatch

set -euo pipefail
umask 022                          # outputs world-readable
module purge
module load python
source ~/venvs/lesion/bin/activate

RAW=${RAW:-/gpfs/helios/projects/ml_project_lesion_tracking/data}            # must contain inputsTr/, targetsTr/
OUT=${OUT:-/gpfs/helios/projects/ml_project_lesion_tracking/data/Longitudinal-CT-processed-team2-aug}

PYTHONPATH=src python -m lesionloc.preprocess "$RAW" "$OUT" --workers "$SLURM_CPUS_PER_TASK" \
    --centre fu --jitter-sd 8 --jitter-max 24 --seed 42 --augment 4 --max-rot 15 --max-scale 0.1
chmod -R o+rX "$OUT"
