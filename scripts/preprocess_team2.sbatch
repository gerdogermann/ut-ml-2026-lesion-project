#!/bin/bash
#SBATCH -J lesion_prep_t2
#SBATCH --partition=main
#SBATCH -t 04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH -o logs/prep-t2-%j.out

# One-time setup on the LOGIN node (compute nodes may lack internet):
#   module load python
#   python -m venv ~/venvs/lesion && source ~/venvs/lesion/bin/activate
#   pip install numpy pandas SimpleITK
# Team 2 data: FU window centred on the true FU lesion + random offset (sd 8 mm, max 24 mm per axis), with masks.
# Raw data is shared (read-only for the team); processed data goes next to it, also shared.
# Run ONCE. Teammates just read OUT.
# Submit from the repo root:  mkdir -p logs && sbatch scripts/preprocess_team2.sbatch

set -euo pipefail
umask 022                          # outputs world-readable
module purge
module load python
source ~/venvs/lesion/bin/activate

RAW=${RAW:-/gpfs/helios/projects/ml_project_lesion_tracking/data}            # must contain inputsTr/, targetsTr/
OUT=${OUT:-/gpfs/helios/projects/ml_project_lesion_tracking/data/Longitudinal-CT-processed-team2}

PYTHONPATH=src python -m lesionloc.preprocess "$RAW" "$OUT" --workers "$SLURM_CPUS_PER_TASK" \
    --centre fu --jitter-sd 8 --jitter-max 24 --seed 42
chmod -R o+rX "$OUT"
