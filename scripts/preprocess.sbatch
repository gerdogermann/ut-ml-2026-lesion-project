#!/bin/bash
#SBATCH -J lesion_prep
#SBATCH --partition=main
#SBATCH -t 04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH -o logs/prep-%j.out

# One-time setup on the LOGIN node (compute nodes may lack internet):
#   module load python
#   python -m venv ~/venvs/lesion && source ~/venvs/lesion/bin/activate
#   pip install numpy pandas SimpleITK
# Raw data is shared (read-only for the team); processed data goes next to it, also shared.
# Run ONCE, as the owner (kristiat). Teammates just read OUT.
# Submit from the repo root:  mkdir -p logs && sbatch scripts/preprocess.sbatch

set -euo pipefail
umask 022                          # outputs world-readable
module purge
module load python
source ~/venvs/lesion/bin/activate

RAW=${RAW:-/gpfs/helios/home/kristiat/data/Longitudinal-CT}            # must contain inputsTr/
OUT=${OUT:-/gpfs/helios/home/kristiat/data/Longitudinal-CT-processed}

PYTHONPATH=src python -m lesionloc.preprocess "$RAW" "$OUT" --workers "$SLURM_CPUS_PER_TASK"
chmod -R o+rX "$OUT"
