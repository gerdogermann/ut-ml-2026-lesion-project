#!/bin/bash
#SBATCH -J t1_register
#SBATCH --partition=main
#SBATCH -t 24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH -o logs/t1-register-%j.out

# Team 1 steps 0-2 and 4 (CPU only): lesion table + bl/propagated centres, then SimpleITK registration.
# Uses the same venv as preprocessing (~/venvs/lesion: numpy pandas SimpleITK).
# Transforms are cached in $OUT/transforms, so a resubmit after a timeout continues where it stopped.
# Quick check on two patients first:  PATIENTS=<pid1>,<pid2> sbatch -t 01:00:00 scripts/team1_register.sbatch
# Submit from the repo root:  mkdir -p logs && sbatch scripts/team1_register.sbatch

set -euo pipefail
umask 002                          # Team 1 members share $OUT: group-writable
module purge
module load python
source ${VENV:-~/venvs/lesion}/bin/activate

RAW=${RAW:-/gpfs/helios/projects/ml_project_lesion_tracking/data}            # must contain inputsTr/, targetsTr/
OUT=${OUT:-/gpfs/helios/projects/ml_project_lesion_tracking/data/team1}
METHODS=${METHODS:-rigid,affine,local,bspline}

[ -s "$OUT/lesions.csv" ] || PYTHONPATH=src python scripts/team1_lesions.py "$RAW" "$OUT"
PYTHONPATH=src python scripts/team1_register.py "$RAW" "$OUT" --methods "$METHODS" --workers 8 \
    ${PATIENTS:+--patients "$PATIENTS"}
PYTHONPATH=src python scripts/team1_eval.py "$OUT/lesions.csv" "$OUT"/centres/*.csv
