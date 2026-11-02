#!/bin/bash
#SBATCH -J t1_ugi
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH -t 12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -o logs/t1-ugi-%j.out

# Team 1 step 5: pretrained uniGradICON registration of every BL scan to its FU scan, no training.
# One-time setup on the login node and a two-patient check on a GPU node: see scripts/unigradicon_check.sbatch.
# Needs $OUT/lesions.csv from scripts/team1_register.sbatch (or scripts/team1_lesions.py).
# Second run with instance optimisation:  IO=50 sbatch scripts/unigradicon.sbatch
# Submit from the repo root:  mkdir -p logs && sbatch scripts/unigradicon.sbatch

set -euo pipefail
umask 002
module purge
module load python
source ${VENV:-~/venvs/unigradicon}/bin/activate

RAW=${RAW:-/gpfs/helios/projects/ml_project_lesion_tracking/data}
OUT=${OUT:-/gpfs/helios/projects/ml_project_lesion_tracking/data/team1}
IO=${IO:-None}

PYTHONPATH=src python scripts/team1_unigradicon.py "$RAW" "$OUT" --io-iterations "$IO" ${PATIENTS:+--patients "$PATIENTS"}
PYTHONPATH=src python scripts/team1_eval.py "$OUT/lesions.csv" "$OUT/centres/ugi_io$IO.csv" "$OUT/centres/propagated.csv"
