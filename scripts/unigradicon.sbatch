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
# IO=50 takes ~6 min per pair (~30 h for all), so split it over 4 GPUs as an array, then one short job collects:
#   IO=50 sbatch --array=0-3 scripts/unigradicon.sbatch                 # prints the array job id A
#   IO=50 sbatch --dependency=afterok:A scripts/unigradicon.sbatch     # all cached: writes ugi_io50.csv + eval
# Submit from the repo root:  mkdir -p logs && sbatch scripts/unigradicon.sbatch

set -euo pipefail
umask 002
module purge
module load python
source ${VENV:-~/venvs/unigradicon}/bin/activate

RAW=${RAW:-/gpfs/helios/projects/ml_project_lesion_tracking/data}
OUT=${OUT:-/gpfs/helios/projects/ml_project_lesion_tracking/data/team1}
IO=${IO:-None}

SHARD=${SLURM_ARRAY_TASK_ID:+$SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT}   # set when submitted with --array

PYTHONPATH=src python scripts/team1_unigradicon.py "$RAW" "$OUT" --io-iterations "$IO" ${PATIENTS:+--patients "$PATIENTS"} \
    ${SHARD:+--shard "$SHARD"}
if [ -n "$SHARD" ]; then exit 0; fi   # the collecting run writes the centres file and evaluates
PYTHONPATH=src python scripts/team1_eval.py "$OUT/lesions.csv" "$OUT/centres/ugi_io$IO.csv" "$OUT/centres/propagated.csv"
