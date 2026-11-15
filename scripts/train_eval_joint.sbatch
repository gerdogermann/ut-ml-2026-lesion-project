#!/bin/bash
#SBATCH -J t2-next
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH -t 10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o logs/train-%j.out

# One Team 2 fixed-schedule run and its evaluation, both on the final weights (last.pt, EMA):
#   train.py "$@"  ->  worst_cases.py (synthetic val)  ->  evaluate_joint.py --two-pass (Team 1 hybrid centres, val).
# Submitted by scripts/submit_team2_next.sh; flags there. Submit from the repo root.
set -euo pipefail
module purge
module load python
source "${VENV:-$HOME/venvs/lesion}/bin/activate"
nvidia-smi
export PYTHONPATH=src PYTHONUNBUFFERED=1

RAW=${RAW:-/gpfs/helios/projects/ml_project_lesion_tracking/data}
TEAM1=${TEAM1:-$RAW/team1}
DATA=${DATA:-$RAW/Longitudinal-CT-processed-team2-big}  # preprocess --big-max 56; augmented copies in $DATA-aug
OUT=runs/$SLURM_JOB_ID

python scripts/train.py --root "$DATA/lesions" --split-file "$DATA/split.json" --aug-root "$DATA-aug/lesions" \
  --model siamese --batch-size 16 --lr 3e-4 --workers "$SLURM_CPUS_PER_TASK" --out "$OUT" "$@"
python scripts/worst_cases.py --ckpt "$OUT/last.pt" --root "$DATA/lesions" --split-file "$DATA/split.json" "$@"
python scripts/evaluate_joint.py --raw "$RAW" --lesions "$TEAM1/lesions.csv" --centres "$TEAM1/centres.csv" \
  --ckpt "$OUT/last.pt" --out "$OUT/joint" --split val --device cuda --two-pass
