#!/bin/bash
#SBATCH -J lesion_joint
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH -t 02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -o logs/joint-%j.out

# Frozen run 6b (template 5, BL mask) on Team 1's cached hybrid centres.
# Submit from the repository root. Extra arguments can select a smoke test.
# mkdir -p logs
# sbatch scripts/evaluate_joint.sbatch --limit 3
# sbatch scripts/evaluate_joint.sbatch
set -euo pipefail
module purge
module load python
source "${VENV:-$HOME/venvs/lesion}/bin/activate"
export PYTHONPATH=src
export OMP_NUM_THREADS=1
RAW=${RAW:-/gpfs/helios/projects/ml_project_lesion_tracking/data}
TEAM1=${TEAM1:-$RAW/team1}
CKPT=${CKPT:-runs/3288501/best.pt}
OUT=${OUT:-runs/joint-$SLURM_JOB_ID}

for c in $CKPT; do  # CKPT may list several checkpoints (ensemble); each needs args.json beside it unless run 6b-like
  if [[ ! -s "$c" ]]; then
    echo "Checkpoint not found: $c" >&2
    echo "To train from scratch and evaluate automatically: sbatch scripts/train_joint.sbatch" >&2
    echo "To evaluate an existing checkpoint: CKPT=/absolute/path/best.pt sbatch scripts/evaluate_joint.sbatch" >&2
    exit 1
  fi
done

python scripts/evaluate_joint.py --raw "$RAW" --lesions "$TEAM1/lesions.csv" \
    --centres "$TEAM1/centres.csv" --ckpt $CKPT --out "$OUT" \
    --split val --device cuda --template-k 5 "$@"
