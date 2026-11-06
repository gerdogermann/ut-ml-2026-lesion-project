#!/bin/bash
#SBATCH -J lesion_no_flips
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH -t 12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o logs/train-%j.out

# Run 6b comparison: same data, model and hyperparameters, without training flips.
# Keep the four offline rotated/scaled variants per lesion. Validation is unflipped.
# Submit from the repo root: mkdir -p logs && sbatch scripts/train_siamese_no_flips.sbatch
set -euo pipefail
module purge
module load python
source "${VENV:-$HOME/venvs/lesion}/bin/activate"
nvidia-smi

DATA=${DATA:-/gpfs/helios/projects/ml_project_lesion_tracking/data/Longitudinal-CT-processed-team2}
AUG_ROOT=${AUG_ROOT:-$DATA-aug/lesions}
export PYTHONPATH=src

git rev-parse HEAD
git status --short
python scripts/train.py --root "$DATA/lesions" --split-file "$DATA/split.json" \
    --model siamese --batch-size 16 --lr 3e-4 --epochs 50 \
    --workers "$SLURM_CPUS_PER_TASK" --out "runs/$SLURM_JOB_ID" \
    --rejitter-sd 8 --bl-mask --lambda-mask 0.5 --aug-root "$AUG_ROOT" \
    --ema 0.999 --template-k 5 --no-flips
