#!/bin/bash
#SBATCH -J lesion_siamese
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH -t 12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o logs/train-%j.out

# Team 2 SiameseNet runs. Extra flags go to every train.py call (later flags win), e.g.
#   sbatch scripts/train_siamese.sbatch                                   # run 2: distance loss only
#   sbatch scripts/train_siamese.sbatch --lambda-hm 1 --dist-warmup 5     # run 3
#   sbatch scripts/train_siamese.sbatch --lambda-hm 1 --dist-warmup 5 --bl-mask                    # run 4
#   sbatch scripts/train_siamese.sbatch --lambda-hm 1 --dist-warmup 5 --bl-mask --lambda-mask 0.5  # run 5
# Setup and venv: see scripts/train.sbatch. Submit from the repo root.

set -euo pipefail
module purge
module load python
source "${VENV:-$HOME/venvs/lesion}/bin/activate"
nvidia-smi

DATA=${DATA:-/gpfs/helios/projects/ml_project_lesion_tracking/data/Longitudinal-CT-processed-team2}  # needs masks
export PYTHONPATH=src

python scripts/train.py --bench --model siamese --batch-size 16 "$@"                     # memory / speed
python scripts/train.py $(ls "$DATA"/lesions/*.npz | head -10) --model siamese --epochs 200 --batch-size 10 \
  --out "runs/$SLURM_JOB_ID-overfit" "$@"                                                # sanity: should reach ~0 mm
python scripts/train.py --root "$DATA/lesions" --split-file "$DATA/split.json" --model siamese --epochs 150 \
  --batch-size 16 --lr 3e-4 --workers "$SLURM_CPUS_PER_TASK" --out "runs/$SLURM_JOB_ID" "$@"
