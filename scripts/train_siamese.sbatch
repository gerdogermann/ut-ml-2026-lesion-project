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
#   ... --rejitter-sd 8                                                   # fresh FU offset every epoch
# Full model (run 5): --rejitter-sd 8 --lambda-hm 1 --dist-warmup 5 --bl-mask --lambda-mask 0.5
# + augmented train data (run 6): add --aug-root "$DATA-aug/lesions" (made by scripts/augment_team2.sbatch)
# Run 6 = run 5 (row 8 of Experiments.md: no heatmap loss) + aug data + weight EMA; 5x more files per epoch, so 50 epochs:
#   --rejitter-sd 8 --bl-mask --lambda-mask 0.5 --aug-root "$DATA-aug/lesions" --ema 0.999 --epochs 50
#   6b: add --template-k 5 (20 mm BL template)     6c: add --prior-sd 12 (centre prior)
# worst_cases.py also prints plain vs local soft-argmax vs flip-averaged (TTA) readouts and per-patient means.
# Setup and venv: see scripts/train.sbatch. Submit from the repo root.

set -euo pipefail
module purge
module load python
source "${VENV:-$HOME/venvs/lesion}/bin/activate"
nvidia-smi

DATA=${DATA:-/gpfs/helios/projects/ml_project_lesion_tracking/data/Longitudinal-CT-processed-team2}  # needs masks
export PYTHONPATH=src

python scripts/train.py --bench --model siamese --batch-size 16 "$@"                     # memory / speed
python scripts/train.py $(ls "$DATA"/lesions/*.npz | head -10) --model siamese "$@" --epochs 200 --batch-size 10 \
  --out "runs/$SLURM_JOB_ID-overfit"                                                     # sanity: should reach ~0 mm
python scripts/train.py --root "$DATA/lesions" --split-file "$DATA/split.json" --model siamese --epochs 150 \
  --batch-size 16 --lr 3e-4 --workers "$SLURM_CPUS_PER_TASK" --out "runs/$SLURM_JOB_ID" "$@"
python scripts/worst_cases.py --ckpt "runs/$SLURM_JOB_ID/best.pt" --root "$DATA/lesions" \
  --split-file "$DATA/split.json" "$@"                                                   # for notebooks/worst_cases.ipynb
