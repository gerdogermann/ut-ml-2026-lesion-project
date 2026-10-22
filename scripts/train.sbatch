#!/bin/bash
#SBATCH -J lesion_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH -t 00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -o logs/train-%j.out

# One-time setup on the LOGIN node (compute nodes may lack internet), same venv as preprocessing:
#   module load python
#   source <venv>/bin/activate           (default ~/venvs/lesion; shared project venv: set VENV, see below)
#   pip install torch
# Submit from the repo root:  mkdir -p logs && sbatch scripts/train.sbatch
# Other data location:        DATA=/path/to/Longitudinal-CT-processed sbatch scripts/train.sbatch
# Other venv:                 VENV=/path/to/venv sbatch scripts/train.sbatch

set -euo pipefail
module purge
module load python
source "${VENV:-$HOME/venvs/lesion}/bin/activate"
nvidia-smi

DATA=${DATA:-/gpfs/helios/home/kristiat/data/Longitudinal-CT-processed}   # contains lesions/ and split.json
export PYTHONPATH=src

for bs in 2 8 16; do
  python scripts/train.py --bench --batch-size "$bs" --steps 20
done
python scripts/train.py data/example_lesions/*.npz --steps 300
python scripts/train.py --root "$DATA/lesions" --split-file "$DATA/split.json" --n 100 --batch-size 8 --steps 2000
