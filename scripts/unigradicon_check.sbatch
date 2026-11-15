#!/bin/bash
#SBATCH -J t1_ugi_check
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH -t 01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o logs/t1-ugi-check-%j.out

# Team 1 step 5, check before the full run: torch / torchvision / CUDA on a GPU node, the unigradicon-register flags,
# then two patients registered and compared with the other methods on the same lesions. Installs nothing.
# One-time setup on the LOGIN node (needs internet; installing and downloading are light, running is not):
#   module load python
#   python -m venv ~/venvs/unigradicon && source ~/venvs/unigradicon/bin/activate
#   pip install --upgrade pip
#   pip install torch==2.14.1 torchvision --index-url https://download.pytorch.org/whl/cu126   # together, before
#   pip install unigradicon pandas SimpleITK                                                 # unigradicon
#   python -c "from unigradicon import get_unigradicon; get_unigradicon()"   # downloads the network weights
# Submit from the repo root:  mkdir -p logs && sbatch scripts/unigradicon_check.sbatch

set -euo pipefail
umask 002
module purge
module load python
source ${VENV:-~/venvs/unigradicon}/bin/activate

RAW=${RAW:-/gpfs/helios/projects/ml_project_lesion_tracking/data}
OUT=${OUT:-/gpfs/helios/projects/ml_project_lesion_tracking/data/team1}
PATIENTS=${PATIENTS:-c9e1074f5b,6d4a60c0cd}    # the registration quick check: compare on the same 51 lesions

echo "node $(hostname)"
pip list 2>/dev/null | grep -i -E "^(torch|torchvision|unigradicon|icon.registration|itk) "
python - <<'EOF'
import torch, torchvision
from torchvision.ops import nms  # fails here if torchvision does not match torch
assert "+cu126" in torch.__version__, f"torch was replaced: {torch.__version__}"
assert torch.cuda.is_available(), "no CUDA device visible"
print("torch", torch.__version__, "torchvision", torchvision.__version__, "gpu", torch.cuda.get_device_name(0))
EOF
unigradicon-register --help

PYTHONPATH=src python scripts/team1_unigradicon.py "$RAW" "$OUT" --io-iterations None --patients "$PATIENTS"
PYTHONPATH=src python scripts/team1_eval.py "$OUT/lesions.csv" --covered \
    "$OUT"/centres/{propagated,rigid,affine,local,bspline8,ugi_ioNone}.csv
