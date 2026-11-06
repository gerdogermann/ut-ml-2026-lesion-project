#!/bin/bash
# Team 2 pretrained-encoder runs (reports/Pretrained encoders for Team 2.md, run matrix P1/P3/P4).
# Run on the LOGIN node from the repo root (compute nodes may lack internet):  bash scripts/submit_pretrained.sh
# Every run = run 6b flags + an encoder variant; each pretrained run has a random-init twin with the same architecture.
set -euo pipefail
module load python
source "${VENV:-$HOME/venvs/lesion}/bin/activate"
python -c "import dynamic_network_architectures" 2>/dev/null || pip install -q dynamic-network-architectures

DATA=${DATA:-/gpfs/helios/projects/ml_project_lesion_tracking/data/Longitudinal-CT-processed-team2}
CKPT=${CKPT:-/gpfs/helios/projects/ml_project_lesion_tracking/pretrained/nnFoundationCNN/checkpoint_final.pth}
if [ ! -f "$CKPT" ]; then  # 410 MB, CC-BY-SA-4.0, https://huggingface.co/MIC-DKFZ/nnFoundationCNN
  mkdir -p "$(dirname "$CKPT")"
  curl -fL -o "$CKPT.part" https://huggingface.co/MIC-DKFZ/nnFoundationCNN/resolve/main/checkpoint_final.pth && mv "$CKPT.part" "$CKPT"
fi
mkdir -p logs

RUN6B=(--rejitter-sd 8 --bl-mask --lambda-mask 0.5 --aug-root "$DATA-aug/lesions" --ema 0.999 --epochs 50 --template-k 5 --amp)
FT=(--freeze early --enc-lr 1e-4 --warmup 0.1)
sub() { echo "$1: $(sbatch -J "t2-$1" -t 24:00:00 --parsable scripts/train_siamese.sbatch "${@:2}")"; }

sub P1      "${RUN6B[@]}" --encoder resenc --pretrained "$CKPT" --freeze all              # frozen 4 mm features + 1x1 proj
sub P1-rand "${RUN6B[@]}" --encoder resenc                                                # same architecture, random init
sub P3      "${RUN6B[@]}" --encoder resenc --coarse --pretrained "$CKPT" "${FT[@]}"       # + 8 mm map, stages 1-2 fine-tuned
sub P3-rand "${RUN6B[@]}" --encoder resenc --coarse --warmup 0.1                          # same architecture, random init
sub P4      "${RUN6B[@]}" --encoder resenc --coarse --pretrained "$CKPT" "${FT[@]}" --enc-mask  # P3 + BL mask in the stem
squeue -u "$USER"
