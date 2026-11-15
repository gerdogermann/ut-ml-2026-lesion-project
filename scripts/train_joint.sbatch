#!/bin/bash
#SBATCH -J lesion_joint_restart
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH -t 12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o logs/joint-restart-%j.out

# Fresh Team 2 run 6b, followed by joint smoke + full validation.
# No old checkpoint is used. Alignment remains a separate cached stage.
# Submit from the repo root: mkdir -p logs && sbatch scripts/train_joint.sbatch
set -euo pipefail
module purge
module load python
source "${VENV:-$HOME/venvs/lesion}/bin/activate"
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

RAW=${RAW:-/gpfs/helios/projects/ml_project_lesion_tracking/data}
TEAM1=${TEAM1:-$RAW/team1}
DATA=${DATA:-$RAW/Longitudinal-CT-processed-team2}
AUG_DATA=${AUG_DATA:-$DATA-aug}
OUT=${OUT:-runs/joint-restart-$SLURM_JOB_ID}
TRAIN_EPOCHS=${TRAIN_EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-16}

# Fail before spending hours training if downstream inputs are unavailable.
for required in "$RAW/data_split.json" "$TEAM1/lesions.csv" "$TEAM1/centres.csv" "$DATA/split.json"; do
    if [[ ! -s "$required" ]]; then
        echo "Missing input: $required" >&2
        echo "See docs/Joint-experiment.md for data preparation and path overrides." >&2
        exit 1
    fi
done
if [[ -e "$OUT" ]]; then
    echo "Output already exists: $OUT. Choose a new OUT for a fresh run." >&2
    exit 1
fi

python - "$DATA" "$AUG_DATA" "$TEAM1" "$RAW" <<'PY'
import sys
from pathlib import Path
import numpy as np
from lesionloc.alignment import CentreTable
from lesionloc.dataset import LesionDataset, load_split
from lesionloc.team1 import read_lesions

data, aug, team1, raw = map(Path, sys.argv[1:])
split = load_split(data / "split.json")
official = load_split(raw / "data_split.json")
for name, patients in split.items():
    if not set(patients) <= set(official[name]):
        raise SystemExit(f"Prepared data {name} patients disagree with the official split")
for root, name, needs_big in ((data, "train", True), (data, "val", False), (aug, "train", True)):
    ds = LesionDataset(root / "lesions", data / "split.json", name, masks=True)
    if not len(ds):
        raise SystemExit(f"No {name} patches in {root}/lesions; prepare data first (see docs/Joint-experiment.md)")
    # Sample the stored format; the loader will check every item during training.
    with np.load(ds.files[0]) as sample:
        keys = {"bl", "fu", "bl_mask", "fu_mask", "target_mm", "has_propagated"}
        if needs_big:
            keys |= {"fu_big", "fu_mask_big"}
        if not keys <= set(sample.files):
            raise SystemExit(f"{ds.files[0]} needs {sorted(keys - set(sample.files))}; regenerate Team 2 data")
    print(f"Input: {len(ds)} {name} patches in {root}", flush=True)
lesions = read_lesions(team1 / "lesions.csv")
val = lesions[lesions["split"] == "val"]
if val.empty or not set(val.patient) <= set(official["val"]):
    raise SystemExit("Alignment validation cohort disagrees with official split")
CentreTable.read(team1 / "centres.csv").require(val)
print(f"Alignment: {len(val)} validation centres", flush=True)
PY

mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/git-commit.txt"
git status --porcelain > "$OUT/git-status.txt"

# Record the exact command before starting. train.py initialises new random
# weights (seed 0) and selects best.pt by validation mean error; no --test.
train_command=(python scripts/train.py --model siamese
    --root "$DATA/lesions" --split-file "$DATA/split.json"
    --aug-root "$AUG_DATA/lesions" --epochs "$TRAIN_EPOCHS"
    --batch-size "$BATCH_SIZE" --lr 3e-4 --workers "$SLURM_CPUS_PER_TASK"
    --rejitter-sd 8 --bl-mask --lambda-mask 0.5 --ema 0.999 --template-k 5
    --out "$OUT/model")
printf '%q ' "${train_command[@]}" > "$OUT/train-command.sh"
printf '\n' >> "$OUT/train-command.sh"
echo "Training from scratch; new checkpoint: $OUT/model/best.pt"
"${train_command[@]}"

# The same independent alignment and prediction modules are used for both runs.
eval_command=(python scripts/evaluate_joint.py --raw "$RAW"
    --lesions "$TEAM1/lesions.csv" --centres "$TEAM1/centres.csv"
    --ckpt "$OUT/model/best.pt" --split val --device cuda --template-k 5)
echo "Running joint smoke test"
"${eval_command[@]}" --out "$OUT/smoke" --limit 3
echo "Running full joint validation"
"${eval_command[@]}" --out "$OUT/validation"
echo "Finished. New checkpoint: $OUT/model/best.pt; results: $OUT/validation/summary.csv"
