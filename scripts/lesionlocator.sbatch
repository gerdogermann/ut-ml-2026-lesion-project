#!/bin/bash
#SBATCH -J lesionlocator
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH -t 12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o /gpfs/helios/projects/ml_project_lesion_tracking/scratch/slurm/%j.out

# Zero-shot LesionLocator (arXiv 2502.20985) on the joint-evaluation cohort. No training.
# Output: $LESION_STORE/experiments/lesionlocator_zeroshot/<split>_<job id>/ (summary.csv, predictions.csv, slurm.log).
#
# One-time setup on the LOGIN node (compute nodes may lack internet):
#   module load python                       # LesionLocator needs Python >= 3.10
#   git clone https://github.com/MIC-DKFZ/LesionLocator ~/LesionLocator
#   python -m venv ~/venvs/lesionlocator && source ~/venvs/lesionlocator/bin/activate
#   pip install -e ~/LesionLocator && deactivate
#   S=/gpfs/helios/projects/ml_project_lesion_tracking/lesionlocator; mkdir -p $S && cd $S
#   wget https://zenodo.org/api/records/15174217/files/LesionLocatorModel.zip/content -O LesionLocatorModel.zip  # 14 GB
#   unzip LesionLocatorModel.zip && ls LesionLocatorCheckpoint   # expect LesionLocatorSeg/ LesionLocatorTrack/
#   (weights: CC BY-NC-SA 4.0, non-commercial)
#
# Submit from the repository root:
#   LIMIT_PAIRS=1 sbatch -t 01:00:00 scripts/lesionlocator.sbatch   # smoke test: one scan pair, check time and bl_check_mm
#   sbatch scripts/lesionlocator.sbatch                              # val split
# Resume a job that ran out of time (scan pairs with a DONE marker are skipped):
#   OUT=<existing output folder> sbatch scripts/lesionlocator.sbatch
set -euo pipefail
module purge
module load python
export PYTHONPATH=src PYTHONUNBUFFERED=1
export LESION_STORE=${LESION_STORE:-/gpfs/helios/projects/ml_project_lesion_tracking}
RAW=${RAW:-$LESION_STORE/data}
TEAM1=${TEAM1:-$RAW/team1}
CKPT=${CKPT:-$LESION_STORE/lesionlocator/LesionLocatorCheckpoint}
LL_VENV=${LL_VENV:-$HOME/venvs/lesionlocator}
SPLIT=${SPLIT:-val}
MODES=${MODES:-mask point}
JOB=${SLURM_JOB_ID:?submit with sbatch}
OUT=${OUT:-$LESION_STORE/experiments/lesionlocator_zeroshot/${SPLIT}_$JOB}
mkdir -p "$OUT"
exec > >(tee -a "$OUT/slurm.log") 2>&1

echo "job $JOB  split $SPLIT  modes $MODES  commit $(git rev-parse HEAD)${LIMIT_PAIRS:+  limit $LIMIT_PAIRS pairs}"
git status --porcelain
nvidia-smi || true
[[ -d $CKPT/LesionLocatorTrack ]] || { echo "no checkpoint at $CKPT (see setup above)" >&2; exit 1; }
TRACK=$LL_VENV/bin/LesionLocator_track
[[ -x $TRACK ]] || { echo "no $TRACK (see setup above)" >&2; exit 1; }

COMMON=(--raw "$RAW" --lesions "$TEAM1/lesions.csv" --out "$OUT" --split "$SPLIT")
[[ $SPLIT == test ]] && COMMON+=(--allow-test)
[[ -n ${LIMIT_PAIRS:-} ]] && COMMON+=(--limit-pairs "$LIMIT_PAIRS")

source "${VENV:-$HOME/venvs/lesion}/bin/activate"
[[ -s $OUT/pairs.tsv ]] || python scripts/lesionlocator_eval.py prepare "${COMMON[@]}"
deactivate

# One LesionLocator call per scan pair and prompt mode. A failed call is logged and skipped; scoring then falls
# back to the registration point for its lesions, so it stays in the count.
failed=0
while IFS=$'\t' read -r pair bl fu; do
  for mode in $MODES; do
    d=$OUT/pairs/$pair/$mode
    [[ -e $d/DONE ]] && continue
    if [[ $mode == mask ]]; then args=(-t prev_mask -p "$OUT/pairs/$pair/bl_mask_prompt.nii.gz")
    else args=(-t point -p "$OUT/pairs/$pair/bl_points.json"); fi
    echo "=== $pair $mode $(date +%T)"
    start=$SECONDS
    if "$TRACK" -bl "$bl" -fu "$fu" "${args[@]}" -o "$d" -m "$CKPT" -device cuda < /dev/null; then  # stdin: keep pairs.tsv
      echo "$((SECONDS - start)) s" > "$d/DONE"
      echo "    done in $((SECONDS - start)) s"
    else
      failed=$((failed + 1))
      echo "    FAILED after $((SECONDS - start)) s" >&2
    fi
  done
done < "$OUT/pairs.tsv"
echo "LesionLocator calls failed: $failed"

source "${VENV:-$HOME/venvs/lesion}/bin/activate"
python scripts/lesionlocator_eval.py score "${COMMON[@]}" --centres "$TEAM1/centres.csv"
chmod -R o+rX "$OUT"
