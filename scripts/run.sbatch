#!/bin/bash
#SBATCH -J lesion
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH -t 10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o /gpfs/helios/projects/ml_project_lesion_tracking/scratch/slurm/%j.out

# One job for Team 2 runs. Everything goes to $LESION_STORE/experiments/<row>_<name>/runs/<job id>/.
# Submit from the repository root:
#   sbatch scripts/run.sbatch train <row>_<name> [train.py flags]
#       train -> worst_cases.py (synthetic val, worst/) -> evaluate_joint.py --two-pass (joint val, joint/), on last.pt
#   sbatch scripts/run.sbatch eval <row>_<name> <run id> [best.pt|last.pt] [evaluate_joint.py flags]
#       joint val evaluation of an existing run, in runs/<run id>/evals/<date>_<job id>/
# The job log is slurm.log in that folder. Add the result as a row in docs/wiki/Experiments.md.
# For long runs: sbatch -t 12:00:00 scripts/run.sbatch train ...
set -euo pipefail
module purge
module load python
source "${VENV:-$HOME/venvs/lesion}/bin/activate"
export PYTHONPATH=src PYTHONUNBUFFERED=1
export LESION_STORE=${LESION_STORE:-/gpfs/helios/projects/ml_project_lesion_tracking}
RAW=${RAW:-$LESION_STORE/data}
TEAM1=${TEAM1:-$RAW/team1}
DATA=${DATA:-$RAW/Longitudinal-CT-processed-team2-big}  # preprocess --big-max 56; augmented copies in $DATA-aug
JOB=${SLURM_JOB_ID:?submit with sbatch}

log_to() { mkdir -p "$1"; exec > >(tee -a "$1/slurm.log") 2>&1; }

cmd=${1:?usage: run.sbatch train|eval <row>_<name> ...}; EXP=${2:?experiment name <row>_<name>}; shift 2
case "$cmd" in
  train)
    OUT=$LESION_STORE/experiments/$EXP/runs/$JOB
    log_to "$OUT"
    echo "job $JOB  experiment $EXP  commit $(git rev-parse HEAD)"
    nvidia-smi || true
    python scripts/train.py --experiment "$EXP" --root "$DATA/lesions" --split-file "$DATA/split.json" \
      --aug-root "$DATA-aug/lesions" --model siamese --batch-size 16 --lr 3e-4 --workers "$SLURM_CPUS_PER_TASK" "$@"
    python scripts/worst_cases.py --ckpt "$OUT/last.pt" --root "$DATA/lesions" --split-file "$DATA/split.json" "$@"
    python scripts/evaluate_joint.py --raw "$RAW" --lesions "$TEAM1/lesions.csv" --centres "$TEAM1/centres.csv" \
      --ckpt "$OUT/last.pt" --out "$OUT/joint" --split val --device cuda --two-pass
    ;;
  eval)
    RUN=${1:?run id}; shift
    CK=last.pt; if [[ ${1:-} == *.pt ]]; then CK=$1; shift; fi
    RUNDIR=$LESION_STORE/experiments/$EXP/runs/$RUN
    OUT=$RUNDIR/evals/$(date +%Y-%m-%d)_$JOB
    log_to "$OUT"
    python scripts/evaluate_joint.py --raw "$RAW" --lesions "$TEAM1/lesions.csv" --centres "$TEAM1/centres.csv" \
      --ckpt "$RUNDIR/$CK" --out "$OUT/joint" --split val --device cuda "$@"
    ;;
  *) echo "unknown command $cmd: use train or eval" >&2; exit 2 ;;
esac
