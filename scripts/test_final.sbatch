#!/bin/bash
#SBATCH -J lesion_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH -t 06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o logs/test-%j.out

# Final test-split numbers for both teams. Run the test split ONCE; nothing here is tuned.
#   Team 1 alone:       team1_eval.py, every centres method (hybrid_ugi_local = centres.csv is the hand-over).
#   Team 2 alone:       worst_cases.py on the synthetic-jitter patches (plain / local / flip TTA).
#   Team 1 + Team 2:    evaluate_joint.py on Team 1's centres. Primary metric, fixed in advance: hybrid_tta.
# Models: N3 (row 24, job 3528825, best joint val) and N0 (row 20, job 3528821, best synthetic val), both last.pt.
# Submit from the repository root:
#   mkdir -p logs
#   SPLIT=val LIMIT=5 sbatch -t 01:00:00 scripts/test_final.sbatch   # check: synthetic val TTA N3 5.40, N0 4.40 mm
#   sbatch scripts/test_final.sbatch                                  # the test run
set -euo pipefail
module purge
module load python
source "${VENV:-$HOME/venvs/lesion}/bin/activate"
nvidia-smi
export PYTHONPATH=src PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1

RAW=${RAW:-/gpfs/helios/projects/ml_project_lesion_tracking/data}
TEAM1=${TEAM1:-$RAW/team1}
DATA=${DATA:-$RAW/Longitudinal-CT-processed-team2-big}
RUNS=${RUNS:-3528825 3528821}
SPLIT=${SPLIT:-test}
ARCH=(--bl-mask --template-k 5 --encoder resenc --coarse)  # rows 20 and 24; worst_cases.py does not read args.json
TEST=(); [[ $SPLIT == test ]] && TEST=(--test)
ALLOW=(); [[ $SPLIT == test ]] && ALLOW=(--allow-test)

echo "git $(git rev-parse HEAD), split $SPLIT, runs $RUNS${LIMIT:+, joint limit $LIMIT}"
git status --porcelain
for job in $RUNS; do
  for f in last.pt args.json; do
    [[ -s runs/$job/$f ]] || { echo "missing runs/$job/$f" >&2; exit 1; }
  done
done

echo "=== Team 1 alone ==="
python scripts/team1_eval.py "$TEAM1/lesions.csv" "$TEAM1"/centres/*.csv "${TEST[@]}"

for job in $RUNS; do
  echo "=== Team 2 run $job alone, synthetic jitter, $SPLIT ==="
  python scripts/worst_cases.py --ckpt "runs/$job/last.pt" --root "$DATA/lesions" --split-file "$DATA/split.json" \
    --split "$SPLIT" "${ARCH[@]}"
  echo "=== Team 1 + Team 2 run $job, $SPLIT ==="
  python scripts/evaluate_joint.py --raw "$RAW" --lesions "$TEAM1/lesions.csv" --centres "$TEAM1/centres.csv" \
    --ckpt "runs/$job/last.pt" --out "runs/$job/joint-$SPLIT-$SLURM_JOB_ID" --split "$SPLIT" "${ALLOW[@]}" \
    --device cuda ${LIMIT:+--limit "$LIMIT"}
done
