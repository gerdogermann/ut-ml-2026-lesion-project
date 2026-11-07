#!/bin/bash
# Team 2 next runs (reports/Team 2 model and data improvements.md). Run on the LOGIN node from the repo root:
#   bash scripts/submit_team2_next.sh            # all three phases, training waits for the data jobs
#   bash scripts/submit_team2_next.sh joint0     # or any subset of: joint0 data train
# joint0: no training. Joint val (Team 1 hybrid centres, two-pass) of P3-rand alone and of the P3 + P3-rand + P4
#         ensemble; needs runs/347222{0,1,2}/best.pt in this checkout. Adds by-start-offset and swap tables.
# data:   Team 1 train residuals (login node, seconds), then FU cubes of 120^3 (+-56 mm room) with other-lesion masks,
#         neighbour centres and volume ratios: base data and 4 augmented copies. Same seed and jitter as before, so the
#         stored val windows are unchanged and synthetic val stays comparable with 4.44 mm. ~45 GB.
# train alone (no "data" in the same call) starts at once: only after the data jobs finished.
# train:  P3-rand architecture, fixed 25 epochs, judged on last.pt, joint val in the same job (~3 h each):
#   N0 control | N1 offset mixture | N2 N1 + grown-lesion target and oversampling | N2b N2 + growth warp
#   N3 N1 + distractor windows and penalty | N4 everything
set -euo pipefail
module load python
source "${VENV:-$HOME/venvs/lesion}/bin/activate"
export PYTHONPATH=src
mkdir -p logs
umask 022

RAW=${RAW:-/gpfs/helios/projects/ml_project_lesion_tracking/data}
TEAM1=${TEAM1:-$RAW/team1}
BIG=${BIG:-$RAW/Longitudinal-CT-processed-team2-big}
RES=$BIG/team1_residuals_train.npy
PHASES=${*:-joint0 data train}
has() { [[ " $PHASES " == *" $1 "* ]]; }

if has joint0; then
  for job in 3472220 3472221 3472222; do  # P3, P3-rand, P4: architecture from the last "args" line of the train log
    [[ -s runs/$job/best.pt ]] || { echo "missing runs/$job/best.pt" >&2; exit 1; }
    python -c "import ast, json, sys; l = [l for l in open(sys.argv[1]) if l.startswith('args ')][-1]
json.dump(ast.literal_eval(l[5:]), open(sys.argv[2], 'w'), indent=1)" "logs/train-$job.out" "runs/$job/args.json"
  done
  echo "J0 P3-rand: $(CKPT=runs/3472221/best.pt OUT=runs/joint-P3rand sbatch --parsable -J t2-J0 \
    scripts/evaluate_joint.sbatch --two-pass)"
  echo "J0 ensemble: $(CKPT='runs/3472220/best.pt runs/3472221/best.pt runs/3472222/best.pt' OUT=runs/joint-ens3 \
    sbatch --parsable -J t2-J0ens scripts/evaluate_joint.sbatch --two-pass)"
fi

DEP=()
if has data; then
  mkdir -p "$BIG"
  python scripts/team1_residuals.py "$TEAM1" "$RES"  # check: train sd / tail not much narrower than val
  prep=$(OUT=$BIG sbatch --parsable -t 08:00:00 scripts/preprocess_team2.sbatch --big-max 56)
  aug=$(OUT=$BIG-aug sbatch --parsable -t 24:00:00 scripts/augment_team2.sbatch --big-max 56)
  echo "data: preprocess $prep, augment $aug"
  DEP=(--dependency=afterok:$prep:$aug)
fi

if has train; then
  BASE=(--rejitter-sd 8 --bl-mask --lambda-mask 0.5 --ema 0.999 --template-k 5 --encoder resenc --coarse
        --warmup 0.1 --amp --epochs 25)
  OFF=(--offsets "$RES" --offset-mix 0.4 0.4 0.2 --wide-mm 56)
  GROW=(--uniform-r 10 --oversample-grown 2)
  DIST=(--p-distractor 0.2 --lambda-other 0.2)
  sub() { echo "$1: $(sbatch --parsable "${DEP[@]}" -J "t2-$1" scripts/train_eval_joint.sbatch "${@:2}")"; }
  sub N0  "${BASE[@]}"
  sub N1  "${BASE[@]}" "${OFF[@]}"
  sub N2  "${BASE[@]}" "${OFF[@]}" "${GROW[@]}"
  sub N2b "${BASE[@]}" "${OFF[@]}" "${GROW[@]}" --p-growth 0.3
  sub N3  "${BASE[@]}" "${OFF[@]}" "${DIST[@]}"
  sub N4  "${BASE[@]}" "${OFF[@]}" "${GROW[@]}" --p-growth 0.3 "${DIST[@]}"
fi
squeue -u "$USER"
