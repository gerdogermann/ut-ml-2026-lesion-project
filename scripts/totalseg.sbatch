#!/bin/bash
#SBATCH -J totalseg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH -t 24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -o logs/totalseg-%j.out

# Team 1 step 3: TotalSegmentator (Wasserthal et al. 2023, 117 CT structures) on every BL and FU scan, once.
# Masks go to $OUT/<scan name>, same geometry as the scan; existing masks are skipped, so resubmit after a timeout.
# One-time setup on the LOGIN node (compute nodes may lack internet); own venv, it pins its own nnU-Net:
#   module load python
#   python -m venv ~/venvs/totalseg && source ~/venvs/totalseg/bin/activate
#   pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126   # V100 needs the cu126 build
#   pip install TotalSegmentator
#   export TOTALSEG_HOME_DIR=/gpfs/helios/projects/ml_project_lesion_tracking/totalseg_home
#   totalseg_download_weights -h      # then download the weights for task total (and the --fast variant)
# --fast (3 mm model) is enough for organ centroids and ~5x quicker; FAST=0 runs the full 1.5 mm model.
# Submit from the repo root:  mkdir -p logs && sbatch scripts/totalseg.sbatch

set -euo pipefail
umask 002
module purge
module load python
source ${VENV:-~/venvs/totalseg}/bin/activate
export TOTALSEG_HOME_DIR=${TOTALSEG_HOME_DIR:-/gpfs/helios/projects/ml_project_lesion_tracking/totalseg_home}

RAW=${RAW:-/gpfs/helios/projects/ml_project_lesion_tracking/data}
OUT=${OUT:-/gpfs/helios/projects/ml_project_lesion_tracking/data/team1/totalseg}
FAST=${FAST:-1}
mkdir -p "$OUT"

n=0; failed=0
for f in "$RAW"/inputsTr/*_img_*.nii.gz; do
    o="$OUT/$(basename "$f")"
    [ -s "$o" ] && continue
    if TotalSegmentator -i "$f" -o "$o" --ml $([ "$FAST" = 1 ] && echo --fast) > /dev/null; then
        n=$((n + 1)); echo "ok $(basename "$f")"
    else
        failed=$((failed + 1)); echo "FAILED $(basename "$f")"; rm -f "$o"
    fi
done
echo "segmented $n, failed $failed, total masks $(ls "$OUT" | wc -l)"
