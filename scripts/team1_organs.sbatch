#!/bin/bash
#SBATCH -J t1_organs
#SBATCH --partition=main
#SBATCH -t 04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -o logs/t1-organs-%j.out

# Team 1 step 3 (CPU): organ centroids from the TotalSegmentator masks, then organ_shift / organ_affine centres.
# Run after scripts/totalseg.sbatch; chain it with  sbatch --dependency=afterok:<totalseg job id> ...
# Organ centroids are cached in $OUT/organs.csv and $OUT/organ_hosts.csv; delete both after new masks arrive.
# Submit from the repo root:  mkdir -p logs && sbatch scripts/team1_organs.sbatch

set -euo pipefail
umask 002
module purge
module load python
source ${VENV:-~/venvs/lesion}/bin/activate

OUT=${OUT:-/gpfs/helios/projects/ml_project_lesion_tracking/data/team1}

PYTHONPATH=src python scripts/team1_organs.py "$OUT" --workers "$SLURM_CPUS_PER_TASK"
PYTHONPATH=src python scripts/team1_eval.py "$OUT/lesions.csv" "$OUT"/centres/organ_*.csv
