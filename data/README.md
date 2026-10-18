# data/

Everything in this folder except this file is git-ignored.

Suggested layout:

```
data/
  raw/         original download from https://fdat.uni-tuebingen.de/records/qe950-g4h94
  processed/   resampled / cropped volumes, lesion pair tables
  splits/      patient-level train/val/test split files
```

Split files (patient ID lists) are small and contain no images. If the team agrees,
they can be committed under `configs/splits/` so everyone uses the same split.


RAW DATA(readable to everybody): /gpfs/helios/home/kristiat/data/Longitudinal-CT/
PREPROCESSED DATA (readable to everybody): /gpfs/helios/home/kristiat/data/Longitudinal-CT-processed/
  lesions/

## Preprocessing (`src/lesionloc/preprocess.py`, run via `scripts/preprocess.sbatch`)

Goal: for each lesion in the baseline (BL) scan, predict where it is in the follow-up (FU) scan.

1. **List patients**: every `inputsTr/<patient>.csv` (one row per lesion).
2. **Filter lesions**: skip if `topology_class` is not `UNCHANGED`/`MERGING` (keep only lesions present in both scans),
   if `linking_unclear` is true, or if the BL/FU centre of gravity (`cog_bl`, `cog_fu`) is missing.
3. **Voxel -> mm**: centres are converted to physical patient coordinates, independent of scan spacing/orientation.
4. **FU search-window centre**: `cog_propagated` (registration guess) if present, else the BL position in mm.
5. **Target**: `target_mm = true FU position - window centre` (mm, x,y,z order); this is what the model predicts.
   Lesions with any component > 56 mm are skipped (true spot too close to the window edge:
   64 vox * 2 mm / 2 = 64 mm, minus an 8 mm margin).
6. **Patches** (BL centred on the BL lesion, FU centred on the window centre):
   64^3 voxels at 2 mm (128 mm cube), axes aligned to the patient (not the scanner), padding -1024 HU,
   clipped to [-1000, 400] HU, scaled to [0, 1], stored as float16 in (z, y, x) order.
7. **Save**: one `lesions/<patient>_<lesion>.npz` per kept lesion with `bl`, `fu`, `target_mm`, `has_propagated`.
8. **Parallel + stats**: patients processed in parallel; the job log prints per-patient and total counts
   (kept and each skip reason).
9. **Split**: official `data_split.json`, filtered to patients with samples, written to `split.json`.
   The script fails if a processed patient is missing from the official split.