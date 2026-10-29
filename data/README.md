
```
RAW DATA(readable to everybody): /gpfs/helios/projects/ml_project_lesion_tracking/data/
PREPROCESSED DATA (readable to everybody): /gpfs/helios/projects/ml_project_lesion_tracking/data/Longitudinal-CT-processed/
  lesions/
```

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
## Team 2 data (`scripts/preprocess_team2.sbatch`)

Output: `/gpfs/helios/projects/ml_project_lesion_tracking/data/Longitudinal-CT-processed-team2/`. Same steps as above, run with
`--centre fu`: the FU window is centred on the true FU lesion plus a random offset (Gaussian, sd 8 mm per axis,
clipped to ±24 mm, fixed per lesion by `--seed`), which simulates Team 1's error. So `target_mm = -jitter_mm`.

Extra fields per npz:
- `bl_mask`, `fu_mask`: uint8 (z, y, x), 1 for this lesion's instance only (nearest-neighbour resampling).
  For `MERGING` lesions the FU mask is the lesion it merged into (`merged_into`).
- `jitter_mm`: the offset of the window centre from the true FU position, mm, (x, y, z).

Extra log counters: `skip_no_mask` (mask file missing), `skip_empty_mask` (label not in BL or FU patch),
`warn_mask_cog_far` (FU mask centre of mass > 4 mm from `target_mm`; a large count means a label or axis bug).
