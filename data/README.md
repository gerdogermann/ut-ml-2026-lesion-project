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
