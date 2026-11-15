# Team 2 model: SiameseNet with a two-scale ResEnc encoder (run P3-rand)

This page describes Team 2's best model so far: what data it sees, how that data is augmented, how the network is
built, and how it is trained and read out. It is run P3-rand, row 16 of [[Experiments]] (job 3472221). On the
validation split it reaches a mean error of **4.44 mm** and a median of **2.54 mm** with flip test-time augmentation
(TTA). The zero baseline is 12.47 mm, and the previous best (run 6b, row 11) was 5.58 mm. Test numbers have not been
computed yet.

Team 2's task (see [[Teams]]): given a baseline (BL) CT patch centred on a lesion and a follow-up (FU) CT patch whose
centre is near, but not exactly at, the same lesion, predict the offset Δ (mm, x/y/z) from the FU patch centre to the
lesion's true FU position.

## 1. Data

### Source

The data is the public **Longitudinal-CT** dataset (Tübingen; paper https://www.nature.com/articles/s41597-026-07466-y,
download https://fdat.uni-tuebingen.de/records/qe950-g4h94). Each patient has a baseline and a follow-up CT, a
per-patient CSV with one row per lesion, and lesion instance masks for both scans. The CSV links each BL lesion to its
FU counterpart. The fields used here are `lesion_id`, `topology_class`, `linking_unclear`, `merged_into`, `cog_bl` and
`cog_fu` (centre of gravity, voxel coordinates) and `cog_propagated` (the organisers' registration guess, used only to
report a `has_propagated` subset, not as an input).

### Which lesions are kept

Preprocessing is `src/lesionloc/preprocess.py`, run with `--centre fu` by `scripts/preprocess_team2.sbatch`
(log `logs/prep-t2-3279118.out`). A lesion pair is kept only if all of the following hold:

- `topology_class` is `UNCHANGED` or `MERGING`, i.e. the lesion exists in both scans. Disappearing and newly appearing
  lesions have no BL-to-FU correspondence and are dropped, as are rows with `linking_unclear` true (1,969 rows dropped).
- Both centres of gravity are present.
- The lesion's instance label is present in both the BL and the FU patch (17 rows dropped as empty masks).
- The true FU position lies at least 8 mm inside the FU window on every axis (no drops on Team 2 data).

For a `MERGING` lesion, the FU mask is the lesion it merged into (`merged_into`), so several BL lesions can share one
FU mask. The target is still that BL lesion's own `cog_fu`.

### Splits

Patients follow the official `data_split.json`, filtered to patients that keep at least one lesion. No patient is in
two splits; `load_split` asserts this.

| Split | Patients | Lesion pairs | Use |
|---|---|---|---|
| train | 227 | 2,159 | training; plus 8,610 offline-augmented copies (section 2) |
| val | 28 | 207 | model selection, all comparisons in [[Experiments]] |
| test | 29 | 286 | final numbers only; not yet used for any Team 2 model |

Validation is concentrated: patient e2c0be2456 contributes 50 lesions (24%) and 3988c7f88e 26 (13%).

### Patch extraction

Every lesion pair becomes one `.npz` file with two 3D patches resampled from the raw scans with SimpleITK:

- **Geometry:** 64³ voxels at 2 mm isotropic, i.e. a 128 mm cube. The axes are aligned to the patient coordinate
  system, not the scanner. Arrays are stored in (z, y, x) order; targets are (x, y, z) in mm.
- **Intensity:** linear interpolation, padding −1024 HU outside the scan, HU clipped to [−1000, 400] and scaled to
  [0, 1] (`(HU + 1000) / 1400`), stored as float16.
- **Masks:** nearest-neighbour resampling on the same grid, 1 for this lesion's instance label only.
- **BL patch:** centred exactly on the BL lesion's centre of gravity.
- **FU patch:** centred on the true FU position plus a random offset ("jitter"), drawn from N(0, 8 mm) per axis and
  clipped to ±24 mm. The jitter stands in for Team 1's window-placement error. The target is
  `target_mm = true FU position − window centre = −jitter`. The stored jitter is fixed per lesion (seeded by patient,
  lesion and `--seed 42`), so val and test always see the same windows.
- **Re-jitter source:** each file also stores `fu_big` and `fu_mask_big`, an 88³ FU cube centred exactly on the true
  FU lesion (64 + 2·12 voxels), from which training cuts a new window every epoch (section 2).

A file therefore holds `bl`, `fu`, `bl_mask`, `fu_mask`, `fu_big`, `fu_mask_big`, `target_mm`, `jitter_mm` and
`has_propagated`. The model input is the 4-channel tensor `[bl, fu, bl_mask, fu_mask]`. In this model the FU mask is
used only by the loss, never as an input (`forward` reads channels 0–2 only; `test_fu_mask_is_not_an_input` checks
this for the small encoder).

## 2. Augmentation

All augmentation is applied to the train split only. Val and test use the stored, un-augmented files.

### Offline: rotation and scale (`scripts/augment_team2.sbatch`)

For every train lesion, 4 extra copies `<patient>_<lesion>_a<k>.npz` are re-sampled from the raw scans (job 3287945,
log `logs/aug-3287945.out`, 8,610 files; 40 copies dropped because a mask fell outside the patch):

- BL and FU each get **their own** random rotation, about a random axis by an angle uniform in [−15°, +15°], and
  **their own** isotropic scale, uniform in [0.9, 1.1]. The rotation is applied about the lesion centre, by rotating
  the sampling grid, so the BL lesion stays at the BL patch centre.
- Because BL and FU are transformed independently, the model also sees a small rigid and scale change between the two
  time points, which is what real follow-up scans look like.
- Offsets are expressed in patch mm (2 mm per patch voxel), the units the model sees. The copy's fixed FU window sits
  at the lesion's stored jitter rounded to whole voxels, and the copy carries its own `fu_big`, so re-jittering works
  on it too.
- Masks are re-sampled with the same grids, so the FU-mask loss and the BL mask template stay consistent.

Each epoch therefore sees 2,159 + 8,610 = 10,769 train pairs, about 5× the original set.

### Online, every time a sample is loaded (`LesionDataset`)

1. **Re-drawn FU jitter (`--rejitter-sd 8`).** The stored FU window is ignored. A fresh offset is drawn from
   N(0, 8 mm) per axis, rounded to whole 2 mm voxels, clipped to ±12 voxels (±24 mm), and a 64³ window plus its mask
   are cut from `fu_big`. The target becomes `−2·shift` mm. The model never sees the same BL/FU offset twice, which
   cut overfitting markedly when it was introduced (row 5).
2. **Random flips.** Each of the three axes is flipped with probability 0.5, all four channels together. The patch
   centre is preserved (index i → 63 − i), so flipping an image axis only negates the matching target component.

No intensity augmentation is used.

## 3. Architecture

The code is `SiameseNet(bl_mask=True, k=5, encoder="resenc", coarse=True)` in `src/lesionloc/model.py`, with
4.00 M parameters, all trained from random initialisation.

```
BL CT 64³ ─┐                          ┌─ stage 1 (4 mm, 32³) ─ 1×1 proj ─ L2 ─┐
           ├─ shared ResEnc encoder ──┤                                        ├─ 3 similarity maps ─ sum ─ softmax ─ soft-argmax ─ Δ (mm)
FU CT 64³ ─┘                          └─ stage 2 (8 mm, 16³) ─ 1×1 proj ─ L2 ─┘        (BL mask: pooled template)
```

### 3.1 Siamese encoder

One encoder with shared weights processes the BL patch and the FU patch separately. Its input is the CT alone, one
channel; the BL mask is not fed to the encoder in this run. The encoder is the first three stages of the ResEnc-L
architecture (Isensee et al., nnU-Net ResEnc; the same layout as the nnFoundationCNN foundation model), built with
`dynamic-network-architectures`. Stages 3–5 of ResEnc-L and its decoder are not used.

| Part | Output | Grid | Content | Params |
|---|---|---|---|---|
| stem | 32 ch, 64³ | 2 mm | 1 conv 3³ (1 → 32) + InstanceNorm + LeakyReLU | 0.001 M |
| stage 0 | 32 ch, 64³ | 2 mm | 1 residual block | 0.06 M |
| stage 1 | 64 ch, 32³ | 4 mm | 3 residual blocks, the first with stride 2 | 0.61 M |
| stage 2 | 128 ch, 16³ | 8 mm | 4 residual blocks, the first with stride 2 | 3.33 M |

Each residual block is conv 3³ (stride 2 in the first block of a stage) → InstanceNorm → LeakyReLU → conv 3³ →
InstanceNorm, plus a skip path, then addition and LeakyReLU. Where the shape changes, the skip path is a 2×2×2
average pool followed by a 1×1 conv and InstanceNorm. InstanceNorm (affine, no running statistics) makes the
encoder independent of batch size and of global intensity offsets.

### 3.2 Projection

Stage 0 (2 mm) is not used for matching. Stage 1 and stage 2 each pass through their own 1×1 conv to 32 channels
(64 → 32 and 128 → 32), and every cell's 32-dimensional feature vector is L2-normalised. Dot products between
features are therefore cosine similarities in [−1, 1].

### 3.3 Similarity maps

The BL lesion sits exactly at the BL patch centre, which falls between the two middle cells of each grid. Every
template is cut after a 2×2×2 mean, which places a cell exactly on the lesion (`centre_template`). BL and FU pass
through the same grid, so any sub-cell offset of the encoder's grid cancels in the BL-to-FU displacement.

Three maps score every FU cell against the BL lesion:

1. **Fine template (4 mm).** A 5³-cell (20 mm) block of BL stage-1 features around the lesion is cross-correlated
   with the 32³ FU stage-1 map, as a per-sample grouped 3D convolution, and divided by 125. This compares local
   appearance and surroundings.
2. **Mask-pooled template (4 mm).** The mean BL stage-1 feature inside the BL lesion mask (the mask is average-pooled
   onto the 32³ grid), L2-normalised, is dot-producted with every FU cell. This summarises the lesion itself,
   whatever its size.
3. **Coarse template (8 mm).** A 3³-cell (24 mm) block of BL stage-2 features is cross-correlated with the 16³ FU
   stage-2 map and divided by 27. The 16³ result is upsampled ×2 with trilinear interpolation onto the 32³ grid. Stage 2
   sees more context than stage 1, which helps tell look-alike structures apart. Adding this map is what lifted the
   model from 5.55 to 4.44 mm (row 14 vs row 16).

Each map is multiplied by its own learnable temperature, `exp(log_scale[i])`, initialised at 10. Cosine values in
[−1, 1] are too flat for a softmax over 32,768 cells without it.

### 3.4 Readout

The three maps are summed into logits over 32³ cells of 4 mm. A softmax over all cells gives a probability volume p,
and a **soft-argmax** gives the prediction Δ̂ = Σ p(v)·x_v, where x_v is the cell's position in mm relative to the
patch centre (−62 … +62 mm per axis). The forward pass returns Δ̂ (B, 3) and the logits (B, 32, 32, 32).

## 4. Training

Settings, from the `args` line of `logs/train-3472221.out` (`scripts/submit_pretrained.sh` → `train_siamese.sbatch`):

- **Loss:** `L = ‖Δ̂ − Δ‖ / 8 mm + 0.5 · L_mask` (`team2_loss`).
  - Distance term: Euclidean distance in mm, the metric itself, divided by the jitter SD so it is O(1).
  - Mask term: `L_mask = −log Σ_{v ∈ FU mask} p(v)`, the negative log of the probability mass inside the FU lesion
    mask (max-pooled onto the 32³ grid; dilated by one cell for lesions with radius < 5 mm). It pushes probability onto
    the lesion early in training. Samples whose window holds no FU mask are skipped.
- **Optimiser:** AdamW, lr 3e-4 for all parameters, weight decay 1e-4. Linear warmup over the first 10% of steps,
  then cosine decay to 0.
- **Batch and length:** batch 16, 50 epochs of 10,769 train pairs (about 675 steps per epoch).
- **Mixed precision:** fp16 autocast with gradient scaling. The softmax, soft-argmax and loss run in fp32.
- **Weight EMA:** an exponential moving average of the weights (decay 0.999) is what is validated and saved.
- **Checkpoint selection:** the EMA weights with the lowest val mean error (plain readout) are kept as `best.pt`.
  That was epoch 18. Val error rises after about epoch 20–30, so 25–30 epochs would be enough.
- **Hardware:** one Tesla V100-SXM2-32GB. Benchmark: 365 ms per step at batch 16, 5.84 GiB peak memory; 255 s per
  epoch, about 3.5 h for 50 epochs.
- **Sanity check:** the job first overfits 10 files and reaches 0.00 mm.

## 5. Inference

The prediction for a lesion is the mean over all 8 flip combinations (TTA, `scripts/worst_cases.py`): each flipped
input is run through the model, the flipped target components are negated back, and the 8 offsets are averaged. TTA
needs no tuned parameters and gains 0.1–0.6 mm across runs. The peak softmax probability is a useful confidence
score: lesions with peak < 0.2 average 9.4 mm error, the rest 3.3 mm.

## 6. Results and known limits

Validation, 207 lesions (row 16 of [[Experiments]]):

| Readout | Mean | Median | CPM@10 | Per-patient mean |
|---|---|---|---|---|
| plain soft-argmax | 4.57 mm | 2.51 mm | 88.9% | 4.00 mm |
| flip TTA | **4.44 mm** | 2.54 mm | 88.4% | 3.88 mm |

- **Against run 6b:** paired TTA difference −1.14 mm, 95% patient-clustered bootstrap CI [−1.85, −0.56]. 74 lesions
  improved by more than 1 mm and 26 got worse by more than 1 mm.
- **Pretraining does not help.** The same architecture initialised from nnFoundationCNN weights (run P3, row 15) ties
  with it (+0.12 mm, CI [−0.38, +0.55]). The random-init model is kept because it needs no external weights.
- **The tail is unchanged.** The 20 worst lesions carry 37% of the total error; the other 187 average 3.09 mm.
  Lesions with radius > 20 mm still average 11.6 mm. Several of them share one FU mask (merged lesions), where the
  target centre is ambiguous. Patient e2c0be2456 (50 lesions, many similar metastases) did not improve
  (6.74 → 6.75 mm).
- **Seed noise** between two otherwise identical runs is about 0.1 mm on the mean (rows 15 and 17). Averaging the
  predictions of the three two-scale runs gives 4.27 mm without TTA.
- **Validation was also used to pick the epoch**, which makes all val numbers slightly optimistic. The window jitter
  is synthetic N(0, 8 mm); the real input will be Team 1's window centres, whose error distribution may differ.

## Reproduce

On the cluster, from the repo root (setup in [[Compute]]; needs `pip install dynamic-network-architectures`):

```bash
DATA=/gpfs/helios/projects/ml_project_lesion_tracking/data/Longitudinal-CT-processed-team2
sbatch -t 24:00:00 scripts/train_siamese.sbatch --rejitter-sd 8 --bl-mask --lambda-mask 0.5 \
  --aug-root "$DATA-aug/lesions" --ema 0.999 --epochs 50 --template-k 5 --amp \
  --encoder resenc --coarse --warmup 0.1
```
