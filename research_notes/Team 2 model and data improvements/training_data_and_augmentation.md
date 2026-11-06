# Training data, sampling and augmentation for Team 2 (BL-to-FU offset regression)

Scope: data-side changes only (offset sampling, window size, negatives, intensity and deformation augmentation, extra
data, label handling, patient weighting). Model-side changes are covered in the sibling notes
`matching_architecture.md` and `training_and_self_supervision.md` in this folder.

Starting point (internal, from the repo): run P3-rand, val 4.44 / 2.54 mm (mean / median, flip TTA) on synthetic
N(0, 8 mm) jitter clipped at ±24 mm ([Team2-Model](../../docs/wiki/Team2-Model.md)). In the joint pipeline, with Team 1's
hybrid centres, the run-6b-class model gives 8.74 / 4.21 mm, against 10.75 / 6.71 mm for Team 1 alone. 13.9% of val
lesions start more than 20 mm off, and 11.5% are still more than 20 mm off after Team 2. Team 1's per-axis SD is
(7.1, 8.2, 11.7) mm, and 100% of hybrid centres are within 56 mm per axis ([Joint experiment results](../../reports/Joint%20experiment%20results.md)).

Already tried on the data side ([Experiments](../../docs/wiki/Experiments.md)):
- Re-drawn jitter every epoch (row 5): big win, 7.52 → 6.78 mm.
- Offline rotation ±15° and scale 0.9–1.1 copies, ×4 (row 10): win, −0.43 mm plain.
- Random flips: in use since row 1.

Model-side ideas already tried: a centre prior σ 12 mm (row 12, tie, dropped because it bets on the synthetic jitter
being right) and a Gaussian heatmap loss (row 6, not better).

Not yet tried on the data side: any offset distribution other than N(0, 8) clipped at ±24, intensity augmentation,
elastic or lesion-growth augmentation, hard-negative or distractor sampling, synthetic single-scan pairs, extra
datasets, MERGING handling, and patient weighting.

## Q1. How should the training offset (jitter) distribution be chosen to match Team 1's heavy-tailed window error?

### Takeaway
This is the highest-value change, and the published evidence for it is the most direct. Train on a mixture: Team 1's
real residuals (or a heavy-tailed fit to them) plus a wide component that covers the whole ±56 mm the window allows,
plus the current N(0, 8 mm) for precision near the centre. Keep the stored val split as it is, and add a second val
readout on Team 1's real val centres for model selection. The current ±24 mm clip means the model has never seen 27%
of the window's legal offset range on each axis. Siamese trackers trained with too small a shift range learn a centre
bias, and their border probabilities collapse to zero.

### Cited Findings
- **Training on the upstream proposal distribution (2026, same dataset family).** The LongiSeg-style
  clinician-verified tracker trains on autoPET/CT IV, i.e. Longitudinal-CT, 285 patients. It simulates the FU prompt
  with a 50/50 split: half are drawn from the GT mask, weighted by 1/d² to the centroid; half are "derived from the
  registered follow-up point (which may fall outside the lesion)". Prompt simulation is the largest single ablation
  gain: the longitudinal concat model goes from DSC 45.8 / LDR 66.2 without it to 54.0 / 76.0 with it, on the autoPET IV
  val set. — [Exploiting Longitudinal Context in Clinician-Verified Interactive Lesion Tracking, arXiv 2605.23118 / MICCAI 2026](https://arxiv.org/pdf/2605.23118)
- **The same lab's segmenter trained with tight shifts only.** Promptable Longitudinal Lesion Segmentation, on
  Longitudinal-CT / autoPET IV Task 2, randomly shifts both scans "by up to 4 voxels in each direction" in training. At
  inference "center locations are perfectly aligned". That design assumes a near-perfect upstream centre, which is the
  regime the 2026 paper above moved away from. — [Promptable Longitudinal Lesion Segmentation in Whole-Body CT, arXiv 2509.00613](https://arxiv.org/pdf/2509.00613)
- **Centre bias from a small shift range (SiamRPN++).** Padding breaks strict translation invariance. With zero shift,
  "probabilities on border areas degrade to zero", i.e. a strong centre bias. Larger uniform shifts make the aggregated
  heatmap "closer to the location distribution of test objects". With a 255 px search region, performance rose with the
  shift range up to about ±64 px. As extracted by automated PDF reading: EAO on VOT2016 / VOT2018 was 0.410 / 0.350 at
  ±48 px, 0.418 / 0.362 at ±64 px and 0.408 / 0.350 at ±96 px. Zero shift gave about 0.15 on VOT2018. Check these
  numbers against the paper's table before quoting them. — [Li et al., SiamRPN++, CVPR 2019](https://arxiv.org/pdf/1812.11703)
- **Train/test quality mismatch in cascades (Cascade R-CNN).** "Each bounding box regressor performs best for examples
  of IoU close to the threshold that the detector was trained on." A u=0.5 detector beats u=0.6 on low-IoU inputs and
  loses at high IoU. Training each stage on the previous stage's output distribution ("resampling") gave 38.9 AP,
  against 35.4 for iterative box refinement applied only at inference, on an FPN+ baseline of 34.9. Gains were 2–4 AP
  across backbones. — [Cai & Vasconcelos, Cascade R-CNN, CVPR 2018](https://arxiv.org/pdf/1712.00726)
- **Robustness to a perturbed start in lesion tracking (DLT).** DLT shifted the template centre by up to 0.25·r_t (9
  shifted centres per pair). The drop in CPM@10 mm was 18.0 points for DEEDS, 17.6 for SiamRPN++, 8.49 for DLT and only
  3.62 for DLT-Mix (supervised + self-supervised pairs with more spatial variety). MED rose by 2.4, 2.3, 1.2 and 0.9 mm
  respectively. Note that DLT perturbed the BL template centre, not the FU search centre. — [Cai et al., Deep Lesion Tracker, CVPR 2021](https://arxiv.org/pdf/2012.04872)
- **Coarse-to-fine re-detection for large errors (DaSiamRPN).** In long-term tracking, DaSiamRPN enlarges the search
  region from 255 to 767 px when the score signals failure (enter at < 0.8, leave at > 0.95). "One step iteration of
  local-to-global is sufficient." — [Zhu et al., DaSiamRPN, ECCV 2018](https://arxiv.org/pdf/1808.06048)
- **Internal facts that set the numbers.** The FU window is 64³ at 2 mm (±64 mm). Preprocessing keeps the true FU point
  ≥ 8 mm inside the window, so legal offsets are ±56 mm per axis. `fu_big` is
  `SIZE + 2·ceil(jitter_max / 2 mm)` = 88³ for `--jitter-max 24`, so online re-jitter cannot go beyond ±24 mm
  (`src/lesionloc/preprocess.py` line 142, `dataset.py` line 44). Team 1's hybrid centres are within 56 mm for 100% of
  val lesions. — [preprocess.py](../../src/lesionloc/preprocess.py); [Joint experiment results](../../reports/Joint%20experiment%20results.md)

### Inferences
- **Recommended sampler (my design, not published).** Per training sample, draw the FU offset from one of three
  components, with weights chosen once and then held fixed:
  - (a) about 40%: N(0, 8 mm) per axis, as now, which keeps the median precision;
  - (b) about 40%: an empirical Team 1 train residual vector, i.e. bootstrap the 3-vector, not the axes separately,
    since z is wider (11.7 mm) and the axes may be correlated. Optionally add N(0, 2 mm) so no residual repeats exactly;
  - (c) about 20%: uniform in the ±56 mm cube, or a per-axis Student-t with ν≈3 and scale 10 mm clipped at ±56. This
    covers offsets that Team 1 produces on test but not on train.

  The Cascade R-CNN finding says a regressor trained only on hard, far inputs loses precision on easy inputs. Component
  (a) guards against that. The 2605.23118 paper's 50/50 GT-near / registration-derived mix is the closest published
  analogue.
- **Leakage check on Team 1 residuals.** If any part of Team 1's hybrid (the gate τ = 25 mm, the UGI model) was fitted
  on the train split, its train residuals will be smaller and less heavy-tailed than its val/test residuals, the usual
  stacking problem. Use out-of-fold residuals (e.g. 5 patient-folds of Team 1) if Team 1 has learned parts. Otherwise
  compare the train residual histogram with the val one before trusting it, and let component (c) cover the gap.
- **Do not tune on synthetic val alone.** A wider sampler will likely worsen the synthetic-jitter val mean a little,
  because the model spends capacity on far offsets. Select checkpoints and mixture weights on a val readout with Team 1's
  real val centres. This is the joint `hybrid_tta` number. It uses val twice (Team 1's gate was also chosen on val), so
  confirm once on test at the end.
- **Curriculum.** I found no lesion-tracking paper that ablates an offset-size curriculum. Cheap option: start with
  component (a) only for 5 epochs, which matches the existing warmup, then switch to the mixture. The mask-mass loss
  already speeds early learning (row 8). A curriculum is optional; the mixture is the main change.
- **Expected effect (back-of-envelope, not measured).** About 24 of 208 joint-val lesions are > 20 mm off. All are
  inside the window. If the model learned to find half of them at the current median-like error (≈ 4–5 mm instead of
  ≈ 25–30 mm), the joint mean would fall by about 12 × 22 / 208 ≈ 1.3 mm, from 8.74 to about 7.4 mm. The synthetic val
  mean may rise by 0.1–0.4 mm. Distractor cases (Q3) will cap the gain, because far offsets put more look-alikes in the
  window.
- **Implementation cost: low.** Re-run `preprocess.py --centre fu --jitter-max 56` for train only, giving `fu_big`
  120³. That is ≈ 3.5 MB per file uncompressed for CT + mask, against ≈ 1.4 MB now. Keep the existing val/test files
  untouched: a larger clip changes the rare stored val jitters > 24 mm. The offline augmented copies can stay at
  ±24 mm (88³). Draw large offsets only from the 2,159 original files, or re-run `augment_team2.sbatch` with the larger
  `fu_big`. In `LesionDataset`, replace the single `np.random.normal` call with the mixture draw (about 10 lines) and
  load a `team1_residuals_train.npy`. No architecture change: the soft-argmax already spans ±62 mm.
- **Test-time complement (free).** Iterative re-centring: run the model, re-cut the FU window at the prediction, run
  again. This is the DaSiamRPN / cascade idea at inference. It turns a 40 mm start into a 2nd-pass start of a few mm if
  the 1st pass lands near the right lesion. It needs a window cut from the raw FU scan at inference, which the joint
  pipeline already does. Gate it on peak probability (< 0.2 means unsure; [Team2-Model](../../docs/wiki/Team2-Model.md)
  section 5).

### Gaps
- No published ablation was found that compares Gaussian vs heavy-tailed vs empirical-residual jitter for a 3D lesion
  tracker. The evidence is from 2D tracking (SiamRPN++), detection cascades, and one 2026 longitudinal-CT segmentation
  paper (prompt simulation).
- The per-lesion joint predictions (`runs/joint-restart-3467988/validation/predictions.csv`) are on the cluster and
  were not inspected. How many > 20 mm lesions have a distractor nearer the window centre than the true lesion is
  unknown. That decides how much of the 1.3 mm estimate is reachable.
- Whether Team 1's hybrid has learned parts fitted on train (residual leakage) was not checked.

## Q2. Should the FU window be larger or multi-scale (96³ / 128³, 2–4 mm) to cover offsets > 24 mm? What do lesion trackers use?

### Takeaway
For Team 1's hybrid centres the 128 mm window is already large enough: 100% of val centres are within 56 mm. The
limit is the training offset range (Q1), not the window. A 96³ window at 2 mm would cost about 3.4× compute (≈ 12 h
per 50 epochs), outside the budget. If a larger field is needed, e.g. for `propagated` centres or a weaker upstream, use
a second 64³ window at 4 mm (256 mm) for a coarse pass followed by the current fine pass. That costs the same per pass.
Published trackers either search near-whole volumes (DLT, TLT, at 1–2 mm with factorised correlation) or rely on
registration to land close (LesionLocator, LongiSeg).

### Cited Findings
- DLT's training template input is (32, 384, 384) voxels. It uses a factorised 3D correlation (kernels 7×11×11 split
  into three 2D kernels) to avoid "the heavy computational loads of 3D cross-correlation", and runs on "whole CT
  volumes" 14× faster than DEEDS. Resolution is 2 mm isotropic or 0.8 × 0.8 × 2 mm. — [Cai et al., DLT](https://arxiv.org/pdf/2012.04872)
- TLT resamples all CTs to 1 mm and works on feature maps at stride 8. The ablation is on modules (SSS, RAAM, global
  regressor: MED 7.58 → 5.98 mm, CPM@10 79.6 → 87.4%). It gives no search-size or sampling ablation. — [Tang et al., Transformer Lesion Tracker, MICCAI 2022](https://arxiv.org/pdf/2206.06252)
- LesionLocator propagates the prompt with a pretrained multi-step registration network (GradICON-initialised UNets).
  It then crops a high-resolution ROI "centered around the propagated prompt" at 0.8 × 0.8 × 1 mm, with patch
  224 × 224 × 160 for segmentation; the propagation module uses a uniform 175³ shape. — [Rokuss et al., LesionLocator, CVPR 2025](https://arxiv.org/html/2502.20985)
- DaSiamRPN uses a 255 px search region normally and 767 px (3×) only after a detected failure. — [DaSiamRPN](https://arxiv.org/pdf/1808.06048)
- SiamRPN++ uses a 127 px template and a 255 px search region; the best shift range was about ±25% of the search
  side. — [SiamRPN++](https://arxiv.org/pdf/1812.11703)
- Internal: 64³ costs 365 ms/step and 5.84 GiB peak at batch 16; 255 s/epoch, about 3.5 h for 50 epochs. — [Team2-Model](../../docs/wiki/Team2-Model.md)

### Inferences
- Applying SiamRPN++'s ±25% to a 128 mm window gives ±32 mm, not ±56 mm. So training to ±56 mm uses a larger fraction
  of the window than the 2D tracking optimum. That is a reason to keep a narrow component in the mixture (Q1) and to
  accept that the outer 8–20 mm band will stay weaker. This is an analogy, not evidence.
- 96³ at 2 mm: about 3.4× voxels. Memory roughly ×3.4 (≈ 20 GiB at batch 16, likely fits 32 GB) and time ≈ 3.4×
  (≈ 12 h / 50 epochs; ≈ 6 h for the 25 epochs that suffice, since val peaks at epoch 18–30). Feasible as one
  confirmation run, not for sweeps. Not recommended unless Q1's mixture leaves many lesions at the edge.
- A 4 mm-spacing 64³ window (256 mm field) as a separate coarse model trained on wide offsets, then the existing fine
  model on a re-centred window, is the cheap multi-scale route. The cost is a second preprocessing pass and a second
  training run of the same length. The existing 8 mm coarse template (stage 2) already gives some context; the gain over
  Q1 alone is uncertain.

### Gaps
- DLT's and TLT's exact training search-volume sizes and their positioning relative to the target were not stated in
  the extracted text.
- No 3D lesion-tracking paper was found that ablates search-window size.

## Q3. Hard-negative / look-alike sampling: patients with many similar metastases, distractor-aware training

### Takeaway
DaSiamRPN's semantic hard negatives and SAM's hard-negative embedding sampling are the best evidence. Both show that
training mostly on easy background yields features that separate "lesion vs background" but not "this lesion vs a
similar one". Longitudinal-CT is exhaustively segmented, so every other FU lesion in the window is a known distractor.
That allows two cheap data-side changes. (1) Oversample windows that contain other lesions, by centring the jitter
toward a neighbouring lesion. (2) Add a loss term on probability mass falling on other lesions' masks. Patient
e2c0be2456 (50 val lesions, no gain from the two-scale model: 6.74 → 6.75 mm) is the target case.

### Cited Findings
- DaSiamRPN: in SiamFC/SiamRPN training "the non-semantic background occupies the majority" of each search area, so
  models learn foreground vs background rather than instance-level features. Adding semantic negative pairs (labelled
  objects of the same and other categories) raised VOT2016 EAO from 0.368 to 0.389; positive pairs from detection data
  had raised it from 0.344 to 0.368. Same-category negatives "make the tracker focus on fine-grained representation".
  Augmentations: translation within 12 px, resize 0.85–1.15, illumination, motion blur, 25% grayscale. — [Zhu et al., DaSiamRPN](https://arxiv.org/pdf/1808.06048)
- SAM (self-supervised anatomical embedding): positives are corresponding voxels in two random overlapping crops (overlap
  ratio 0.2–1, CT crop 32 × 96 × 96). Global negatives are any voxel > 3 mm away. Local hard negatives are the top
  candidates on the combined global + local similarity map, randomly subsampled for diversity. Removing hard-negative
  sampling "lowers average accuracy in all tasks". Sampling only the hardest negatives (no diversity) "increases maximum
  errors considerably". SAM reports 91.1% lesion-matching accuracy on DLS, against 82.1% for DLT in its table. — [Yan et al., SAM, arXiv 2012.02383 / IEEE TMI 2022](https://arxiv.org/pdf/2012.02383)
- DLT's anatomical signal encoder exists to stop matches with "visually similar but spurious regions". Removing it
  raises val MED by 1.15 mm, to 8.23 mm. — [DLT](https://arxiv.org/pdf/2012.04872)
- Longitudinal-CT segmentation "was exhaustive, i.e. all present lesions were segmented on all CT volumes". There was no
  minimum size. The median is 7 lesions per study (range 1–95). — [Gatidis et al., Sci Data 2026](https://www.nature.com/articles/s41597-026-07466-y)
- LesionLocator: "accurate distinction of closely located or merging lesions remains a challenge". — [LesionLocator](https://arxiv.org/html/2502.20985)
- Auto-MBT (liver metastases) reports most errors "with clustered lesions". Matched F1 for > 10 lesions/scan was 0.716
  for deformable + overlap matching vs 0.984 for multi-cue matching with neighbour context. — [Karunanayake et al., Eur Radiol Exp 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13369105/)

### Inferences
- **Distractor-centred jitter (data only).** For a fraction p ≈ 0.2 of samples, choose another FU lesion of the same
  patient within about 40 mm of the true one. Place the window centre on it, or halfway between, plus small jitter. The
  window then contains the distractor at or near the centre, and the target off-centre. That is exactly the failure
  where the model snaps to the nearest look-alike. It requires `fu_big` large enough (Q1, 120³) and a per-patient list of
  FU lesion centres, which is available from the CSV `cog_fu` of all lesions, including NEWLYAPPEARING ones.
- **Distractor-mass loss.** Store a second FU mask channel, "all other lesions in this FU window". This needs the full
  FU instance map in preprocessing, a one-line change in `mask_patch` (label ≠ fu_label and > 0). Add
  λ·Σ_{v ∈ other lesions} p(v), or a hinge on it, to `team2_loss`. This is the dense analogue of DaSiamRPN's semantic
  negatives. Expected to help the e2c0be2456-type patients most. The effect size is unknown. Cost: one preprocessing
  re-run plus about 15 lines.
- **Patient-dense oversampling.** Sample patients with many lesions more often in the hard-negative branch only, so
  that the overall patient weighting (Q8) is not distorted.
- **Caution from SAM.** Hardest-only sampling increased maximum errors, so mix distractor-centred samples with ordinary
  ones rather than replacing them.

### Gaps
- No lesion-tracking paper was found that ablates distractor-centred window sampling. The closest evidence is 2D
  tracking (DaSiamRPN) and dense embedding (SAM).
- How many FU lesions sit within 64 mm of each val target is unknown without the full FU label maps. That count would
  give the share of the tail that is distractor-driven.

## Q4. Intensity augmentation for CT (contrast phase, noise, gamma, multi-window channels, kernels)

### Takeaway
Expect a small effect. Longitudinal-CT is single-centre, single-vendor (Siemens), portal-venous, 120 kV, 3 mm slices
with a medium-smooth kernel for every scan. Scanner and kernel shift is therefore small. Residual BL-vs-FU differences
come from contrast timing, dose modulation and therapy-induced change. If added, use HU-respecting random windowing
(shift level, scale width) and mild Gaussian noise, applied **independently to BL and FU**. Avoid naive brightness and
contrast multiplication, which a 2025 study found can hurt CT tumour segmentation. A second, narrow soft-tissue input
window may help small, low-contrast lesions more than augmentation does.

### Cited Findings
- Acquisition: Siemens SOMATOM Sensation 64 / Definition AS / Flash / Force / Biograph128. Standard portal-venous phase
  with body-weight-adapted Ultravist 370. 120 kV, reference 240 mAs with CARE Dose. Reconstructed at 3 mm with a
  "medium-smooth kernel to ensure consistent soft-tissue representation". — [Gatidis et al., Sci Data 2026](https://www.nature.com/articles/s41597-026-07466-y)
- Random windowing (window-level shift + window-width scaling before clipping), against nnU-Net's intensity pipeline
  (contrast, brightness, gamma, inverse gamma) and against none:
  - HepaticVessel, poor contrast-enhancement timing: DSC 0.365 with no intensity augmentation, 0.431 with nnU-Net's
    pipeline, 0.450 with random windowing.
  - CRLM, poor timing: 0.501, 0.528 and 0.546.
  - Single augmentations on LiTS tumour 3D: no intensity augmentation 0.692, intensity scaling 0.650, contrast 0.668,
    window scaling 0.701.

  The authors write that traditional schemes "not respecting the unit of intensity in the CT modality, in fact can hurt
  performance in certain settings". — [Random Window Augmentations for Deep Learning Robustness in CT, arXiv 2510.08116](https://arxiv.org/pdf/2510.08116)
- Window shifting, i.e. placing viewing windows at random around the region of interest during training, "improves
  liver lesion segmentation performance and robustness". Abstract only, numbers not extracted. — [View it like a radiologist, arXiv 2311.14990](https://arxiv.org/abs/2311.14990)
- Trackers that used intensity augmentation:
  - DLT's self-supervised pairs: Gaussian noise (variance 0–0.05), Gaussian blur (σ 0.5–1.5), elastic, rotation,
    scale, crop, each with p = 0.5. — [DLT](https://arxiv.org/pdf/2012.04872)
  - LesionLocator: noise (p = 1.0, variance 0–0.05), blur (p = 0.1), multiplicative brightness (p = 0.15, 0.75–1.25),
    contrast (p = 0.15, 0.75–1.25). — [LesionLocator](https://arxiv.org/html/2502.20985)
  - SAM includes "intensity jitter" in its augmentation ablation. — [SAM](https://arxiv.org/pdf/2012.02383)
- Internal: input is HU clipped to [−1000, 400] and scaled to [0, 1]. The encoder uses InstanceNorm (affine, no running
  stats), which removes global offsets per channel and sample. — [Team2-Model](../../docs/wiki/Team2-Model.md)

### Inferences
- Because of InstanceNorm, a global brightness shift is almost a no-op for this encoder. Window-width and level changes
  (non-linear after clipping) and noise are the parts that still reach the features.
- Apply windowing independently to BL and FU, e.g. level ±40 HU and width ×(0.8–1.2) around the clip window, p ≈ 0.5.
  Add Gaussian noise of about 10–20 HU SD, p ≈ 0.3. Mismatched BL/FU contrast is what Siamese matching must ignore.
  Implementation needs the raw HU; reverse the stored scaling (`x·1400 − 1000`) in the dataset. About 15 lines, no
  preprocessing. Expected gain: a few tenths of a mm at most. Seed noise is 0.1 mm, so one run cannot resolve it. Use
  the paired, patient-clustered bootstrap.
- **Multi-window input.** Lesions of 20–150 HU occupy about 10% of the current [0, 1] range. A 2nd channel at a
  soft-tissue window (W400/L40) gives the stem more contrast on small lesions (8 of the 20 worst in row 8 were r < 5 mm).
  Cost: 2-channel stem (re-init, since the model is random-init anyway), and either re-preprocessing (store raw HU) or
  derive it from the stored [0, 1] values, which keep the HU within [−1000, 400]. This is my inference. I found no
  tracking-specific evidence.
- Kernel and slice-thickness augmentation (blur along z, e.g. σ 0.5–1 voxel) is low priority: all scans use 3 mm slices
  and the same kernel class.

### Gaps
- No study was found that measures intensity augmentation for lesion tracking or offset regression specifically. All
  evidence is from segmentation or registration.
- The per-scan contrast timing and BL-FU interval are not in the dataset CSV, so how much contrast-phase mismatch there
  is between BL and FU is unknown.

## Q5. Elastic / non-rigid deformation and synthetic follow-up generation (growth/shrink, single-scan pairs)

### Takeaway
This is the second-strongest evidence after Q1. On a Tübingen melanoma longitudinal dataset, LesionLocator's synthetic
follow-ups (anatomy-informed lesion growth/shrink deformations) cut tracking MED from 5.09 to 3.89 mm. A 2026 follow-up
reports up to +6 Dice from synthetic longitudinal pretraining. DLT's self-supervised single-scan pairs were worse alone
but made the tracker much more robust to a perturbed start. For Team 2, two cheap uses follow:
- online lesion-local growth/shrink deformation of the FU patch;
- self-pairs (BL from a scan, FU from the same scan, independently augmented) from all 1,969 dropped
  DISAPPEARING / NEWLYAPPEARING / unclear lesions and all FU lesions.

### Cited Findings
- **LesionLocator synthetic follow-ups.**
  - Method: deformation field V = ∇(G_σs ∗ S_lesion) · A(x, y, z) around each lesion. A random voxel-wise amplitude
    field r ∈ U(−3.5, 3.5) is smoothed by G_σr (σ = 3). Amplitudes are sampled from [−22, −18] (shrink) or [15, 25]
    (grow), and σ_s = 4–5.5 depends on lesion size. Several small stages are stacked. Image-level elastic (magnitude
    0.05), rotation ±5°, scale 0.95–1.05 and translation ±5 px are added.
  - Data: 2,728 scans from the lesion-segmentation pool were given a synthetic second time point.
  - Result on the in-house melanoma longitudinal set (60 patients, 159 scans), point prompts: real data only, Dice 62.62
    / CPM@25 83.62 / MED 5.09 mm; with synthetic data, 64.33 / 85.40 / 3.89 mm.

  — [Rokuss et al., LesionLocator, CVPR 2025](https://arxiv.org/html/2502.20985)
- The "Lesion Dataset with Synthetic Follow-Ups" is released (≈ 700 GB, simulated follow-ups "with consistent instance
  labels"). — [MIC-DKFZ/LesionLocator GitHub](https://github.com/mic-dkfz/lesionlocator); [DKFZ repository record](https://inrepo02.dkfz.de/record/299862)
- Pretraining on the LesionLocator synthetic longitudinal corpus (2,625 patients) gave "an improvement of up to 6 Dice
  points" over training from scratch on Longitudinal-CT / autoPET IV. — [arXiv 2509.00613](https://arxiv.org/pdf/2509.00613)
- In the 2026 tracker, pretraining on 2,606 synthetic pairs adds DSC 54.0 → 56.5 and LDR 76.0 → 77.7, on top of prompt
  simulation. — [arXiv 2605.23118](https://arxiv.org/pdf/2605.23118)
- **DLT self-supervised pairs from unpaired images.** Augmentations: elastic (scale 0–0.25), xy rotation ±10°, scale
  0.75–1.25, random crop, noise, blur, each p = 0.5.
  - Clean test: DLT-SSL alone 8.8 mm MED and 71.0% CPM@10, against 7.0 mm and 78.9% for supervised DLT. DLT-Mix (both)
    7.1 mm and 78.7%.
  - Under start perturbation: DLT-Mix lost only 3.6 CPM points, against 8.5 for DLT.

  — [DLT](https://arxiv.org/pdf/2012.04872)
- Synthetic displacement fields are a standard source of supervised registration training pairs; RegNet adds simulated
  respiratory motion to artificial DVFs for chest CT. — [Sokooti et al., RegNet, arXiv 1908.10235](https://arxiv.org/abs/1908.10235)

### Inferences
- **Lesion growth/shrink on the FU patch (online).** Apply the LesionLocator field to the FU patch and mask only (BL
  unchanged), with amplitudes in their range. It adds the appearance change (therapy response) that rotation/scale
  copies do not. The target stays the deformed mask's centre: for a symmetric field about the lesion this is about the
  same point, but recompute the COG from the warped mask to be safe. Cost: about 40 lines (scipy `gaussian_filter` +
  `map_coordinates` on 88³–120³), CPU cost per sample manageable with workers. Expected gain: moderate on SHRUNK/TINY
  cases. Unproven for offset regression.
- **Mild elastic on BL and FU independently.** For example nnU-Net-style, small magnitude, applied only to the window
  around, not on, the lesion. This mimics breathing and organ shift. Low priority: rotation/scale copies already work.
  Elastic warps of the full window change the target if the lesion moves, so warp, then recompute the target.
- **Single-scan self-pairs as extra data.** Take any annotated lesion in any train scan. BL patch = scan at the lesion
  COG, with augmentation A. FU patch = the same scan at COG + offset (Q1 sampler), with independent augmentation B plus
  the growth/shrink field. The target is known exactly. This turns the 1,969 dropped rows (DISAPPEARING BL lesions,
  NEWLYAPPEARING FU lesions, linking_unclear) and every FU-scan lesion into extra pairs: roughly 4,079 + 3,103 lesions
  in total across splits, of which only the train-split patients may be used. DLT's results say self-pairs alone are
  weaker than real pairs (identical anatomy, no real deformation). So mix them at about 20–30% of each batch, as in
  DLT-Mix, and do not replace real pairs. Cost: one preprocessing mode (BL and FU from the same image), reusing
  `augment()`.
- **Using the released 700 GB synthetic set.** That is too large to move for one student run. Subsample a few hundred
  scans with lesions in thorax/abdomen, cut 64³ pairs offline, and pretrain for about 10 epochs before fine-tuning.
  Pretraining with nnFoundationCNN weights gave nothing (row 15 vs 16), but that was generic MAE pretraining, not
  task-matched pairs. Cost: high in disk and data handling. Medium priority.

### Gaps
- No measurement exists of lesion growth/shrink augmentation alone on a centre-offset metric. LesionLocator's MED gain
  mixes extra data volume with the deformation type.
- LesionLocator's evaluation set is a 60-patient Tübingen melanoma set. Whether it overlaps with Longitudinal-CT
  patients (also Tübingen melanoma) is not stated in what was read. If it overlaps, it is in-domain evidence.

## Q6. Extra data: other public longitudinal CT lesion datasets, and using DISAPPEARING / NEW / unclear / unlabelled lesions

### Takeaway
The realistic extra sources are:
- the LesionLocator synthetic follow-up set (public, ≈ 700 GB);
- DeepLesion-derived DLS (3,891 pairs, but 2D RECIST bookmarks and mixed slice thickness, a poor fit for 3D COG
  targets);
- self-pairs from this dataset's own non-paired lesions (Q5).

PanTrack (161 pancreatic CT exams, 45 patients) is announced for release with the MICCAI 2026 paper. NLST-New-lesion-
LongCT (TCIA 2025) annotates only new lung lesions. autoPET IV Task 2 is the same Longitudinal-CT data, not extra.
DISAPPEARING and NEW lesions are most useful as distractor masks (Q3) and as self-pair sources (Q5). As "no-match"
negatives they only pay off if the model gets an absence output.

### Cited Findings
- Longitudinal-CT: 300 patients, 600 CTs, 7,182 lesions (4,079 BL, 3,103 FU), CC BY, exhaustive segmentation. It is the
  data of the autoPET IV challenge 2025. — [Gatidis et al., Sci Data 2026](https://www.nature.com/articles/s41597-026-07466-y)
- autoPET/CT IV dataset: 285 melanoma patients (670 images). PanTrack: 45 pancreatic adenocarcinoma patients, 161 CT
  exams (2–11 per patient). "Code, model and dataset will be released at https://github.com/MIC-DKFZ/LongiSeg." —
  [arXiv 2605.23118](https://arxiv.org/pdf/2605.23118); [MICCAI 2026 paper PDF](https://papers.miccai.org/miccai-2026/paper/3814_paper.pdf)
- DLS (from DeepLesion): 3,008 / 403 / 480 train / val / test lesion pairs. DeepLesion annotations are RECIST bookmarks
  on single axial slices. — [DLT](https://arxiv.org/pdf/2012.04872); [DeepLesion](https://pmc.ncbi.nlm.nih.gov/articles/PMC6052252/)
- NLST-New-lesion-LongCT: a "newly annotated longitudinal dataset derived from the National Lung Screening Trial" of new
  lung lesions in low-dose CT, with point annotations (XLSX), version 1 updated 2025-07-08. — [TCIA](https://www.cancerimagingarchive.net/analysis-result/nlst-new-lesion-longct/)
- LesionLocator synthetic follow-ups: ≈ 700 GB, public. — [GitHub](https://github.com/mic-dkfz/lesionlocator)
- Internal: 1,969 rows are dropped as DISAPPEARING / NEWLYAPPEARING / linking_unclear. — [Team2-Model](../../docs/wiki/Team2-Model.md)

### Inferences
- **DLS** has 2D-bookmark centres, a different population (NIH, mixed protocols) and lesions only on key slices, so FU
  "centres" are 2D. Low priority for fine-tuning. It could serve for pretraining the matcher at 4 mm, where a few-mm
  centre error matters less. Cost: high (download of DeepLesion, about 220 GB, plus conversion).
- **NEWLYAPPEARING FU lesions** never get a BL template, but they are real look-alikes in FU windows: add them to the
  distractor mask (Q3). **DISAPPEARING BL lesions** give templates whose true FU match is absent. Training these as
  negatives (target "no lesion") needs a presence logit or a "null" cell in the softmax. That is a model change, useful
  for the pipeline to flag failures, but it does not lower the error on kept lesions. Medium-low priority for this
  metric.
- **linking_unclear rows**: keep excluded from distance supervision. They may serve as self-pairs (Q5), where the target
  is defined by construction.
- **No train patients from val/test** may be used in self-pairs, even unlabelled ones. The patient split must hold.

### Gaps
- PanTrack's release status as of November 2026 was not confirmed; only the paper's statement was found.
- I did not confirm whether any 2024–2026 public dataset other than Longitudinal-CT gives 3D lesion masks with BL-FU
  correspondences in whole-body CT. None was found in the searches.
- The repo cites FDAT record `qe950-g4h94`, while search found `qwsry-7t837` for "Longitudinal-CT". They may be versions
  of the same release; this was not checked.

## Q7. Handling MERGING lesions (several BL lesions share one FU mask)

### Takeaway
There is no published fix. The literature calls merging a known failure mode, and pipelines assume 0/1 correspondence.
The cheapest sound change: for MERGING pairs, drop or down-weight the distance term and keep the mask-mass term. Any
point inside the merged FU mask then counts as correct. Report MERGING separately. Do not fix it with model changes.

### Cited Findings
- LesionLocator names "closely located or merging lesions" as a remaining challenge. — [LesionLocator](https://arxiv.org/html/2502.20985)
- Hering et al. (earlier Tübingen cohort): lesions "can split or merge" while the pipeline assumes 0 or 1
  correspondence; very close lesions "could be wrongly assigned". — [Hering et al., MIDL 2021](https://proceedings.mlr.press/v143/hering21a/hering21a.pdf)
- Longitudinal-CT links (correspondence, merging, splitting) "were established manually by expert radiologists based on
  spatial proximity, anatomical context, and visual assessment". The topology classes have no SPLITTING value. — [Gatidis et al.](https://www.nature.com/articles/s41597-026-07466-y)
- Internal: for MERGING, the FU mask is the `merged_into` lesion, and the target is the BL lesion's own `cog_fu`.
  Preprocessing warns when the mask COG is > 4 mm from the target (`warn_mask_cog_far`). Large lesions (r > 20 mm)
  average 11.6 mm error, and several share one FU mask. — [Team2-Model](../../docs/wiki/Team2-Model.md); [preprocess.py](../../src/lesionloc/preprocess.py)

### Inferences
- Options, in order of cost:
  1. **Down-weight.** For MERGING samples, set the distance-term weight to about 0.25 and keep L_mask. Cost: a
     per-sample weight from the npz (store `topology_class`), about 5 lines.
  2. **Mask-hit target.** Replace the distance with the distance to the nearest point of the merged FU mask (0 if
     inside). This is the CPM@radius logic as a loss.
  3. **Exclude** MERGING with `warn_mask_cog_far` from training.

  First count MERGING and `warn_mask_cog_far` among the train 2,159 and the val tail. If MERGING is a few % of pairs but
  a large share of > 20 mm errors, option 1 is enough.
- Evaluation must keep the official target (`cog_fu`) for comparability. Report the val metric with and without MERGING.

### Gaps
- The share of MERGING pairs in train/val and their share of the error tail were not computed here. The data is on the
  cluster.
- How `cog_fu` is defined for a merged lesion (the merged blob's COG, or this lesion's own part) is not documented in
  the dataset paper text that was read.

## Q8. Patient-level imbalance (one val patient has 24% of lesions): sampling and weighting

### Takeaway
This is mostly an evaluation problem, not a training one. Report patient-cluster bootstrap CIs and the per-patient mean
(already done). For training, a mild reweighting such as sampling probability ∝ 1/√(n_lesions of patient) limits
domination by 50–95-lesion patients without discarding their many hard look-alike pairs. Those pairs should instead go
to the hard-negative branch (Q3). No published ablation for this setting was found.

### Cited Findings
- Lesion counts per study: median 7 (range 1–95); BL median 8 (1–86), FU median 6 (1–95). — [Gatidis et al.](https://www.nature.com/articles/s41597-026-07466-y)
- Patient-cluster bootstrap (patients as the resampling unit, 10,000 replicates) "avoids treating multiple lesions from
  the same patient as independent observations, preventing high lesion-burden patients from artificially narrowing
  confidence intervals". — [Karunanayake et al., Eur Radiol Exp 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13369105/)
- Ignoring hierarchical structure (many samples per patient) is a named metrics pitfall; per-patient aggregation can
  reveal what pooled averaging hides. — [Maier-Hein / Reinke et al., Nat Methods 2024](https://pubmed.ncbi.nlm.nih.gov/38347140/)
- Internal: per-patient mean < lesion mean in every run (e.g. 3.88 vs 4.44 mm), so patients with many lesions are
  harder. — [Team2-Model](../../docs/wiki/Team2-Model.md)

### Inferences
- Full inverse-frequency patient weighting (1/n) would cut the effective sample size and down-weight the very cases (dense
  metastases) where the model fails. The 1/√n compromise is a judgement call: cost 3 lines (`WeightedRandomSampler`).
  Expected effect on the lesion-level mean: ~0, possibly slightly negative. On the per-patient mean: small positive.
  Only worth it if the per-patient mean is the reported headline.
- Selection noise matters more. With 28 val patients and e2c0be2456 at 24%, checkpoint selection on the lesion mean is
  partly selection on one patient. Select on the per-patient mean, or on the mean with that patient's weight capped,
  which is a zero-cost change. Also report the val mean excluding e2c0be2456 as a sensitivity check.

### Gaps
- No study was found that compares patient-weighted vs uniform lesion sampling for lesion tracking or detection.

## Q9. Overall: which data-side changes to try first, with expected effect and cost

### Takeaway
In this order:
1. Mixture offset sampler with Team 1 residuals and ±56 mm coverage (Q1).
2. Distractor-centred windows plus a distractor-mass loss (Q3).
3. FU lesion growth/shrink deformation plus single-scan self-pairs (Q5).
4. MERGING down-weighting (Q7).
5. HU-respecting window/noise augmentation (Q4).

Larger windows, extra external datasets and patient reweighting are low priority within one V100 and 3.5 h runs.

### Cited Findings
- Training on the upstream (registration-derived) prompt distribution gave the largest ablation gain in the 2026
  longitudinal-CT tracker: LDR 66.2 → 76.0. — [arXiv 2605.23118](https://arxiv.org/pdf/2605.23118)
- Semantic hard negatives: +0.021 EAO on VOT2016. — [DaSiamRPN](https://arxiv.org/pdf/1808.06048)
- Synthetic follow-ups: MED 5.09 → 3.89 mm. — [LesionLocator](https://arxiv.org/html/2502.20985)
- CT intensity augmentation: +2–6 Dice points in segmentation, with naive scaling sometimes negative. — [arXiv 2510.08116](https://arxiv.org/pdf/2510.08116)

### Inferences
| # | Change | Targets | Expected effect (my estimate) | Cost |
|---|---|---|---|---|
| 1 | Offset mixture: 40% N(0, 8) + 40% Team 1 train residuals + 20% wide (±56 mm); `fu_big` 120³ for train; select on joint val | joint tail (> 20 mm, 11.5%) | joint mean −0.5 to −1.5 mm; synthetic val ±0.3 mm | 1 preprocessing job + about 10 lines; 1 run |
| 1b | Iterative re-centring at inference, gated on peak prob | far starts | part of #1's gain with no retraining; adds to it | about 20 lines in the joint eval |
| 2 | Distractor-centred windows (p ≈ 0.2) + distractor-mass loss from the full FU label map | dense-metastasis patients (e2c0be2456) | −0.1 to −0.5 mm val mean, mostly in the tail | preprocessing re-run + about 30 lines |
| 3 | LesionLocator-style growth/shrink on FU + 20–30% single-scan self-pairs | shrunk/tiny lesions, data volume | −0.1 to −0.4 mm | about 60 lines + 1 preprocessing mode; CPU-heavy loader |
| 4 | MERGING: distance weight 0.25, keep mask term | large / merged lesions | small on the mean; cleaner gradients | about 5 lines |
| 5 | Independent BL/FU random windowing + noise; optional soft-tissue 2nd channel | small low-contrast lesions | ≤ 0.3 mm, hard to resolve against 0.1 mm seed noise | about 15 lines (2nd channel: + stem change) |
| 6 | 96³ window or 4 mm coarse model | starts > 56 mm (none with hybrid) | ~0 with Team 1 hybrid | 3.4× compute or a 2nd model |
| 7 | External data (DLS, synthetic 700 GB set) | general features | uncertain; generic pretraining gave 0 here | high (disk, conversion) |
| 8 | Patient 1/√n sampling; select on per-patient mean | evaluation stability | ~0 on the lesion mean | 3 lines |

- Run #1 alone first. It changes the input distribution, and every later ablation should be measured on the joint
  readout it introduces. Use paired patient-clustered bootstrap CIs, since seed noise is about 0.1 mm and val CIs are
  about ±0.5–1 mm.
- Every estimate in the table is an inference from the cited analogues plus the internal tail counts. None is a
  measured result for this model.

### Gaps
- None of the effect sizes are measured on this task. The per-lesion joint predictions (cluster) would let #1's ceiling
  be bounded before training: count lesions > 20 mm and check whether the heatmap has any mass on the true lesion.
