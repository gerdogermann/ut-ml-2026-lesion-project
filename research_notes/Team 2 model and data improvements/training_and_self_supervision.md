# Training procedure, self-/semi-supervision and honest evaluation for the Team 2 Siamese lesion matcher

Scope: training-side changes for SiameseNet run P3-rand (val mean 4.44 mm TTA, median 2.54 mm; 4.0 M params; 2,159 train
pairs + 8,610 offline aug copies; AdamW 3e-4, wd 1e-4, warmup + cosine, batch 16, 50 epochs, EMA 0.999; best epoch 18,
val error rising after ~20–30 epochs; seed noise ~0.1 mm; 3-model average 4.27 mm without TTA) — see
[Team2-Model.md](../../docs/wiki/Team2-Model.md). Supervised nnFoundationCNN pretraining (row 15 vs 16: +0.12 mm,
CI [−0.38, +0.55]) and the loss families are already covered in
[Pretrained encoders for Team 2](../../reports/Pretrained%20encoders%20for%20Team%202.md) and
[Lesion patch localisation losses](../../reports/Lesion%20patch%20localisation%20losses.md); not repeated here.
Tool budget note: about 17 search/fetch calls; several classic papers (SWA, stochastic depth, label smoothing, PixPro,
VADeR) were not re-fetched and are flagged where cited from memory.

## 1. Self-supervised matching pretraining on synthetic single-CT pairs

### Takeaway
The strongest published evidence for this task family is SAM-style pixel-wise contrastive pretraining on unlabeled CT,
where one volume yields two crops with known correspondence: with zero lesion labels it beat supervised trackers on
DeepLesion lesion matching (91.1% vs 82.1%). The data volumes were 20 k DeepLesion sub-volumes up to >50 k CTs, far more
than this project's ~570 scans, and no paper isolates the gain inside a correlation tracker that already starts within
±24 mm. Expect fewer gross misses, not a better median.

### Cited Findings
- SAM (Yan et al., TMI 2022) makes positive pairs from a single unlabeled CT: "cropping two 3D patches with random
  location and size, resizing them to the same shape", with corresponding pixels in the overlap. CT augmentations were
  random crop, scaling and intensity jitter (deformation and rotation ≤30° only for X-ray). Loss is InfoNCE, τ = 0.5,
  with 500 online hard negatives (most cosine-similar non-matching embeddings) + 1,000 random negatives for the global
  embedding, and 500 negatives for the local one. Convergence took about 25 k iterations; GPU hours are not stated —
  [SAM, arXiv 2012.02383](https://arxiv.org/html/2012.02383v3)
- SAM pretraining data: DeepLesion (≈20 k sub-volumes), NIH-Lymph Node (176 scans) and an in-house chest CT set (94
  patients) — [SAM](https://arxiv.org/html/2012.02383v3)
- SAM lesion-matching accuracy on 3,556 manually annotated DeepLesion lesion pairs: SAM 91.1% vs supervised lesion
  embedding 80.7%, LesaNet 82.1%, DLT 82.1% — [SAM](https://arxiv.org/html/2012.02383v3);
  [SAM PDF](https://www.cs.jhu.edu/~lelu/publication/SAM__Self_supervised_Learning_of_Pixel_wise_Anatomical_Embeddings_in_Radiological_Images.pdf)
- SAM ablations (lesion-matching accuracy): full 91.1%; without coarse-to-fine 84.1%; without hard-negative sampling
  89.0%; without "diverse negative sampling" 92.2% (higher on matching but much worse max landmark error, 58.7 vs 16.4 mm
  on contrast chest CT); trained on NIH-LN alone (176 scans) 91.4%; reported robust to embedding dimension (default
  c = 128) — [SAM](https://arxiv.org/html/2012.02383v3)
- On the DLS tracking benchmark SAM (no lesion labels) reaches 6.1 ± 6.7 mm, CPM@10 86.0%; supervised DLT trained on ~3 k
  pairs 7.0 ± 8.9 mm, 78.9%; SAM++ 5.4 ± 6.0 mm, 88.3% — as compiled in the project's earlier report from
  [SAM++, arXiv 2306.13988](https://arxiv.org/pdf/2306.13988) and [DLT, arXiv 2012.04872](https://arxiv.org/html/2012.04872v2)
- UAE (MedIA 2025, SAM follow-up) raises single-modality lesion tracking from 81.87% to 84.06% over SAM —
  [UAE, arXiv 2311.15111](https://arxiv.org/html/2311.15111v3)
- DLT itself already combines supervision with self-supervision: it compares a template image with augmented copies
  `T_aug(I_t)` of the same image, using non-longitudinal DeepLesion images as unlabeled data; augmentations are elastic
  deformation, rotation, random scale, random crop, Gaussian noise and blur — [DLT](https://arxiv.org/html/2012.04872v2)
- DLT component analysis (Table 3, MED in mm): removing the anatomy signal encoder (ASE) or the learned global kernel
  gives 9.3 mm; feature-dimension and kernel-size changes move MED between 7.4 and 9.4 mm. The fetch did not return a
  "without self-supervision" row — [DLT](https://arxiv.org/html/2012.04872v2)
- Vizitiu et al. (MICCAI 2023, Siemens) trained multi-scale self-supervised tracking on >50,000 CTs, with optional
  supervision from automatically extracted landmarks; 92% DeepLesion matching accuracy and "localization accuracy that
  is nearly 10% higher than the state-of-the-art". Reviewers noted the pipeline is "almost identical to that of SAM" and
  that there is no ablation, so the source of the gain is unknown —
  [MICCAI 2023 paper page and reviews](https://conferences.miccai.org/2023/papers/452-Paper3115.html)
- Generic CT SSL gains are modest: VoCo pretrained Swin-UNETR on ~1.6 k CTs (BTCV 24 + TCIA-Covid19 722 + LUNA 843) and
  lifted BTCV Dice from 80.53 (scratch) to 83.85 (+3.32) —
  [VoCo, CVPR 2024, arXiv 2402.17300](https://arxiv.org/html/2402.17300v2). OpenMind's best SSL (MAE) added +2.47 Dice to
  ResEnc-L, and a frozen MAE encoder scored 30.38 vs 68.36 Dice fine-tuned (from the earlier report,
  [OpenMind, arXiv 2412.17041](https://arxiv.org/abs/2412.17041))
- Public unlabeled CT pools (sizes from memory, not re-fetched this session): DeepLesion 10,594 studies of 4,427 patients
  ([DLT/DeepLesion summary](https://www.researchgate.net/publication/355885490_Deep_Lesion_Tracker_Monitoring_Lesions_in_4D_Longitudinal_Imaging_Studies));
  TotalSegmentator ~1.2 k CTs ([arXiv 2208.05868](https://arxiv.org/abs/2208.05868)); AbdomenCT-1K ~1.1 k CTs
  ([arXiv 2010.14808](https://arxiv.org/abs/2010.14808))

### Inferences
- The proposed pretext task (crop + affine/elastic deformation of one CT, known dense correspondence) is SAM's recipe
  with DLT's augmentations. It can train the exact SiameseNet (same encoder, same 32³ correlation + soft-argmax head) end
  to end. The pretext target is the same Δ, so no head mismatch arises, unlike the nnFoundationCNN transfer.
- Pretext design that mimics the real task: sample a "lesion-like" anchor (random voxel inside the body, or
  intensity-weighted towards soft-tissue blobs), cut BL 64³ around it, and cut FU from an independently rotated (±15°),
  scaled (0.9–1.1), elastically deformed copy with jitter N(0, 8 mm). This matches the current offline-aug and jitter
  statistics, so pretraining and fine-tuning see the same geometry.
- Compute fits easily: at the logged 365 ms/step (batch 16) one V100 day is ≈ 237 k steps ≈ 3.8 M synthetic pairs,
  about 10× SAM's 25 k iterations. The bottleneck will be CPU resampling. Precompute 2 mm, HU-clipped volumes (or
  ~128³ cubes) once, then deform on the GPU (`grid_sample`).
- Data-size realism: the project's own scans (~227 train patients × 2 ≈ 450 volumes; 284 × 2 ≈ 570 including val/test)
  are small, but SAM's NIH-LN-only run (176 scans) still reached 91.4%. So scan count may matter less than variety of
  anchors. Adding TotalSegmentator/AbdomenCT-1K (~2.3 k CTs) is cheap insurance.
- Leakage rule: pretrain only on train-patient scans plus public data. Using val/test scans in SSL is transductive and
  would contaminate the comparison set even without labels. Public datasets overlap with LIDC/TCIA sources; check that
  the Tübingen cohort is not among them (it is not a TCIA collection, as far as known — unverified).
- What synthetic pairs miss: real tumour growth/shrinkage, contrast-phase change and breathing. They teach anatomy
  correspondence, which is the part that removes look-alike gross misses (SAM's coarse-to-fine ablation, 91.1 → 84.1%).
  Run 6b/P3 notes already say the remaining mean comes from a few gross misses.
- Expected effect size (inference): the P3 vs P3-rand tie shows a segmentation-pretrained encoder gave ~0; a
  matching-pretrained encoder is better aligned, but a plausible gain is a few tenths of a mm on the mean, which is at or
  below the detectable difference on val (section 7). Budget for patient-level CV to judge it.

### Gaps
- No paper found that pretrains a Siamese correlation tracker on synthetic pairs and then fine-tunes it on ~2 k real
  pairs, with a with/without-pretraining ablation. DLT's self-supervision ablation number was not retrieved.
- SAM/SAMv2 GPU hours and batch size were not found in the fetched text.
- Registration-style pretraining (uniGradICON) was not researched in this session; no evidence gathered on whether
  registration features help point matching.

## 2. Unlabeled lesions and dense pixel-level contrastive losses

### Takeaway
Dense (every-voxel) contrastive supervision is what makes SAM work, and hard negatives are worth ~2 points of matching
accuracy. On real BL/FU pairs only the lesion correspondence is labelled, so dense losses belong on synthetic pairs, or on
real pairs only through registration-derived correspondence (noisy at the ~4 mm level).

### Cited Findings
- SAM's loss is pixel-wise: every sampled overlap pixel is a positive pair, with online hard negatives; removing hard
  negatives drops matching accuracy 91.1 → 89.0% — [SAM](https://arxiv.org/html/2012.02383v3)
- DenseCL (pixel-level contrastive on natural images) beats MoCo-v2 by 1.1 AP pretrained on COCO and 0.5 AP on ImageNet;
  the authors report "up to +2% AP and +3% mIoU" on dense tasks —
  [DenseCL, arXiv 2011.09157](https://arxiv.org/html/2011.09157v2); [DenseCL GitHub](https://github.com/WXinlong/DenseCL)
- Registration-propagated centres on an earlier 206-pair version of the same Tübingen cohort sat 3.79 / 3.16 / 4.49 mm
  (x/y/z) from manual FU centres (from the earlier report) —
  [Hering et al., MIDL 2021](https://proceedings.mlr.press/v143/hering21a/hering21a.pdf)

### Inferences
- Cheapest dense add-on: during synthetic pretraining, apply InfoNCE to the 4 mm and 8 mm projected features at N
  sampled voxels (e.g. 256 per pair), not just the centre. Negatives are the 500 most similar non-matching FU cells (SAM
  recipe). The existing projection + L2 norm is already the right embedding.
- Real pairs, hard negatives from other lesions: patients like e2c0be2456 (50 lesions, many similar metastases) are where
  the model fails. Other lesions of the same patient that fall inside the FU window are explicit hard negatives. Add a
  term pushing softmax mass off their FU masks (e.g. `−log(1 − Σ_{v∈other masks} p(v))`). Same patient, no extra labels.
  Not tested in any paper found.
- Multiply real supervision: each BL patch can be paired with FU windows centred on several jitters per step (already
  done via re-jitter). A second positive is the FU→BL direction (swap roles, target −Δ), which doubles pairs at no
  annotation cost. Inference only.
- Dense correspondence on real pairs via `cog_propagated` or a registration field is possible, but its ~4 mm error is
  comparable to the model's median 2.54 mm. Use it only as a soft, low-weight auxiliary, or not at all.

### Gaps
- PixPro and VADeR numbers were not fetched this session. No medical dense-contrastive paper (other than the SAM family)
  with a lesion-matching result was retrieved.
- No evidence on whether hard negatives from other annotated lesions help a supervised tracker.

## 3. Regularisation for the observed overfitting

### Takeaway
The model already has EMA and heavy augmentation. Val bottoms at epoch 18 of 50 because cosine decay keeps lr high past
the optimum. The cheapest proven lever is a shorter schedule fixed in advance (e.g. 20–25 epochs, last EMA weights). SAM
(sharpness-aware) is the only flat-minimum optimiser with consistent small medical gains, at ~2× cost. No medical evidence
for any of these was found on regression/matching tasks.

### Cited Findings
- Hassan et al. (IEEE Access 2025) compared SAM, ASAM, GSAM, WSAM and CR-SAM against Adam on breast ultrasound (2,405 and
  562 images), CheXpert and PathMNIST. Only plain SAM "consistently enhances generalization" across ResNet50, VGG16, ViT
  and Swin. On the 2,405-image BUS set, test accuracy went 80.2 → 84.0% (ResNet50), 84.0 → 87.5% (VGG16), 80.0 → 82.5%
  (ViT). ASAM collapsed on transformers (60.7–69.6%). SAM cost ≈ 1.8–2.1× Adam time; the variants cost 4.2–5.9× —
  [Hassan et al., PMC12121992](https://pmc.ncbi.nlm.nih.gov/articles/PMC12121992/)
- SWA (Izmailov et al. 2018) averages weights along a constant or cyclical lr tail to reach flatter optima — cited from
  memory, not re-fetched: [arXiv 1803.05407](https://arxiv.org/abs/1803.05407). Stochastic depth (Huang et al. 2016)
  randomly drops residual blocks during training — from memory: [arXiv 1603.09382](https://arxiv.org/abs/1603.09382)
- Data sampling variance dominates benchmark variance; init variance is "generally less than 50% of the variance of
  bootstrap" — [Bouthillier et al., MLSys 2021](https://proceedings.mlsys.org/paper_files/paper/2021/file/0184b0cd3cfb185989f858a1d9f5c1eb-Paper.pdf)
  (relevant: regularisation tweaks worth < 0.3 mm will be hard to see; section 7)

### Inferences
- Ranked by expected value per GPU hour (all inference, none tested on this model):
  1. **Shorter schedule, fixed a priori** (epochs 20–25, cosine to 0 at the end, keep last EMA). This also removes
     epoch selection on val, which is the main optimism bias. The cosine tail lands at the low-lr region where EMA/SWA
     help most.
  2. **SWA over the last ~5 epochs on top of EMA**: cheap (no extra steps), but EMA 0.999 at ~675 steps/epoch already
     averages ~1.5 epochs. SWA over the final epochs is a wider average; expect a small, maybe nil gain.
  3. **Weight-decay sweep** {1e-4, 1e-3, 1e-2, 5e-2} with AdamW. wd 1e-4 is low for AdamW (decoupled, multiplied by lr).
     InstanceNorm layers make conv weights scale-invariant, so wd mostly changes effective lr. Exclude norm and
     `log_scale` parameters from decay.
  4. **Stochastic depth** 0.1–0.2 on the 7 residual blocks of stages 1–2. This is cheap. Dropout on cosine features is
     a poor fit, because the dot product needs stable features.
  5. **Sharpness-aware minimisation** (ρ ≈ 0.05): ~2× step time (≈ 7 h for 50 epochs, ≈ 3.5 h for 25). Medical evidence
     is classification only, with +0.5 to +3.6 accuracy points.
  6. **Lower lr for the encoder than for projection/temperature** matters mainly for pretrained encoders (section 1),
     not for random init.
  7. **Smaller projection dim** (32 → 16): SAM was robust to embedding dimension. DLT's MED moved 7.9–8.7 mm with feature
     dimension. Low priority.
- The train/val gap (row 10: train L_dist ≈ 3.95 mm vs val ~6 mm) is partly label noise (FU centre fuzzy at ~3–4 mm,
  per the Hering figure). Some of the "overfitting" is the model learning annotation idiosyncrasies of train patients,
  which no regulariser fixes.

### Gaps
- No medical-imaging study found on SWA, stochastic depth or SAM optimiser for localisation/regression or Siamese
  matching. The SWA/stochastic-depth original papers were not re-fetched.

## 4. Temperature, soft targets and curriculum

### Takeaway
SAM uses a fixed InfoNCE temperature τ = 0.5. Team 2 uses a learnable scale initialised at 10 over 32,768 cells, a very
different regime. Watch the learned scale, because a softmax that grows ever peakier is a classic overfitting channel.
The project already found a wide Gaussian heatmap target blurs the map and hurts the median (row 6), so any σ curriculum
must end small.

### Cited Findings
- SAM: InfoNCE, τ = 0.5 (logit scale 2) over ~1,500 negatives — [SAM](https://arxiv.org/html/2012.02383v3)
- Project row 6: CE to a Gaussian with σ = clip(0.5·r_FU, 4, 12) mm gave val mean 6.92 vs 6.78 mm and median 5.80 vs
  4.74 mm. It learned faster early (epoch 11: 8.57 vs ~10.7 mm) and had more lesions within 10 mm —
  [Experiments.md row 6](../../docs/wiki/Experiments.md)

### Inferences
- Log `exp(log_scale[i])` per epoch. If it rises steadily while val worsens after epoch ~20, clamp it (e.g. max 30) or
  freeze it after warmup. The scale controls peak sharpness and, with soft-argmax, how much far-off mass pulls the mean.
- Curriculum σ: the row 6 result (fast early learning, blurry end point) argues for annealing λ_hm or σ to zero over the
  first ~30% of training rather than a fixed σ. The L_mask term already plays the "early coarse target" role, so the
  marginal value is probably small.
- Label-smoothing analogue: mixing the one-cell target with ε·uniform is equivalent to bounding the temperature. It is
  cheaper to bound the scale directly.

### Gaps
- No paper found on learnable temperature dynamics in 3D correlation/soft-argmax trackers. CLIP-style learnable
  temperature behaviour and label-smoothing papers were not fetched.

## 5. Multi-task auxiliary objectives

### Takeaway
No retrieved paper isolates an auxiliary head (FU mask, segmentation, reconstruction) as improving lesion matching. The
nearest evidence is indirect: DLT's anatomy signal and SAM's anatomy embedding help, and generic reconstruction/MAE
pretraining helps segmentation only modestly. The existing FU-mask loss already is the most task-aligned auxiliary.

### Cited Findings
- DLT's anatomy signal encoder (SAM-derived anatomical signal) is a component whose removal gave MED 9.3 mm in its
  component analysis — [DLT](https://arxiv.org/html/2012.04872v2)
- Generic SSL (MAE/reconstruction) gains on CT segmentation: +2.47 Dice (OpenMind best) and +3.32 Dice (VoCo vs scratch
  Swin-UNETR) — [OpenMind](https://arxiv.org/abs/2412.17041); [VoCo](https://arxiv.org/html/2402.17300v2)
- LesionLocator's synthetic longitudinal pretraining cut point-prompt centre error from 5.09 to 3.89 mm on 60 melanoma
  patients (from earlier report) — [LesionLocator, arXiv 2502.20985](https://arxiv.org/html/2502.20985)
- Project row 8: adding L_mask (FU-mask probability mass) tied on the end point (−0.06 mm) but sped up early learning —
  [Experiments.md](../../docs/wiki/Experiments.md)

### Inferences
- A segmentation decoder head on the shared encoder (predict BL mask from BL, FU mask from FU) adds dense labels: every
  lesion voxel, ~2 k lesions × 2 scans. It may regularise stage-1 features towards lesion boundaries. Cost: a small
  decoder, ~1.2× step time. A plausible but unproven gain; test only after cheaper items.
- Reconstruction (Models Genesis/MAE-style) is the least aligned with matching; skip under the 1–2 day budget.
- The matching-specific auxiliary with the best prior is the synthetic-pair contrastive loss itself, run jointly with
  supervised training (mixed batches, e.g. 50% real, 50% synthetic) instead of as a separate pretraining stage. That is
  how DLT used self-supervision.

### Gaps
- No ablation found of a segmentation or reconstruction auxiliary in a lesion-tracking model.

## 6. Ensemble distillation and multi-seed ensembling

### Takeaway
Ensembling is the most reliable gain already measured in-project (3 models: 4.27 mm without TTA vs ~4.5–4.6 single
plain). Distillation of segmentation ensembles into one CNN retains ensemble performance in medical imaging. For a
course project where 3× inference cost is trivial, keep the ensemble; distil only if one model is required.

### Cited Findings
- Noothout et al. (J Med Imaging 2022): "Knowledge distillation can compress segmentation ensembles of uniform or diverse
  composition into a single CNN while maintaining the performance of the ensemble" —
  [PubMed 35692896](https://pubmed.ncbi.nlm.nih.gov/35692896/); [PDF](https://repository.ubn.ru.nl/bitstream/handle/2066/252007/252007.pdf)
- Standard ensemble distillation keeps accuracy but loses the ensemble's diversity information (knowledge uncertainty).
  Ensemble Distribution Distillation addresses this; 5-model ensembles were too few to distil knowledge uncertainty well,
  20 did better — [Malinin et al., ICLR 2020, OpenReview](https://openreview.net/forum?id=BygSP6Vtvr)
- Project: averaging three two-scale runs gives 4.27 mm without TTA; single-run plain readout 4.57 mm —
  [Team2-Model.md](../../docs/wiki/Team2-Model.md)

### Inferences
- Cost/benefit: one extra seed ≈ 2–3.5 h V100; inference is milliseconds per lesion. Ensembling 3–5 seeds, plus flip TTA
  on each, is the cheapest sub-mm gain available. It also gives a free uncertainty signal: the spread of member
  predictions, alongside peak probability.
- If distilling: target is the ensemble's averaged probability volume (KL between softmax maps) plus the Δ loss. Train the
  student on real train pairs plus unlabeled synthetic/extra-jitter windows, where teacher labels are free. Teacher labels
  on train data are overfit (train error ≪ val error), so the extra unlabeled windows matter. Inference only.
- Do not compare a 3-model ensemble against a single model when judging a training change. Compare like with like
  (single vs single, averaged over seeds; or ensemble vs ensemble).

### Gaps
- No distillation evidence found for regression/localisation or Siamese tracking specifically. The size of the retained
  gain in Noothout et al. (numbers) was not extracted.

## 7. Evaluation practice: selection vs comparison, CV, bootstrap, seeds

### Takeaway
Val (207 lesions, 28 patients) is too small to resolve changes below ~0.6–0.7 mm with patient-clustered paired bootstrap.
Seed noise (~0.1 mm, estimated from one pair of runs) is minor next to patient sampling. The fix is to stop selecting
epochs on val (fix the schedule), compare configurations with 5-fold patient-level CV on the 255 train+val patients
(~2,370 evaluated lesions), and run ≥3 seeds only for the final candidates. Test stays untouched until one final model.

### Cited Findings
- Bouthillier et al. (MLSys 2021): randomise "weight initialization, data sampling, random data augmentation and the whole
  hyperparameter optimization"; data sampling (bootstrap) "stands out as the most important source of variance"; init
  variance is "generally less than 50% of the variance of bootstrap". Fixing seeds to arbitrary values induces correlation
  between trainings, which increases estimator variance. Compare with probability of improvement P(A > B), with
  percentile-bootstrap CI: significant if the lower bound is > 0.5, practically meaningful if the upper bound is > γ = 0.75.
  Randomising more sources of variation approximated the ideal estimator at "a 51 times reduction in compute cost" —
  [MLSys PDF](https://proceedings.mlsys.org/paper_files/paper/2021/file/0184b0cd3cfb185989f858a1d9f5c1eb-Paper.pdf);
  [arXiv 2103.03098](https://arxiv.org/abs/2103.03098)
- Varoquaux & Cheplygina (npj Digit Med 2022) recommend "enough data in the test set to bring statistical power, at least
  several hundreds samples, ideally thousands or more", with confidence intervals on the reported metric —
  [npj Digital Medicine](https://www.nature.com/articles/s41746-022-00592-y)
- Project paired comparisons on val: P3-rand vs 6b −1.14 mm, 95% patient-clustered CI [−1.85, −0.56] (half-width ≈
  0.65 mm). P3 vs P3-rand +0.12 mm, CI [−0.38, +0.55] (half-width ≈ 0.47 mm). Val is concentrated: one patient has 24%
  of lesions and two patients have 37% — [Team2-Model.md](../../docs/wiki/Team2-Model.md)

### Inferences
- **Minimum detectable difference on val (own calculation).** From the P3 vs P3-rand CI, SE ≈ 0.47 / 1.96 ≈ 0.24 mm
  for a paired, patient-clustered difference between similar models. At α = 0.05 two-sided and 80% power,
  MDD ≈ 2.8 × SE ≈ **0.67 mm**. Changes of 0.1–0.3 mm (most regularisation tweaks) cannot be confirmed on val alone.
- **Seed noise estimate is weak.** "~0.1 mm" comes from one pair of runs (1 degree of freedom). The SD estimate
  σ̂ ≈ 0.1/√2 ≈ 0.07 mm has a 95% CI of roughly 0.45×–32× σ̂ (χ² with 1 df). Before relying on it, run 3–5 seeds of the
  baseline once; that number then sets everything else.
- **Seeds per configuration.** With σ_seed ≈ 0.1 mm, the seed-only SE of a k-vs-k mean difference is
  0.1·√(2/k) ≈ 0.08 mm at k = 3 and 0.06 mm at k = 5. That is small next to the 0.24 mm patient-sampling SE, which more
  seeds do not reduce. So: 1 seed for screening, 3 seeds for finalists, and spend the rest of the budget on CV folds
  (more evaluated patients), in line with Bouthillier's finding that data sampling dominates.
- **Patient-level K-fold CV on 255 train+val patients.** 5 folds of ~51 patients each give out-of-fold predictions for
  all ~2,366 lesions. Pooled SE scales roughly with √(207/2366) ≈ 0.30 of val's, so MDD ≈ 0.2–0.25 mm. Clustering into
  more, smaller patients is likely to help further. Cost at a fixed 25-epoch schedule (~1.8 h/run): ≈ 9 h per
  configuration, ≈ 2–3 configurations per V100 day. Note that offline aug copies and any SSL pretraining must be
  regenerated or restricted per fold, so that no held-out patient's scans enter training or pretraining.
- **Nested selection.** Either (a) fix epochs and hyperparameters a priori and take the last EMA checkpoint (no inner
  loop needed; recommended), or (b) inside each training fold hold out ~10% of patients for early stopping. Never pick
  the epoch on the fold used for comparison. Current val numbers are optimistic by the epoch-selection bias. This is
  likely small for a smooth EMA curve, but unknown.
- **Bootstrap protocol.** Resample patients (not lesions) with replacement, 10 k resamples, recompute the paired mean
  difference over all lesions of the drawn patients. Report both lesion-weighted and patient-weighted means. With
  e2c0be2456 at 24% of val, a leave-one-patient-out sensitivity check is cheap and informative. Also report
  P(A better than B per lesion) as a Bouthillier-style probability of improvement.
- **Multiple comparisons.** With ~17 runs compared against the same val, the best-so-far value is itself selection-biased
  (winner's curse). Before opening test, re-evaluate the final pick and 1–2 runners-up on CV, or with fresh seeds.
- **Test set (286 lesions, 29 patients)** is "several hundred" by Varoquaux's bar only at the lesion level. At the
  patient level it is 29 clusters, so report its CI with the patient bootstrap and open it exactly once.

### Gaps
- The per-lesion paired-difference SD between two seeds of the same config is not in the wiki. It would give a direct
  seed-vs-sampling variance split; compute it from the existing per-lesion val error files (commit b05c710).
- No literature found that gives a specific recommended number of seeds for medical localisation tasks; the k
  recommendation above is derived, not cited.
