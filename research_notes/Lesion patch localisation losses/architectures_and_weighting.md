# Architectures for longitudinal 3D CT lesion tracking and multi-loss weighting

Scope: predict the follow-up (FU) lesion centre from a BL patch + BL mask + FU patch (64^3 @ 2 mm, FU centre jittered within ±24 mm). Planned model: Siamese 3D encoder, correlation, heatmap, soft-argmax, L = L_dist + λ_hm·L_hm + λ_mask·L_mask. Notes compiled 2026-10-29.

Source-verification note: the DLT (CVPR 2021) and TLT (MICCAI 2022) full texts were read (pdftotext). For LesionLocator (CVPR 2025) and the 2026 autoPET IV winner paper, the arXiv HTML was read. Loss-weighting papers (Kendall, GradNorm, DWA, unitary-scalarization critiques) and DSNT/soft-argmax details were **not** fetched this session. They are cited to their arXiv pages from prior knowledge and marked [not re-verified].

## Q1. Deep Lesion Tracker (Cai et al., CVPR 2021)

### Takeaway
DLT is a 3D Siamese tracker. A 3D DenseNet+FPN encodes the image, and an anatomy-signal encoder (ASE) encodes Gaussian maps placed at the lesion and at its affine-projected location. The two are fused multiplicatively and cross-correlated over the whole search volume. Training uses a CenterNet-style focal loss on a Gaussian heatmap. Mean error was about 7 mm and CPM@10mm 78.9% on 3,891 DeepLesion pairs. Removing the anatomy signal noticeably hurt accuracy.

### Cited Findings
- Problem setup: given template image I_t, lesion centre µ_t and radius r_t, plus search image I_s, predict µ_s. This matches our task, except we also have a mask. — [DLT arXiv 2012.04872](https://arxiv.org/abs/2012.04872)
- Image encoder "3D DenseFPN": 2D DenseNet kernels inflated to 3D (duplicated along the third axis, weights divided by the number of duplications), plus an FPN that outputs 3 scales. The authors found it more effective than 3D UNet for universal lesion appearance. — [DLT](https://arxiv.org/abs/2012.04872)
- Anatomy Signal Encoder (ASE) inputs are Gaussian heatmaps G(µ, n·r), with n = 4 working best. The template map is centred on µ_t. The search map is centred on the affine-projected T_Aff(µ_t) with size T_Aff(r_t). Affine registration (L1, SimpleElastix) is computed beforehand. Rationale: "directly implementing lesion tracking with Siamese networks can produce matches with visually similar but spurious regions." — [DLT](https://arxiv.org/abs/2012.04872)
- Fusion: F = ψ(I_t) ⊙ φ(G_t), i.e. element-wise multiplication of image and anatomy features. This "performs better than channel-wise concatenation." — [DLT](https://arxiv.org/abs/2012.04872)
- Correlation: a 3×3×3 template kernel K and a larger global kernel K_g (7×11×11) are cropped from the fused template features at µ_t. Correspondence M = K⋆S + K_g⋆S, with S = ψ(I_s) ⊙ φ(G_s). "Fast cross-correlation" decomposes K_g into three learned flattened kernels (1×11×11, 7×1×11, 7×11×1). This cuts over 65% of FLOPs and works better than taking the middle slices. — [DLT](https://arxiv.org/abs/2012.04872)
- Output: Ŷ = sigmoid(Wᵀ(M1 + U2 + U3) + b), where the coarser scales are upsampled. The prediction is the argmax of Ŷ (hard argmax, not soft-argmax). — [DLT](https://arxiv.org/abs/2012.04872)
- Loss: a single-term focal loss (CenterNet style, α=2, β=4) against a Gaussian target Y = G(µ_s, r_s) downsampled to the output size. There is no separate distance loss. — [DLT](https://arxiv.org/abs/2012.04872)
- "Center augmentation": µ_t is randomly shifted within ‖Δµ_t‖ ≤ 0.25·r_t, because clinician centre labels are inexact. — [DLT](https://arxiv.org/abs/2012.04872)
- Self-supervised learning (SSL) on unpaired images: the search image is an augmented copy of the template. Augmentations, each with p = 0.5:
  - elastic deformation, scale 0–0.25
  - in-plane rotation ±10°
  - scaling 0.75–1.25
  - random crops
  - Gaussian noise, variance 0–0.05
  - Gaussian blur, σ 0.5–1.5

  Mixed training (DLT-Mix) randomly switches between the SSL loss and the supervised loss (SSL when λ ≤ τ = 0.25, with λ ~ U[0,1]). — [DLT](https://arxiv.org/abs/2012.04872)
- Data: the DLS benchmark has 3,008 / 403 / 480 train/val/test lesion pairs from DeepLesion. The training template size is (32, 384, 384), at 0.8 mm in-plane after affine alignment. — [DLT](https://arxiv.org/abs/2012.04872)
- Metrics: CPM@Radius counts a hit if the error is below the lesion radius. CPM@10mm uses threshold min(r, 10 mm), chosen because 55% of test lesions have radius above 10 mm. MED is the mean Euclidean distance, also reported as MED_X/Y/Z. — [DLT](https://arxiv.org/abs/2012.04872)
- DLS test results:

  | Method | CPM@10mm | CPM@Radius | MED |
  |---|---|---|---|
  | DLT | 78.85 | 86.88 | 7.0±8.9 mm |
  | DLT-Mix | 78.65 | 88.75 | 7.1 mm |
  | DEEDS deformable registration | 71.88 | 85.52 | 7.4 mm |
  | 3D SiamRPN++ | 68.85 | 80.31 | 8.3 mm |
  | Affine | 48.33 | 65.21 | 11.2 mm |
  | VoxelMorph | 49.90 | 65.59 | 10.9 mm |

  DLT-SSL (no paired labels) reached CPM@10mm 71.04. The external clinical set (100 studies) gave 88% radiologist acceptance. DLT was about 14× faster than DEEDS. — [DLT](https://arxiv.org/abs/2012.04872); test table as reproduced in [TLT paper](https://zrongguo.github.io/files/2022_MICCAI_Transformer%20Lesion%20Tracker.pdf)
- Ablation (Table 6, parsed from PDF text, so row alignment is approximate): final config test MED 6.98 mm. Variants without the global kernel, without ASE, or without learned K_g flattening had test MED of about 7.9–9.4 mm. The "w/o ASE" row appears to be about 9.3 mm. Multiply-fusion beat concat (valid MED 6.69 vs 6.85). — [DLT](https://arxiv.org/abs/2012.04872)

### Inferences
- Our FU patch is already roughly centred (±24 mm), which plays the role of DLT's affine prior. A cheap analogue of ASE is a Gaussian "prior" channel centred on the FU patch centre. It could also be an implicit centre bias, e.g. a learned or fixed log-prior added to the correlation logits.
- DLT got to about 7 mm with a heatmap-only loss and hard argmax. A distance term is therefore not essential, but heatmap supervision is. A plain regression CNN failing is consistent with DLT/TLT both using dense heatmaps.
- Our data (about 2,500 pairs) is similar in size to DLS training (3,008 pairs). DLT's capacity and training regime are a realistic reference. SSL with synthetic pairs, i.e. BL patch → augmented copy as pseudo-FU, is a proven way to add data.

### Gaps
- DLT's optimizer, learning rate, batch size and epochs were not found in the extracted text, which may be in the supplement.

## Q2. Transformer Lesion Tracker, LesionLocator, and 2024–2026 work / autoPET IV

### Takeaway
TLT (2022) replaced correlation with cross-attention and added a heatmap-weighted soft-argmax "global regression." It reached 6.0 mm MED and CPM@10mm 87.4% on DLS. Its loss is L1 on the soft-argmax centre plus a focal heatmap loss, which is essentially the planned design. LesionLocator (CVPR 2025) and the autoPET IV Task 2 winners (2025/2026) recast tracking as registration-propagated prompt plus promptable segmentation. They report MED of about 3–5 mm on whole-body melanoma CT and train with Dice+CE.

### Cited Findings
**TLT (Tang et al., MICCAI 2022, LNCS 13436 pp. 196–206; code github.com/TangWen920812/TLT)**
- Encoder: shared-weight 3D ResNet18 with the last stage removed, first-conv stride 1, and channels halved "considering the parameter redundancy and overfitting in 3D networks." Output stride 8, C = 192, 5.98M parameters total. — [TLT PDF](https://zrongguo.github.io/files/2022_MICCAI_Transformer%20Lesion%20Tracker.pdf)
- Sparse Selection Strategy: a Gaussian G(c_t, r_t) with denominator (2r)² is built, resized to the feature grid, and thresholded at Tr. Only template features where G > Tr are kept as tokens. Tr = 0.7 was best (CPM@10mm 87.37), against 83.57 at 0.9 and 85.01 at 0.4. This is template = features pooled/selected inside the lesion region rather than a fixed centre crop. — [TLT](https://zrongguo.github.io/files/2022_MICCAI_Transformer%20Lesion%20Tracker.pdf)
- Cross-Attention Transformer: N = 3 cross-attention blocks on the template and search paths. A Registration-based Anatomical Attention Module (RAAM) adds an affine-projected Gaussian as an additive attention bias: softmax(QKᵀ/√d + M_A). Affine was chosen over non-rigid registration because non-rigid "is slow and provides restriction to the attention." — [TLT](https://zrongguo.github.io/files/2022_MICCAI_Transformer%20Lesion%20Tracker.pdf)
- Head and losses:
  - Classification map Ŷ plus per-voxel regressed coordinates Ĉ (3×DHW).
  - Final centre ĉ = Σ softmax(Ŷ)·Ĉ ("global regression"), i.e. a soft-argmax over per-voxel coordinate predictions.
  - L_r = ‖ĉ − c_s‖₁ (L1).
  - Auxiliary focal loss with α = β = 2 on a min-max normalised Gaussian label G(c_s, r_s).
  - No loss weights are stated in the text, which suggests equal weighting.

  — [TLT](https://zrongguo.github.io/files/2022_MICCAI_Transformer%20Lesion%20Tracker.pdf)
- Training: Adam, lr 1e-4, 300 epochs, batch size 4, 1 mm isotropic resampling. — [TLT](https://zrongguo.github.io/files/2022_MICCAI_Transformer%20Lesion%20Tracker.pdf)
- DLS test (960 directed pairs, since each pair is tracked both ways):

  | Method | CPM@10mm | CPM@Radius | MED | MED_Z |
  |---|---|---|---|---|
  | TLT | 87.37 | 95.32 | 6.0±7.7 mm | 1.7 mm |
  | TransT baseline | 79.59 | 88.99 | 7.6 mm | – |
  | DLT | 78.85 | – | 7.0 mm | – |

  — [TLT](https://zrongguo.github.io/files/2022_MICCAI_Transformer%20Lesion%20Tracker.pdf)
- TLT doubles training data by tracking pairs in both directions (BL→FU and FU→BL). — [TLT](https://zrongguo.github.io/files/2022_MICCAI_Transformer%20Lesion%20Tracker.pdf)

**LesionLocator (Rokuss et al., CVPR 2025, arXiv 2502.20985; code github.com/MIC-DKFZ/LesionLocator)**
- Pipeline:
  - A GradICON-based registration network (pretrained, input resampled to 175³) warps the BL prompt (point, box or previous mask) to FU.
  - A promptable 3D ResEnc-L U-Net then segments a high-resolution crop (0.8×0.8×1 mm) centred on the propagated prompt.
  - Prompts are fed as extra input channels.
  - Tracking is autoregressive, using the previous predicted mask as the next prompt.

  — [LesionLocator](https://arxiv.org/html/2502.20985v1)
- Loss: L = L_seg (CE + soft Dice) + L_prop (NCC + λ·GradICON regulariser), trained end-to-end. Simple unweighted sum of the two modules. — [LesionLocator](https://arxiv.org/html/2502.20985v1)
- Training:
  - Pretraining: nnU-Net, 4,000 epochs, patch 192³, batch 24, SGD lr 1e-2 poly.
  - Fine-tuning: 2,000 epochs, batch 3, lr 1e-3, patch 224×224×160.
  - Augmentation includes elastic deformation, rotation ±5°, and scaling 0.95–1.05.
  - Point prompt encoded as a 5-voxel-radius ball. This beat a single pixel by a large margin (Dice 79.71 vs 66.56).

  — [LesionLocator](https://arxiv.org/html/2502.20985v1)
- Synthetic follow-ups: anatomy-informed deformation (gradient of a Gaussian-smoothed lesion mask × amplitude in {−22, −18, 15, 25}) to grow or shrink lesions. 2,728 scans got a synthetic timepoint. Real longitudinal data: 60 melanoma patients, 159 scans. — [LesionLocator](https://arxiv.org/html/2502.20985v1)
- Tracking results (melanoma whole-body, 5-fold CV):

  | Method | MED | CPM@25 | Dice |
  |---|---|---|---|
  | Yan et al. point tracker | 6.92 mm | – | – |
  | Hering et al. (nnU-Net + Elastix) | 4.45 mm | 82.50 | – |
  | Registration + ULS | 5.81 mm | – | – |
  | LesionLocator (point) | 5.09 mm | 83.62 | – |
  | + synthetic data | 3.89 mm | – | – |
  | + previous-mask prompt | 3.12 mm | 85.96 | 68.31 |

  — [LesionLocator](https://arxiv.org/html/2502.20985v1)

**Dataset and autoPET IV Task 2 (longitudinal CT)**
- The Longitudinal-CT dataset (Sci Data, published 18 May 2026) contains:
  - 300 metastatic melanoma patients from University Hospital Tübingen, with 600 CT studies (BL + FU each).
  - 7,182 manually segmented lesions (4,079 BL, 3,103 FU), with longitudinal correspondence labels.
  - Portal-venous phase, Siemens scanners, 3 mm slices.
  - An nnU-Net v2 baseline.

  — [Univ. Lübeck research portal](https://research.uni-luebeck.de/en/publications/a-longitudinal-whole-body-ct-dataset-with-manually-annotated-tumo/); data at [FDAT qwsry-7t837](https://fdat.uni-tuebingen.de/records/qwsry-7t837) (one FDAT record says 333 patients, which conflicts with 300).
- Challenge timeline (2025): Task 2 baseline 29 April, data 8 May, final test phase opened 29 August. — [autoPET IV](https://autopet-iv.grand-challenge.org/)
- Kirchhoff et al. (arXiv 2509.00613, Aug 2025) submission extended LongiSeg with point and mask prompts. Final score 63.71 vs 59.57 for pretrained LesionLocator weights. The largest gain came from pretraining on synthetic longitudinal data, and the test submission used a 5-fold ensemble. — [arXiv 2509.00613](https://arxiv.org/html/2509.00613v1) (search snippet; metric definition not verified)
- "Exploiting Longitudinal Context in Clinician-Verified Interactive Lesion Tracking" (arXiv 2605.23118, May 2026) claims 1st place in MICCAI autoPET IV. Design:
  - Shared ResEnc U-Net encoder.
  - Early fusion: image concatenated with a Gaussian prompt heatmap (σ = 1).
  - Latent temporal fusion via a "Difference Weighting Block" at every skip: x′_t = x_t·InstNorm(x_t − x_0) + x_t.
  - uniGradICON registration to propose the FU prompt.
  - Dice+CE loss, SGD, 1,000 epochs, synthetic pretraining on 2,606 pairs.

  Training-prompt simulation is 50/50 between points sampled from the GT mask (weighted 1/d²) and registered points, which can fall outside the lesion. — [arXiv 2605.23118](https://arxiv.org/html/2605.23118)
- 2605.23118 results (autoPET IV test, automatic):
  - Ours: DSC 60.7, lesion detection rate (LDR) 77.7
  - Hering et al.: 56.6, LDR 76.7
  - LesionLocator: 44.1, LDR 61.1
  - nnInteractive: 43.3

  Ablation on autoPET IV val: concatenation from scratch 45.8 DSC, concatenation with pretraining 56.5, full model 58.5. Synthetic pretraining is worth up to +4.5 Dice. Data used: 285 patients / 670 images. — [arXiv 2605.23118](https://arxiv.org/html/2605.23118)

### Inferences
- Our planned loss L_dist (on soft-argmax) + λ_hm·L_hm is the TLT recipe: L1 on the soft-argmax centre plus a focal heatmap. TLT appears to use λ = 1, which is a reasonable default.
- On this exact dataset, the strong methods are segmentation-based with Dice+CE. That argues for keeping L_mask (predict the FU mask) as an auxiliary term: dense supervision helps small data. The centre-of-mass of the predicted mask could also act as an alternative or ensemble estimate of the centre.
- The "concat from scratch" vs "concat + pretraining" gap (45.8 vs 56.5 DSC) in 2605.23118 is direct evidence that on this dataset size, naive BL/FU fusion from scratch underperforms. Pretraining (synthetic pairs) mattered more than the fusion design.
- A prompt encoded as a ball/Gaussian rather than a single voxel was a large win in LesionLocator. We already have the full BL mask, which is better than a point.

### Gaps
- No official autoPET IV Task 2 leaderboard numbers or metric definitions were retrieved. The challenge evaluated segmentation (Dice/detection), not centre distance, so no published "mm error" exists for this dataset in a patch-centre setting.
- "Unstable Prompts, Unreliable Segmentations" ([arXiv 2507.19230](https://arxiv.org/pdf/2507.19230)) is relevant to autoPET IV but was not read.
- Other 2024–2026 point trackers (e.g. Yan et al., cited in LesionLocator at 6.92 mm MED) were not traced to their primary papers.

## Q3. Siamese correlation designs, template choice, mask channel, pretrained 3D encoders

### Takeaway
Correlation-based trackers (SiamFC → SiamRPN++ → DLT) use a small template kernel cropped around the target, correlated over search features. Newer ones (TLT) select template tokens inside the lesion region and use cross-attention. Both use the lesion extent via Gaussian/mask gating, and both benefit from a positional prior. Supervised-pretrained CT encoders (STU-Net on TotalSegmentator, nnU-Net ResEnc) are the most defensible starting points for small data. SAM-Med3D is weaker as a generic encoder.

### Cited Findings
- SiamFC/CFNet use the template feature patch as a kernel for a full cross-correlation over the search feature map. This encodes the relative position of the target. — [DLT related-work section](https://arxiv.org/abs/2012.04872); original [SiamFC arXiv 1606.09549](https://arxiv.org/abs/1606.09549) [not re-verified]
- DLT: template = 3×3×3 plus a 7×11×11 crop around µ_t, at multiple FPN scales. A 3D SiamRPN++ with a (3,5,5) kernel was clearly worse (CPM@10mm 68.85 vs 78.85). Large-context kernels help. — [DLT](https://arxiv.org/abs/2012.04872)
- Translation invariance issue: "pre-registering will have minimal effect on the translation-invariant cross-correlation." DLT therefore injects the anatomy prior into the features (multiplicative fusion) rather than only aligning images. — [DLT](https://arxiv.org/abs/2012.04872)
- TLT: template tokens = features inside a Gaussian of the lesion (threshold 0.7), with the anatomical prior as an attention bias. Ablation thresholds ranged 83.6–87.4 CPM@10mm. Template definition matters by a few points. — [TLT](https://zrongguo.github.io/files/2022_MICCAI_Transformer%20Lesion%20Tracker.pdf)
- Mask/prompt as an extra input channel is standard in LesionLocator (prompt channels at the highest resolution) and in the autoPET IV winner (Gaussian heatmap concatenated with image). — [LesionLocator](https://arxiv.org/html/2502.20985v1); [arXiv 2605.23118](https://arxiv.org/html/2605.23118)
- STU-Net was supervised-pretrained 4,000 epochs on TotalSegmentator (1,204 CTs, 104 structures). Fine-tuning gains were largest on small and large datasets and modest on medium ones. — [OpenMEDLab arXiv 2402.18028](https://arxiv.org/pdf/2402.18028)
- Models Genesis is a self-supervised 3D U-Net of about 16M parameters, pretrained on a small CT set (623 volumes per a later benchmark). It was comparable to MedicalNet (p = 0.05) on liver segmentation, where MedicalNet's supervised LiTS pretraining helped. — [Models Genesis arXiv 2004.07882](https://arxiv.org/pdf/2004.07882)
- A 2025 benchmark found supervised-pretrained 3D models transfer better than self-supervised ones (full-data fine-tuning). — [arXiv 2501.11253](https://arxiv.org/pdf/2501.11253)
- SAM-Med3D (point-promptable) scored only 48.76 average Dice in LesionLocator's lesion benchmark, against 79.71 for LesionLocator (point) and 74.25 for ULS. — [LesionLocator](https://arxiv.org/html/2502.20985v1); [SAM-Med3D arXiv 2310.15161](https://arxiv.org/pdf/2310.15161)
- LesionLocator, LongiSeg and nnInteractive weights are public (MIC-DKFZ GitHub). The LesionLocator encoder is a ResEnc-L trained on lesion data. — [LesionLocator](https://arxiv.org/html/2502.20985v1)

### Inferences
Concrete design for the 64³ @ 2 mm patches:
- **Input:** a 2-channel BL input (CT + BL mask, or the mask as a Gaussian/ball) and a 1-channel FU input. A shared encoder is possible if FU gets a zero mask channel.
- **Encoder and correlation:** a stride-4 encoder gives a 16³ map with 8 mm cells. Correlate there, then upsample/refine to stride 2 (32³) for a finer heatmap. ±24 mm is ±12 voxels at 2 mm, or ±3 cells at stride 4.
- **Template:** use two templates. One is mask-pooled (average of BL features inside the mask, which is size-invariant and gives a 1×1×1 kernel, i.e. cosine similarity). The other is a centre crop (e.g. 3³–5³ cells) for context. Sum the two correlation maps, DLT-style K + K_g. Pure mask-pooling loses context, and pure centre crop ignores lesion size.
- **Prior:** add a fixed or learned centre prior (log-Gaussian over the ±24 mm range) to the logits. This is the ASE/RAAM analogue.
- **Pretrained weights:** STU-Net-S/B or a LesionLocator/nnU-Net ResEnc encoder are the best-supported options. Their native spacing (1–1.5 mm) differs from our 2 mm, so expect some domain shift. Fine-tune with a lower LR on the encoder.

### Gaps
- No head-to-head comparison of mask-pooled vs centre-crop templates in 3D lesion tracking was found.
- No source compared these pretrained encoders at about 2.5k pairs.

## Q4. Augmentation and data size

### Takeaway
Published trackers use elastic deformation, small rotations (±5–10°), scaling (0.75–1.25 or 0.95–1.05), noise and blur, and centre jitter of the template. Generating synthetic pairs (SSL or synthetic follow-ups) was the single most effective data multiplier on DLS and on this exact dataset.

### Cited Findings
- DLT augmentation: elastic 0–0.25, xy-rotation ±10°, scale 0.75–1.25, crops, Gaussian noise var ≤ 0.05, blur σ 0.5–1.5, each p = 0.5. Template centre jitter is within 0.25·r_t. — [DLT](https://arxiv.org/abs/2012.04872)
- SSL-only DLT (no paired labels) reached CPM@10mm 71.04, close to DEEDS (71.88). Mixed training gave the best CPM@Radius (88.75) and more robustness (75.03 vs 70.36 CPM@10mm under perturbation, Table 2). — [DLT](https://arxiv.org/abs/2012.04872)
- LesionLocator augmentation: elastic (p = 1), rotation ±5°, scaling 0.95–1.05. Synthetic follow-ups improved MED from 5.09 to 3.89 mm. — [LesionLocator](https://arxiv.org/html/2502.20985v1)
- Synthetic longitudinal pretraining: +4.5 Dice ([2605.23118](https://arxiv.org/html/2605.23118)), and the largest gain in Kirchhoff et al. ([2509.00613](https://arxiv.org/html/2509.00613v1)).
- TLT trains on both directions of each pair. — [TLT](https://zrongguo.github.io/files/2022_MICCAI_Transformer%20Lesion%20Tracker.pdf)

### Inferences
Recipe for this project:
- **Flips:** apply the same flip to BL, BL mask and FU, and negate the target-offset component on the flipped axis. Left-right flips are anatomically plausible. Avoid z (superior-inferior) flips, or use them only for SSL pairs. A flip applied to only one timepoint breaks correspondence.
- **Swap direction:** swap BL and FU (needs the FU mask, which the dataset provides) and negate the offset. This roughly doubles the pairs, as in TLT.
- **Re-jitter:** re-sample the FU crop offset each epoch within ±24 mm, or wider. Each epoch then gives new targets, which is the strongest cheap augmentation. Check that the training jitter distribution matches the test one.
- **Independent augmentation:** apply intensity augmentation independently per timepoint, since contrast and scanner differ: brightness/contrast, gamma, noise, blur. Small independent rotations (≤10°) and scaling (0.9–1.1) on FU only also work, as long as the target is transformed with the same matrix.
- **SSL pairs:** BL patch → augmented, shifted copy as pseudo-FU. Also use lesion-growth/shrink deformation, as in LesionLocator. Any lesion, including ones with no FU match, becomes a training sample.
- **Baseline to beat:** "predict patch centre." For a uniform offset in a ±24 mm cube, mean error ≈ 0.96·24 ≈ 23 mm, which is an analytic estimate. The real jitter distribution may differ, so compute it on the validation set. DLT/TLT-level accuracy is about 6–7 mm MED, and registration+segmentation on this dataset is about 3–5 mm.
- **Data size:** about 2,500 pairs from about 224 patients sits below the regime where training 3D trackers from scratch is comfortable (DLS 3,008 pairs plus SSL). Use patient-level splits, synthetic pairs and pretrained encoders.

### Gaps
- No ablation isolating flip augmentation for lesion tracking was found.

## Q5. Multi-loss weighting

### Takeaway
The closest published systems (TLT, LesionLocator, the autoPET IV winner) all use plain unweighted sums of 2 terms. Recent critiques find that tuned fixed weights (unitary scalarization) match fancy multi-task weighting. Recommendation:
- Normalise each term to O(1), e.g. L_dist in voxels or divided by a scale such as 10 mm, not raw mm.
- Grid-search λ coarsely on a log scale.
- Optionally try Kendall uncertainty weighting as a near-free alternative.
- Use a short heatmap-first warm-up.

### Cited Findings
- TLT: L1 (soft-argmax centre) + focal heatmap, no weights reported. DLT: focal only. LesionLocator: L_seg + L_prop unweighted, with λ only inside the registration regulariser. — [TLT](https://zrongguo.github.io/files/2022_MICCAI_Transformer%20Lesion%20Tracker.pdf); [DLT](https://arxiv.org/abs/2012.04872); [LesionLocator](https://arxiv.org/html/2502.20985v1)
- **Homoscedastic uncertainty weighting.** For a regression term the weighted loss is L/(2σ²) + log σ, and for a classification term L/σ² + log σ. In practice one learns s = log σ² for stability. The authors report it beating grid-searched fixed weights on Cityscapes semantic + instance + depth. — [Kendall, Gal, Cipolla, CVPR 2018, arXiv 1705.07115](https://arxiv.org/abs/1705.07115) [not re-verified]
- **GradNorm.** Learns task weights so that gradient norms at the last shared layer match a target. The target is scaled by each task's relative inverse training rate to the power α (the asymmetry hyperparameter). Reported to match or beat exhaustive grid search with a single hyperparameter. — [Chen et al., ICML 2018, arXiv 1711.02257](https://arxiv.org/abs/1711.02257) [not re-verified]
- **Dynamic Weight Averaging (DWA).** λ_k(t) = K·exp(w_k(t−1)/T) / Σ_i exp(w_i(t−1)/T), with w_k = L_k(t−1)/L_k(t−2) and temperature T = 2. It uses only loss values, not gradients. — [Liu, Johns, Davison, MTAN, CVPR 2019, arXiv 1803.10704](https://arxiv.org/abs/1803.10704) [not re-verified]
- **Critiques.** With proper tuning and regularisation, plain fixed-weight sums match specialised multi-task optimisers on standard benchmarks. — [Kurin et al., "In Defense of the Unitary Scalarization", NeurIPS 2022, arXiv 2201.04122](https://arxiv.org/abs/2201.04122); [Xin et al., "Do Current Multi-Task Optimization Methods in Deep Learning Even Help?", NeurIPS 2022, arXiv 2209.11379](https://arxiv.org/abs/2209.11379) [both not re-verified]
- **DSNT.** A differentiable soft-argmax "spatial-to-numerical transform" with no trainable parameters. It works well at low heatmap resolution and beat heatmap matching and FC regression in pose estimation. The paper pairs the coordinate loss with a heatmap regulariser (a divergence to a Gaussian target). — [Nibali et al., arXiv 1801.07372](https://arxiv.org/abs/1801.07372) (abstract verified; regulariser specifics [not re-verified])
- Integral regression (soft-argmax) for pose combines a heatmap loss with the integral coordinate loss. — [Sun et al., ECCV 2018, arXiv 1711.08229](https://arxiv.org/abs/1711.08229) [not re-verified]
- The focal-heatmap loss with α = 2, β = 4 comes from CenterNet; DLT adopts it. — [DLT](https://arxiv.org/abs/2012.04872); [CenterNet arXiv 1904.07850](https://arxiv.org/abs/1904.07850) [not re-verified]

### Inferences
Practical scheme for L = L_dist + λ_hm·L_hm + λ_mask·L_mask:
1. **Scale terms first.**
   - L_dist: Euclidean or smooth-L1 of the soft-argmax in voxel units (2 mm) or divided by 10 mm. Raw mm errors of 10–25 mm early on would dominate.
   - L_hm: a focal loss summed over voxels scales with the number of voxels and positives. Normalise by the number of positives (CenterNet convention), or use CE/KL between softmax(logits) and a normalised Gaussian target, which is naturally O(1) and pairs well with soft-argmax.
   - L_mask: Dice+BCE, already O(1).
2. **Defaults:** λ_hm = 1 (the TLT setting), λ_mask = 0.5–1. Grid search λ_hm ∈ {0.1, 0.3, 1, 3, 10} and λ_mask ∈ {0, 0.3, 1}, with short runs on one fold. Select by validation MED/CPM, not total loss.
3. **Kendall weighting:** 2–3 learnable scalars, about 5 lines of code. Initialise s = 0 and put s in a parameter group without weight decay. Risk: it can down-weight the hardest term (L_dist) when it is noisy. Clamp s.
4. **GradNorm / DWA:** probably not worth it for 3 terms. GradNorm needs extra backward passes in 3D, and the critiques suggest small gains.
5. **Curriculum:** epochs 0–N use L_hm (+ L_mask) only, then ramp λ_dist from 0 to 1. Soft-argmax over an untrained, near-uniform heatmap is biased toward the patch centre, and L_dist gradients are uninformative then. Also anneal the softmax temperature (or sharpen with learnable β·logits). This avoids the known soft-argmax failure where multimodal maps average to a point between peaks.
6. **Heatmap target:** a Gaussian at the FU centre with σ ≈ lesion radius/2–1 in feature cells. DLT used a radius-scaled Gaussian G(µ_s, r_s). A minimum σ of about 1 cell avoids targets that are too sharp at stride 4.

### Gaps
- No lesion-tracking paper ablated λ values or adaptive weighting. All recommendations above are transferred from general multi-task literature.
- Kendall/GradNorm/DWA numbers and the critiques were not re-read this session.

## Q6. Training tips (LR, batch size, AMP, overfitting)

### Takeaway
Reported 3D tracker settings:
- TLT: Adam lr 1e-4, batch 4, 300 epochs, 6M-parameter model.
- nnU-Net-family (LesionLocator, winner): SGD lr 1e-2 poly from scratch or 1e-3 for fine-tuning, batch 2–3 for large patches.

64³ patches are small, so batch 8–32 fits on one GPU with AMP. Prefer small models (channels halved, as TLT did explicitly to reduce overfitting), strong augmentation, and patient-level CV.

### Cited Findings
- TLT: Adam, lr 1e-4, 300 epochs, batch 4, 5.98M parameters. Channels were reduced "considering the parameter redundancy and overfitting in 3D networks." — [TLT](https://zrongguo.github.io/files/2022_MICCAI_Transformer%20Lesion%20Tracker.pdf)
- LesionLocator: pretraining SGD lr 1e-2 poly, batch 24; fine-tuning lr 1e-3, batch 3; ran on A100 40 GB, PyTorch 2.3.1. — [LesionLocator](https://arxiv.org/html/2502.20985v1)
- autoPET IV winner: SGD, 1,000 epochs, ResEnc U-Net, with one-third of patients held out for test and an 80/20 train/val split. — [arXiv 2605.23118](https://arxiv.org/html/2605.23118)
- Kirchhoff et al.: a 5-fold ensemble stabilised test performance. — [arXiv 2509.00613](https://arxiv.org/html/2509.00613v1)

### Inferences
- **Optimizer and schedule:** AdamW lr 1e-4–3e-4 (from scratch) with weight decay 1e-4–1e-2 and cosine decay plus warm-up. Use a 10× lower LR on a pretrained encoder.
- **Batch and normalisation:** batch 8–16 at 64³ with AMP (bf16, or fp16 with GradScaler). Run the correlation/softmax and the soft-argmax in fp32 to avoid overflow in exp. Use InstanceNorm or GroupNorm rather than BatchNorm if the batch is small.
- **Early stopping:** use validation MED/CPM@10mm. Monitor the train/val gap.
- **Overfitting controls:** about 1–5M parameters, dropout on the head, re-jittered FU crops each epoch, bidirectional pairs, SSL/synthetic pairs, and patient-level splits. Lesions from the same patient must not cross folds.
- **Sanity checks:**
  - Overfit 10 pairs to about 0 mm.
  - Report the zero-offset baseline.
  - Check the coordinate convention (voxel index order zyx vs xyz, and the sign of the offset). Sign/axis bugs are a common reason "plain CNN regression failed".

### Gaps
- DLT training hyperparameters were not located.
- No source studied batch size or AMP specifically for 3D Siamese trackers.
