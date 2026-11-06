# Matching architecture improvements for the Team 2 Siamese lesion-offset model

Scope: architecture changes to `SiameseNet(bl_mask=True, k=5, encoder="resenc", coarse=True)` (`src/lesionloc/model.py`, 158 lines). The current head computes three cosine-similarity maps on a 32³ × 4 mm grid: a 5³ fine template, a mask-pooled template, and a 3³ coarse 8 mm template upsampled ×2. Each map has a learnable temperature. The maps are summed, a global softmax runs over 32³ cells, and a soft-argmax gives the offset. Val: 4.44 mm mean / 2.54 mm median with flip TTA (row 16 of `docs/wiki/Experiments.md`). Already tried, so not re-recommended here as-is: heatmap CE loss (row 6), centre prior (row 12), local soft-argmax ±8 mm (row 9: mean +0.26 mm; row 11: tta_local 5.54 vs 5.58, a tie), frozen and fine-tuned nnFoundationCNN (rows 13, 15), and the BL mask as an encoder channel (row 7 small gain; row 17 invalid test, stem frozen). Losses and pretrained encoders are covered in `reports/Lesion patch localisation losses.md` and `reports/Pretrained encoders for Team 2.md`.

Labelling convention: "Cited Findings" are what the source says. "Inferences" are the researcher's own reasoning about this codebase; none of them has been tested on this data.

## Q1. Cost-volume processing: does a learned 3D CNN over stacked correlation maps beat fixed summation?

### Takeaway
In optical flow and large-displacement CT registration, learned processing of a correlation volume clearly beats using raw correlations. The closest lesion tracker (DLT) still sums its maps and puts only a single learned linear layer on top. For this model, the cheap, low-risk version is to stack the three maps (plus a few extra correlation channels) as channels and run a small 3D CNN over the 32³ volume before the softmax. Expected gain: modest, about 0.2–0.5 mm (inference), concentrated where the summed map has several peaks.

### Cited Findings
- DLT (Cai et al., CVPR 2021) fuses three scales by `Ŷ = σ(Wᵀ(M1 + U2 + U3) + b)`. That is a sum of the per-scale correspondence maps (coarser maps upsampled), then one learned fully convolutional layer and a sigmoid. The prediction is the global argmax. At each scale the map is itself the sum of a 3×3×3 local-kernel correlation and a large "global kernel" (7,11,11) correlation — [DLT, arXiv 2012.04872](https://arxiv.org/abs/2012.04872)
- DLT ablation (test MED): without the global kernel 9.3 mm vs 7.9 mm with it. Making the global-kernel decomposition non-learned (taking middle slices instead of learned flattening convs) gives 9.3 mm vs 7.9 mm. A larger global kernel (7,15,15) gives 7.7 mm but runs about 7× slower (24.1 vs 3.51 s per volume) — [DLT, arXiv 2012.04872](https://arxiv.org/abs/2012.04872)
- GMFlow (CVPR 2022), Things val EPE: a correlation volume followed by 0 conv residual blocks gives 18.83. With 4 blocks it gives 10.99, with 8 blocks 9.59, with 18 blocks 8.67. Most of the gain comes in the first 4 blocks — [GMFlow, arXiv 2111.13680](https://arxiv.org/abs/2111.13680)
- RAFT ablation (Sintel clean / final EPE): feeding correlation lookups to the update operator gives 1.63, against 2.27 for feeding warped features. A lookup radius of 0, i.e. a single correlation value, gives 3.41 against 1.63 at radius 4. Pooling the correlation volume into a pyramid gives 1.63 against 1.95 without — [RAFT, arXiv 2003.12039](https://arxiv.org/abs/2003.12039)
- RWCNet (3D CT/MRI registration, Learn2Reg 2022) uses a cost volume plus a GRU. On NLST lung CT (large displacement), TRE is 2.11 mm with correlation and 4.10 mm without. On OASIS brain (small displacement) the cost volume barely matters: 80.7 vs 80.0 Dice. The authors conclude that cost volumes and recurrent refinement help mainly for large displacements — [RWCNet, arXiv 2302.02283](https://arxiv.org/abs/2302.02283)
- SAMConvex builds 6D correlation volumes from SAM features and refines a flow field coarse-to-fine by lookups on the correlation pyramid. It reports beating learning- and optimisation-based methods on abdomen, head-and-neck and lung CT in about 2 s per pair — [SAMConvex, arXiv 2307.09727](https://arxiv.org/abs/2307.09727)
- Correlate-and-Excite (stereo) guides cost-volume channels by image features and uses top-k selection before soft-argmin. Both are cheap, and they improved accuracy at real-time speed — [CoEx, arXiv 2108.05773](https://arxiv.org/abs/2108.05773)

### Inferences
- Concrete design: `C = stack[fine xcorr, pooled-mask dot, coarse xcorr up×2, (optional) per-channel or group-wise correlations]`. Group-wise correlation means splitting the 32 projection channels into 4–8 groups and correlating each group separately (GwcNet-style), which gives 4–8 extra maps at no encoder cost. Then use 3–4 residual 3³ conv blocks (16–32 ch, InstanceNorm) and a 1×1 conv to one logit map, followed by the existing softmax and soft-argmax. Initialise the last layer so that the output equals the current weighted sum (identity-like init), so training starts from the 4.44 mm solution rather than from scratch. Params < 0.1 M; the 32³ grid makes this cheap (about 2–5% of current step time).
- Why it could help here: the sum is a fixed linear rule. A CNN can learn non-linear combinations, for example "fine peak only counts if the coarse map agrees nearby" (AND-like gating) and "suppress isolated high-fine/low-coarse peaks". That is exactly the look-alike failure mode of patient e2c0be2456 (many similar metastases). The DLT ablation shows that learned flattening beats fixed slicing (7.9 vs 9.3 mm), which is weak but direct evidence that learned map combination matters in 3D lesion tracking.
- Optionally concatenate FU stage-1 features (context, as in RAFT) into the cost-volume CNN. Risk: it lets the head learn FU-only appearance priors ("lesion-like spot") that ignore BL. That can help with merged or large lesions, but can also overfit with 2,159 pairs.
- Risk: low to medium. The main risk is overfitting, since the train/val gap is already about 2 mm. Use a tiny CNN, dropout on the cost volume, and keep the EMA.
- Cost: about 30–50 lines in `SiameseNet.forward`; one 4–5 h run plus a seed repeat (seed noise ≈ 0.1 mm, but the CI half-width in row 16 is ≈ 0.6 mm, so a single run proves little).

### Gaps
- No lesion-tracking paper ablates a learned cost-volume CNN against summation directly. The optical-flow evidence comes from data sets 100–1000× larger.
- Not found: any 3D lesion-tracking work after TLT (2022) that reports a cost-volume aggregation network on DeepLesion/DLS.

## Q2. Cross-attention / transformer matching (TransT, STARK, MixFormer, LoFTR; TLT in 3D)

### Takeaway
The Transformer Lesion Tracker beat DLT (6.0 vs 7.0 mm MED) on a training set of a similar size (3,008 pairs). Most of its gain came from template sparse selection, a Gaussian mask on the template features, not from attention itself. In GMFlow, cross-attention is the single most important component. A small cross-attention block on the 16³ (8 mm) grid is affordable on a V100 and is the most promising "big" change. It is higher risk than Q1 because of data size.

### Cited Findings
- TLT (MICCAI 2022): shared modified 3D ResNet-18 (features at 1/8 resolution, C = 192), sparse selection strategy (SSS), cross-attention transformer (CAT, 3 cross-attention layers per path), registration-based anatomical attention module (RAAM), global regressor. 5.98 M params, Adam 1e-4, 300 epochs, batch 4, CT at 1 mm. Trained on DLS: 3,008 train / 403 val / 480 test pairs — [TLT, arXiv 2206.06252](https://arxiv.org/abs/2206.06252)
- TLT test results: MED 6.0 ± 7.7 mm, CPM@10 mm 87.37%, against DLT 7.0 ± 8.9 mm / 78.85% and TransT (baseline) 7.6 ± 7.9 mm / 79.59%. Transformer methods mainly cut the z error: MEDz is 1.7 mm (TLT) and 1.8 mm (TransT), against 4.0 mm for DLT — [TLT, arXiv 2206.06252](https://arxiv.org/abs/2206.06252)
- TLT ablation (val): CAT alone 7.58 mm, + SSS 6.76, + RAAM 6.30, + global regressor 5.98. SSS gave the largest single gain. It keeps only template features where a Gaussian of the template lesion (σ ∝ 2r) exceeds a threshold of 0.7; thresholds 0.4–0.9 gave 6.39–6.80 mm — [TLT, arXiv 2206.06252](https://arxiv.org/abs/2206.06252)
- TLT failure modes: when the affine registration fails, TLT often fails too, because RAAM makes it depend on registration. Similar lesions at similar locations (e.g. two nodules a few slices apart) still confuse it — [TLT, arXiv 2206.06252](https://arxiv.org/abs/2206.06252)
- GMFlow ablation (Things val EPE): full model 6.67; without cross-attention 10.84; without positional encoding 8.38; without self-attention 7.04. With equal parameter budget, "Transformer + softmax" with 6 blocks (4.2 M params) gives 6.67, against 9.04 for a 12-block conv over the cost volume (11.5 M). Global matching beats local 9×9 matching (6.67 vs 19.88 EPE), and the gap is mostly on large motions (s40+: 17.37 vs 61.06) — [GMFlow, arXiv 2111.13680](https://arxiv.org/abs/2111.13680)
- GMFlow benefits from longer training (200 k → 800 k iterations), which the authors relate to known transformer data hunger — [GMFlow, arXiv 2111.13680](https://arxiv.org/abs/2111.13680)
- LoFTR interleaves self- and cross-attention on 1/8-resolution features with positional encoding. The authors say positional encoding makes features "position-dependent, which is crucial … to produce matches in indistinctive regions". They use linear attention to keep cost O(N) — [LoFTR, arXiv 2104.00680](https://arxiv.org/abs/2104.00680)

### Inferences
- Feasible design: on the 16³ stage-2 grid (4,096 FU tokens), run 2 blocks of [FU self-attn, FU→BL cross-attn over the BL tokens inside about a 5³–7³ window plus the mask-pooled token], with 3D sinusoidal or learned positional encodings. Then compute the coarse correlation from the attended features. Memory: 4,096 × (≤343) cross-attention is trivial. Even full 4,096² self-attention at batch 16, 4 heads, fp16 is about 2 GB, which fits beside the 5.84 GiB peak. On the 32³ grid (32,768 tokens) full attention is not feasible; use windowed or linear attention, or stay at 16³.
- The TLT ablation suggests the template-restriction trick (SSS) carries much of the value. This model already does a version of it via `centre_template` and mask pooling. So expect less gain from attention here than TLT's +1 mm over DLT. A plausible gain is 0–0.5 mm (inference). The tail cases (look-alike metastases) are the likely beneficiaries, because cross-attention lets each FU cell compare against the whole BL context, not just a fixed 5³ kernel.
- Positional encoding is a double-edged sword here. Absolute position in the FU window encodes the window prior (≈ the centre prior, which tied in row 12 and is risky under Team 1's real error distribution). Use relative or rotary encodings, or none in cross-attention, to keep the translation equivariance that makes the Siamese design robust to far offsets.
- Risk: medium-high. Transformers on 2,159 pairs (≈ 10.8 k with offline augmentation) may overfit or train slowly; GMFlow needed 800 k iterations on synthetic data. Mitigation: insert the block with a zero-initialised residual gate (output = input at start), keep the 3 existing maps, low LR on the block.
- Cost: about 60–100 lines, one new module. 1–2 runs of about 4–5 h each. Probably +20–40% step time.

### Gaps
- No public TLT code details were checked (repo [TangWen920812/TLT](https://github.com/TangWen920812/TLT) exists but was not read). The z-axis gain mechanism is the authors' speculation.
- STARK, MixFormer and TransMatch were not read in this pass; no numbers from them are given.
- No source isolates the effect of training-set size on transformer vs correlation matching in 3D medical data.

## Q3. Coarse-to-fine / iterative re-centring (cascade, RAFT-style updates, ROI re-crop)

### Takeaway
Iterative refinement is among the most strongly supported architectural ideas: RAFT's error falls from 4.04 to 1.43 EPE going from 1 to 32 updates, and RWCNet's TRE nearly halves with more recurrent steps. Here, though, the joint-pipeline tail comes from starting offsets beyond the training jitter (±24 mm clip) and window, and re-centring helps only if the first pass lands within the basin of the second. The cheapest high-value version is a test-time two-pass re-crop: predict, re-cut the FU window around the prediction, predict again, accept only if the result is consistent. It needs no new weights. A trained residual stage is a second step.

### Cited Findings
- RAFT inference updates (Sintel clean EPE): 1 update 4.04, 3 updates 2.14, 8 updates 1.61, 32 updates 1.43, 100 updates 1.41. Tied weights across iterations beat untied ones (1.63 vs 1.96, 4.8 M vs 32.5 M params). A GRU update beats plain conv (1.63 vs 2.04) — [RAFT, arXiv 2003.12039](https://arxiv.org/abs/2003.12039)
- RAFT argues that a single high-resolution field with iterative updates avoids coarse-to-fine cascades' "difficulty of recovering from errors at coarse resolutions" — [RAFT, arXiv 2003.12039](https://arxiv.org/abs/2003.12039)
- RWCNet (3D CT): 2-timestep GRU TRE 5.17 mm vs 2.11 mm with full steps (12/12/4 steps per resolution). Single resolution 4× gives 5.52 mm vs 2.11 mm multi-resolution — [RWCNet, arXiv 2302.02283](https://arxiv.org/abs/2302.02283)
- GMFlow refinement: upsample the 1/8 flow to 1/4, warp the second feature, and learn the residual by local 9×9 window matching, sharing the transformer weights with the global stage. Sharing improved generalisation. One refinement already beat 31-iteration RAFT on Sintel clean (generalisation from Chairs+Things) — [GMFlow, arXiv 2111.13680](https://arxiv.org/abs/2111.13680)
- DLT robustness test: shifting the template centre by up to 0.25 r raised DLT-Mix MED by only 0.9 mm and lowered CPM by 3.62%. DEEDS registration rose by 2.4 mm and dropped 18% in CPM — [DLT, arXiv 2012.04872](https://arxiv.org/abs/2012.04872)
- LesionLocator (CVPR 2025) propagates prompts by a deformation field from a registration module and segments autoregressively, using the previous scan's predicted mask as the prompt for the next. Mask propagation was best: retrieval 86%, Dice 79, centre distance about 3 mm, against point or box prompts. It still names "closely located or merging lesions" as a limitation — [LesionLocator, arXiv 2502.20985](https://arxiv.org/abs/2502.20985)
- Joint pipeline (this project): with Team 1's hybrid centres, joint val mean is 8.74 mm. 11.5% of lesions remain > 20 mm off (Team 1 alone: 13.9%). Team 1's per-axis SD is (7.1, 8.2, 11.7) mm, and some offsets exceed the ±24 mm training clip — `reports/Joint experiment results.md` (internal)

### Inferences
- Test-time re-centring (no retraining): run the model, shift the FU window centre by the rounded prediction, re-cut a 64³ window from the raw FU scan (the preprocessing path from centres already exists), run again and add the offsets. Two or three passes. Accept pass k only if the softmax peak probability rises, or if |Δ_k| is shrinking. Peak prob is already a strong confidence signal: peak < 0.2 gives 9.4 mm vs 3.3 mm. Expected effect, by inference: small on synthetic val, where most lesions start within ±24 mm and already reach 2.5 mm median. Potentially large on the joint pipeline for lesions starting 20–40 mm off: one pass moves them into the ±24 mm range the model was trained on. Risk: drift onto a look-alike lesion. The peak-prob gate and a max total shift (e.g. 40 mm) limit this.
- Within-patch refinement without re-cropping (LoFTR/GMFlow style): after the global soft-argmax on 4 mm cells, crop a 7³–9³ window of stage-0 (2 mm, 64³) or stage-1 features around the argmax. Correlate a small stage-0 BL template there and take a local soft-argmax for the residual. Train the residual with its own L1 loss, as RAFT/GMFlow supervise every stage. This differs from the failed ±8 mm local soft-argmax (row 9), which re-weighted the same coarse map. Here the refinement sees new, finer features and is supervised directly.
- Trained 2-stage cascade: stage 2 sees FU windows centred on stage-1 predictions. Train it on stage-1 residuals: run the trained model on train data, collect the residual distribution, and draw stage-2 jitter from it. Weights can be shared (RAFT/GMFlow found sharing better) or the stage-2 model can be a fine-tuned copy. Cost: an extra 4–5 h run plus a dataset pass. The `fu_big` 88³ cube supports only ±12 voxels (±24 mm) re-jitter around the true FU centre, so stage-1-centred crops outside that need re-cutting from the raw scans.
- Implementation cost: test-time re-centring is about 50 lines in the eval script plus access to raw volumes; the residual fine head is about 40 lines in the model; the full cascade is a day of work.

### Gaps
- No source found for DLT using cascaded refinement. The DLT paper read here has affine registration as initialisation plus a single Siamese pass. The "cascaded refinement in DLT" premise in the brief could not be confirmed.
- No published numbers on test-time window re-centring for 3D lesion tracking.

## Q4. Using 2 mm stage-0 features; local vs global soft-argmax; bias under multimodal heatmaps

### Takeaway
Global soft-argmax is biased toward the window centre and averages between modes. This is documented in pose estimation (BCIR) and stereo (multimodal soft-argmin), and it is a plausible cause of part of this model's tail. The failed ±8 mm local soft-argmax shows that simply truncating the same 4 mm map is not enough. Better options: top-k or peak-window readout combined with a learned refinement on 2 mm features (Q3), or a training-time fix. Expected gain: median −0.2 to −0.5 mm from sub-cell refinement; mean gain depends on the tail fixes.

### Cited Findings
- Bias-Compensated Integral Regression: combining softmax and expectation "induces a bias" that pulls predictions toward the heatmap centre. The bias forces networks to learn "degenerately localized heatmaps". BCIR compensates for the bias and adds a Gaussian prior loss, which trained faster and was more accurate than plain integral regression on body and hand pose — [BCIR, arXiv 2301.10431](https://arxiv.org/abs/2301.10431)
- Stereo matching: soft-argmax-trained networks are "prone to being multimodal" because nothing constrains the distribution's shape. Sampling-Gaussian supervision improved 5 baselines on 2 data sets — [Sampling-Gaussian, arXiv 2410.06527](https://arxiv.org/abs/2410.06527). CMD adds anisotropic soft-argmin and uncertainty regularisation to push predictions toward unimodal distributions under domain shift — [CMD, arXiv 2504.21302](https://arxiv.org/abs/2504.21302)
- CoEx uses top-k selection of the cost volume before soft-argmin to compute disparity — [CoEx, arXiv 2108.05773](https://arxiv.org/abs/2108.05773)
- LoFTR's fine level crops a w×w window around each coarse match on fine features, correlates the centre vector with the window, and takes the expectation over the resulting heatmap for sub-pixel accuracy. The fine loss is an ℓ2 loss weighted by inverse heatmap variance — [LoFTR, arXiv 2104.00680](https://arxiv.org/abs/2104.00680)
- Heatmap-in-heatmap (face alignment): quantisation error from neglecting sub-pixel coordinates accounted for "more than 1/3 of the whole prediction errors" of SOTA methods; nested integer + decimal heatmaps fixed it — [HIH, arXiv 2104.03100](https://arxiv.org/abs/2104.03100)
- SAM (Yan et al., TMI) cuts the FPN connection between its coarsest level and the level above, to keep global and local embeddings independent. Not cutting it worsened lesion matching from 2.7 to 3.0 px MRE (max error 28.8 → 44.8) — [SAM, arXiv 2012.02383](https://arxiv.org/abs/2012.02383)

### Inferences
- In this model the soft-argmax runs over ±62 mm with background mass spread across 32,768 cells. Any residual mass far from the peak pulls the estimate toward the window centre and toward secondary modes. On lesions with peak prob < 0.2 (38 of 207 val lesions, mean 9.4 mm) this is likely a real contributor. That is an inference; it could be checked offline on saved logits by comparing argmax, soft-argmax and top-k readouts per lesion.
- Option A, a readout-only check costing no training: top-k (k ≈ 27–125 cells) soft-argmax, or a peak-window soft-argmax taken around the argmax of the TTA-averaged logits. Row 9 shows that the peak-window version alone can lock onto the wrong peak. So it is useful only once the map is unimodal more often (after Q1 or Q2).
- Option B: a stage-0 residual head (Q3 second bullet). Stage 0 is 32 ch at 2 mm, already computed and currently unused. A 5³ stage-0 template (10 mm) correlated within a 9³ window around the coarse peak, plus a local soft-argmax trained with the distance loss, should cut quantisation and smoothing error on small lesions (r < 5 mm: currently 2.94 mm). Cost: about 40 lines, +10% step time.
- Option C: an auxiliary loss on the pre-softmax logits that sharpens the map (e.g. the FU-mask NLL already present; a small Gaussian CE with σ ≈ 4 mm, the λ_hm 0.3 retry already listed in row 6). Losses are out of scope here; see the losses report.

### Gaps
- No lesion-tracking paper reports a direct comparison of global vs local soft-argmax readouts.
- BCIR's exact compensation formula was not read, only the abstract.

## Q5. Deeper stages (16 mm) and anatomical / global position embeddings to separate look-alike lesions

### Takeaway
Two independent lesion-tracking sources show that global context mainly cuts large errors: DLT's global kernel, and SAM's global embedding, which cut max error from 130 to 29 px. This project already saw a 1.1 mm gain from adding the 8 mm map. A fourth, 16 mm map (stage 3, an 8³ grid on the 64³ patch) is the cheapest continuation. But the 128 mm FU window limits how much context it can add. A larger FU search window (asymmetric template/search sizes) is the architectural change most aligned with the joint-pipeline tail.

### Cited Findings
- SAM lesion-matching ablation (DeepLesion follow-up matching, MRE/max in px): full coarse-to-fine 2.7 / 28.8, accuracy 91.1%. Global embedding only: 6.6 / 29.9, accuracy 52.5%. Local embedding only: 3.2 / 130.4, accuracy 90.1%. Single fine-scale embedding without the coarse-to-fine structure: 3.5 / 41.1, accuracy 84.1%. Local-only embeddings "may also highlight distant areas with similar local textures if the global similarity is not considered, leading to a large maximum error" — [SAM, arXiv 2012.02383](https://arxiv.org/abs/2012.02383)
- SAM backbone: 3D ResNet-18 + FPN, global embedding from the coarsest FPN level and local embedding from the finest, 128-d each, L2-normalised. A U-Net backbone was worse (lesion MRE 3.5 vs 2.7). Without ImageNet init, MRE was 3.0 vs 2.7 — [SAM, arXiv 2012.02383](https://arxiv.org/abs/2012.02383)
- DLT: removing the global (7,11,11) kernel raised val MED from 7.08 to 8.77 mm and test MED from 7.95 to 9.29 mm — [DLT, arXiv 2012.04872](https://arxiv.org/abs/2012.04872)
- DLT's anatomy signal encoder (ASE) multiplies image features by encoded Gaussian heatmaps: around the template lesion, and around the affine-projected location in the search image. Without ASE, test MED is 9.3 vs 7.9 mm. A Gaussian σ of 4r was best; 8r "loses its specificity" (8.5 mm) — [DLT, arXiv 2012.04872](https://arxiv.org/abs/2012.04872)
- GMFlow: global matching beats local matching especially for large motions (s40+ EPE 17.37 vs 61.06 for a 9×9 local window) — [GMFlow, arXiv 2111.13680](https://arxiv.org/abs/2111.13680)
- MedDIFT (2025) uses multi-scale features from a pretrained latent medical diffusion model as training-free voxel descriptors for 3D correspondence. Multi-level fusion helped in its ablation (lung CT) — [MedDIFT, arXiv 2512.05571](https://arxiv.org/abs/2512.05571)

### Inferences
- Stage 3 (16 mm): ResEnc-L stage 3 has 256 ch, an 8³ grid on 64³ input. A 3³ template there spans 48 mm, which is over a third of the window. Its correlation map covers only 8³ positions and needs ×4 upsampling. Expected gain: smaller than the 8 mm map's 1.1 mm, by inference, because the 8 mm map with a 24 mm template already captures much organ-level context and the window caps context at 128 mm. Cost: about 15 lines (add the stage, a projection, a 4th `log_scale`). Params grow by several M (a 256-ch stage), which raises overfitting risk; a narrower stage (128 ch) is an option. Try it together with Q1, so the CNN can learn when to trust the 16 mm map.
- Larger FU search window: correlation is translation-equivariant, so the FU patch can be larger than the BL patch without changing the template (SiamFC-style template/search asymmetry; DLT correlates against the whole search volume). For example, keep BL 64³ and use FU 96³ at 2 mm (192 mm cube) with jitter widened to Team 1's real residuals. Cost: re-preprocessing the FU patches (`fu_big` must grow to about 120³), 3.4× FU voxels; memory is likely to fit, since the current peak is 5.84 GiB. This is the architectural change most directly aimed at the > 20 mm tail (inference). It overlaps with the data recommendation in the joint report (train on Team 1's residuals).
- Anatomical-position embeddings: SAM-style global embeddings learned self-supervised on whole CTs separate "same texture, different body part". Within a 128 mm window, the coarse map already plays that role. A cheap proxy is a second, independent coarse projection head trained with a contrastive loss on the train pairs (BL lesion vs FU positions) on stage 2/3 features. The pretrained SAM checkpoints were already assessed in the pretrained-encoders report and are not repeated here.
- Do not encode absolute window coordinates (see Q2): it re-introduces the centre prior.

### Gaps
- SAM++ / SAMv2 / UAE numbers on lesion tracking were not re-read in this pass (they are in the pretrained-encoders report).
- No source quantifies how much context beyond 128 mm helps lesion matching at 2 mm.

## Q6. BL mask as input channel vs pooling only; mask-conditioned FiLM / attention on the template

### Takeaway
Both closest 3D trackers inject the BL lesion location into the features themselves, and both report about 0.8–1.4 mm gains from it: DLT multiplies the features by a Gaussian (ASE), and TLT keeps only template features inside a Gaussian (SSS). This model's version (mask pooling plus a centred template) is weaker: it never tells the encoder which BL cells are lesion. The cheapest test is a gated template: multiply the BL template by a soft mask or Gaussian before correlation, i.e. weight the template cells. A properly trainable mask stem channel is the second test; row 17 was not a valid test.

### Cited Findings
- DLT fuses image features with anatomy-signal features by element-wise multiplication, F = ψ(I_t) ⊙ φ(G_t), on both template and search. Multiplication beat channel concatenation (val MED 6.69 vs 6.85 mm for the final configurations). Gaussian σ = 4r was best, 2r gave 7.4 mm and 8r 8.5 mm (test) — [DLT, arXiv 2012.04872](https://arxiv.org/abs/2012.04872)
- TLT's SSS keeps template features where a radius-scaled Gaussian exceeds 0.7, which lowered val MED from 7.58 to 6.76 mm. The authors attribute the gain to small lesions (e.g. lung nodules), which shrink to a single voxel after downsampling — [TLT, arXiv 2206.06252](https://arxiv.org/abs/2206.06252)
- This project: BL mask as a second encoder channel plus the pooled template gave −0.17 mm (row 7, small encoder, near noise). Row 17 (`--enc-mask` on ResEnc) was invalid because `--freeze early` froze the stem, so the mask weights stayed at zero — `docs/wiki/Experiments.md` (internal)

### Inferences
- Gated template (cheapest): weight the 5³ fine template and the 3³ coarse template cell-wise by `w = α + (1−α)·G(r_BL)`, where G is a Gaussian from the BL mask centre and radius (σ ≈ r, clipped to the template), or by the pooled BL mask itself, with α learnable (init 1, i.e. the current model). This is DLT's ⊙ fusion, restricted to the template. It mostly helps lesions much smaller or much larger than the fixed 20/24 mm templates. Large lesions (r > 20 mm, 11.6 mm mean error) may need the opposite: a template size that scales with r. Cost: about 15 lines; risk low.
- Mask-conditioned FiLM: compute γ, β per channel from the pooled BL lesion feature (or from r and mask volume) and modulate FU stage-1/2 features before correlation. Not supported by any source found. Risk: it makes FU features BL-dependent, which breaks the shared-encoder symmetry and doubles FU feature compute per BL query (fine at batch 16). Low priority.
- Trainable mask stem channel: rerun row 17 with the stem unfrozen (`--enc-mask` on random init, as in row 16). It costs one run and is already on the team's list.

### Gaps
- No source compares mask-as-input vs mask-weighted template in 3D lesion matching.

## Q7. Multi-hypothesis / mixture outputs vs single soft-argmax

### Takeaway
Mixture outputs are well supported in stereo (SMD-Nets) and fit this task's real ambiguity: merged lesions sharing one FU mask, and look-alike metastases. Their value here is mainly as a better readout and a confidence signal under an averaging metric: predict K = 2–3 modes and weights, then output the top mode. They also give a principled fallback to Team 1's centre when uncertain. Expected mean gain is small unless the top-mode choice is better than the current soft average. The bigger payoff is calibrated uncertainty for the joint pipeline.

### Cited Findings
- SMD-Nets output a bimodal mixture density per pixel. This gives "sharp and precise disparity estimates near discontinuities while explicitly modeling the aleatoric uncertainty", and it improved several 2D and 3D stereo backbones — [SMD-Nets, arXiv 2104.03866](https://arxiv.org/abs/2104.03866)
- Adaptive multi-modal cross-entropy (stereo): unimodal ground-truth assumptions ignore that edge pixels can be multimodal. Modelling multimodal targets plus an improved estimator took GANet to first place on KITTI 2015/2012 — [ADL, arXiv 2306.15612](https://arxiv.org/abs/2306.15612)
- LesionLocator lists "accurate distinction of closely located or merging lesions" as an open challenge — [LesionLocator, arXiv 2502.20985](https://arxiv.org/abs/2502.20985)
- This project: lesions with r > 20 mm average 11.6 mm; several of them share one FU mask (merged lesions), where the target centre is ambiguous. Peak prob < 0.2 marks a 9.4 mm subgroup — `docs/wiki/Team2-Model.md` (internal)

### Inferences
- The cheapest version uses no new head: extract the top-K local maxima of the 32³ logits (3D NMS, e.g. 3³ max-pool), compute each mode's mass and local soft-argmax, and output the highest-mass mode. This is "argmax over modes" instead of a global mean. It is a readout experiment on saved logits, run in minutes. Train-time version: a winner-takes-all loss over K heads (Rupprecht-style multiple hypotheses), but with 2k pairs the extra heads may collapse.
- For merged lesions the metric itself is ambiguous: the target is the BL lesion's own `cog_fu`, while the FU mask is shared. No architecture fixes label ambiguity; mode selection may even raise error there. Report these lesions separately.
- Use mode weights as a gate in the joint pipeline: if the top mode has mass < τ, fall back to Team 1's centre or flag the lesion. Tune τ on train/val residuals.

### Gaps
- No lesion-tracking paper with mixture/multi-hypothesis outputs was found.
- No source for Rupprecht et al. multiple-hypothesis numbers was read in this pass.

## Q8. Ensembling and TTA beyond flips

### Takeaway
Ensembling is the most reliable gain available: three seeds give 4.27 mm without TTA, against 4.44–4.57 mm for single models. It costs only compute. Rotation and multi-crop TTA are untested here. Multi-crop (shifted FU windows) is attractive because it also probes robustness to the window offset, which is the joint pipeline's weakness. No external evidence specific to lesion tracking was found.

### Cited Findings
- This project: flip TTA gains 0.1–0.6 mm across runs (row 9: −0.58 mm; joint pipeline: −0.57 mm). The mean of the plain predictions of three two-scale runs (rows 15–17) is 4.27 / 2.39 mm — `docs/wiki/Experiments.md`, `reports/Joint experiment results.md` (internal)
- SAM: adding flip augmentation in training hurt X-ray landmark matching badly (hand MRE 13.1 → 64.9) and slightly hurt CT lesion matching (2.6 → 2.8). Rotation and deformation augmentation helped hand X-ray (22.6 → 13.1) — [SAM, arXiv 2012.02383](https://arxiv.org/abs/2012.02383)
- RAFT: tied-weight iterative application at test time keeps improving up to about 32–100 iterations — [RAFT, arXiv 2003.12039](https://arxiv.org/abs/2003.12039)

### Inferences
- Multi-crop TTA: run the model on 4–8 FU windows shifted by ±8 mm along axes, add back the shifts, and average or median the predictions, optionally weighting by peak probability. Each crop sees the lesion at a different window position, which averages out window-position bias (Q4). Cost: needs the raw FU volume or `fu_big` (±24 mm room only around the *true* centre, so for real inference re-cut from raw). Combines naturally with Q3 re-centring.
- Rotation TTA: ±10° rotations about the FU window centre (the training data already has ±15° offline augmentation). Rotate the predicted offset back. Cost: resampling at inference (SimpleITK path exists). Expected gain, by inference: ≤ 0.2 mm; flips already give the cheap symmetry gain. SAM's flip result suggests that left/right anatomy cues matter. This model's flip TTA works empirically, but check per-axis whether x-flips help less than z-flips.
- Ensembles: average 3–5 seeds, or use snapshot or SWA weights from epochs 15–30 (val rises after epoch 20–30, so a cosine schedule with restarts or SWA over epochs 15–30 fits). Prefer averaging logits, then one soft-argmax, over averaging offsets. Logit averaging keeps one mode when models disagree; offset averaging lands between modes. This is an inference, testable offline. Combine with flip TTA: 3 seeds × 8 flips = 24 forward passes, cheap at inference.
- Ensemble gains do not fix the joint-pipeline tail (starting offsets beyond training range). That needs Q3, Q5's larger window, or training on Team 1 residuals.

### Gaps
- No published comparison of TTA variants for 3D lesion tracking was found.
- Whether logit averaging beats offset averaging for this model is untested.

## Q9. Prioritisation across all options (expected gain, risk, cost)

### Takeaway
Ranked by expected gain per GPU-hour, and with the joint-pipeline tail in mind:
1. Offline readout experiments on saved logits (no training): top-K mode readout, top-k soft-argmax, logit- vs offset-averaged seed ensembles.
2. Test-time re-centring and multi-crop TTA on the joint pipeline.
3. Learned cost-volume CNN head (Q1) plus gated template (Q6).
4. Stage-0 residual refinement head (Q3/Q4).
5. Larger FU search window with wider jitter (Q5).
6. Cross-attention block on 16³ (Q2).
7. Stage 3 / 16 mm map (Q5).
8. A trained cascade.

### Cited Findings
- Learned correlation processing and iterative refinement pay off most for large displacements: RWCNet NLST vs OASIS ([RWCNet](https://arxiv.org/abs/2302.02283)); GMFlow s40+ ([GMFlow](https://arxiv.org/abs/2111.13680)).
- Global context mainly removes gross errors: SAM max error 130 → 29 px ([SAM](https://arxiv.org/abs/2012.02383)); DLT without the global kernel +1.4 to +1.7 mm ([DLT](https://arxiv.org/abs/2012.04872)).
- Transformer matching gave about 1 mm over DLT on a similar-size data set, mostly through template selection and z accuracy ([TLT](https://arxiv.org/abs/2206.06252)).

### Inferences
- Expected gains (val, synthetic jitter), with all numbers the researcher's estimates:
  - cost-volume CNN: −0.2 to −0.5 mm
  - gated template: 0 to −0.3 mm
  - stage-0 refinement: median −0.2 to −0.5 mm, mean −0.1 to −0.3 mm
  - cross-attention: 0 to −0.5 mm, with high variance
  - stage 3: 0 to −0.3 mm
  - seed ensemble: about −0.2 to −0.3 mm (observed 4.27 vs about 4.5)
  - re-centring and a larger window: small on synthetic val, but potentially several mm on the joint-pipeline subset that starts > 20 mm off (the 11.5% tail)
- Noise floor: seed noise is ≈ 0.1 mm, but paired CIs on 207 val lesions / 28 patients are about ±0.6 mm. Changes under about 0.5 mm need two seeds or a 3-seed ensemble comparison to be credible. Evaluate the joint pipeline (Team 1 centres) too, not only synthetic jitter, since that is where the architecture's limits show.
- Runs that fit the V100 4–8 h budget: Q1 + Q6 together (one run), the stage-0 head (one run), stage 3 (one run), the larger window (one re-preprocess plus one run), cross-attention (1–2 runs). Use 30 epochs instead of 50, since val peaks at epochs 13–28.

### Gaps
- None of the expected-gain numbers above has external support specific to this data set. They should be treated as hypotheses for the ablation plan.
