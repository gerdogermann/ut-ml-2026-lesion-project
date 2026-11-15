# Mask- and shape-aware loss functions for lesion localisation in 3D CT follow-up patches

Setting assumed throughout: Siamese 3D CNN, input = BL patch (64^3 @ 2 mm) + BL mask, FU patch (64^3, lesion within +-24 mm of centre). Output = logits z(v) over FU voxels, p = softmax(z) over the patch (or sigmoid heatmap h(v) in [0,1]), point x_hat = soft-argmax = sum_v p(v) x_v. FU mask M (binary, this lesion only), centroid c = mean of x_v over v in M, equivalent radius r = (3|M| dv / 4pi)^(1/3) (dv = 8 mm^3). Notes written 2026-10-29.

Provenance tags: "[verified]" = read in a fetched page this session; "[snippet]" = from search-result summary only; "[prior knowledge]" = well-known formula from the cited paper, not re-read this session — report writer should treat as needing a check.

## Q1. Probability-mass-in-mask loss (MIL / "any point inside")

### Takeaway
L_mass = -log sum_{v in M} p(v) is the natural MIL-style "any point inside the lesion" objective for a softmax-over-voxels head; it is directly analogous to bag-level MIL losses used in box-supervised segmentation, but I found no paper using exactly this loss for longitudinal lesion tracking. It should be combined with a centroid term or entropy/concentration term, because by itself it is satisfied by a mass spread over the whole lesion, which makes soft-argmax drift toward the mask centroid only by accident (and outside the mask for non-convex/merged lesions).

### Cited Findings
- MIL view of box supervision: Hsu et al. (NeurIPS 2019) treat each sweeping line crossing a tight box as a positive bag (at least one pixel is foreground) and lines outside as negative bags; loss = unary MIL term (bag max-probability, BCE-style) + pairwise smoothness term; integrated into Mask R-CNN and on PASCAL VOC "performs favorably against existing weakly supervised methods and even surpasses some fully supervised methods" [snippet] — [NeurIPS 2019](https://papers.neurips.cc/paper_files/paper/2019/hash/e6e713296627dff6475085cc6a224464-Abstract.html); [code](https://www.github.com/chengchunhsu/WSIS_BBTP)
- Soft-argmax/DSNT: heatmap normalised to a distribution, coordinate = expectation; no trainable parameters, fully differentiable; authors add a regulariser on the heatmap shape (reportedly a Jensen-Shannon divergence to a Gaussian, secondary-summary only) which "increases prediction accuracy"; ResNet-50+DSNT comparable to 8-stack hourglass at 3x speed [snippet] — [Nibali et al. 2018, arXiv 1801.07372](https://arxiv.org/abs/1801.07372v2); also implemented in [Kornia](https://kornia.readthedocs.io/en/v0.7.0/_modules/kornia/geometry/subpix/dsnt.html)
- LesionLocator's own point prompts are sampled as "a random pixel within the foreground", i.e. the field treats any interior voxel as a valid identifier of the lesion [verified] — [Rokuss et al. CVPR 2025, arXiv HTML](https://arxiv.org/html/2502.20985)

### Inferences
- Formulas (my derivations, consistent with the above):
  - Mass-in-mask (softmax head): L_mass = -log( sum_{v in M} p(v) + eps ). Gradient pushes logit mass from outside M to inside M, no preference within M. Equivalent to MIL "noisy-OR over bag" in the softmax limit; contrast with max-pooling MIL: L_max = -log max_{v in M} p(v) (more peaked, less stable).
  - Sigmoid-head MIL variants: positive bag L+ = -log(1 - prod_{v in M}(1 - h(v))) (noisy-OR) or -log(LSE_tau over M); negative bag: L- = -sum_{v not in dilate(M,k)} log(1 - h(v)).
  - Fix the "spread mass" problem: add (a) entropy term lambda*H(p) or DSNT-style divergence to a compact Gaussian, (b) variance term sum_v p(v)||x_v - x_hat||^2 (Nibali-style variance regulariser), or (c) a small centroid regression term ||x_hat - c||/r.
  - Soft-argmax pitfall: x_hat is the mean of p; if p has mass on two disconnected components (merged lesion, or lesion + distractor), x_hat lands between them, possibly outside M. Mass-in-mask is computed on p, not on x_hat, so it does not detect this. Pair with a point-level term (Q3) or use an argmax/top-k-local soft-argmax at inference.
- Pros: scale-invariant in the sense that a tiny and a huge lesion both reach loss 0 when all mass is inside; matches "hit" metric (Q4); tolerant of ambiguous centroid for irregular lesions. Cons: weak gradient on location within lesion; degenerate for very large lesions covering much of the 64^3 patch (a lesion of 5 cm diameter at 2 mm voxels ~ 25 voxels across, ~6% of patch volume — easy to satisfy); undefined if FU mask empty (vanished lesion) — needs a "no lesion" token or skip.

### Gaps
- No published longitudinal-CT tracker found that trains with -log sum_{v in M} p(v); evidence for its benefit vs a Gaussian heatmap is absent. Would need an own ablation.

## Q2. Dice / Tversky / BCE between heatmap and mask; mask (or blurred mask / distance transform) as heatmap target

### Takeaway
The standard lesion tracker (DLT) uses a Gaussian heatmap target whose width scales with lesion radius plus CenterNet-style focal loss; LesionLocator instead predicts the full FU mask with CE+Dice and gets the point from the mask. Using the mask (or a mask-shaped soft target) as heatmap target is a legitimate middle ground: it encodes size and shape, but makes the "peak" ill-defined, so the point should then be derived from the predicted region (centroid of thresholded prediction) rather than from a sharp peak.

### Cited Findings
- DLT target = Gaussian kernel centred at target lesion centre with spread tied to lesion radius, Y = G(mu_s, r_s), G(mu, n r) = exp(-sum_i (i - mu^i)^2 / (2 (n r)^2)); trained with focal loss (alpha=2, beta=4): positives (1-y_hat)^alpha log(y_hat), negatives (1-y)^beta y_hat^alpha log(1-y_hat), which down-weights negatives near the centre; centre jitter during training ||Delta mu|| <= 0.25 r; no mask used (DeepLesion has only RECIST) [verified] — [Cai et al. CVPR 2021, ar5iv](https://ar5iv.labs.arxiv.org/html/2012.04872)
- DLT results on DeepLesion tracking benchmark (3891 lesion pairs): CPM@10mm 78.85%, CPM@Radius 86.88%, MED 7.0 +- 8.9 mm vs DEEDS registration 71.88 / 85.52 / 7.4 mm [verified] — [Cai et al.](https://ar5iv.labs.arxiv.org/html/2012.04872)
- LesionLocator trains segmentation with L_seg = L_CE + L_Dice (plus registration loss L_prop = NCC + lambda*GradICON); prompts (point ball r=5 px, box, or previous-timepoint mask) are concatenated as input channels at the first network level [verified] — [Rokuss et al. 2025](https://arxiv.org/html/2502.20985)
- Regional losses (Dice, CE) "have values that differ considerably – typically of several orders of magnitude – across segmentation classes" under heavy imbalance; motivates boundary/distance losses [snippet] — [Kervadec et al., MIDL 2019 / MedIA 2021](https://arxiv.org/abs/1812.07032v4)
- Tversky index T = TP / (TP + a FP + b FN), a+b=1, b>a weights recall for small lesions [prior knowledge] — [Salehi et al. 2017, arXiv 1706.05721](https://arxiv.org/abs/1706.05721)

### Inferences
- Options and formulas (h = sigmoid heatmap, or p normalised):
  1. Soft Dice: L = 1 - 2 sum h M / (sum h + sum M). Size-invariant per lesion (good: small lesions not ignored). Bad for localisation alone: near-zero gradient until overlap exists; tiny lesions (1-3 voxels at 2 mm) give extremely noisy Dice.
  2. Tversky with b>a: favours covering the lesion (recall) — appropriate if downstream point is derived from predicted region.
  3. Voxel BCE vs M: strong per-voxel gradients but dominated by background (64^3 = 262k voxels vs maybe tens of lesion voxels) — use pos_weight or focal BCE, or combine with Dice (the nnU-Net / LesionLocator CE+Dice recipe).
  4. Target shapes T(v) for a heatmap regression/focal loss:
     - Gaussian at centroid, sigma = k*r (DLT-style; k ~ 0.25-0.5 suggested by DLT's 0.25r jitter). Size-aware, peak well defined.
     - Gaussian-blurred mask T = M * G_sigma (normalised to max 1): keeps shape, smooths edges, peak near the "thickest" part, which for non-convex lesions is inside the lesion (unlike the centroid).
     - Distance-transform target T = exp(-d_in(v)... ) or T = D_in(v)/max D_in, where D_in = Euclidean distance to the mask boundary inside the lesion: peak at the medial axis / "pole of inaccessibility" — guaranteed inside M even for crescent/merged shapes. Good if the hit metric is point-in-mask.
     - Plain binary mask as target: no unique peak; soft-argmax of a uniform-on-mask distribution = mask centroid, so with softmax+KL(T||p) and T = M/|M| the expected point equals the centroid. This is a neat equivalence: KL to the normalised mask = cross-entropy H(M/|M|, p) = -(1/|M|) sum_{v in M} log p(v) (arithmetic mean of log-probs in mask; compare Q1, which is log of the sum). The mean-log version forces mass to be spread uniformly over the lesion; the log-sum version only requires total mass inside.
- Recommended practical combo (inference): L = H(T_gauss_or_DT, p) + lambda1 * L_mass + lambda2 * ||x_hat - c||/max(r, r_min) ; optionally aux Dice on a separate seg head (Q7).

### Gaps
- No head-to-head study found comparing Gaussian vs mask vs distance-transform heatmap targets specifically for longitudinal lesion tracking in CT.

## Q3. Distance-transform-based losses: point-to-mask distance, boundary loss, Hausdorff losses, signed distance maps

### Takeaway
Distance maps of the FU mask give smooth, everywhere-informative gradients even when the prediction is far from the lesion — the main weakness of Dice/mass losses. For a point output, the simplest is L = D_out(x_hat) (distance to mask, 0 inside), computed by trilinear interpolation of a precomputed EDT; for a heatmap, Kervadec's boundary loss sum_v phi(v) p(v) is the same idea in expectation.

### Cited Findings
- Boundary loss: distance metric on contours expressed as an integral over regions using the network's softmax outputs; L_B = sum_{q in Omega} phi_G(q) s_theta(q), where phi_G is the signed distance map of the GT (negative inside) [formula: prior knowledge]; combined with generalized Dice, improves Dice by up to 8% and Hausdorff by up to 10% on ISLES and WMH, more stable training; code github.com/LIVIAETS/surface-loss [snippet] — [Kervadec et al.](https://arxiv.org/abs/1812.07032v4); [PMLR MIDL 2019](https://proceedings.mlr.press/v102/kervadec19a.html)
- Hausdorff losses: three differentiable HD estimates (distance transform, morphological erosion, spherical convolution kernels); evaluated on prostate (US, MR), liver and pancreas (CT); reduce HD vs standard losses; best combined with Dice [snippet] — [Karimi & Salcudean, IEEE TMI 2020, arXiv 1904.10030](https://pdf.arxiv.org/pdf/1904.10030). DT-based form: L_HD-DT = (1/|Omega|) sum ( (p - q)^2 * (d_p^alpha + d_q^alpha) ), d = distance maps of prediction and GT boundaries, alpha ~ 2 [prior knowledge].
- Weighted Hausdorff Distance: modified average Hausdorff between predicted probability map and GT point set; no bounding boxes or max object count; used for pupil centre, head and plant centre localisation; plain HD not differentiable wrt FCN output, hence weighting by probabilities [snippet] — [Ribera et al. CVPR 2019](https://mlanthology.org/cvpr/2019/ribera2019cvpr-locating/); [code](https://github.com/HaipengXiong/weighted-hausdorff-loss)
- Overview of distance-transform maps in segmentation CNNs (Ma et al. MIDL 2020, "How distance transform maps boost segmentation CNNs") appeared in search; not read — [typeset](https://typeset.io/pdf/how-distance-transform-maps-boost-segmentation-cnns-an-2coioutg4r.pdf)

### Inferences
- Concrete formulas for this project (D_out(v) = EDT of complement of M in mm, 0 inside; D_in(v) = EDT inside M; SDM phi = D_out - D_in):
  - Point-to-mask: L_pt = D_out(x_hat) via trilinear sampling (torch grid_sample on precomputed D_out). Gradient = -unit normal toward lesion; zero once inside. Add margin variant: L = max(0, phi(x_hat) + m) to push point at least m mm inside.
  - Expected distance (heatmap): L_ed = sum_v p(v) D_out(v) (boundary-loss form with D_out instead of signed phi) — penalises every bit of mass outside proportional to its distance; unlike L_mass, it is linear in distance so far-away false peaks get strong gradient. Signed version sum_v p(v) phi(v) additionally rewards mass deep inside (toward medial axis).
  - Size normalisation: divide by r (or max(r, 5 mm)) so big and small lesions contribute comparably.
  - Merging lesions: EDT on this-lesion-only mask is fine; if FU mask is a merged blob containing other lesions, D_out is 0 over the whole blob — loss cannot tell which part; need instance labels.
- Boundary/HD losses in their segmentation form only matter if a segmentation head exists (Q7). Kervadec recommends ramping boundary-loss weight up from a regional loss (alpha schedule) [prior knowledge].
- Cost: EDT is precomputed once per training sample (scipy.ndimage.distance_transform_edt with sampling=2 mm) — trivial for 64^3; but must be recomputed after spatial augmentation (or augment the EDT with the same transform; rotation/flip/scale commute up to scale factor).

### Gaps
- Karimi's exact quantitative gains not confirmed (summary gave garbled "18-45" figure). No study of distance-based losses for point localisation in longitudinal CT.

## Q4. Point-in-mask hit rate as metric, and differentiable surrogates

### Takeaway
The literature's standard tracking metrics are centre-distance based (CPM@radius, CPM@10mm, CPM@25mm, MED), not point-in-mask; point-in-mask is a natural stricter/shape-aware alternative when FU masks exist. Surrogates: sigmoid of signed distance at x_hat, or probability mass inside mask.

### Cited Findings
- DLT metrics: CPM = % lesions where ||c_gt - c_pred|| < threshold; CPM@Radius uses threshold r = half longest diameter; CPM@10mm uses min(r, 10 mm); MED = mean Euclidean distance (mm), also per-axis MEDX/Y/Z [verified] — [Cai et al.](https://ar5iv.labs.arxiv.org/html/2012.04872)
- LesionLocator tracking metrics: CPM@25 (centres within 25 mm), Dice@25 (Dice on matched lesions), MED, overall Dice; best (prev-mask prompt + synthetic data): Dice 68.31, CPM@25 85.96, Dice@25 79.02, MED 3.12 mm; point prompt 64.33 / 85.40 / 74.61 / 3.89 mm; Hering et al. baseline 58.66 / 82.50 / 69.27 / 4.45; no centre-in-mask criterion defined [verified] — [Rokuss et al.](https://arxiv.org/html/2502.20985)
- autoPET IV Task 2 (longitudinal CT, 300 patients with BL/FU + masks + per-lesion localisation): metrics DSC@last, FPV@last, FNV@last with lesion centre given as prompt in FU; ranking weights 0.5/0.25/0.25 [snippet; page slightly inconsistent about FNV] — [autoPET IV eval](https://autopet-iv.grand-challenge.org/eval/); [challenge](https://autopet-iv.grand-challenge.org/)
- Hering et al. (MIDL 2021) point-prompt pipeline: registration finds corresponding FU region, CNN segments; 125 lesions / 43 melanoma patients; succeeded for 96% BL and 80% FU lesions [snippet] — [PMLR 143](https://proceedings.mlr.press/v143/hering21a.html)

### Inferences
- Report several metrics: (1) hit@mask = 1[x_hat in M] (nearest voxel); (2) hit@dilated mask (M dilated by 1-2 voxels to forgive boundary for tiny lesions); (3) CPM@radius and CPM@10mm (DLT, comparable to literature); (4) MED in mm; (5) normalised error e/r. Note: our centroid can lie outside M for non-convex lesions, so hit@mask(centroid) < 100% — report this "oracle" rate as an upper bound.
- Note autoPET IV uses the GT FU lesion centre as the prompt — i.e., exactly the output our localiser would provide to a downstream segmenter; hit@mask matters because a centre prompt outside the lesion likely breaks the segmenter.
- Differentiable surrogates: S1 = sigmoid(-phi(x_hat)/tau) (soft hit), loss -log S1 (= softplus(phi/tau)); S2 = sum_{v in M} p(v) (expected hit if sampling point ~ p) — this is exactly Q1's loss; S3 = soft-dilated mask via Gaussian blur of M sampled at x_hat.
- Tiny lesions: at 2 mm voxels, a 4 mm lesion is ~2 voxels across; mask-hit is near-binary and dominated by resampling/partial-volume. Use dilated hit or radius floor r_min (~5 mm, cf. DLT's 10 mm cap logic).

### Gaps
- No public benchmark found that reports point-in-mask hit rate for lesion tracking.

## Q5. Target point: centroid vs any point inside; lesion-size-normalised errors

### Takeaway
Literature trackers use a single centre: RECIST-endpoint mean (DLT) or mask centre (LesionLocator CPM); size-normalisation via radius thresholds (CPM@Radius) is standard. With masks, a hybrid is best: regress toward a point guaranteed inside (centroid if inside, else DT maximum), and normalise errors by equivalent radius with a floor.

### Cited Findings
- DLT centre = mean of the four RECIST diameter endpoints; radius = half longest diameter; CPM@Radius normalises acceptance by lesion size, CPM@10mm caps it [verified] — [Cai et al.](https://ar5iv.labs.arxiv.org/html/2012.04872)
- LesionLocator samples point prompts uniformly from the foreground (any point inside identifies the lesion for the segmenter) [verified] — [Rokuss et al.](https://arxiv.org/html/2502.20985)

### Inferences
- Centroid c: unique, smooth target, matches literature metrics; but for crescent, necrotic-ring, or merged lesions c may be outside M; also changes discontinuously under merge (centroid jumps to blob centre).
- "Any point inside": better matches downstream prompting (autoPET, LesionLocator-style), via L_mass / L_ed; ambiguous within lesion.
- Medial point m* = argmax D_in: always inside, robust to shape; can jump between disconnected parts of near-equal thickness.
- Proposed hybrid target t = c if c in erode(M,1) else m*; loss ||x_hat - t||/max(r, r_min) + L_ed/max(r, r_min).
- Normalised error e/r: unstable for tiny lesions (r ~ 2 mm); use max(r, r_min) with r_min ~ 5 mm or report both mm and normalised. Equivalent radius from volume (sphere) vs half-longest-diameter (RECIST, as in DLT) differ for elongated lesions — state which.
- Size-weighted loss: dividing distance by r makes errors on small lesions count more — desirable since a 5 mm error misses a 6 mm lesion but not a 40 mm lesion.

### Gaps
- No ablation found comparing centroid vs inside-point targets for tracking.

## Q6. Growth/shrinkage/merging; BL mask as prompt input (LesionLocator, promptable tracking)

### Takeaway
Feeding the BL mask (or warped BL mask) as an input channel is the current SOTA approach (LesionLocator, autoPET IV winner lineage); growth/shrinkage is handled in training mainly through data (real pairs + synthetic deformation), and merging remains an acknowledged open problem.

### Cited Findings
- LesionLocator: prompts (point ball r=5 px, box expanded by U[0,10] px, or previous-timepoint mask) concatenated with image at first level; tracking: Prompt Propagation Module (GradICON backbone, pretrained) estimates deformation Phi_t = g(I_t, I_t+1) and warps the prompt; autoregressive Y_hat_t+1 = f(I_t+1, Phi_t o Y_hat_t); during training uses GT previous mask, inference uses own predictions [verified] — [Rokuss et al. CVPR 2025](https://arxiv.org/html/2502.20985)
- Using previous-timepoint mask as prompt gives the best tracking (Dice 68.31, CPM@25 85.96, MED 3.12 mm) vs point (64.33/85.40/3.89) or box (66.49/83.95/3.25) [verified] — [Rokuss et al.](https://arxiv.org/html/2502.20985)
- Growth/shrinkage in synthetic longitudinal data: amplitudes {-22,-18,15,25} (negative = shrinkage, positive = growth) with voxel-wise random scaling (-3.5, 3.5) for non-uniform change [verified] — [Rokuss et al.](https://arxiv.org/html/2502.20985)
- Merging: "accurate distinction of closely located or merging lesions remains a challenge"; suggest re-identification with an additional click; competing models "fail to correctly identify vanishing lesions" [verified] — [Rokuss et al.](https://arxiv.org/html/2502.20985)
- Trained on 23,262 annotated scans + synthetic 4D data; released ~700 GB dataset with simulated follow-ups and consistent instance labels; featured in winning autoPET IV (Promptable Longitudinal Lesion Tracking) solution [snippet] — [GitHub MIC-DKFZ/LesionLocator](https://github.com/MIC-DKFZ/LesionLocator); [CVPR paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Rokuss_LesionLocator_Zero-Shot_Universal_Tumor_Segmentation_and_Tracking_in_3D_Whole-Body_CVPR_2025_paper.pdf)
- DLT handles tracking without masks via Siamese appearance + anatomy signals (self-supervised training possible from single time point) [verified] — [Cai et al.](https://arxiv.org/pdf/2012.04872)
- A 2026 DKFZ preprint (arXiv 2605.23118, "Promptable Longitudinal Lesion Segmentation in Whole-Body CT"?) reportedly won autoPET IV with ~60.7 Dice automatic / 73.7 clinician-verified tracking — third-party summary only, unverified — [emergentmind](https://www.emergentmind.com/papers/2605.23118); [DKFZ record](https://inrepo02.dkfz.de/record/313306)
- ULS23 baseline evaluated on longitudinal melanoma CT (Radboudumc, with Hering) — [arXiv 2507.19230](https://arxiv.org/html/2507.19230v1)

### Inferences
- Our BL mask channel is the right design (matches LesionLocator's dense prompt). Loss implications:
  - Growth: mask target scales automatically; Gaussian target with sigma ~ r_FU (not r_BL) adapts; L_mass easier for grown lesions.
  - Shrinkage to tiny/vanished: FU mask may be few voxels or empty — need radius floor, dilated mask for hit, and a rule for empty masks (skip, or add a "lesion absent" logit / null bin in the softmax so -log p(null) is trainable).
  - Merging: if "this lesion only" mask comes from an instance split of a merged blob, the boundary between instances is arbitrary; the centroid of the split part is ill-posed. Consider: accept a hit anywhere in the merged component (mass-in-component) or down-weight merged cases. Report merged cases separately.
  - BL mask also allows size-conditioning: predicted sigma or target scale can be set from BL radius at inference.
- Could augment like LesionLocator: random elastic growth/shrink of FU mask + image is out of scope for a course project; simpler: random scaling of BL prompt mask (dilate/erode) to make the model robust to size change.

### Gaps
- LesionLocator's handling of merges/vanishing at the loss level is not described. The autoPET IV winner details unverified.

## Q7. Auxiliary segmentation head as multi-task regulariser for localisation

### Takeaway
Evidence from medical object detection (Retina U-Net) supports that adding dense segmentation supervision improves detection, especially with small datasets; evidence from MULAN is mixed (segmentation was not shown to help detection there). For this project an auxiliary FU-mask head (CE+Dice, maybe + boundary loss) on the decoder is cheap and likely helpful; it also enables deriving the point from the predicted mask.

### Cited Findings
- Retina U-Net: RetinaNet + U-Net segmentation branch "recaptures discarded supervision signals"; mask-loss gradients in Mask R-CNN only flow from one pyramid level up, Retina U-Net lets segmentation loss reach the whole network; gain "grows in the limit of small data sets"; matches more complex two-stage detectors; 2D and 3D code (medical detection toolkit) [snippet] — [Jaeger et al., ML4H/PMLR 116](https://proceedings.mlr.press/v116/jaeger20a); [arXiv 1811.08661](https://arxiv.org/pdf/1811.08661)
- MULAN (MICCAI 2019, DeepLesion): joint detection/tagging/segmentation on Mask R-CNN; tagging improved detection via score refinement; detection improved tagging but "impaired segmentation"; segmentation trained on pseudo-masks (DeepLesion has no GT masks) — no evidence segmentation helps detection [snippet] — [Yan et al. arXiv 1908.04373](https://arxiv.org/pdf/1908.04373)
- LesionLocator achieves CPM/MED purely from a segmentation model (point derived from predicted mask), MED 3.1-3.9 mm vs DLT-style point trackers ~7 mm on a different dataset (DeepLesion) — not directly comparable [verified] — [Rokuss et al.](https://arxiv.org/html/2502.20985); [Cai et al.](https://ar5iv.labs.arxiv.org/html/2012.04872)
- Boundary loss combined with GDL improves Dice/HD on small, unbalanced lesions (ISLES/WMH) — relevant if an aux seg head is used for small lesions [snippet] — [Kervadec et al.](https://arxiv.org/abs/1812.07032v4)

### Inferences
- Recommended: shared encoder/decoder, two heads: (a) localisation logits -> softmax p -> soft-argmax; (b) seg sigmoid -> L_seg = BCE + Dice (+ lambda*boundary loss, ramped). Total L = L_loc + w*L_seg with w ~ 0.5-1. Ablate w=0 vs w>0.
- Alternative "segment-then-localise": point = centroid (or DT max) of thresholded seg output near the predicted peak — essentially LesionLocator's paradigm; strong baseline to compare against.
- Risk: seg head may dominate on large lesions and under-fit tiny ones; Dice per-sample (not batch) keeps small lesions weighted.
- DLT vs LesionLocator MED numbers come from different datasets/annotations, so they do not prove that segmentation beats point regression.

### Gaps
- No controlled ablation found showing aux segmentation improves point-localisation accuracy specifically for longitudinal lesion tracking. Karimi exact numbers, Tversky/boundary-loss formulas marked prior knowledge not re-read this session.
