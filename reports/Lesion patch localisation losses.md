# Heatmaps, not regression, will find the lesion

Team 2 should stop treating the follow-up (FU) offset as a number to regress and treat it as a position to find. The recommended model is a Siamese 3D encoder whose baseline (BL) lesion template is correlated with every FU position. A softmax turns the correlation map into a probability volume, and a soft-argmax in millimetres turns that into the offset Δ̂. Training uses three normalised terms: `L = ‖Δ̂ − Δ‖/8 mm + 1.0·CE(Gaussian target, p) + 0.5·(−log Σ_{v∈FU mask} p(v))`, with a short heatmap-only warm-up. This is the published recipe of the closest analogues, the Deep Lesion Tracker and the Transformer Lesion Tracker, extended with the FU mask that this project has and they did not. Those trackers reach about **6–7 mm mean error on DeepLesion**, against roughly **12.7 mm** for predicting zero under the new N(0, 8²) jitter. For the team interface, Team 1 should hand over **centres (`centres.csv`), not patches**, because the repo's `preprocess.py` already cuts patches from centres and keeping one cutting path removes a whole class of train/test mismatch. Experiment 1 (14.37 vs 14.51 mm) showed no learned signal. Changing the loss shape on a direct-regression head will not fix that. A spatial output will, and most of the gain should come at the step from regression to correlation plus soft-argmax, before any loss refinements. All loss weights below come from 2D pose and general multi-task papers; no lesion-tracking paper ablates them, so the ablation plan is a necessary part of the recommendation.

## Team 1 should send centres, and Team 2 should match their error distribution

The repo already fixes the interface: Team 1 writes `centres.csv` (`patient, lesion, x_mm, y_mm, z_mm`), and `src/lesionloc/preprocess.py` cuts the 64³ × 2 mm patches around those centres. Keep it that way, for four reasons. First, a centre is the complete output of Team 1's task. Its error `‖true − c‖` is exactly the zero baseline Team 2 must beat, so it can be scored without any patches. Second, one cutting code path means training patches (`--centre fu`) and evaluation patches (from Team 1's centres) share identical resampling, axis order, intensity handling and mask-label logic. That logic includes the merged-lesion label rule and the `skip_outside_window` filter, which drops any lesion further than 56 mm from the centre on an axis. Two teams cutting patches independently is the most likely source of a silent sign or axis bug, which the notes name as a common reason why plain regression "fails" ([TLT, MICCAI 2022](https://zrongguo.github.io/files/2022_MICCAI_Transformer%20Lesion%20Tracker.pdf) on training practice; [DLT](https://arxiv.org/abs/2012.04872) on anisotropic axis errors). Third, Team 2 keeps control of the crop. It can later cut larger FU windows for on-the-fly re-jittering, add context, or change spacing, with no new deliverable from Team 1. Fourth, a CSV is tiny and versionable; a set of patch npz files is neither.

One addition is worth asking for: an optional per-lesion uncertainty column (for example `sd_mm`) or at least Team 1's validation residuals. Team 2 trains on synthetic offsets, so the training jitter should match Team 1's real residual distribution. The current registration guess gives a 14.51 mm mean error on the old patches. If those residuals were isotropic Gaussian, that mean would correspond to σ ≈ 9.1 mm per axis. The real residuals are probably heavier-tailed, so treat that figure only as a rough scale, not a fitted value. If Team 1's final residuals are clearly wider or heavier-tailed than N(0, 8²), Team 2 should widen its jitter, or train on a mixture, before the joint evaluation. The jitter can never exceed the window: the repo clips at ±24 mm and discards true positions beyond 56 mm.

The 14.51 mm figure is not a data error. It was measured on the old `cog_propagated` patches. On the new data, where `target_mm = −jitter_mm` with per-axis N(0, 8²) clipped to ±24 mm, ‖Δ‖ follows an almost exactly Maxwell distribution. A Monte-Carlo check (10⁶ draws, own computation) gives a **mean of 12.75 mm, a median of 12.30 mm, 5.8% within 5 mm, 33% within 10 mm and 90% within 20 mm**, with a per-lesion SD of 5.3 mm. With a few hundred validation lesions, the measured zero baseline should land within about ±0.7 mm of 12.75. Re-measure it on the exact stored validation jitters (step 0 below). Every model must be compared against that number, paired lesion by lesion, not against 14.51.

## Every loss family, and what each one actually optimises

Notation: Δ is the true offset in mm (from `target_mm`, the dataset `cog_fu` minus the patch centre). Δ̂ is the prediction and e = Δ̂ − Δ. M is the FU mask on the output grid. p(v) is the softmax probability of FU position v, and x_v is that position's coordinate in mm relative to the patch centre. r is the FU equivalent radius `(3|M|·8 mm³ / 4π)^{1/3}`. Note that `target_mm` is the CSV centre of gravity, not the mask centroid. Preprocessing warns when the two differ by more than 4 mm (`warn_mask_cog_far`), and the losses should keep Δ as the target, since that is what is evaluated.

### Coordinate losses measure the metric but cannot rescue a blind network

| Loss | Formula | Gradient / minimiser | Use here |
|---|---|---|---|
| MSE | `‖e‖²` | grad 2e, conditional mean; outlier-sensitive, weak near 0 | avoid |
| per-axis L1 | `Σ_i |e_i|` | grad ±1, per-axis median; not rotation-invariant | fine |
| Euclidean | `sqrt(‖e‖² + ε²)` | unit-length grad, geometric median; equals the reported metric | **default L_dist** |
| smooth-L1 / Huber (δ) | `½e²/δ` if `|e|<δ`, else `|e| − ½δ` | quadratic small, linear large | set δ in mm (2–5 mm); PyTorch's `beta=1` is effectively L1 here |
| Wing (w, ε) | `w·ln(1+|e|/ε)` if `|e|<w`, else `|e| − C` | amplifies small errors | late-stage polish, w≈10 mm, ε≈2 mm (extrapolated) |

Across landmark benchmarks the L1 family beats L2. On AFLW, normalised error was **L2 2.41, L1 2.00, smooth-L1 2.02, Wing 1.88** ([Feng et al. 2018](https://arxiv.org/abs/1711.06753)). For soft-argmax outputs, "L1 loss works consistently better than L2 loss" ([Sun et al. 2018](https://arxiv.org/abs/1711.08229)). DSNT uses the non-squared Euclidean distance because it "directly optimizes the metric we are interested in" ([Nibali et al. 2018](https://arxiv.org/abs/1801.07372)). These differences are a few percent. They cannot explain Experiment 1. A network without input-dependent features minimises every one of these losses with a constant, and because Δ is zero-centred that constant is about 0. That is exactly the 14.37 vs 14.51 mm result. The real cause is the global-pool/fully-connected head. Its weights "are highly dependent on the spatial distribution of the inputs during training… [causing] particularly bad performance on small datasets", and FC regression reached only **22% PCKh when trained on 1024 samples** while DSNT and heatmap matching did far better ([Nibali et al. 2018](https://arxiv.org/abs/1801.07372)). CoordConv's toy experiments show that ordinary convolution "does not lead to coordinate transform generalization even to neighboring pixels" ([Liu et al. 2018](https://arxiv.org/abs/1807.03247)). Compute the loss in mm but divide it by 8 mm (the jitter SD), so the term is O(1) and Huber/Wing thresholds have a known meaning.

### Soft-argmax makes the coordinate loss spatial

Soft-argmax (integral regression, DSNT) computes `Δ̂ = Σ_v softmax(β·z)(v) · x_v` over a fixed grid in mm. It is differentiable, sub-voxel and parameter-free, and it keeps the network fully convolutional: shifting the FU patch shifts the response. Its known flaw is bias. Softmax mass on far voxels pulls the expectation toward the map centre, more so when the map is diffuse ([Gu et al. 2023, BCIR](https://arxiv.org/abs/2301.10431)). In this task the map centre is Δ = 0, so the bias shrinks predictions toward the zero baseline. It reduces the gain but does not cause wild errors. The fixes are a learnable temperature β, a heatmap regulariser, and, during early training, a warm-up so the distance gradient does not act on a uniform map. Multimodal maps are the second risk: two peaks average to a point between them, possibly outside the lesion. TLT's variant takes a softmax-weighted average of per-voxel coordinate predictions. Trained with L1 plus a focal heatmap loss, it reached **6.0 mm MED and CPM@10mm 87.4%** on DeepLesion ([TLT](https://zrongguo.github.io/files/2022_MICCAI_Transformer%20Lesion%20Tracker.pdf)).

### Heatmap losses supply the dense supervision small data needs

Dense heatmap supervision is easier to learn than coordinate regression "because the latter [heatmap] is supervised by dense pixel information" ([Sun et al. 2018](https://arxiv.org/abs/1711.08229)). The options differ mainly in how they handle the extreme imbalance of 3D maps. With σ = 2 voxels, only about 0.1% of a 64³ grid has a Gaussian target above 0.5, against 1.2% foreground in the 2D case that motivated Adaptive Wing ([Wang et al. 2019](https://arxiv.org/abs/1904.07399)).

| Loss | Formula | Pros | Cons |
|---|---|---|---|
| MSE to Gaussian (SCN) | `Σ_v (h(v) − γ·G_σ(v))²`, optional learned σ with `+α σ²` | medical default; learned σ encodes uncertainty ([Thaler et al. 2021](https://arxiv.org/abs/2109.09533)) | background-dominated; "an optimization step which improves MSE loss" need not improve localisation ([Nibali](https://arxiv.org/abs/1801.07372)); nnLandmark reports MSE far worse than BCE-TopK ([arXiv 2504.06742](https://arxiv.org/abs/2504.06742)) |
| Softmax CE / KL to normalised Gaussian | `−Σ_v q(v) log p(v)`, `q = G_σ/ΣG_σ` | one answer per patch; per-sample so no imbalance; O(1) scale; pairs with soft-argmax | σ choice; floor = entropy of q (harmless) |
| DSNT JS regulariser | `JS(p ‖ N(Δ, σ_t²))`, σ_t ≈ 1 cell, λ = 1 | "marked performance improvements"; robust to λ ([Nibali](https://arxiv.org/abs/1801.07372)) | similar effect to CE; slightly more code |
| Penalty-reduced focal (CenterNet, DLT) | `−Σ (1−ŷ)^α log ŷ` at peak; `(1−y)^β ŷ^α log(1−ŷ)` elsewhere; α=2, β=4; σ = lesion radius | the published 3D lesion-tracking loss ([DLT](https://arxiv.org/abs/2012.04872)) | per-voxel sigmoid allows multiple peaks; must normalise by positives |
| Adaptive Wing + weighted map | intensity-adaptive Wing, W=10 on dilated foreground | handles imbalance in MSE form | 2D-tuned (α=2.1, ω=14, ε=1, θ=0.5), unvalidated in 3D |
| Balanced logistic (SiamFC) | `log(1+exp(−y·v))`, y=+1 within radius R | matches a hit-within-radius metric | coarse; no sub-voxel signal ([Bertinetto et al. 2016](https://arxiv.org/abs/1606.09549)) |

For a single lesion per patch, the softmax-CE form fits the problem best. It enforces exactly one answer, it is naturally normalised, and it shares the softmax with the soft-argmax readout. The focal loss is the DLT-replicating alternative. DLT used focal-only supervision with hard argmax and reached **MED 7.0 ± 8.9 mm and CPM@10mm 78.85%** on 3,008 training pairs, beating 3D SiamRPN++ (8.3 mm) and DEEDS registration (7.4 mm) ([DLT](https://arxiv.org/abs/2012.04872)). That training set is similar in size to ours. No paper compares CE and focal losses head-to-head on 3D lesions, so this choice belongs in the ablation plan. The heatmap's resolution matters. At 32³ (4 mm cells), argmax would carry about 1.7 mm RMS quantisation error (own estimate), while soft-argmax stays sub-voxel. DSNT is "largely robust to heatmap size" ([Nibali](https://arxiv.org/abs/1801.07372)).

### Mask losses encode the lesion's size and shape, which no published tracker exploited

DLT and TLT had only RECIST marks. This project has the FU mask, and LesionLocator showed that a full mask prompt beats points: on whole-body melanoma CT it achieved **MED 3.12 mm with the previous mask vs 3.89 mm with a point**, training only Dice+CE ([Rokuss et al. 2025](https://arxiv.org/html/2502.20985)). The candidate mask losses on a softmax map are the following.

The **mass-in-mask (MIL) loss** is `L_mass = −log(Σ_{v∈M} p(v) + ε)`. It rewards putting all the probability inside the lesion, with no preference within it. It is scale-invariant, it matches a hit-in-mask metric, and it tolerates an ambiguous centroid. On its own it allows mass to spread across a large lesion, so it needs a centroid or concentration term next to it. The closest precedent is MIL box supervision ([Hsu et al. 2019](https://papers.neurips.cc/paper_files/paper/2019/hash/e6e713296627dff6475085cc6a224464-Abstract.html)). No lesion tracker trains with this exact loss, so its benefit is untested.

The **mask-as-target CE** is `−(1/|M|) Σ_{v∈M} log p(v)`, which is KL to M/|M|. Its optimum places mass uniformly over the lesion, so the soft-argmax lands on the mask centroid. That is neat, but it forces spread, and spread worsens the soft-argmax bias.

The **expected distance loss** is `L_ed = Σ_v p(v)·D_out(v)/max(r, 5 mm)`, where D_out is the Euclidean distance transform (EDT) to the mask, in mm and 0 inside. This is Kervadec's boundary loss with the inside part dropped ([Kervadec et al.](https://arxiv.org/abs/1812.07032v4)). Unlike L_mass, it grows linearly with distance, so a far false peak gets a strong gradient. It requires a precomputed EDT per sample (`scipy.ndimage.distance_transform_edt`, `sampling=2`).

The **point-to-mask distance** samples D_out at Δ̂ with trilinear interpolation, `L_pt = D_out(Δ̂)`. Its margin variant is `max(0, φ(Δ̂) + m)`, where φ is the signed distance (negative inside). It is a direct surrogate of the hit-in-mask metric, but its gradient is zero once the point is inside. The soft-hit variant `softplus(φ(Δ̂)/τ)` is smoother.

The **auxiliary segmentation head** adds a second decoder output trained with per-sample BCE + soft Dice on M. It passes the most dense signal to the encoder, and Retina U-Net reports that such gains grow "in the limit of small data sets" ([Jaeger et al.](https://proceedings.mlr.press/v116/jaeger20a)). MULAN found no detection gain from segmentation ([Yan et al.](https://arxiv.org/pdf/1908.04373)), so the evidence is mixed. It costs a decoder.

Two data facts limit all mask losses. First, for `MERGING` lesions `preprocess.py` uses the label the lesion merged into, so M is the merged blob. The losses then cannot tell which part is "this" lesion, and those cases should be reported separately. Second, empty masks are already skipped, so a "vanished lesion" class is not needed. Tiny lesions (2–3 voxels at 2 mm) make mask terms near-binary. Use a 1-cell dilation and a radius floor of 5 mm.

### Probabilistic losses add calibrated confidence

A per-axis Laplace NLL head predicts a scale b_i = exp(s_i) with `L = Σ_i (|e_i|/b_i + log b_i)`, clamping s to [log 0.5, log 50] mm. On COCO, a learnable-σ Laplace head scored **67.4 AP vs 58.1 for plain L1**. The full RLE flow reached 70.5, and the Laplace forms beat the Gaussian ones ([Li et al. 2021, RLE](https://arxiv.org/abs/2107.11291)). The risk on small data is that a network without signal "explains away" every case with a large σ. Gaussian NLL also down-weights hard samples until the mean fit stalls. The fix is β-NLL, which weights each sample by stop-grad(σ²)^β with β = 0.5 ([Seitzer et al. 2022](https://arxiv.org/abs/2203.09168)), or adding σ only after the mean beats the baseline. Its payoff is flagging: predicted σ correlated with correctness at Pearson ≈ 0.52–0.55 in RLE, a useful signal for the joint pipeline. The heatmap's own spread offers a free alternative ([Nibali](https://arxiv.org/abs/1801.07372)). Homoscedastic uncertainty weighting, `L_k/(2σ_k²) + log σ_k` per term ([Kendall et al. 2018](https://arxiv.org/abs/1705.07115)), is the same idea applied to the λ values rather than to the samples.

## The proposed model: correlation, soft-argmax and three normalised terms

**Architecture.** The BL input has 2 channels (CT + BL mask), and the FU input has 1 channel (CT, with a zero mask channel if the encoder is shared). Use a shared 3D residual encoder of about 1–5M parameters with GroupNorm, at stride 2 (32³ map, 4 mm cells). TLT deliberately halved its channels "considering the parameter redundancy and overfitting in 3D networks" ([TLT](https://zrongguo.github.io/files/2022_MICCAI_Transformer%20Lesion%20Tracker.pdf)). Build two templates from BL features and add their correlation maps, as DLT combines its K and K_g kernels. The first is mask-pooled: the average of the features inside the BL mask, giving a 1×1×1 kernel that is size-invariant. The second is a centre crop of 3³–5³ cells, for context. Add a learnable temperature β (initialised at 1, computed in fp32) and an optional fixed log-Gaussian centre prior over ±24 mm, a cheap analogue of DLT's anatomy signal, which "w/o ASE" ablations show matters ([DLT](https://arxiv.org/abs/2012.04872)). The output is a softmax p over 32³, and Δ̂ = Σ p(v)·x_v with x_v in mm. The template design and the centre prior are own inferences and are not ablated in the literature.

**Loss.**

```
L = w(t) · L_dist + λ_hm · L_hm + λ_mask · L_mask

L_dist = sqrt(‖Δ̂ − Δ‖² + 0.1²) / 8 mm
L_hm   = −Σ_v q(v) log p(v),   q = normalised Gaussian at Δ,  σ = clip(0.5·r_FU, 4 mm, 12 mm)
L_mask = −log( Σ_{v ∈ M↓} p(v) + 1e−6 ),   M↓ = FU mask max-pooled to 32³, dilated 1 cell if r_FU < 5 mm
w(t)   = 0 for epochs 0–4, linear ramp to 1 over epochs 5–9, then 1
λ_hm = 1.0,  λ_mask = 0.5
```

**Why these weights.** Each term is O(1) by construction: L_dist is in units of the jitter SD, CE is per-sample, and L_mass is a log-probability. This follows the evidence that tuned fixed weights match adaptive schemes such as GradNorm and DWA ([Kurin et al. 2022](https://arxiv.org/abs/2201.04122); [Xin et al. 2022](https://arxiv.org/abs/2209.11379)). It also matches every close tracker: TLT, LesionLocator and the 2026 autoPET IV winner all use plain sums ([arXiv 2605.23118](https://arxiv.org/html/2605.23118)). λ_hm = 1 is the TLT setting. λ_mask = 0.5 is lower because L_mass overlaps with L_hm for compact lesions. The warm-up keeps L_dist from pulling on a near-uniform map, where soft-argmax is biased toward 0. The σ floor of 4 mm (one cell) follows DSNT's 0.5–1 px recommendation for σ_t. The size-scaling follows DLT's radius-tied Gaussian. The floor and the scaling factor are own choices.

**Training defaults.** AdamW, lr 3e-4, weight decay 1e-4, cosine schedule with 5 warm-up epochs, about 150 epochs, batch 16, AMP with the softmax and soft-argmax in fp32, early stopping on validation mean error. These are inferences from TLT (Adam 1e-4, batch 4, 300 epochs). Augmentation: left-right flip of BL, BL mask, FU and FU mask together, negating Δ_x; independent intensity jitter per timepoint; BL↔FU swap with Δ negated, which doubles the pairs as in TLT. A sanity check comes first: overfit 10 pairs to about 0 mm before any real run.

**Re-drawing jitter is the main deferred improvement.** The current data store one fixed jitter per lesion, chosen for simplicity. The network therefore sees each lesion at exactly one offset, which invites memorisation. Re-drawing the offset each epoch gives unlimited, exactly labelled augmentation that targets translation equivariance directly. The cleanest implementation is to have `preprocess.py` cut a larger FU patch (for example 112³) centred on the true FU lesion, then crop the 64³ window at a fresh N(0, 8²) offset inside the dataloader. Validation and test must keep the fixed stored jitters, so that model and zero baseline see identical inputs.

## Ablation plan: one change per run, from zero to the full model

The first six steps follow `docs/wiki/Teams.md`. Each later step changes one thing relative to the best configuration so far. All runs use the same jittered validation set and are logged with their λ values.

| # | Change | What it tests |
|---|---|---|
| 0 | Predict zero; predict train mean, on the new jittered data | Re-measured bar (expect ≈12.7 mm mean, ≈12.3 mm median) |
| 1 | Experiment 1 direct regression (Euclidean / smooth-L1), retrained on the new data | Whether the FC head learns anything once the target is pure jitter |
| 2 | Siamese correlation + softmax + soft-argmax, `L = L_dist` only | Spatial output vs FC head, the step expected to carry most of the gain |
| 3 | + L_hm (CE to Gaussian, λ_hm = 1) with the 5+5-epoch warm-up | Dense supervision |
| 4 | + BL mask input channel (mask-pooled template enabled) | Which lesion to follow, and its size |
| 5 | + L_mask (mass-in-mask, λ_mask = 0.5) | FU shape supervision |
| 6 | λ sweep: λ_hm ∈ {0.3, 1, 3}, λ_mask ∈ {0.25, 0.5, 1}, short runs | Weight sensitivity |
| 7 | L_hm: CE → penalty-reduced focal (α=2, β=4, σ = r_FU) | DLT's loss vs softmax CE |
| 8 | L_mask: mass-in-mask → expected distance L_ed | Distance-graded mask loss |
| 9 | Per-epoch re-drawn jitter (larger stored FU patch) | Fixed vs fresh offsets |
| 10 | + flip and BL↔FU swap augmentation | Data multiplication |
| 11 | + per-axis Laplace NLL head (β-NLL, β = 0.5) | Confidence for flagging; accuracy change |
| 12 | + auxiliary FU segmentation head (BCE + Dice, weight 0.5) | Dense multi-task regulariser |
| 13 | Pretrained encoder (e.g. STU-Net or nnU-Net ResEnc) at 10× lower lr | Transfer to a 2.5k-pair regime |

Step 9 is a candidate to move earlier if steps 2–5 show a large train/validation gap. Any alternative to this order (for example Kendall uncertainty weighting instead of step 6) should still be one change per run.

## Metrics that make the comparison honest

Report each run as a paired comparison against the step 0 zero baseline on identical validation offsets. Include: mean ± SD, median and 90th-percentile Euclidean error in mm; the success rate within 5, 10 and 20 mm; CPM@Radius and CPM@10mm, with the threshold `min(r, 10 mm)`, for comparability with DLT/TLT ([DLT](https://arxiv.org/abs/2012.04872)); and the hit rate of the predicted point inside the FU mask and inside the mask dilated by one voxel. Report the mask hit rate of the true centre as an oracle ceiling, because centroids of non-convex lesions fall outside the mask. Add per-axis MAE and the percent improvement `(E_zero − E_model)/E_zero`, with a patient-level bootstrap 95% CI on the paired per-lesion difference. Break results down by lesion-size tercile and by topology class, keeping `MERGING` separate. For runs with a σ head, add the Spearman correlation of σ with error and an error-versus-coverage curve. For the final joint evaluation, report `‖c_Team1 + Δ̂ − true‖` against `cog_propagated` alone. That evaluation is also where a mismatch between Team 1's residuals and the training jitter would show up.

## Conclusion

The project's failure so far was architectural, not a matter of loss choice. Every loss in the catalogue collapses to "predict zero" when the head cannot represent position. That reorders priorities. The large expected gain is in steps 2–3, where the output becomes a map. The mask terms are refinements whose value no published tracker has measured, and this project is unusually well placed to measure it, because it has instance FU masks that DLT and TLT lacked. The literature's 6–7 mm (DeepLesion) and 3–5 mm (segmentation-based, melanoma CT) figures come from other datasets and protocols. They indicate the plausible range, not a target this setup is guaranteed to reach.

The interface decision and the training-data decision are the same decision. Because Team 1 sends centres and Team 2 synthesises its own offsets, the dangerous gap is a mismatch between N(0, 8²) and Team 1's real residuals, not the model. Once Team 1 has validation residuals, the cheapest high-value experiment is to match the training jitter to them. Re-drawing that jitter every epoch makes this a one-line change instead of a new preprocessing run.
