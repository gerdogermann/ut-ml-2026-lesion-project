# Coordinate / distance-based loss functions for single 3D point (lesion centre) prediction

Project framing used throughout: input = BL CT patch 64^3 @ 2 mm (128 mm cube) + BL lesion mask + FU patch of same size whose centre is offset from the true FU lesion centroid by Δ ~ roughly N(0, 8^2 mm^2) per axis (clipped near ±24 mm). Target = Δ in mm. ~2500 lesion pairs, ~224 training patients. Prior result: direct-regression CNN with Euclidean / smooth-L1 loss gave val mean error 14.4 mm vs 14.5 mm for predicting zero.

Source dates: CoordConv (NeurIPS 2018), DSNT (arXiv Jan 2018), Integral Human Pose (ECCV 2018), Wing (CVPR 2018), Adaptive Wing (ICCV 2019), Kendall & Gal (NeurIPS 2017), SCN (MedIA 2019), Noothout et al. (IEEE TMI 2020), Deep Lesion Tracker (CVPR 2021), RLE (ICCV 2021), β-NLL (ICLR 2022), BCIR (TPAMI 2023), nnLandmark (MIDL 2026). Current date 2026-10-29.

## Q1. Point losses: L1, L2/MSE, Euclidean distance, smooth-L1/Huber, Wing, Adaptive Wing — gradients, robustness, small vs large error

### Takeaway
For direct coordinate regression, L1-type losses (L1, smooth-L1, Wing) consistently beat L2/MSE in landmark papers; L2 is outlier-sensitive and its gradient vanishes for small errors. Wing amplifies small/medium errors and is the best of these for direct regression, but loss choice gives a few-percent gain only — it cannot rescue a model that has collapsed to the mean (which is what "14.4 vs 14.5 mm" indicates).

### Cited Findings
- Definitions used in the Wing paper: loss summed over coordinates, `loss(s,s') = Σ_i f(s_i − s'_i)`; L1: `f(x)=|x|`; L2: `f(x)=½x²`; smooth-L1: `½x²` if `|x|<1`, else `|x|−½` (quadratic small, linear large; a special case of Huber). — [Feng et al. 2018, Wing loss, arXiv:1711.06753](https://arxiv.org/abs/1711.06753)
- Gradient analysis: L1 gradient magnitude is constant (1), L2 gradient is |x|; "in both cases the update towards the solution will be dominated by larger errors… it is hard to correct relatively small displacements." A log-shaped loss has gradient 1/x, increasing near zero, restoring influence of small errors. — [Feng et al. 2018](https://arxiv.org/abs/1711.06753)
- Wing loss: `wing(x) = w·ln(1+|x|/ε)` if `|x|<w`, else `|x| − C`, with `C = w − w·ln(1+w/ε)`. w sets the non-linear range (−w, w), ε limits curvature; ε must not be too small or training becomes unstable / gradients explode at small errors. — [Feng et al. 2018](https://arxiv.org/abs/1711.06753)
- Empirical (AFLW, CNN-6, normalised error ×10⁻²): L2 2.41, L1 2.00, smooth-L1 2.02, Wing 1.88. Best Wing settings around w=10, ε=2 (1.88); grid range 1.88–2.02 — fairly insensitive. Note the units: AFLW error normalised by face size, w in pixels. — [Feng et al. 2018](https://arxiv.org/abs/1711.06753)
- L2 is "well-known… sensitive to outliers", which is why Fast R-CNN (Girshick) and others use smooth-L1. — [Feng et al. 2018](https://arxiv.org/abs/1711.06753)
- Wing has a gradient discontinuity at 0 and large gradient magnitude around it; this makes it unusable for heatmap (per-pixel) regression — "training… with the Wing loss is never able to converge" — because small errors on many background pixels dominate. — [Wang et al. 2019, Adaptive Wing, arXiv:1904.07399](https://arxiv.org/abs/1904.07399)
- Adaptive Wing: `AWing(y,ŷ) = ω·ln(1 + |(y−ŷ)/ε|^(α−y))` if `|y−ŷ|<θ`, else `A|y−ŷ| − C`. Exponent depends on target pixel intensity y∈[0,1]: behaves like MSE for background (y=0), like Wing for foreground (y=1), and is smooth at zero error. Designed for heatmaps, used with a Weighted Loss Map emphasising foreground and hard background pixels. — [Wang et al. 2019](https://arxiv.org/abs/1904.07399)
- Integral regression paper: for coordinate loss after soft-argmax, "L1 loss works consistently better than L2 loss". — [Sun et al. 2018, Integral Human Pose Regression, arXiv:1711.08229](https://arxiv.org/abs/1711.08229)
- Probabilistic reading: L2 = Gaussian NLL with constant σ, L1 = Laplace NLL with constant σ. On COCO with ResNet-50+FC, constant-variance Gaussian (ℓ2) 36.6 AP vs constant-variance Laplace (ℓ1) 58.1 AP — a huge gap in favour of L1. — [Li et al. 2021, RLE, arXiv:2107.11291](https://arxiv.org/abs/2107.11291)
- DSNT uses non-squared Euclidean distance `L_euc(μ,p) = ||p − μ||₂` because it "directly optimizes the metric we are interested in". — [Nibali et al. 2018, DSNT, arXiv:1801.07372](https://arxiv.org/abs/1801.07372)
- Noothout et al. regress displacement vectors with mean absolute error; a log-transform of the displacement further improved localisation because it penalises errors for patches near the landmark more heavily than far ones. — [Noothout et al. 2020, arXiv:2007.05295](https://arxiv.org/abs/2007.05295)

### Inferences
- Gradients w.r.t. prediction vector e = Δ̂ − Δ (my derivation, standard calculus):
  - MSE `||e||²`: ∇ = 2e. Vanishes as e→0, grows with outliers. Minimiser of expected loss = conditional mean.
  - Per-axis L1 `Σ|e_i|`: ∇ = sign(e) (each axis ±1). Minimiser = per-axis conditional median. Not rotation-invariant.
  - Euclidean `||e||₂`: ∇ = e/||e|| (unit vector, magnitude 1 regardless of error size; undefined at 0, so add ε: `sqrt(||e||²+ε²)`). Minimiser = geometric median. Rotation-invariant; matches the reported metric exactly.
  - Smooth-L1/Huber with threshold δ: ∇ = e/δ inside, sign(e) outside. With mm targets and δ=1 (PyTorch default `beta=1.0`), the quadratic zone is only ±1 mm, i.e. effectively L1 for this task. Choose δ in mm consciously (e.g. 2–5 mm ≈ 1–2.5 voxels) or normalise targets.
  - Wing: ∇ = w/(ε+|x|)·sign(x) inside |x|<w — up to w/ε (=5 for w=10, ε=2) times larger than L1 near zero. Parameters are in target units; Wing's w=10 was in pixels on ~face-size crops, so for mm targets something like w≈10 mm, ε≈2 mm is a reasonable starting point (not tested in literature for this task).
- Why the previous model hit "predict-zero" performance: with an uninformative representation, every one of these losses is minimised by a constant (mean/median ≈ 0 because Δ is zero-centred). So 14.4 vs 14.5 means the network learned essentially no input-dependent signal — a representation/architecture problem (see Q2), not a loss-shape problem. Switching L2→Wing will not fix that on its own.
- Recommended direct-regression loss when used: Euclidean (with ε) or per-axis L1/smooth-L1 on targets normalised by 8 mm (the offset sd), so δ≈0.25–0.5 in normalised units.

### Gaps
- No found paper compares Wing vs L1 specifically for 3D CT point regression in mm; parameter transfer to mm is my extrapolation.
- Barron's general robust loss (CVPR 2019) covers L2/Charbonnier/Cauchy as one family but I did not retrieve it in this session.

## Q2. Why direct coordinate regression from a CNN often fails, and fixes

### Takeaway
Global-pool/FC heads must learn a non-translation-equivariant mapping from features to numbers; CNNs are poor at this "coordinate transform" (CoordConv) and FC heads do not share weights across positions, so they generalise poorly on small data (DSNT). Fixes: keep the output spatial (heatmap + soft-argmax/DSNT, or heatmap + argmax), add coordinate channels (CoordConv), use per-voxel displacement voting (Noothout), or use similarity/correlation between BL template and FU search volume (Deep Lesion Tracker).

### Cited Findings
- FC coordinate heads: "the weights of the fully-connected layer are highly dependent on the spatial distribution of the inputs during training… although the convolutional part of the model is spatially invariant, the model as a whole will not generalize well… This is an inefficient usage of the training data, and causes particularly bad performance on small datasets." — [Nibali et al. 2018](https://arxiv.org/abs/1801.07372)
- DSNT property table: heatmap matching — spatially generalisable but not fully differentiable and poor at low resolution; FC — differentiable but not spatially generalisable and has parameters; DSNT — all desirable properties, no parameters. — [Nibali et al. 2018](https://arxiv.org/abs/1801.07372)
- On MPII with ResNet-34, DSNT beats heatmap matching by 90.5% (relative) at 7×7 heatmaps and by 2.0% at 56×56; heatmap matching only reached 44% PCKh at 7×7 due to argmax quantisation. — [Nibali et al. 2018](https://arxiv.org/abs/1801.07372)
- CoordConv toy result: supervised coordinate regression (one-hot pixel image → (x,y)): a conv+FC net (85k params) fits uniform split (<0.5 px error) "but that same architecture completely fails on the quadrant split"; a fully convolutional net (12k params) reached ~5 px error on quadrant split but was poor on uniform; a 900-parameter CoordConv model generalises on both. "The translation invariance in ordinary convolution does not lead to coordinate transform generalization even to neighboring pixels." — [Liu et al. 2018, CoordConv, arXiv:1807.03247](https://arxiv.org/abs/1807.03247)
- CoordConv mechanism: add hard-coded input channels containing the coordinates of each position, letting the network learn full translation invariance or any degree of dependence; reported e.g. 24% better IoU for Faster R-CNN on MNIST detection. — [Liu et al. 2018](https://arxiv.org/abs/1807.03247)
- Adaptive Wing adds CoordConv (incl. boundary coordinates) to heatmap networks for face alignment. — [Wang et al. 2019](https://arxiv.org/abs/1904.07399)
- Integral regression rationale: "regression learning is more difficult than heat map learning, because the latter is supervised by dense pixel information." — [Sun et al. 2018](https://arxiv.org/abs/1711.08229)
- BCIR: direct regression methods "are less effective than those which locate joints by estimating a likelihood heatmap… working with heatmaps allows neural networks to keep a fully convolutional architecture and therefore retain critical spatial structures." — [Gu et al. 2023, BCIR, arXiv:2301.10431](https://arxiv.org/abs/2301.10431)
- RLE shows direct regression can be competitive if the loss is a learned likelihood: ResNet-50+FC with RLE gains +12.4 mAP over the same model with L1 on COCO, reaching 71.3 AP vs 71.0 for heatmap SimplePose (with heatmap-pretrained backbone). — [Li et al. 2021](https://arxiv.org/abs/2107.11291)
- Medical 3D displacement voting: FCNN predicts, for each patch, a displacement vector to the landmark (regression) plus a landmark-presence probability (classification); final location = probability-weighted average of the voted positions. Joint regression+classification beat both regression-only and classification-only on 3D CCTA; regression-only was hurt by inaccurate votes from far patches. — [Noothout et al. 2020](https://arxiv.org/abs/2007.05295)
- Heatmap regression is "the current state-of-the-art formulation" for 3D medical landmark detection; nnLandmark uses an EDT-shaped heatmap (radius 15 voxels), sigmoid output, BCE-TopK20 loss, argmax decoding. Ablation: MSE heatmap loss gave 12.06 / 52.48 mm MRE vs ~1.6 / ~2.9 mm with BCE-TopK on two datasets — loss choice on heatmaps matters a lot. — [nnLandmark, arXiv:2504.06742, MIDL 2026](https://arxiv.org/abs/2504.06742)
- SCN (Payer et al.) splits heatmap prediction into a local-appearance and a spatial-configuration component multiplied together; it greatly outperforms a localisation U-Net when training images are reduced to 100/50/10 — evidence that heatmap designs with structural priors help on small medical datasets. — [Payer et al. 2019, arXiv:1908.00748](https://arxiv.org/abs/1908.00748)
- Closest task analogue: Deep Lesion Tracker (DeepLesion, 3891 lesion pairs) tracks a BL lesion into FU CT using a Siamese template-vs-search formulation with an anatomical signal encoder; mean centre error ~7 mm, 5% better than DEEDS registration and 14× faster; 89% matching accuracy on 480 test pairs; robust to inaccurate initial centre (−1.9% accuracy vs −10% for SiamRPN++/DEEDS). — [Cai et al. 2021, arXiv:2012.04872](https://arxiv.org/abs/2012.04872)

### Inferences
- For this project the natural fully-convolutional formulation is: produce a 3D score volume over FU-patch positions (e.g. 32^3 or 16^3) and read out Δ by soft-argmax (DSNT/integral) or argmax; or correlate a BL-lesion embedding against FU feature map (DLT-style cross-correlation) and soft-argmax the correlation map. Both keep translation equivariance: shifting the FU patch shifts the response map.
- Because offsets are synthetic (FU patch centre = true centre + random shift), training data can be regenerated with new random shifts every epoch. This is a free, unlimited augmentation that directly targets equivariance; with fixed pre-extracted offsets the network sees each lesion at one offset only and memorises.
- If keeping a global-pool head, add CoordConv channels (3 channels of normalised x,y,z in the FU patch) at the input and/or before the head — cheap and directly addresses the coordinate-transform failure.

### Gaps
- No found ablation of FC vs soft-argmax specifically for 3D CT lesion re-localisation with ~2500 pairs; DLT's loss details (it trains on heatmap-like similarity maps) were not retrieved in this session.

## Q3. Soft-argmax / integral regression / DSNT / spatial softmax: border bias, temperature β, combination with distance loss

### Takeaway
Soft-argmax = expectation of coordinates under a softmax-normalised heatmap; differentiable, sub-voxel, works at low resolution, and can be trained with only a coordinate loss (L1 or Euclidean). Pitfalls: (i) softmax mass on all voxels pulls the expectation toward the map centre (bias), more so for diffuse maps and targets far from centre; (ii) unconstrained heatmap shape → weak supervision, slow convergence; fixes are a heatmap regulariser (JS divergence to a Gaussian, variance reg.), learnable/large β, bias compensation, or sampling-argmax.

### Cited Findings
- Integral regression: `J = Σ_p p·H̃(p)`, `H̃(p) = exp(H(p)) / Σ_q exp(H(q))`; native 3D form sums over D×H×W. Trained with L1 on J. Can be trained with or without an extra heatmap loss: for 2D the variant with heatmap loss (I1) was best, for 3D the coordinate-only variant (I*) was best. More robust to heatmap resolution than argmax. — [Sun et al. 2018](https://arxiv.org/abs/1711.08229)
- DSNT: input is a normalised heatmap Ẑ (non-negative, sums to 1) so predictions always lie within the heatmap extent; output = Ẑ·X and Ẑ·Y with fixed coordinate grids. Softmax worked best among rectification functions. Loss: `L = ||p − DSNT(Ẑ)||₂ + λ·L_reg(Ẑ)`. — [Nibali et al. 2018](https://arxiv.org/abs/1801.07372)
- DSNT regularisation: heatmap spread does not affect output, so many heatmaps give the same coordinate; adding a regulariser yields "marked performance improvements". Options: variance regularisation, or divergence to a target Gaussian; Jensen–Shannon with σ_t=1 px and λ=1 performed best. — [Nibali et al. 2018](https://arxiv.org/abs/1801.07372)
- Heatmap MSE vs coordinate metric mismatch: improving pixel MSE can worsen argmax localisation (DSNT Fig. 4). — [Nibali et al. 2018](https://arxiv.org/abs/1801.07372)
- Bias: combining softmax with expectation induces a bias — "heatmaps with large standard deviations will shift the prediction from the localized center toward the center of the image"; the further the target from the centre the greater the bias. Only extremely concentrated heatmaps are unbiased, so integral regression learns degenerately peaked heatmaps and converges slower than detection. — [Gu et al. 2023, BCIR](https://arxiv.org/abs/2301.10431)
- Temperature β in softmax(β·H): small β spreads mass (more bias from zero pixels), large β concentrates mass, approaching argmax as β→∞ but gradients of far pixels vanish — a trade-off between bias and learning signal. — [Gu et al. 2023](https://arxiv.org/abs/2301.10431)
- BCIR compensation (2D, map size h×w, normaliser C(β)=Σ exp(β·Ĥ)): `Ĵ_ro = C/(C − wh)·Ĵ_re − [hw²/(2(C−wh)), h²w/(2(C−wh))]ᵀ`, plus a Gaussian prior loss on the heatmap for faster training; L1 applied to compensated coordinate. — [Gu et al. 2023](https://arxiv.org/abs/2301.10431)
- Sampling-argmax: soft-argmax leaves the probability map shape unconstrained, causing degradation; sampling-argmax minimises the expected localisation error under the output distribution via differentiable sampling, as a drop-in replacement. — [Li et al. 2021, Sampling-Argmax, arXiv:2110.08825](https://arxiv.org/abs/2110.08825)
- Integral Pose on COCO: 63.0–63.4 AP vs 71.0 for heatmap SimplePose — integral regression underperforms well-tuned heatmap detection on hard 2D benchmarks. — [Li et al. 2021, RLE tables](https://arxiv.org/abs/2107.11291)
- Adding an auxiliary heatmap loss to direct regression "barely brings performance improvements" (RLE appendix, COCO). — [Li et al. 2021](https://arxiv.org/abs/2107.11291)

### Inferences
- Geometry of this task makes the border bias benign-but-real: true centres lie within ~±24 mm of a ±64 mm patch, so targets are never near the border; the softmax bias pulls predictions toward patch centre = toward Δ=0, i.e. shrinkage toward the zero baseline. It reduces gains rather than causing gross errors. Mitigate by (a) a learnable β (init ~1, or compute softmax on logits with a learned scale), (b) JS/Gaussian regulariser with σ_t ≈ 1–2 voxels at the heatmap resolution, (c) cropping the response map to the plausible ±32 mm region before softmax.
- Coordinate grid should be in mm relative to the FU patch centre (e.g. voxel index → (i − (N−1)/2)·spacing_mm). Then soft-argmax directly outputs Δ in mm and the distance loss is in mm.
- Practical recipe: 3D U-Net/encoder-decoder or Siamese correlation → logits at 16^3–32^3 (4–8 mm cells) → softmax(β·logits) → expected mm coordinate → Euclidean or L1 loss + λ·JS(heatmap, Gaussian at GT, σ≈4–8 mm). Low resolution is fine because DSNT/integral are sub-voxel.

### Gaps
- No 3D CT-specific study of β or regulariser strength found; values above are extrapolated from 2D pose papers.
- BCIR derivation is 2D; 3D version is an analogous extension (not checked).

## Q4. Heteroscedastic / probabilistic losses (Gaussian or Laplace NLL with predicted σ; RLE) and uncertainty flagging

### Takeaway
Predicting σ alongside Δ and minimising NLL gives learned loss attenuation and a per-case confidence; Laplace (L1-like) NLL beats Gaussian, and RLE (normalising-flow residual density) beats both. Pitfall: plain Gaussian NLL can stall the mean fit by down-weighting hard samples — use β-NLL (β≈0.5) or warm-start with plain L1/Euclidean. σ correlates moderately with error (Pearson ~0.5 in RLE), useful for flagging uncertain predictions.

### Cited Findings
- Gaussian heteroscedastic loss: `L = 1/N Σ [ ||y_i − f(x_i)||² / (2σ(x_i)²) + ½ log σ(x_i)² ]`; in practice predict `s = log σ²` for numerical stability: `L = Σ ½ exp(−s)||y − ŷ||² + ½ s`. "Similarly for l₁ loss" (Laplace version). — [Kendall & Gal 2017, arXiv:1703.04977](https://arxiv.org/abs/1703.04977)
- Learned loss attenuation: exp(−s) down-weights residuals on inputs predicted as uncertain, making the model robust to noisy labels; the log term prevents predicting infinite uncertainty. Aleatoric-only models cannot detect out-of-distribution inputs (needs epistemic, e.g. MC dropout). Reported 1–3% gains over non-Bayesian baselines on depth/segmentation. — [Kendall & Gal 2017](https://arxiv.org/abs/1703.04977)
- RLE: MLE view `L = −log P(μ_g | I)`; RLE models density via a normalising flow on the normalised residual (μ_g − μ̂)/σ̂ with a residual design: `L_rle = −log Q(μ̄_g) − log G_φ(μ̄_g) − log s + log σ̂`, Q = preset Laplace (default) or Gaussian; flow (RealNVP, e.g. 3×64 FC) used only in training — zero test-time cost. σ̂ via sigmoid; confidence `ĉ = 1 − mean σ̂`. — [Li et al. 2021](https://arxiv.org/abs/2107.11291)
- RLE ablation (COCO AP): const-var Gaussian (ℓ2) 36.6; const-var Laplace (ℓ1) 58.1; learnable-σ Gaussian 60.2; learnable-σ Laplace 67.4; RLE-Gaussian 70.0; RLE-Laplace 70.5. Learnable σ alone gives large gains; Laplace > Gaussian. — [Li et al. 2021](https://arxiv.org/abs/2107.11291)
- RLE on MPII (ResNet-50+FC): direct regression PCKh@0.1 23.6 → RLE 26.7 (heatmap 25.4); RLE notably better at high-precision thresholds. Human3.6M MPJPE: DLE 51.0 → RLE 48.6 mm. — [Li et al. 2021](https://arxiv.org/abs/2107.11291)
- Confidence correlation with prediction correctness (Pearson with OKS): heatmap max-value 0.479, Gaussian-σ 0.476, Laplace-σ 0.522, RLE-Gaussian 0.540, RLE-Laplace 0.553. — [Li et al. 2021](https://arxiv.org/abs/2107.11291)
- β-NLL pitfall: Gaussian NLL scales down gradients of poorly predicted points (high σ), effectively undersampling them, so training gets stuck in sub-optimal mean fits (fails even on a noiseless sine). Fix: weight each sample's NLL by stop-grad(σ²)^β; β=0 = NLL, β=1 = MSE gradients for the mean while still learning σ; "β = 0.5 will be a good starting point." — [Seitzer et al. 2022, β-NLL, arXiv:2203.09168](https://arxiv.org/abs/2203.09168)
- β-NLL notes that the high-uncertainty points NLL ignores were "the most important data points to model correctly" in their dynamics example; Student-t alternatives get better likelihoods but are heavier to implement. — [Seitzer et al. 2022](https://arxiv.org/abs/2203.09168)
- Anisotropic uncertainty in medical landmarks: Pyramid Covariance Predictor predicts the Cholesky factor of a full covariance per landmark, improving detection. — [Uncertainty-aware U-Net, arXiv:2303.10349](https://arxiv.org/abs/2303.10349)
- Heatmap-based alternative to σ: DSNT's softmax heatmap variance itself is a measure of spread. — [Nibali et al. 2018](https://arxiv.org/abs/1801.07372)

### Inferences
- 3D Laplace NLL for this task (per axis, predict b_i = exp(s_i) in mm): `L = Σ_{i∈xyz} ( |Δ̂_i − Δ_i| / b_i + log b_i )`. Gaussian version: `Σ ½(e_i²·exp(−s_i) + s_i)`. Clamp s to e.g. [log(0.5 mm), log(50 mm)].
- Risk on this small dataset: if the network cannot find signal, NLL lets it "explain away" everything with large σ — same collapse to Δ̂≈0 but with σ≈8 mm. Recommended order: train mean with L1/Euclidean (or β-NLL β=0.5–1) first; add σ head after the mean beats the zero baseline.
- Flagging: report (a) Spearman correlation between predicted σ (or soft-argmax heatmap spread) and actual error, (b) error vs coverage curve (discard the top-k% most uncertain; mean error on the remainder), (c) calibration: fraction of cases with |e_i| within predicted interval. With ~250 val lesions these estimates will be noisy — give bootstrap CIs.
- RLE is feasible (tiny RealNVP over 3D residual) but likely overkill for a course project; learnable-σ Laplace captures most of RLE's reported gain (58.1→67.4 of the 58.1→70.5 total on COCO).

### Gaps
- No 3D medical-imaging paper retrieved that reports RLE or β-NLL for point localisation in CT; transfer is assumed.
- Calibration quality of σ on small data (~2500 samples) not documented in found sources.

## Q5. Scale/units: mm vs voxel vs normalised; anisotropy; per-axis vs vector loss

### Takeaway
Compute the loss in a physically meaningful, isotropic unit (mm, or mm/scale) so that each axis counts equally and hyperparameters (Huber δ, Wing w/ε) have known meaning; normalise targets (e.g. divide by 8 mm offset sd) for optimisation. Per-axis L1 and vector Euclidean are both standard; Euclidean matches the metric and is rotation-invariant.

### Cited Findings
- DSNT uses vector Euclidean distance; Wing/L1/L2 in landmark papers are applied per coordinate and summed. — [Nibali et al. 2018](https://arxiv.org/abs/1801.07372); [Feng et al. 2018](https://arxiv.org/abs/1711.06753)
- Wing loss parameters (w, ε) are defined in target units; best w≈10, ε≈2 for pixel coordinates on AFLW (results evaluated as face-size-normalised error). — [Feng et al. 2018](https://arxiv.org/abs/1711.06753)
- RLE predicts σ through a sigmoid (σ∈(0,1)), i.e. on coordinates normalised to the crop. — [Li et al. 2021](https://arxiv.org/abs/2107.11291)
- DLT notes most clinical CTs have 5 mm slice thickness and the z-direction error can be hidden in 2D visualisation — anisotropy matters for lesion tracking. — [Cai et al. 2021](https://arxiv.org/abs/2012.04872)
- Medical landmark annotations can be anisotropic; full covariance prediction helps. — [arXiv:2303.10349](https://arxiv.org/abs/2303.10349)
- nnLandmark reports MRE in mm and defines heatmap radius in voxels (EDT 15 voxels; robust range 11–23 voxels), after nnU-Net resampling. — [nnLandmark, arXiv:2504.06742](https://arxiv.org/abs/2504.06742)

### Inferences
- Data is already resampled isotropically to 2 mm, so mm and voxel losses differ only by a factor of 2; but source z-resolution was probably coarser (often 5 mm) so label noise in z is likely larger. Check per-axis error and per-axis σ; a per-axis Laplace NLL naturally absorbs this.
- Normalisation: train on Δ/8 mm (unit-variance targets per axis); then smooth-L1 δ≈0.25–0.5 and Wing w≈1.25, ε≈0.25 correspond to the mm values suggested in Q1. Always report metrics back in mm.
- Soft-argmax grids should be built in mm (spacing-aware) so anisotropic resampling, if introduced later, does not silently distort the loss.

### Gaps
- No source found that directly compares mm vs normalised loss scaling for 3D landmarks; the guidance is based on how loss hyperparameters are defined.

## Q6. Evaluation metrics: mean/median Euclidean error, hit rates, improvement vs zero baseline

### Takeaway
Report mean ± sd and median Euclidean error (mm), success rate within radii (SDR@r, e.g. 5/10/20 mm) and "inside lesion" hit rate (CPM@Radius as in DLT), per-axis errors, and relative improvement over the predict-zero baseline, with patient-level bootstrap CIs. The zero baseline has an analytic expected value for the sampling distribution.

### Cited Findings
- Mean Radial Error `MRE = 1/N Σ ||x − x̂||₂` and Success Detection Rate `SDR@t = % landmarks with error ≤ t` (nnLandmark reports SDR@2/3/4 mm). — [nnLandmark, arXiv:2504.06742](https://arxiv.org/abs/2504.06742)
- Lesion tracking metrics: mean centre-distance error (DLT ≈ 7 mm on DeepLesion) and CPM@Radius — whether the predicted centre falls inside the lesion (radiologist judged inside/outside on 100 external studies: DLT 88.4% vs DEEDS 85.6%). — [Cai et al. 2021](https://arxiv.org/abs/2012.04872)
- Pose literature uses PCK@threshold relative to an object scale (PCKh@0.5/0.1) and AUC of the cumulative error curve; RLE reports both. — [Li et al. 2021](https://arxiv.org/abs/2107.11291); [Nibali et al. 2018](https://arxiv.org/abs/1801.07372)
- Cumulative error distribution (CED) curves are the standard visual comparison in landmark papers. — [Feng et al. 2018](https://arxiv.org/abs/1711.06753); [Payer et al. 2019](https://arxiv.org/abs/1908.00748)
- Uncertainty quality: Pearson correlation between confidence and correctness. — [Li et al. 2021](https://arxiv.org/abs/2107.11291)

### Inferences
- Zero-baseline analytics (my derivation): if Δ ~ N(0, σ²I₃) with σ=8 mm, ||Δ|| is Maxwell-distributed: mean = 2σ√(2/π) ≈ 12.8 mm, median ≈ 1.54σ ≈ 12.3 mm, P(||Δ||≤10 mm) ≈ 33%, P(||Δ||≤20 mm) ≈ 90%. The observed 14.5 mm zero baseline is above 12.8, so either the effective offset distribution is wider than sd 8, or val offsets include label/centroid variation — worth checking; quote the empirical zero baseline on the exact same val offsets.
- Because offsets are random, evaluate on a fixed, seeded set of val offsets (or several draws averaged) so model and baseline see identical inputs; report paired differences (per-lesion error_model − error_zero) with patient-level bootstrap CI, plus % improvement `(E_zero − E_model)/E_zero`.
- Recommended table: mean, median, 90th percentile error (mm); SDR@5, @10, @20 mm; hit-inside-FU-lesion-mask rate (and by lesion-size bins — small lesions are where "inside lesion" is hardest); per-axis MAE; % improvement vs zero; optionally error-vs-coverage curve if σ is predicted. Also a "registration-only"/oracle-free baseline if available.

### Gaps
- No standard benchmark protocol for this exact synthetic-offset re-localisation setup was found; metrics are adapted from landmark detection and DLT.
