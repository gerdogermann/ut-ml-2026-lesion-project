# Heatmap-based loss functions for single-point 3D lesion localisation (Siamese BL to FU patch)

Notes compiled 2026-10-29. Setting assumed throughout: 64^3 FU patch at 2 mm isotropic, GT FU centroid offset within about ±24 mm (±12 voxels) of the patch centre, GT FU mask available, about 2500 pairs from about 224 patients.

Provenance: items marked **[verified]** were read from the full paper text this session (arXiv PDFs converted to text). Items marked **[from memory]** come from the researcher's prior knowledge of well-known papers and were not re-read this session. The report writer should treat [from memory] hyperparameters as likely correct but unconfirmed.

## 1. MSE (L2) heatmap regression to a Gaussian target: σ choice, learned σ, class imbalance, resolution and quantisation

### Takeaway
Plain pixel/voxel-wise L2 to a Gaussian target is the medical-landmark default (Payer SCN). σ is the key hyperparameter: too small hurts robustness, too large hurts accuracy. Payer et al. (MedIA 2019) learn one σ per landmark with an α·σ² penalty. Thaler et al. (2021) extend this to anisotropic Gaussians with init σ=3 px, α=5, γ=100. Argmax decoding has quantisation error tied to the voxel size. Plain L2 is also dominated by background voxels.

### Cited Findings
- **Target and loss (SCN family) [verified]:** target h_i(x;σ_i) = γ/(2πσ_i²) · exp(−‖x−x_i‖²/(2σ_i²)) (2D form; the d-dimensional form normalises by (2π)^{d/2}σ^d). The loss is Σ_i Σ_x ‖ĥ_i(x;w) − h_i(x;σ_i)‖². "A scaling factor γ is used to avoid numerical instabilities during training." — [Thaler, Payer, Urschler, Štern, MELBA 2021, arXiv 2109.09533](https://arxiv.org/abs/2109.09533)
- **Learned σ (Payer et al. 2019) [verified]:** argmin over w, σ_1..σ_N of Σ‖ĥ_i − h_i(σ_i)‖² + α Σ σ_i². Gradients flow through the target into σ_i. The α term prevents the trivial solution σ→∞, h≈0. σ_i is shared across all images of a landmark, not set per image. "suboptimal σi lead to either reduced prediction accuracy or reduced robustness". — [Thaler et al. 2021](https://arxiv.org/abs/2109.09533), describing [Payer et al., MedIA 2019, doi 10.1016/j.media.2019.03.007](https://doi.org/10.1016/j.media.2019.03.007)
- **Learned σ encodes uncertainty [verified]:** "The more certain the network is in its prediction of a specific landmark, the smaller the target heatmap size will be." Thaler extends the target to an anisotropic Gaussian Σ_i = R_i Σ*_i R_iᵀ. Hyperparameters: γ=100, σ_maj=σ_min=3 initialised, θ=0, α=5, L2 weight decay 0.001, SCN trained 40,000 iterations, on 2D hand radiographs and cephalograms. — [Thaler et al. 2021](https://arxiv.org/abs/2109.09533)
- **SCN decoding and small-data claim [verified]:** the predicted coordinate is the argmax of the heatmap. SCN multiplies a local-appearance heatmap by a spatial-configuration heatmap. The authors claim SCN beats a localisation U-Net when trained on 100/50/10 hand radiographs. The dataset is 895 hand radiographs with 37 landmarks. — [Payer et al., MIDL 2019 abstract, arXiv 1908.00748](https://arxiv.org/abs/1908.00748)
- **MSE is an indirect objective [verified]:** "despite the Euclidean loss having a global minimum when the MSE loss is zero, we aren't guaranteed that an optimization step which improves MSE loss will improve our results". In the paper's Fig. 4 example, a single correct-location pixel has MSE 100. A near-perfect blob with a wrong-place max has MSE 16. — [Nibali et al. 2018, DSNT, arXiv 1801.07372](https://arxiv.org/abs/1801.07372)
- **Quantisation [verified]:** heatmap matching with argmax got only 44% PCKh at a 7×7 heatmap (ResNet-34, MPII), against roughly 85%+ for DSNT. HM improves as resolution rises, "evidence of the quantization effects inherent to calculating coordinates via a pixel-wise argmax". HM inference also used "small adjustments ... based on the brightness of adjacent pixels" (the Newell quarter-pixel shift). — [Nibali et al. 2018](https://arxiv.org/abs/1801.07372)
- **Background imbalance [verified]:** for a 64×64 heatmap with a 7×7 Gaussian, "foreground pixels only constitute 1.2% of all the pixels". Equal weighting "could make the training process slow to converge and result in an inferior performance". — [Wang et al., Adaptive Wing, ICCV 2019, arXiv 1904.07399](https://arxiv.org/abs/1904.07399)
- **Original heatmap regression [from memory]:** Tompson et al. (NeurIPS 2014) trained ConvNets with MSE against 2D Gaussian targets (σ about 1 px) at reduced output resolution, plus a graphical-model stage. — [Tompson et al. 2014, arXiv 1406.2984](https://arxiv.org/abs/1406.2984)
- **Sub-pixel decoding fixes [from memory]:** DARK (Zhang et al., CVPR 2020) decodes with a second-order Taylor expansion around the max of a log-heatmap. It also encodes non-quantised (sub-pixel) Gaussian centres in the target. Reported gains are largest at low output resolution. — [Zhang et al. 2020, arXiv 1910.06278](https://arxiv.org/abs/1910.06278)

### Inferences
- In our setting, 1 voxel = 2 mm. Argmax at full 64³ resolution gives a worst-case per-axis quantisation of ±1 mm, about 0.87 mm RMS in 3D for a uniform error. That is small compared with the about 7 mm MED reported by DLT. At 1/2 resolution (32³, 4 mm voxels) it becomes about 1.7 mm RMS. At 1/4 (16³, 8 mm voxels) it becomes about 3.5 mm RMS, which is no longer negligible. Use sub-voxel decoding (soft-argmax, local centroid, Taylor) or an offset head below full resolution.
- σ in voxels: a σ of 1.5–3 voxels (3–6 mm) at 2 mm spacing matches the SCN init σ=3. A lesion-size-adaptive σ is another option, e.g. σ ∝ equivalent radius from the FU mask, as in DLT's G(µ_s, r_s) and CenterNet's size-adaptive σ. With 64³ = 262,144 voxels and σ=2 vox, about 0.1% of voxels have a target above 0.5. Background imbalance is therefore much worse in 3D than in the 2D 1.2% case, which argues for focal or weighted losses or a softmax/CE formulation.
- The FU mask can define an alternative target, e.g. the mask itself or a distance-transform-based heatmap. No source was found that evaluates this specifically. See Gaps.

### Gaps
- Exact Payer 2019 MedIA hyperparameters for the 3D spine CT experiments (γ, α, σ init) were not re-read. Only the 2D MELBA follow-up values (γ=100, α=5, σ_init=3) were verified.
- No head-to-head study of the 3D σ choice for lesions (as opposed to anatomical landmarks) was found.

## 2. Softmax heatmap + CE/KL to normalised Gaussian; DSNT with Jensen–Shannon regulariser; one-hot vs Gaussian targets

### Takeaway
Normalise the heatmap with a spatial softmax over all 64³ voxels. Then either (a) take the expectation (DSNT / soft-argmax) and apply a Euclidean loss plus a JS regulariser to N(p, σ_t²I), or (b) apply CE/KL to a normalised Gaussian or one-hot target. DSNT found JS regularisation best, with σ_t of 0.5–1 px and λ=1, and DSNT is robust to low resolution. Softmax naturally solves the background-dominance issue, because the loss is per-sample, not per-voxel.

### Cited Findings
- **DSNT definition [verified]:** Ẑ = φ(Z), normalised to sum to 1. Coordinate grids X_{i,j} = (2j−(n+1))/n and Y analogous, in [−1,1]. The prediction is µ = [⟨Ẑ,X⟩_F, ⟨Ẑ,Y⟩_F], i.e. the mean, not the mode. It is differentiable, gives sub-pixel precision, and adds no trainable parameters. — [Nibali et al. 2018](https://arxiv.org/abs/1801.07372)
- **Rectification choice [verified]:** validation PCKh (ResNet-34@28px, no reg) was softmax 86.81%, abs 86.48%, ReLU 86.69%, sigmoid 86.71%. "softmax works best". — [Nibali et al. 2018](https://arxiv.org/abs/1801.07372)
- **Loss [verified]:** L = ‖p − µ‖₂ + λ L_reg(Ẑ). Options are a variance regulariser L_var = (Var[c_x] − σ_t²)² + (Var[c_y] − σ_t²)², or a distribution regulariser L_D = D(p(c) ‖ N(p, σ_t² I)) with D = KL or JS. The JS regulariser is "similar to incorporating the usual heatmap matching objective into the DSNT loss function". — [Nibali et al. 2018](https://arxiv.org/abs/1801.07372)
- **Regulariser comparison [verified, table partly garbled in extraction]:** JS achieved the highest accuracy, about 87.6% PCKh at σ_t=1 against about 85.9–86.9% unregularised. Exact per-cell values could not be cleanly parsed from the PDF. Final choice: "Jensen-Shannon divergence with σt = 1 and λ = 1". "The accuracy of the model was found to be quite robust with respect to the regularization strength, λ". For σ_t, "values in the range of half a pixel to one pixel were found to work well". Variance regularisation with σ_t=2 split the heatmap into four blobs. — [Nibali et al. 2018](https://arxiv.org/abs/1801.07372)
- **Resolution robustness [verified]:** "DSNT is largely robust to heatmap size". Regularisation helps "in all cases except the lowest resolution, where boundary effects come into play" (a 1 px σ Gaussian clips in a 7×7 map). — [Nibali et al. 2018](https://arxiv.org/abs/1801.07372)
- **Mean vs mode [verified]:** DSNT uses the mean, which is sensitive to outliers. The authors argue the network learns to suppress low-confidence outliers because they directly hurt the Euclidean loss. — [Nibali et al. 2018](https://arxiv.org/abs/1801.07372)
- **Integral regression [from memory]:** Sun et al. (ECCV 2018) propose essentially the same soft-argmax ("integral") approach for 2D/3D pose, extended to 3D volumetric heatmaps. They combine an L1 coordinate loss with optional heatmap supervision. — [Sun et al. 2018, arXiv 1711.08229](https://arxiv.org/abs/1711.08229)

### Inferences
- For one point per patch, a spatial softmax over the 64³ correlation map plus CE to a normalised Gaussian is the most direct "classification over positions" formulation: L = −Σ_x q(x) log p(x), with q = Gaussian/Σ Gaussian. KL(q‖p) differs from this CE only by the constant entropy of q. The one-hot target is the limit σ→0, which gives a sharper but noisier gradient and ignores annotation uncertainty about the centroid. A Gaussian with σ≈1–2 voxels is a reasonable compromise. This is a principled inference, not tested in a cited source for 3D lesions.
- A practical recipe is DSNT-style soft-argmax in mm units, with a Euclidean or L1 loss in mm plus λ·JS(p ‖ N(gt, σ_t²)), λ=1, σ_t≈1 voxel. It optimises the actual metric (mm error) and stays sub-voxel even at 1/2 or 1/4 resolution.
- Softmax needs a temperature or learnable scale on raw correlation scores. Unscaled dot products can be too flat or too peaked. This is a general practice, not sourced here.
- Our FU centre is within ±12 voxels of a 64³ patch centre, so the soft-argmax boundary effects noted by DSNT should be minor. The positions sit well inside the map.

### Gaps
- No source found that directly compares CE-to-Gaussian against MSE-to-Gaussian on 3D medical landmarks.

## 3. Focal-style heatmap losses (CornerNet/CenterNet), Adaptive Wing

### Takeaway
The CornerNet/CenterNet penalty-reduced focal loss (α=2, β=4, sigmoid per voxel, Gaussian-reduced negatives, normalised by the number of positives) is the standard fix for background dominance. Deep Lesion Tracker uses exactly this loss on its 3D lesion-centre heatmap. Adaptive Wing is an L2 alternative from 2D face alignment that up-weights foreground and near-foreground pixels.

### Cited Findings
- **DLT's loss (CenterNet-style focal) [verified]:** L = −Σ_i { (1−ŷ_i)^α log ŷ_i if y_i = 1; (1−y_i)^β ŷ_i^α log(1−ŷ_i) otherwise }, with α=2, β=4. Target Y = G(µ_s, r_s) = exp(−Σ_{i∈{x,y,z}} (i−µ^i)² / (2 r_s²)), i.e. a Gaussian with σ equal to the lesion radius, downsized to the output resolution. The output Ŷ is sigmoid. "The ground-truth heat map is < 1 everywhere except at the lesion center voxel." The loss down-weights negatives near µ_s. Decoding is the global argmax. — [Cai et al., Deep Lesion Tracker, CVPR 2021, arXiv 2012.04872](https://arxiv.org/abs/2012.04872)
- **CenterNet (Objects as Points) [from memory]:** the same focal loss, normalised by the number of keypoints N, with a size-adaptive σ_p. Output stride R=4. A local offset head Ô is trained with L1 on (p/R − p̃) at the GT centre only. Weights: λ_off=1, λ_size=0.1. — [Zhou et al. 2019, arXiv 1904.07850](https://arxiv.org/abs/1904.07850)
- **CornerNet [from memory]:** introduced the penalty-reduced focal loss variant. Negatives within a radius of the GT corner get reduced penalty through an unnormalised Gaussian. The radius is chosen so boxes within it keep IoU ≥ 0.3 with GT, and σ = radius/3. CornerNet also introduced an offset head for quantisation. — [Law & Deng, ECCV 2018, arXiv 1808.01244](https://arxiv.org/abs/1808.01244)
- **Adaptive Wing [verified]:** AWing(y,ŷ) = ω ln(1 + |y−ŷ|/ε)^{α−y} if |y−ŷ| < θ; otherwise A|y−ŷ| − C. A and C make the function continuous and smooth at θ. Here y is the GT heatmap value, so the loss shape adapts: it is Wing-like on foreground (y≈1) and MSE-like on background (y≈0). Values: α=2.1, ω=14, ε=1, θ=0.5. — [Wang, Bo, Fuxin, ICCV 2019, arXiv 1904.07399](https://arxiv.org/abs/1904.07399)
- **Weighted loss map [verified]:** M = 1 where the 3×3 grey-dilated GT heatmap is ≥ 0.2, else 0. Loss_weighted = Loss ⊗ (W·M + 1), with W=10. This focuses on foreground and "difficult background pixels". — [Wang et al. 2019](https://arxiv.org/abs/1904.07399)

### Inferences
- DLT is the closest published analogue to this project: 3D CT, Siamese correlation, lesion-centre heatmap. Its σ = lesion radius (from RECIST) can be replaced here by an equivalent radius from the FU mask, (3V/4π)^{1/3}, or a fraction of it. DLT's ablation only varied the anatomy-signal Gaussian width n (n=2: 7.4 mm, n=4: 7.9 mm, n=8: 8.5 mm MED). It did not ablate the loss itself, so focal vs MSE is untested in that paper.
- Focal loss uses per-voxel sigmoid, so several local maxima can coexist. That suits multi-object detection, but the softmax formulation enforces a single answer, which matches our "exactly one FU lesion in the patch" prior better. Inference only.
- For a 3D Adaptive Wing, use a 3×3×3 dilation for the mask. Its hyperparameters were tuned for 2D faces with 64×64 heatmaps and are unvalidated for 3D.

### Gaps
- No cited 3D medical study compares penalty-reduced focal, AWing and MSE head-to-head.

## 4. Heatmap + sub-voxel offset regression head; coarse-to-fine refinement

### Takeaway
At reduced output resolution, CenterNet/CornerNet add a 3-channel offset head trained with L1 only at the GT peak. Medical methods (Noothout 2020) use global-to-local cascades that combine patch classification with displacement regression. DLT itself has no offset head and uses argmax on an upsampled multi-scale map.

### Cited Findings
- **CenterNet offset [from memory]:** L_off = (1/N) Σ_p |Ô_{p̃} − (p/R − p̃)|, supervised only at the low-resolution centre p̃. At inference, the peak position plus the offset gives sub-pixel coordinates. — [Zhou et al. 2019](https://arxiv.org/abs/1904.07850)
- **Global-to-local regression + classification (3D) [verified via search summary]:** a global FCNN localises roughly. Specialised FCNNs then refine by combining displacement regression (vectors from patch centres to the landmark) with classification (landmark-in-patch probability). The final position is the classification-weighted average of the displacements. Evaluated on 8 landmarks in CCTA, 2 in olfactory MR and 19 in cephalometric X-ray, with performance "similar to a second observer". — [Noothout et al., IEEE TMI 39(12) 2020, arXiv 2007.05295](https://arxiv.org/abs/2007.05295)
- **DLT multi-scale fusion [verified]:** correspondence maps M1, M2, M3 from three FPN scales. Ŷ = σ(Wᵀ(M1 + U2 + U3) + b), with U2/U3 upsampled by (1,2,2)/(1,4,4). µ_p is the global argmax. — [Cai et al. 2021](https://arxiv.org/abs/2012.04872)
- **Score-map upsampling [verified]:** SiamFC found that "upsampling the score map using bicubic interpolation, from 17 × 17 to 272 × 272, results in more accurate localization". — [Bertinetto et al. 2016, arXiv 1606.09549](https://arxiv.org/abs/1606.09549)
- **Thaler fits Gaussians at inference [verified]:** they fit an anisotropic Gaussian to the predicted heatmap, initialised at its maximum. This gives a sub-pixel coordinate and an uncertainty estimate. — [Thaler et al. 2021](https://arxiv.org/abs/2109.09533)

### Inferences
- Simplest options, cheapest first: (1) full-res 64³ heatmap with soft-argmax or local-window centroid decoding, needing no extra head; (2) 32³ heatmap with a CenterNet offset head (3 channels, L1 in voxel units at the GT voxel only, λ_off=1); (3) two-stage, where a coarse heatmap is followed by re-cropping around the peak and a second refinement network. Option 3 roughly doubles the compute and is probably unnecessary when the search window is only ±24 mm.
- The offset target in mm or voxel units must use the same physical scale across axes when voxels are anisotropic.

### Gaps
- No source quantifies how much an offset head helps over soft-argmax in 3D medical volumes.

## 5. 3D specifics: memory, 1/2 or 1/4 resolution heatmaps, anisotropic voxels

### Takeaway
3D heatmaps are expensive, which is why many 3D methods output at reduced resolution. DSNT-style expectation or offset heads recover the precision lost by downsampling. With isotropic 2 mm voxels, anisotropy matters only if the data are resampled differently. Gaussians and losses should be defined in mm.

### Cited Findings
- **Memory pressure in 3D heatmaps [verified]:** Thaler et al. note that prior spinal CT work with more than a hundred landmarks did "not incorporate upsampling layers, thus reducing the accuracy of their heatmap regression outputs" because of memory. — [Thaler et al. 2021](https://arxiv.org/abs/2109.09533)
- **DLT 3D cost [verified]:** a full large 3D kernel is costly. DLT decomposes the (7,11,11) global kernel into three flattened kernels (fast cross-correlation, FCC). The (7,15,15) kernel takes 24.1 s per volume against 3.51 s for the decomposed (7,11,11), at MED 7.7 vs 7.4 mm. DLT's template training input is (32, 384, 384), its search images are 0.8×0.8×2 mm, and its correlation kernel is 3×3×3 cropped at the template centre. — [Cai et al. 2021](https://arxiv.org/abs/2012.04872)
- **Low resolution vs accuracy [verified]:** "higher heatmap resolution is beneficial at any depth. However, ... increasing resolution with dilations has a large impact on memory consumption and computational cost." — [Nibali et al. 2018](https://arxiv.org/abs/1801.07372)
- **Anisotropic DLT errors [verified]:** DLT MED_Z = 4.0±6.1 mm against MED_X 3.5 and MED_Y 2.9. These are with 2 mm slices, compared with 0.8 mm in-plane. The z axis is the weakest. — [Cai et al. 2021](https://arxiv.org/abs/2012.04872)

### Inferences
- A 64³ float32 heatmap is 1 MB per sample per channel, which is trivial. The memory cost is in the decoder features at full resolution, not the heatmap. A correlation output at 1/2 resolution (32³) with soft-argmax or offset decoding is a sensible default. Full 64³ is also affordable with a light 3D U-Net decoder at batch 8–16 on a single GPU (rough estimate, not sourced).
- If voxels are anisotropic, define the Gaussian with σ in mm per axis: σ_vox,axis = σ_mm / spacing_axis. Compute soft-argmax coordinates in mm, so the loss is not biased toward the coarse axis. This project uses isotropic 2 mm, which avoids the issue.

### Gaps
- No 3D-specific memory benchmarks for heatmap heads were found.

## 6. Correlation/cost-volume heatmaps in Siamese trackers: SiamFC, SiamRPN, DLT supervision

### Takeaway
SiamFC supervises the dense correlation score map with a class-balanced logistic loss, where positives lie within radius R of the centre. SiamRPN replaces this with anchor classification (CE) plus box regression (smooth L1). DLT (3D lesions) supervises the sigmoid correlation map with the CenterNet focal loss against a lesion-radius Gaussian. DLT beats a 3D SiamRPN++ (78.85 vs 68.85 CPM@10mm).

### Cited Findings
- **SiamFC loss [verified]:** ℓ(y,v) = log(1 + exp(−yv)), y ∈ {+1,−1}, averaged over the score map, L = (1/|D|) Σ_u ℓ(y[u], v[u]). Labels: y[u] = +1 if k‖u − c‖ ≤ R (k = network stride), else −1. "The losses of the positive and negative examples in the score map are weighted to eliminate class imbalance." Stride is 8, the score map is 17×17, and it is upsampled bicubically to 272×272 at test. — [Bertinetto et al., ECCV-W 2016, arXiv 1606.09549](https://arxiv.org/abs/1606.09549)
- **SiamFC radius [from memory]:** R = 16 px in the original paper. Not confirmed in the extracted text above.
- **SiamRPN [from memory]:** a correlation-based region proposal network. The classification branch uses CE over anchors (positive IoU > 0.6, negative < 0.3) and the regression branch uses a normalised smooth-L1 loss on box deltas. — [Li et al., CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Li_High_Performance_Visual_CVPR_2018_paper.html)
- **DLT supervision [verified]:** described in section 3. Also: "center augmentation" randomly shifts the template centre within ‖Δµ_t‖ ≤ 0.25 r_t for robustness to inexact clinician centres. A self-supervised variant pairs an image with its augmented copy and uses the same loss. — [Cai et al. 2021](https://arxiv.org/abs/2012.04872)
- **DLT results [verified]:** on the DeepLesion DLS test set (960 directed pairs), DLT reaches CPM@10mm 78.85%, CPM@Radius 86.88% and MED 7.0±8.9 mm. Comparisons: SiamRPN++ (3D DenseFPN) 68.85 / 80.31 / 8.3 mm, DEEDS deformable registration 71.88 / 85.52 / 7.4 mm, affine 48.33 / 65.21 / 11.2 mm. DLT-Mix (supervised + SSL) reaches 78.65 / 88.75 / 7.1 mm. The DLS dataset has 3008/403/480 lesion pairs for train/val/test. Ground-truth centre = mean of the four RECIST endpoints. Radius = half the longest diameter. — [Cai et al. 2021](https://arxiv.org/abs/2012.04872)
- **DLT ablation on Gaussian width of anatomy prior [verified]:** n=2 gives 7.4 mm, n=4 gives 7.9 mm and n=8 gives 8.5 mm MED. Removing the anatomy-signal encoder raises MED to 9.3 mm. The table also shows feature width 32/64/128 at 8.7/7.9/7.9 mm (row labels inferred from garbled PDF extraction). — [Cai et al. 2021](https://arxiv.org/abs/2012.04872)

### Inferences
- The DLT pipeline maps almost one-to-one to this project. The BL mask plays the role of the template "anatomy signal" (a Gaussian at µ_t with width n·r_t). The FU patch plays the role of the search image. A focal-loss Gaussian target with σ ≈ FU radius is the published baseline to beat. DLT's training set (3008 pairs) is comparable in size to ours (about 2500), which is encouraging for feasibility.
- The SiamFC binary disc label is a good fit when only a disc of "acceptable" positions is wanted. A disc with radius equal to the FU lesion radius, or derived from the FU mask, directly matches DLT's CPM@Radius metric. A simple variant uses the FU mask voxels as positives with balanced BCE.

### Gaps
- The DLT paper does not ablate the focal loss against MSE or CE alternatives.
- SiamRPN numbers were not re-verified this session.

## 7. Evidence: heatmap vs direct coordinate regression in (medical) landmark detection

### Takeaway
The evidence consistently favours fully convolutional heatmap or expectation-based outputs over FC coordinate regression, especially with little training data. DSNT gives the clearest controlled comparison (natural images). In medical imaging, SCN/U-Net heatmaps dominate, and hybrid displacement-plus-classification methods (Noothout) also work well in 3D.

### Cited Findings
- **Controlled comparison, small data [verified]:** on ResNet-34@28px MPII without augmentation, FC regression "exhibits very poor spatial generalization, achieving the extremely low PCKh score of 22% when trained on 1024 samples". "both DSNT and heatmap matching perform much better with fewer samples". FC was worst at higher resolutions and "worse than DSNT in general". — [Nibali et al. 2018](https://arxiv.org/abs/1801.07372)
- **DSNT efficiency [verified]:** DSNTr ResNet-50@28px gets 89.5% MPII test PCKh at 18.6 ms and 636 MiB. The 8-stack hourglass with heatmap matching gets 90.9% at 41.3 ms and 1476 MiB. Accuracy is competitive with a much smaller model. — [Nibali et al. 2018](https://arxiv.org/abs/1801.07372)
- **Medical small data [verified]:** SCN (heatmap) outperforms related methods "on size-limited datasets", with training subsets of 100/50/10 hand radiographs. — [Payer et al. 2019 MIDL abstract](https://arxiv.org/abs/1908.00748)
- **Payer 2016 [from memory]:** "Regressing Heatmaps for Multiple Landmark Localization Using CNNs" (MICCAI 2016) introduced CNN heatmap regression for medical landmarks (hand X-ray and MR) and reported it beating random-forest baselines. — [Payer et al. 2016, doi 10.1007/978-3-319-46723-8_27](https://doi.org/10.1007/978-3-319-46723-8_27)
- **Hybrid regression + classification in 3D [verified via search summary]:** works across 2D/3D and modalities. The authors argue that heatmap regression "often requires combining a large number of predictions" (e.g. voting), which makes it computationally expensive. — [Noothout et al. 2020](https://arxiv.org/abs/2007.05295)
- **Lesion tracking [verified]:** heatmap-based DLT (7.0 mm MED) beats anchor-based box regression SiamRPN++ (8.3 mm). It also beats detector + embedding pipelines (7.8–9.3 mm) on 3D lesions. — [Cai et al. 2021](https://arxiv.org/abs/2012.04872)

### Inferences
- With about 2500 pairs, a direct FC regression head (global pool then 3 numbers) is the riskiest choice, given the DSNT small-data result. A fully convolutional heatmap with softmax and soft-argmax, or with focal loss and argmax, keeps translation equivariance. Since the BL-feature correlation map is already a spatial heatmap, regression would discard that structure.
- A cheap, strong combination for this project: spatial softmax over the correlation map, then L = CE(p, Gaussian_σ=1–2 vox) + λ·‖soft-argmax(p) − gt‖_mm. A DSNT-style alternative is Euclidean loss + JS(σ_t=1, λ=1). The CenterNet focal loss with σ = f(FU radius) serves as a DLT-replicating baseline. Evaluate with MED (mm) and CPM@Radius / CPM@10mm to stay comparable to DLT.

### Gaps
- No 3D medical paper was found with a controlled heatmap vs FC-regression ablation on the same backbone. The DSNT evidence comes from 2D human pose.
- Recent (2024–2026) heatmap-loss surveys for medical landmarks were not searched due to the tool-call budget.
