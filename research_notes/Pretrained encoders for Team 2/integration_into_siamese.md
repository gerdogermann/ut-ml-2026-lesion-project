# Integrating a pretrained 3D CT encoder into Team 2's SiameseNet

Scope: how to plug a pretrained 3D encoder into `src/lesionloc/model.py::SiameseNet` without changing its interface (`forward(x) -> (delta_mm (B,3), logits (B,32,32,32))`), and how to fine-tune it on ~2.2k lesions (+8.6k augmented copies). Code facts below come from reading the repo at commit 7a074b4 (`model.py`, `losses.py`, `dataset.py`, `scripts/train.py`, `docs/wiki/Experiments.md` rows 2–12). External facts carry inline sources. Code snippets are proposals; they have not been run.

Repo facts that constrain the design (from the code, not external sources):
- Input `x (B,4,64,64,64)` = `[bl, fu, bl_mask, fu_mask]`, intensities `(HU+1000)/1400` clipped to [0,1] (`preprocess.py`: `HU_MIN, HU_MAX = -1000, 400`), 2 mm isotropic.
- The current stem is `Conv3d(k=4, s=2, p=1)`. `cell_mm()` assumes cell *i* is centred on voxel `2i+0.5`, so the 32 cells are symmetric about the patch centre (voxel 31.5).
- **`losses.py` line 35 hard-codes the same grid**: `F.max_pool3d(fu_mask, 4, stride=2, padding=1)  # same grid as SiameseNet's stem`. The model hard-codes it too: `F.avg_pool3d(mbl, 4, stride=2, padding=1)`. A backbone with a different stride-2 kernel must change both.
- The template is `avg_pool3d(f_bl, 2, stride=1)` sliced around index 15, plus a mask-pooled vector (`--bl-mask`). Logits = learnable temperature × cosine xcorr. `soft_argmax` runs over `self.coords`.
- Training: AdamW, lr 3e-4, weight decay 1e-4, cosine schedule per step, no warmup, a single param group (`train.py`). There is no AMP and no gradient checkpointing.
- Run 2 benchmark: 278 ms/step and 2.2 GiB at batch 16. Epochs take ~180 s with the aug data and are data-loading bound (Experiments.md rows 4 and 10).
- Best: run 6b, val TTA 5.58 / 3.49 mm. Experiments.md treats differences of ~0.06–0.2 mm on the 207 val lesions as noise.

---

## Q1. Input adaptation: spacing, intensity normalisation, and the BL-mask channel

### Takeaway
Spacing is a small problem: 2 mm is close to the 1.5 mm that most CT checkpoints saw, and nnFoundation was pretrained without resampling. Intensity handling depends on the norm layer. InstanceNorm CNNs (ResEnc / nnFoundationCNN / nnU-Net-style STU-Net) are nearly invariant to a global affine intensity map. Swin-based checkpoints (MONAI SSL, VoCo, SuPreM-Swin) need their exact HU window, which is an exact rescale of our [0,1] data. Add the BL mask as a zero-initialised second stem channel only when the stem is trainable. With a frozen stem, inject the mask only through the existing mask-pooled template.

### Cited Findings
**Spacing and intensity used by each checkpoint**
- nnFoundationCNN expects "Single-channel 3D volumes, **Z-score normalised**, with **no resampling**". Its recommended downstream patch is 192³ — [HF model card](https://huggingface.co/MIC-DKFZ/nnFoundationCNN).
- nnFoundation's pretraining data "was not resampled to a fixed target spacing, as the heterogeneous datasets and downstream tasks require substantially different spatial resolutions. Z-score normalization was applied to all images." The pretraining set is CT, MRI and PET, 2.1M volumes, MAE objective — [nnFoundation paper, arXiv 2609.26924](https://arxiv.org/abs/2609.26924).
- The nnFoundationCNN `adaptation_plan.json` lists `"normalization_schemes": ["ZScoreNormalization"]`, `"use_mask_for_norm": [false]`, `"spacing": null` (config `noresample`), `"pretrain_num_input_channels": 1`, `"arch_class_name": "ResEncL"`, `"key_to_stem": "encoder.stem"`, `"key_to_encoder": "encoder.stages"` — [adaptation_plan.json](https://huggingface.co/MIC-DKFZ/nnFoundationCNN/resolve/main/adaptation_plan.json).
- The OpenMind ResEnc-L / Primus checkpoints were pretrained at "a fixed target spacing of cubic 1mm" with z-score normalisation — [OpenMind paper, arXiv 2412.17041](https://arxiv.org/abs/2412.17041). The nnssl README's example `pretrain_data.json` is `Dataset745__OpenNeuro` with T1w images. This fits OpenMind being an MRI (OpenNeuro) collection, so its checkpoints are a poor fit for CT — [nnssl README](https://github.com/MIC-DKFZ/nnssl).
- nnU-Net's own nnssl adapter has 4 spacing modes: `like_pretrained` ("often 1×1×1 mm isotropic", recommended default), `default_nnunet` (dataset median), `fixed`, `no_resample` — [nnU-Net finetuning_from_nnssl_checkpoints.md](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/finetuning_from_nnssl_checkpoints.md). In the nnFoundation downstream segmentation benchmark, "All data were resampled to the dataset-specific target spacing determined by nnU-Net", not to a pretraining spacing — [arXiv 2609.26924](https://arxiv.org/abs/2609.26924).
- MONAI Swin UNETR SSL pretraining (`model_swinvit.pt`, ~5,050 CTs) defaults: `a_min=-1000`, `a_max=1000` (ScaleIntensityRanged to [0,1]), spacing `1.5 × 1.5 × 2.0` mm, ROI 96, `in_channels=1`, `feature_size=48` — [research-contributions SwinUNETR/Pretrain main.py](https://github.com/Project-MONAI/research-contributions/blob/main/SwinUNETR/Pretrain/main.py) and [README](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/Pretrain). The BTCV fine-tuning script uses `a_min=-175`, `a_max=250`, the same spacing, and `optim_lr=1e-4` — [SwinUNETR/BTCV main.py](https://github.com/Project-MONAI/research-contributions/blob/main/SwinUNETR/BTCV/main.py).
- SuPreM TotalSegmentator fine-tuning defaults: `a_min=-250`, `a_max=250` to [0,1], spacing 1.5 mm isotropic, ROI 96, `lr=1e-4`, `in_channels=1` — [SuPreM target_applications/totalsegmentator/train.py](https://github.com/MrGiovanni/SuPreM/blob/main/target_applications/totalsegmentator/train.py).
- CT-FM feature extractor preprocessing: `Orientation("SPL")` and `ScaleIntensityRange(a_min=-1024, a_max=2048, b_min=0, b_max=1, clip=True)`. The example applies no spacing transform — [HF ct_fm_feature_extractor](https://huggingface.co/project-lighter/ct_fm_feature_extractor). Its pretraining patch is [24,128,128] on 148k CTs, contrastive — [nnFoundation Table 1](https://arxiv.org/abs/2609.26924).
- STU-Net checkpoints come from nnU-Net v1 on TotalSegmentator (1,204 CTs), at 128³ crops — [STU-Net README](https://github.com/uni-medical/STU-Net). The README does not state the plan's spacing or normalisation; they are in the shipped `plan_files/*.pkl`.

**Norm layers**
- ResEnc-L in nnssl uses `norm_op=nn.InstanceNorm3d` (affine), LeakyReLU, `strides=[[1,1,1],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2]]`, `features_per_stage=[32,64,128,256,320,320]`, `n_blocks_per_stage=[1,3,4,6,6,6]` — [nnssl architecture_registry.py](https://github.com/MIC-DKFZ/nnssl/blob/nnFoundation/src/nnssl/architectures/architecture_registry.py). nnFoundationCNN shows the same "6 stages, features 32–64–128–256–320–320" (102M) — [HF card](https://huggingface.co/MIC-DKFZ/nnFoundationCNN).

**Extra input channels**
- nnU-Net's `PretrainedTrainer` "repeats the stem across input channels when they differ" when loading nnssl weights — [nnU-Net nnssl fine-tuning doc](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/finetuning_from_nnssl_checkpoints.md). STU-Net's nnU-Net-2.2 fine-tuning script was updated for "loading weights with multimodal input data to support networks with multiple input channels" — [STU-Net repo commit message](https://github.com/uni-medical/STU-Net).
- Deep Lesion Tracker (DLT) does not feed location into the image stem. It encodes the lesion as a Gaussian heatmap with σ = 4r in a separate "anatomy signal encoder" at 3 scales. It fuses that branch with image features by element-wise product, which "performs better than channel-wise concatenation". Removing the anatomy signal "considerably increases the MED" — [DLT, arXiv 2012.04872](https://arxiv.org/abs/2012.04872).

### Inferences
**Spacing.** Keep the 2 mm patches. 2 mm vs 1.5 mm is a 0.75× zoom, close to the 0.9–1.1 scale aug the offline augmentation already uses. nnFoundation saw mixed native spacings. Re-cutting at 1 mm gives 128³ patches and ~8× compute. Do that only if a 1 mm-trained checkpoint (OpenMind-style) clearly needs it, and it is MRI anyway.

**Intensity, InstanceNorm CNNs (ResEnc, nnFoundationCNN, STU-Net).** The stem conv is linear and is followed by per-sample, per-channel InstanceNorm. A global map `a·x+b` (a>0) of the input therefore changes the stem output by `a·conv(x) + const`, and InstanceNorm removes both terms. Only zero-padded borders feel the difference. The exact z-score constants barely matter. Use a fixed affine so air, soft tissue and bone land roughly where z-scored CT puts them:
```python
hu = x * 1400.0 - 1000.0              # undo preprocess.py scaling (values > 400 HU were clipped)
z  = (hu - MU) / SD                   # MU, SD: mean/std of a few raw training CTs, computed once
```
The real mismatch is the clip at 400 HU. The pretrained model saw bone, contrast agent and metal unclipped. To test that, re-cut patches with a wider window. That is a preprocessing change, so make it a later ablation, not the first experiment.

**Intensity, Swin / ViT checkpoints (LayerNorm, no per-channel instance stats).** Match the window exactly. All of these are exact rescales of our data, because their lower bound is ≥ −1000 HU:
- MONAI Swin SSL `[-1000,1000]`: `x_swin = 0.7 * x`. Exact, because our HU ≤ 400 < 1000.
- SuPreM `[-250,250]`: `x_s = ((x*1400 - 1000) + 250).div(500).clamp(0,1)`.
- BTCV-style `[-175,250]`: `x_b = ((x*1400 - 1000) + 175).div(425).clamp(0,1)`.
- CT-FM `[-1024,2048]`: `x_c = (x*1400 - 1000 + 1024) / 3072`. Lossy above 400 HU.

**BL-mask channel. Three options, in order of preference for a minimal diff:**
1. **Frozen stem, mask in the head only.** Keep the stem 1-channel and do not feed `bl_mask` to the backbone. The existing `--bl-mask` mask-pooled template already uses the mask. This is a zero-risk first step.
2. **Trainable stem, zero-initialised extra channel.** The function at step 0 is then exactly the pretrained one. Prefer this over nnU-Net's "repeat the stem" default, which suits extra *image* modalities: repeating the weights onto a binary mask would add a large, out-of-distribution signal at init. Pitfall, verified in dynamic-network-architectures: `ConvDropoutNormReLU.forward` returns `self.all_modules(x)`, and `self.conv` is an alias of `all_modules[0]` ([simple_conv_blocks.py](https://github.com/MIC-DKFZ/dynamic-network-architectures/blob/main/dynamic_network_architectures/building_blocks/simple_conv_blocks.py); the HF card also warns that `all_modules.*` keys alias the named modules). Replace both references:
   ```python
   blk = enc.stem.convs[0]                      # ConvDropoutNormReLU
   old = blk.conv
   new = nn.Conv3d(2, old.out_channels, old.kernel_size, old.stride, old.padding, bias=old.bias is not None)
   with torch.no_grad():
       new.weight.zero_(); new.weight[:, :1] = old.weight
       if old.bias is not None: new.bias.copy_(old.bias)
   blk.conv = new; blk.all_modules[0] = new     # forward uses all_modules, not .conv
   ```
3. **Separate mask branch (DLT style).** A tiny conv branch maps `bl_mask` to a (B,C,32,32,32) gate. Multiply it into the BL features before template pooling: `f_bl * (1 + g)`, with `g` zero-init so step 0 is the identity. This keeps the pretrained stem untouched and frozen-compatible. DLT reports that a product beats concatenation, but its branch encodes a broad location prior (σ = 4r), not a binary lesion mask.

### Gaps
- No source measured how a 1.5 mm or 1 mm pretrained encoder degrades at 2 mm input for correspondence or matching tasks. The claim that a 0.75× zoom is benign is an inference.
- VoCo's fine-tuning HU window and spacing were not verified. The VoCo README only shows the weight-loading code (SwinUNETR, `feature_size=48`, `use_v2=True`, renaming `swin_vit` to `swinViT`, `strict=False`) — [VoCo README](https://github.com/Luffy03/VoCo).
- CT-FM's norm type was not verified. Its `config.json` sets only `blocks_down=[1,2,2,4,4]`, `init_filters=32`, `in_channels=1` ([HF](https://huggingface.co/project-lighter/ct_fm_feature_extractor)). MONAI `SegResEncoder` defaults to `norm="batch"` ([MONAI segresnet_ds.py](https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/segresnet_ds.py)). If CT-FM kept BatchNorm, its running stats must stay frozen (`.eval()`) during fine-tuning.
- I did not find the STU-Net plan's target spacing and normalisation without downloading the `.pkl` plans.

---

## Q2. Which stage to tap for the 4 mm grid, decoder vs encoder skips, and multi-scale fusion (DLT / SAM / TLT designs)

### Takeaway
With 2 mm input, the **stride-2 stage of a CNN encoder is the native 4 mm grid**. For ResEnc-L / nnFoundationCNN that is stage 1: 64 channels, 3 residual blocks, 32³ for a 64³ input. Stage 2 (8 mm, 128 ch, 16³) gives coarse context. Do not build a U-Net decoder. Pretrained decoders are generally not transferred, so it would be randomly initialised. Use encoder stages directly, SAM/DLT-style: a fine 4 mm embedding plus a coarse 8 mm embedding, each with its own small projection, L2 norm and temperature, and **sum the correlation maps** after upsampling the coarse one. One grid pitfall: ResEnc and SegResNet downsample with a k3/s2/p1 conv, which shifts cell centres by 1 mm relative to the current `cell_mm()`.

### Cited Findings
- ResEnc-L stage strides are `[1,2,2,2,2,2]` with features `[32,64,128,256,320,320]` and blocks `[1,3,4,6,6,6]` — [nnssl architecture_registry.py](https://github.com/MIC-DKFZ/nnssl/blob/nnFoundation/src/nnssl/architectures/architecture_registry.py). `ResidualEncoder.forward` runs `self.stem` and then iterates `self.stages`, appending each stage output. With `return_skips=True` it returns all of them — [residual_encoders.py](https://github.com/MIC-DKFZ/dynamic-network-architectures/blob/main/dynamic_network_architectures/building_blocks/residual_encoders.py).
- In nnssl's encoder-only mode the model is `model.encoder`, with `key_to_encoder` / `keys_to_in_proj` re-rooted — [nnssl get_network_by_name.py](https://github.com/MIC-DKFZ/nnssl/blob/nnFoundation/src/nnssl/architectures/get_network_by_name.py). The checkpoint is a dict with `network_weights` (state_dict), `nnssl_adaptation_plan` and `citations`, and loads with `weights_only=True` — [HF card](https://huggingface.co/MIC-DKFZ/nnFoundationCNN).
- Decoders are not transferred. OpenMind "chose to discard the decoder, as previous work showed this to be slightly superior" — [arXiv 2412.17041](https://arxiv.org/abs/2412.17041). In nnFoundation, "only encoder weights were transferred from the pretrained foundation model, while decoder weights were initialized randomly" — [arXiv 2609.26924](https://arxiv.org/abs/2609.26924). nnU-Net's adapter "skips the segmentation head" — [nnU-Net doc](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/finetuning_from_nnssl_checkpoints.md).
- MONAI `SwinUNETR.forward`: `hidden_states_out = self.swinViT(x_in, self.normalize)`, `enc0 = self.encoder1(x_in)` (full res), `enc1 = self.encoder2(hidden_states_out[0])`, and so on. Inputs must be divisible by `patch_size**5` (32 for the default `patch_size=2`) — [MONAI swin_unetr.py](https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/swin_unetr.py). The SSL weights cover the `swinViT` backbone ("self-supervised pre-trained weights for Swin UNETR backbone") — [Pretrain README](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/Pretrain).
- **SAM (Self-supervised Anatomical eMbedding)** uses a 3D ResNet-18 (I3D-inflated ImageNet weights) with an FPN. A **global** embedding comes "from the coarsest FPN level" and a **local** one "from the finest FPN level", each through "separate 3×3×3 convolutions and L2 normalization layers". The authors "cut the connection between the coarsest FPN feature and its upper level to make F^g and F^l more independent". At inference, "we upsample S^g, S^l to the size of the original image, and find the peak of S^g+S^l". Local hard negatives are mined on the combined map S^g+S^l, which is consistent with inference. SAM reports 91% accuracy on whole-body follow-up lesion matching — [SAM, arXiv 2012.02383](https://arxiv.org/abs/2012.02383).
- **DLT** uses a 3D DenseFPN (2D DenseNet inflated to 3D) that outputs 3 scales, plus an anatomy-signal encoder at the same 3 scales. At each scale it correlates a 3×3×3 template kernel and a larger global kernel (7×11×11, decomposed into 3 learned flattened kernels). It then sums the scales: `Ŷ = σ(Wᵀ(M1 + U2 + U3) + b)`, with U2 and U3 upsampled from the coarser scales. Removing the learnable decomposition of the large kernel hurt — [DLT, arXiv 2012.04872](https://arxiv.org/abs/2012.04872).
- In the OpenMind ResEnc-L pretraining variants, "For dense-outputs the highest-resolution output was used (MAE, S3D, VF, MG)". For global objectives, skip outputs were average-pooled and concatenated — [arXiv 2412.17041 App. D](https://arxiv.org/abs/2412.17041).

### Inferences
**Stage arithmetic at 2 mm input (64³):**

| Backbone | 4 mm tap (32³) | 8 mm tap (16³) | Notes |
|---|---|---|---|
| ResEnc-L / nnFoundationCNN / OpenMind ResEnc-L | `stages[1]`, 64 ch, 3 blocks | `stages[2]`, 128 ch, 4 blocks | Stages 3–5 (most of the 102M params) are discarded. |
| CT-FM `SegResEncoder` (`blocks_down=[1,2,2,4,4]`, `init_filters=32`) | level 1 (expected 64 ch) | level 2 (expected 128 ch) | Channel counts assume MONAI's doubling. Verify by a shape check. |
| STU-Net (nnU-Net-style, stride-1 first stage) | stage 1 | stage 2 | Check the plan's strides; nnU-Net plans can be anisotropic. |
| MONAI Swin SSL / VoCo / SuPreM-Swin | `hidden_states_out[0]` (patch-embedding output, no Swin block) | `hidden_states_out[1]` (one Swin stage) | The 4 mm Swin tap is semantically shallow. Use `hidden[1]` (8 mm) upsampled plus a trainable fine branch, or feed 1 mm input (8× cost). Swin is the weakest fit for a 4 mm grid. |

The fact that `hidden_states_out[0]` is the patch-embedding output comes from reading MONAI's `SwinTransformer.forward` (`x0 = patch_embed(x)`, and the following `layers1` block downsamples). The table's "patch embedding only" label is an inference from that code, not an external claim.

**Receptive field (rough estimate, not from a source).** ResEnc stem + stage 0 + stage 1 has roughly 30 voxels of RF, about 60 mm. That is *smaller* than the current encoder's ~100 mm (dilations 1/2/4). Adding stage 2 brings it to roughly 100+ voxels, about 200 mm or more. Stage 1 alone may therefore lose the context the current model gets from its dilations. That is one reason to add the stage-2 coarse map.

**Grid-alignment pitfall (code-level).**
- The current stem (k4, s2, p1) centres cell *i* on voxel 2i+0.5. ResEnc's and SegResNet's stride-2 convs (k3, s2, p1) centre cell *i* on voxel **2i**. Cell *i* is then at `(2i - 31.5)·2 mm = cell_mm()[i] - 1 mm` per axis.
- If `coords` is left unchanged, every prediction is biased by +1 mm per axis, about 1.7 mm Euclidean. Flip TTA would hide this, because the bias flips sign.
- Swin's k2/s2 patch embedding centres on 2i+0.5, the same as the current code, so it needs no fix.

Fixes:
```python
# model.py: let the grid offset depend on the backbone
self.register_buffer("coords", cell_mm() - (1.0 if stride2_kernel == 3 else 0.0), persistent=False)
# BL lesion (voxel 31.5) sits at cell 15.75: weights 0.25/0.75 per axis instead of avg_pool3d(·, 2, stride=1)
w1 = torch.tensor([0.25, 0.75])
k = (w1[:, None, None] * w1[None, :, None] * w1[None, None, :])          # (2,2,2), sums to 1
t_full = F.conv3d(f_bl, k.expand(C, 1, 2, 2, 2).to(f_bl), groups=C)       # index 15 -> cell 15.75
# mask pooling onto the new grid (model.py and losses.py line 35):
F.avg_pool3d(mbl, 3, stride=2, padding=1)     # model: was (4, 2, 1)
F.max_pool3d(fu_mask, 3, stride=2, padding=1) # losses.team2_loss: was (4, 2, 1); must match the model's grid
```
Shortcut alternative: keep the 2×2×2 mean template (0.5 voxel = 1 mm off) and only fix `coords`.

**Fusion: sum two correlation maps rather than concatenating features.** This follows SAM (S^g + S^l) and DLT (M1 + U2 + U3). It reuses `xcorr` unchanged and keeps separate temperatures. A minimal head:
```python
p4 = nn.Conv3d(64, D, 1)                         # 4 mm "local" embedding
p8 = nn.Conv3d(128, D, 1)                        # 8 mm "global" embedding (no top-down link, as in SAM)
f4 = F.normalize(p4(s1), dim=1); f8 = F.normalize(p8(s2), dim=1)
# template at 4 mm (k=5 as in run 6b); at 8 mm use k=3 around the centre of the 16^3 map
logits = s_l * xcorr(f4_fu, t4) / k**3 + s_g * F.interpolate(xcorr(f8_fu, t8)[:, None], scale_factor=2, mode="trilinear", align_corners=False)[:, 0] / 27
```
- Pitfall: trilinear upsampling with `align_corners=False` from a k3/s2/p1 grid onto another k3/s2/p1 grid shifts by a quarter cell. Check it on a synthetic delta, or accept it, because the 4 mm map sets the peak.
- Concat + 1×1 projection (`cat([s1, up(s2)]) -> Conv1x1 -> D`) is the cheaper alternative and a single-map ablation.
- D = 32 matches the current width. SAM and DLT found the embedding dimension uncritical "as long as [it] is large enough" ([DLT ablation](https://arxiv.org/abs/2012.04872)).

**Large-context template (DLT's K_g).** The current 5³ template already beat 3³ (run 6b). DLT's learned decomposed 7×11×11 kernel is a later option if gross misses persist. It is not part of the encoder swap.

### Gaps
- **TLT** (Transformer-based Lesion Tracker) was not retrieved. I cannot report its embedding-head design.
- SAM's exact embedding dimension and FPN strides were not in the retrieved passages. A public SAM checkpoint is purpose-built for CT point matching and could be a candidate backbone, but its framework, weights and spacing were not checked here.
- RF numbers above are hand estimates. Measure them with a gradient-based RF probe if they matter.

---

## Q3. Fine-tuning strategy for small data

### Takeaway
The evidence points against a *fully frozen* encoder for dense tasks and in favour of **full or partial fine-tuning with a warmup and a lower peak LR**. Use frozen features only as a cheap first probe, and expect them to underperform, especially MAE-pretrained ones. Concretely: freeze stem + stage 0 for good. Fine-tune the tapped stages (1 and 2) at ~1e-4 to 3e-5 under a 10–15% linear warmup. Train the new head at 3e-4. Keep InstanceNorm as is; it is batch-size independent. Keep EMA. LoRA is a fallback, not the default, for a ~4M-param truncated CNN.

### Cited Findings
- OpenMind (ResEnc-L, segmentation, 4 datasets) compared Default, **Frozen** (decoder only), Warm-Up, Valley and Sawtooth schedules. Average DSC: Default 68.36, **Frozen 30.38**, Warm-Up 68.76, Valley 67.45, Sawtooth 67.42. MAE was the best pretraining both frozen (45.72) and fine-tuned (76.65–76.78). The final runs used "Sawtooth for ResEncL and Warm-Up for Primus-M". The LR was reduced from nnU-Net's 1e-2 to 1e-3 "for the transfer learning setting". Warm-up stages last 15 of 150 epochs. "Pre-training accelerates convergence": the largest gains came within the first 150 epochs — [arXiv 2412.17041](https://arxiv.org/abs/2412.17041).
- nnFoundation downstream segmentation used a "linear warm-up over the first 15% of epochs, followed by polynomial learning-rate decay", AdamW, with peak LRs from small sweeps: nnFoundationCNN 1e-3, CT-FM 2e-4, SwinUNETR 3e-4, VoCo-B 3e-4, VISTA3D 5e-5, MERLIN 5e-5. "If a baseline showed unstable training … the peak learning rate was reduced by a factor of 10." Detection used ResEnc-L at 3e-4 with a 10,000-step warm-up. Classification used a 10% warm-up + cosine. Some models overfit, so "we selected the best checkpoint based on validation performance" — [arXiv 2609.26924](https://arxiv.org/abs/2609.26924).
- In nnFoundation's table, "VoCo-L did not converge in our setup, while VoCo-H exceeded the available GPU memory" — [arXiv 2609.26924](https://arxiv.org/abs/2609.26924).
- "Contrary to the conventional wisdom of fine-tuning only the last few layers … fine-tuning intermediate layers is more effective". The second quarter (25–50%) was best for contrastive SSL and the third quarter for restorative SSL. Fine-tuning a shallower network made of "the first three quarters (0–75%)" gave up to +5.48% over end-to-end — [Revisiting Fine-Tuning Strategies for SSL Medical Imaging, arXiv 2307.10915](https://arxiv.org/abs/2307.10915).
- "with the encoder kept frozen, pre-trained models still fall short of nnU-Net" for 3D segmentation — [Curia-MAE abstract, arXiv 2608.05844](https://arxiv.org/abs/2608.05844).
- Frozen backbone + per-task LoRA has been used for few-shot 3D brain MRI continual learning — [arXiv 2602.23533](https://arxiv.org/abs/2602.23533). This is a different setting (continual, MRI); it is not evidence that LoRA beats fine-tuning here.
- Norms: ResEnc-L uses InstanceNorm3d (affine) — [architecture_registry.py](https://github.com/MIC-DKFZ/nnssl/blob/nnFoundation/src/nnssl/architectures/architecture_registry.py). MONAI SegResEncoder defaults to BatchNorm — [segresnet_ds.py](https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/segresnet_ds.py).

### Inferences
**Recommended recipe (fits `train.py` with a small change):**
```python
bb, head = model.backbone, [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
for p in list(bb.stem.parameters()) + list(bb.stages[0].parameters()):
    p.requires_grad_(False)                                 # first ~quarter: low-level, frozen for good
groups = [{"params": head, "lr": a.lr},                     # 3e-4, as now
          {"params": bb.stages[2].parameters(), "lr": a.lr * a.bb_lr_mult},          # e.g. 0.3 -> ~1e-4
          {"params": bb.stages[1].parameters(), "lr": a.lr * a.bb_lr_mult * 0.5}]    # layer-wise decay 0.5
opt = torch.optim.AdamW(groups, weight_decay=1e-4)
warm = int(0.1 * a.epochs * len(train_dl))                  # 10-15% linear warmup (OpenMind, nnFoundation)
sched = torch.optim.lr_scheduler.SequentialLR(opt, [
    torch.optim.lr_scheduler.LinearLR(opt, 1e-3, 1.0, warm),
    torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * len(train_dl) - warm)], [warm])
```
- **Two-phase "Sawtooth" variant**, in OpenMind's spirit: train the head alone (backbone `requires_grad=False`) for ~5 epochs with warmup, then unfreeze stages 1–2 with a second warmup. It protects the pretrained features from the large gradients of a randomly initialised projection and temperature. Warm-Up alone was as good on average in OpenMind, so treat Sawtooth as an ablation.
- **Peak backbone LR.** The literature range for AdamW fine-tuning is 5e-5 to 1e-3. With ~10k samples per epoch × 50 epochs and a truncated encoder, start at 1e-4 for stage 2 and 5e-5 for stage 1. If val diverges early, apply nnFoundation's ×0.1 rule.
- **Which layers.** The 2307.10915 result (tune intermediate layers, drop or freeze the extremes) maps directly onto "freeze stem + stage 0, tune stages 1–2, discard 3–5".
- **Norm handling.** InstanceNorm is per-sample, so batch 16 is irrelevant; keep it. Do not swap it for GroupNorm, because that discards pretrained affine stats. For any BatchNorm backbone (possibly CT-FM), call `.eval()` on the BN modules every epoch after `model.train()`. `predict()` toggles `model.train()` back on, so the BN-eval override must be reapplied there too.
- **EMA.** Keep `--ema 0.999`. `AveragedModel` averages all parameters, frozen ones included, which is harmless. Note the EMA lag during warmup; best epochs in runs 6/6b were ~32–35 of 50.
- **Regularisation option.** Weight decay pulls fine-tuned weights toward 0, not toward the pretrained point. If overfitting shows (train/val gap > the current ~2.2 mm), try L2-SP (penalise ‖θ−θ₀‖²) or lower the backbone LR. This is an inference with no source retrieved here.
- **LoRA/adapters.** For a ~4M-param truncated CNN, partial fine-tuning is already cheap. LoRA on 3×3×3 convs is uncommon and adds code. Consider it only for a large ViT backbone (Primus/nnFoundationViT, 674M), which is a poor fit for 64³ / 4 mm anyway (8³ tokens means 8 mm cells at 1 mm, or 16 mm at our 2 mm).

### Gaps
- No retrieved study fine-tunes a pretrained encoder specifically for **Siamese correspondence / lesion tracking** with a frozen-vs-fine-tuned comparison. DLT used ImageNet-inflated weights and SAM used I3D ImageNet init, but neither ablated the fine-tuning schedule.
- The OpenMind frozen-vs-FT numbers are for MRI segmentation with an MAE encoder plus a random decoder. Frozen contrastive features (CT-FM, VoCo) may do relatively better for matching. This is plausible but unverified.

---

## Q4. Feature caching with a frozen backbone (re-jittered FU windows, flips, offline aug)

### Takeaway
Do not cache. It is impractical here, and it buys little because training is already data-loading bound. Run the frozen backbone on the fly under `torch.no_grad()` and fp16 autocast. The forward pass is cheap and largely hides behind the CPU loader. Caching is only worth it for a fixed eval set (val/test, 207/286 lesions) or for fast linear-probe sweeps on BL features without flips.

### Cited Findings
- No external source addresses this setup. The findings below are repo facts and arithmetic:
- `LesionDataset.__getitem__` cuts `fu` from `fu_big` (88³) at a fresh integer-voxel offset `s ~ round(N(0, sd/2))`, clipped to ±12 voxels. It then applies **random flips on 3 axes** to all 4 channels (`dataset.py`).
- The offline aug set (`*_a{k}.npz`) holds 4 rotated and scaled copies per train lesion, i.e. 8,610 files, with their own `fu_big` (`preprocess.py: augment`).
- Run 6/6b epochs take ~182 s and are bound by data loading (Experiments.md row 10).

### Inferences
- **Why caching fails:**
  1. **Flips.** Convolutions are not flip-equivariant, so `features(flip(x)) ≠ flip(features(x))`. Each lesion would need 8 cached flip variants per side.
  2. **Half-cell phase.** The FU window offset is an integer number of 2 mm voxels, but cells are 2 voxels. Cropping a cached `fu_big` feature map (88³ in, 44³ cells) only aligns for even offsets. Odd offsets need a second phase per axis, i.e. 8 phase variants. Restricting jitter to even voxels quantises targets to 4 mm, which conflicts with the soft-argmax precision goal.
  3. **Size.** One phase of stage-1 features for `fu_big` is 64 ch × 44³ × 2 B ≈ 10.9 MB per lesion. × 10,769 train files ≈ 117 GB, before the ×8 flip and ×8 phase factors. The 8 mm stage-2 map is 8× smaller but still large.
- **The fu_big trick still has a narrow use.** For the *non-flipped, even-offset* case, run the backbone once on `fu_big` and crop `[m+s)/2 : (m+s)/2 + 32]` from the 44³ map. Interior cells match, but borders differ because zero padding sits at different places. This only helps an online-jitter-on-GPU variant. It is not worth the code now.
- **Cache BL only?** BL has no jitter, but it is flipped together with FU, and the aug copies differ. The same flip problem applies.
- **Eval caching.** A one-off precompute of val features for the 8 flip-TTA variants (207 × 8 forwards) is trivial and only speeds repeated readouts. `worst_cases.py` could use it. Optional.

### Gaps
- No measurement exists yet of GPU forward time for a truncated ResEnc-L at batch 32 (16 BL + 16 FU) on V100 versus the ~180 s per epoch of loading. Measure with the existing `--bench` flag.

---

## Q5. Compute and memory on V100 for ResEnc-L / Swin UNETR / STU-Net-B at 64³, batch 16 × 2

### Takeaway
A **truncated** ResEnc-L / nnFoundationCNN (stem → stage 2) is cheap: roughly 4M params, single-digit GB even when fine-tuned, and far below the 102M-param full encoder. Full Swin UNETR (62M, `feature_size=48`) and STU-Net-B (58M) at 64³ × 32 samples are feasible in fp16 with gradient checkpointing, but heavier. V100 has no bf16, so use `torch.autocast(dtype=torch.float16)` + `GradScaler`.

### Cited Findings
- STU-Net sizes: S 14.6M / 0.13 TFLOPs, **B 58.26M / 0.51 TFLOPs**, L 440M / 3.81 TFLOPs, H 1.46B / 12.6 TFLOPs, each at a 128³ crop — [STU-Net README](https://github.com/uni-medical/STU-Net).
- Swin UNETR (Tang et al.) = 62.19M params (SuPreM table). SuPreM U-Net = 19.08M and SegResNet = 4.70M — [SuPreM README](https://github.com/MrGiovanni/SuPreM).
- nnFoundationCNN = 102M (6 stages); nnFoundationViT = 674M — [HF card](https://huggingface.co/MIC-DKFZ/nnFoundationCNN). CT-FM feature extractor = 77.8M — [HF](https://huggingface.co/project-lighter/ct_fm_feature_extractor).
- MONAI Swin UNETR SSL pretraining ran on "8 X 32G V100 GPUs" at batch 1 per GPU, ROI 96. A `--use_checkpoint` (gradient checkpointing) single-GPU mode exists — [Pretrain README](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/Pretrain). VoCo's loading snippet enables `use_checkpoint=True` by default — [VoCo README](https://github.com/Luffy03/VoCo).
- VoCo-H exceeded GPU memory and VoCo-L did not converge in nnFoundation's benchmark — [arXiv 2609.26924](https://arxiv.org/abs/2609.26924).
- Current model: 2.2 GiB, 278 ms/step at batch 16 (run 2). 6b is 310 ms/step (Experiments.md).

### Inferences (estimates; verify with `scripts/train.py --bench`)
- **Truncated ResEnc-L, stem to stage 2.** Params ≈ stage 2 (4 blocks × 2 convs × 27 × 128²) ≈ 3.5M + stage 1 ≈ 0.7M + stem/stage 0 ≈ 0.06M, about **4.3M** in total.
- **Activations.** Stage 0 runs at full 64³ with 32 ch, about 8.4M values per tensor per sample, and dominates. Freezing stem + stage 0 (Q3) means no stored activations there. Stages 1–2 at fp16, ~3 tensors kept per conv, 32 samples: very roughly 2–4 GB. **Full fine-tune including stage 0** is very roughly 6–9 GB, which fits a 16 GB V100.
- **Frozen backbone** under `no_grad` keeps only the transient forward memory.
- **Full Swin UNETR / VoCo-B** at 64³ × 32 with gradient checkpointing likely fits 32 GB in fp16; 16 GB is tight without checkpointing. Since only `hidden_states_out[0..1]` are needed, run `model.swinViT` alone and stop early: call `patch_embed` and `layers1` manually instead of the full forward. This cuts most of the cost.
- **STU-Net-B, full encoder.** At 0.51 TFLOPs per 128³ forward, a 64³ forward is ~1/8, ≈ 64 GFLOPs. × 32 samples × ~3 for fwd+bwd ≈ 6 TFLOPs per step. On a V100 (~30–60 effective fp16 TFLOPS with conv3d) that is roughly 0.1–0.3 s per step: feasible. Truncate to the first 2–3 stages anyway.
- **AMP snippet** (`train.py` currently has none):
  ```python
  scaler = torch.cuda.amp.GradScaler()
  with torch.autocast("cuda", dtype=torch.float16):
      loss, parts = step_loss(model, x, t, a, epoch - 1)
  scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
  ```
  `soft_argmax` and `team2_loss` already cast logits to fp32, so the softmax stays stable. Pitfall: the cosine xcorr divides by k³ and multiplies by `exp(log_scale)` (~10). That is fine in fp16, but keep `F.normalize` outputs in fp16 only if the channel count is small. Otherwise cast to fp32 before `xcorr`.
- **Gradient checkpointing** for the CNN: `torch.utils.checkpoint.checkpoint_sequential(bb.stages[1], 3, x, use_reentrant=False)`. It is only needed if stage 0 is unfrozen.
- **Throughput.** Training is data-loading bound at ~180 s per epoch, so a backbone that adds ≤ ~100 ms per step may cost almost no wall time. If it becomes GPU-bound, raise `--workers` first, or move flips to the GPU.

### Gaps
- No published memory or throughput numbers for these backbones at 64³ × batch 32 on V100 were found. All numbers above are estimates.

---

## Q6. Minimal-diff experimental plan

### Takeaway
Start with **one frozen-backbone run that keeps everything else of run 6b fixed**, paired with a **random-init control of the same truncated architecture**. Only the pair separates the pretraining effect from the architecture change. Go further (partial fine-tuning, the coarse map, a mask channel) if pretrained beats random-init by more than the noise (≥ ~0.3 mm on the TTA mean, or a paired per-lesion test). A frozen MAE encoder may lose to the 6b baseline and still be worth fine-tuning.

### Cited Findings
- nnU-Net's nnssl tooling provides a `--from_scratch` flag: "To train with the **same configuration** but **without** loading the pre-trained weights (a fair baseline)" — [nnU-Net doc](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/finetuning_from_nnssl_checkpoints.md).
- `DynamicPretrainedTrainer` "prints how many of the network's encoder parameters were loaded from the checkpoint". nnU-Net now "Refuse[s] nnssl checkpoints whose architecture does not match the base plan" (commit 42b4fd8, 2026-11-05) — [nnU-Net doc](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/finetuning_from_nnssl_checkpoints.md). Silent partial loads are a known failure mode; check the counts.
- The frozen schedule underperforms full fine-tuning by a large margin for MAE ResEnc-L segmentation (Q3) — [arXiv 2412.17041](https://arxiv.org/abs/2412.17041).
- Experiments.md: differences of 0.04–0.06 mm were called "noise" or "tie"; TTA is a "paired comparison on the same lesions, so it is less noisy". The best 6b epoch is ~34 of 50.

### Inferences
**Step 0: plumbing checks (minutes, no SLURM job).**
1. Load the checkpoint:
   ```python
   ck = torch.load(path, map_location="cpu", weights_only=True)
   sd = {k[len("encoder."):]: v for k, v in ck["network_weights"].items() if k.startswith("encoder.")}
   enc = get_res_enc_l(1, 1).encoder
   missing, unexpected = enc.load_state_dict(sd, strict=False)
   ```
   Assert `missing` is empty for stem / stages 0–2. (`get_res_enc_l` is in `nnssl.architectures.architecture_registry`.)
2. Shape check: `stages[1]` output is `(B,64,32,32,32)` and `stages[2]` is `(B,128,16,16,16)` for a 64³ input.
3. Grid check: put a synthetic delta at voxel (31.5 + 2j) and confirm `soft_argmax` returns 4j mm with the corrected `coords`.
4. Run the existing overfit check on 10 files (expect ≈ 0 mm) and `--bench` for memory and ms/step.

**Code diff (smallest version).**
- Add a `backbone` option to `SiameseNet.__init__`. When set, replace `self.encoder` with a module that returns stage-1 features followed by `Conv3d(64, 32, 1)` (trainable projection).
- `forward` stays as is, except:
  - The encoder input is 1 channel (CT only). Mask handling is option 1 of Q1: the mask is used only in the mask-pooled template.
  - `coords` gets the −1 mm offset.
  - Mask pooling switches to `(3, 2, 1)` in both `model.py` and `losses.py`. Pass the kernel from the model, e.g. a `model.pool_k` attribute that `step_loss` forwards to `team2_loss`.
- Add the `--backbone {none,resenc_ckpt_path}`, `--freeze-backbone` and `--bb-lr-mult` flags to `train.py`.

**Run matrix.** All runs use run-6b settings: `--rejitter-sd 8 --bl-mask --lambda-mask 0.5 --aug-root … --ema 0.999 --epochs 50 --template-k 5`. Readout: TTA mean / median, CPM@10 and per-patient mean.

| Run | Backbone | Train | Purpose |
|---|---|---|---|
| P1 | nnFoundationCNN stem→stage 1, **frozen**, + 1×1 proj | head only | Do pretrained 4 mm features carry matching signal as they are? |
| P1-rand | same architecture, **random init**, trainable | all | Architecture-only control (nnU-Net `--from_scratch` analogue) |
| P2 | P1 + stage 2 coarse map, summed logits (Q2) | head only | Does 8 mm context help? Stage 1 RF is smaller than the current ~100 mm. |
| P3 | P2, **stages 1–2 fine-tuned** (bb-lr 1e-4 / 5e-5, 10% warmup) | partial | Expected best, per OpenMind / nnFoundation evidence |
| P3-rand | P3 architecture, random init, same schedule | all | The pretraining effect under fine-tuning |
| P4 | P3 + zero-init BL-mask stem channel (needs stem + stage 0 trainable at low LR) | partial+ | Mask in the encoder vs head-only |

**Decision rules.**
- If P3 < P3-rand and P3 < 6b by ≥ 0.3 mm (TTA mean), keep the backbone and try a second checkpoint, e.g. CT-FM SegResEncoder level 1 (CT-only contrastive).
- If P1 ≈ P1-rand and P3 ≈ P3-rand, pretraining is not helping at 2 mm / 64³. Stop and report it.
- Run a paired per-lesion comparison (Wilcoxon on the 207 val errors, or a bootstrap of the mean difference) rather than eyeballing means. Patients e2c0be2456 and 3988c7f88e dominate val (37%), so also report the per-patient mean.

**Cheap ablations (no new code once P3 exists).** Projection dim 32 vs 64; Sawtooth vs plain warmup; frozen stage 1 + fine-tuned stage 2 only; concat + 1×1 fusion vs summed logits.

**Expensive ablations (new preprocessing).** Wider HU window (e.g. [-1000, 2000]) to match the pretraining intensity range. A 1.5 mm or 1 mm re-cut.

**Backbone shortlist for this code, ranked by integration effort.**
1. **nnFoundationCNN.** CT+MRI+PET, ResEnc with InstanceNorm, no-resampling pretraining matches 2 mm, `weights_only` checkpoint, stride-2 stage at 64 ch.
2. **CT-FM SegResEncoder.** CT-only contrastive, one `from_pretrained` call. Check for BatchNorm.
3. **STU-Net-B, or SuPreM U-Net / SegResNet.** Supervised on TotalSegmentator / AbdomenAtlas organ labels, so the features are organ-semantic. That suits anatomy-consistent matching, but lesion texture may be under-represented.
4. **MONAI Swin SSL / VoCo / SuPreM-Swin.** Awkward 4 mm tap (Q2). Only via `hidden[1]` + upsampling.

OpenMind ResEnc-L checkpoints are MRI-pretrained; deprioritise them for CT.

### Gaps
- No prior work found applies an nnssl/nnFoundation encoder inside a Siamese correlation tracker. The expected effect size is unknown.
- The thresholds above (0.3 mm, Wilcoxon) are a proposal based on the noise level Experiments.md reports, not on a power analysis.
