# Pretrained 3D CT encoders / foundation models as the shared encoder of a 3D Siamese lesion-localisation model

Scope note: the task uses 64³ patches at 2 mm isotropic (128 mm cube), HU clipped to [-1000, 400] and scaled to [0, 1]. The current encoder gives a 32³ grid of 4 mm cells. "Fit" below means: can the model give dense, discriminative features near 4 mm on such a crop, and can it run on a V100?

Research budget: about 17 tool calls (Nov 2026). Many per-model preprocessing details (spacing, HU window, strides) were not on the pages fetched. They are listed under Gaps and not stated as fact.

## Q1. What is "nnFoundation"? How does it relate to nnssl and OpenMind? Which checkpoints exist, and are they CT or MRI?

### Takeaway
"nnFoundation" is a real, named model pair from DKFZ, released in late 2026 (arXiv 2609.26924, Harsy, Wald, Gotkowski, Kirchhoff, Knopp, Rokuss et al.). It contains nnFoundationCNN (ResEnc, 102M, MAE) and nnFoundationViT (Primus, 674M, MAE). Both were pretrained on 2.16M CT/MRI/PET volumes, of which only 26.6% are CT. The code lives in the MIC-DKFZ/nnssl repo. The earlier OpenMind checkpoints (ICCV 2025) are MRI-only, so they are ruled out for this CT task.

### Cited Findings
**nnssl / nnFoundation**
- The nnssl repo is a self-supervised learning framework for 3D medical imaging. Its preprocessing follows nnU-Net, and data goes in through a `pretrain_data.json`. The repo "holds the code for nnFoundation", a "pair of 3D radiology foundation models". — [MIC-DKFZ/nnssl](https://github.com/MIC-DKFZ/nnssl)
- Released checkpoints: nnFoundationCNN is a ResEnc with 6 stages, features 32-64-128-256-320-320, 102M params ([HF](https://huggingface.co/MIC-DKFZ/nnFoundationCNN)). nnFoundationViT is a Primus with 40 layers, embedding dim 1056, 16 heads, 8³ patch tokens, 674M params ([HF](https://huggingface.co/MIC-DKFZ/nnFoundationViT)). — [MIC-DKFZ/nnssl](https://github.com/MIC-DKFZ/nnssl)
- Paper: "nnFoundation: 3D Foundation Models for Radiology" (Harsy et al., 2026), arXiv:2609.26924. — [arXiv](https://arxiv.org/abs/2609.26924)
- Pretraining data: 2,158,570 volumes after filtering, from 3,271,454 series and 125 sources. Modalities: MRI 71.7%, CT 26.6%, PET 1.7%. That is about 574k CT volumes (26.6% × 2.16M). — [arXiv HTML](https://arxiv.org/html/2609.26924v1)
- Objective: MAE for both models. CNN: mask ratio 0.75 with 16³ masked patches, SGD, batch 8. ViT: mask ratio 0.8, 8-layer decoder, AdamW, 625k steps, global batch 96. Pretraining input patch 192³ for both. — [arXiv HTML](https://arxiv.org/html/2609.26924v1)
- Spacing and normalisation: pretraining data was "not resampled to a fixed target spacing". "Z-score normalization was applied to all images." No CT-specific clipping is mentioned. Downstream classification uses 1 mm isotropic, and segmentation uses the nnU-Net dataset-specific spacing. — [arXiv HTML](https://arxiv.org/html/2609.26924v1)
- The HF card for nnFoundationCNN says: z-score normalised single-channel 3D volumes, do not resample, recommended downstream patch 192³. The checkpoint `checkpoint_final.pth` (410 MB) holds `network_weights`, the `nnssl_adaptation_plan` (architecture and preprocessing plan) and citations. `all_modules.*` keys alias the same tensors, and deduplicating them breaks loading. — [HF nnFoundationCNN](https://huggingface.co/MIC-DKFZ/nnFoundationCNN)
- Licence: CC-BY-SA-4.0 for the repo and the CNN weights card. Attribution and share-alike apply. Commercial use is not excluded. — [HF nnFoundationCNN](https://huggingface.co/MIC-DKFZ/nnFoundationCNN), [nnssl](https://github.com/MIC-DKFZ/nnssl)
- Integration: fine-tuning goes through nnU-Net (segmentation), nnDetection and an image_classification repo. Usage: `nnUNetv2_preprocess_like_nnssl -pc https://huggingface.co/MIC-DKFZ/nnFoundationCNN -am like_pretrained`, or `hf_hub_download("MIC-DKFZ/nnFoundationCNN", "checkpoint_final.pth")`. — [HF nnFoundationCNN](https://huggingface.co/MIC-DKFZ/nnFoundationCNN)
- Compute: ViT pretraining took about 12,500 A100 GPU-hours, CNN pretraining about 500 A100 GPU-hours. — [arXiv HTML](https://arxiv.org/html/2609.26924v1)
- Results:
  - Over 34 segmentation datasets, nnFoundationCNN beat CTFM by 3.0 Dice points and nnU-Net by 1.9.
  - The authors say "the convolutional model dominates spatially localized tasks, whereas the transformer model excels in tasks requiring global semantic reasoning and in frozen-feature settings".
  - Low data: about +3 Dice with 10 training cases.
  - Baselines compared, with their params: CTFM 87M, CURIA 86M, MERLIN 271M, MIS-FM 142M, SwinUNETR 102M, VISTA3D 218M, VoCo-B 54M.
  - [arXiv HTML](https://arxiv.org/html/2609.26924v1)
- The fetched part of the paper (first ~100k of 208k chars) had no registration, correspondence or lesion-tracking evaluation and no frozen-feature or linear-probe numbers. — [arXiv HTML](https://arxiv.org/html/2609.26924v1)

**OpenMind (predecessor benchmark)**
- "An OpenMind for 3D Medical Vision Self-supervised Learning" (Wald et al., ICCV 2025) releases a 114k-volume public 3D MRI pretraining set. Sources describe the anatomy differently: "brain" in one place, "head-and-neck" in a figure caption. — [arXiv 2412.17041](https://arxiv.org/pdf/2412.17041), [ICCV listing](https://mlanthology.org/iccv/2025/wald2025iccv-openmind)
- Checkpoints cover ResEnc-L (CNN) and Primus-M (transformer). Methods: VoCo, VolumeFusion, Models Genesis, MAE, Spark 3D (CNN only), SimMIM (transformer only), SwinUNETR-SSL, SimCLR. Repos sit under AnonRes/* and MIC-DKFZ/*, for example `MIC-DKFZ/ResEncL-OpenMind-SwinUNETR`, `MIC-DKFZ/PrimusM-OpenMind-MG`, `AnonRes/ResEncL-OpenMind-VF`. — [HF AnonRes/PrimusM-OpenMind-MG](https://huggingface.co/AnonRes/PrimusM-OpenMind-MG/blob/main/README.md), [HF MIC-DKFZ/ResEncL-OpenMind-SwinUNETR](https://huggingface.co/MIC-DKFZ/ResEncL-OpenMind-SwinUNETR)
- Segmentation: ResEnc-L MAE scored 70.91 overall vs 70.47 for ResEnc-L from scratch (1k epochs). Primus-M MAE scored 70.42. Contrastive methods did better on classification, reconstruction methods on segmentation. — [arXiv 2412.17041](https://arxiv.org/pdf/2412.17041) (via search summary)
- The OpenMind HF model cards say the checkpoints are "not recommended for direct feature extraction". They point to the nnssl and nnU-Net fine-tuning repos instead. Licence: cc-by-4.0. — [HF AnonRes/ResEncL-OpenMind-VF](https://huggingface.co/AnonRes/ResEncL-OpenMind-VF/blob/main/README.md)

### Inferences
- If the user says "nnFoundation", they almost certainly mean nnFoundationCNN/ViT (the 2026 nnssl release), not OpenMind. OpenMind is MRI-only, so rule it out for CT.
- nnFoundationCNN is the best-fitting DKFZ option. The six ResEnc stages follow nnU-Net convention: stage 0 at stride 1, then stride 2 per stage. On a 2 mm, 64³ input that gives grids of 64³, 32³ (4 mm, 64 ch), 16³ (8 mm, 128 ch), 8³, 4³ and 2³. The stride-2 stage matches the current 4 mm cell, but its receptive field is small. A U-Net-style fusion (upsample stage 3/4 to 32³ and concatenate with stage 1) would be needed for context. This stride layout is inferred from the nnU-Net ResEnc convention. Confirm it by reading `nnssl_adaptation_plan` in the checkpoint.
- nnFoundationViT uses 8³ tokens: 16 mm at 2 mm spacing, so a 64³ crop gives only 8³ tokens. That is too coarse for 4 mm matching without a decoder. At 674M params it is also heavy for a V100 (no bf16, no native flash attention).
- Intensity mismatch: the pipeline stores [0,1] from [-1000, 400] HU. nnFoundation expects z-scored, unresampled volumes. Recover HU as `x*1400-1000`, then z-score (per-patch or with dataset statistics; which one nnssl used for CT is unknown). Clipping at 400 HU removes dense bone and contrast detail the model saw in pretraining. It is probably a minor shift for soft-tissue lesions.
- Spacing mismatch: pretraining used native spacing (typically ~0.7–1 mm in-plane, 1–5 mm slices). The 2 mm patches are coarser than most pretraining data, so the effective receptive field in mm is about 2× larger than in pretraining. Re-extracting patches at about 1–1.5 mm (96–128³ voxels for the same 128 mm cube) would match the pretraining distribution better but costs about 3–8× the compute.

### Gaps
- The exact CT share of each checkpoint's training batches and the per-stage strides of nnFoundationCNN were not stated on the pages read.
- Licence of the nnFoundationViT weights was not checked separately (the repo is CC-BY-SA-4.0).
- Whether the second half of the nnFoundation paper has registration or correspondence results is unknown, because it was not read.
- The pycad blog post on nnFoundation was not read.

## Q2. Other candidates with CT weights: catalogue

### Takeaway
Released, usable CT encoders include:
- **Dense-correspondence specialist:** SAM / UAE (Alibaba DAMO), MIT licence.
- **General segmentation or SSL encoders:** nnFoundationCNN, VoCo (SwinUNETR), SuPreM (U-Net / SegResNet / SwinUNETR), STU-Net (nnU-Net), CT-FM (SegResEncoder).
- **Global or semantic models:** Merlin (I3D ResNet152, CLIP-style), FMCIB (lesion-centred SimCLR), SAM-Med3D (promptable ViT).

Only SAM/UAE was trained for the exact property needed: pixel-wise anatomical embeddings that match across CT scans, including follow-up lesion matching, at 2 mm spacing.

### Cited Findings
**SAM: Self-supervised Anatomical eMbedding (Yan et al., IEEE TMI) and successors UAE-S / UAE-M**
- Method: SAM "generates semantic embeddings for each image pixel that describes its anatomical location or body part". It uses pixel-level contrastive learning, coarse-to-fine matching and hard-negative selection. — [arXiv 2012.02383](https://arxiv.org/abs/2012.02383)
- Architecture: "A lightweight 3D ResNet-18 is adopted as backbone with a feature pyramid network (FPN)", initialised from ImageNet via I3D inflation. A global embedding comes from the coarsest FPN level and a local embedding from the finest. Each is c = 128 channels, produced by 3×3×3 convs and L2-normalised. — [arXiv HTML v3](https://arxiv.org/html/2012.02383v3)
- Preprocessing: "We normalize CT images to an isotropic spacing of 2mm." Training crop 32×96×96, batch 8 pairs, about 25k iterations, RAdam with lr 1e-4. Negatives must be more than 3 mm from the anchor. — [arXiv HTML v3](https://arxiv.org/html/2012.02383v3)
- Training data (no labels): DeepLesion (~20k sub-volumes), NIH-LN (176 scans) and an in-house ChestCT set (94 patients). — [arXiv HTML v3](https://arxiv.org/html/2012.02383v3)
- Lesion matching on 3,556 DeepLesion follow-up lesion pairs: 91.1% of matched points fall inside the lesion's 3D box. Baselines scored 80.7% (lesion embedding), 82.1% (LesaNet) and 82.1% (DLT). Mean radial error was 2.7±2.5, in pixels (not mm). — [arXiv HTML v3](https://arxiv.org/html/2012.02383v3)
- ChestCT landmarks: MRE 4.3±3.0 px in 0.23 s, vs DEEDS registration 4.6±3.3 px in 50 s and VoxelMorph 7.3±3.6 px. — [arXiv HTML v3](https://arxiv.org/html/2012.02383v3)
- Repo `self-supervised-anatomical-embedding-v2` releases three checkpoints:
  - `SAM.pth`, trained on NIH-LN.
  - `SAMv2_iter_20000.pth`, the UAE-S single-modality model, trained on NIH-LN and TotalSegmentator.
  - `CrossSAM_iter_20000_ct_mr_inter_1.pth`, the UAE-M CT-MRI model.
  - Weights download via ModelScope `snapshot_download('viggin/Self-supervised_Anatomical_Embeddings', revision='v1.0.0')`.
  - Licence MIT.
  - Stack: PyTorch 1.9.0, mmcv 1.4.7, mmdetection 2.20, TorchIO.
  - Inputs are assumed to be in LPS+ orientation.
  - Activity: 7 commits, 84 stars, no releases.
  - [GitHub](https://github.com/alibaba-damo-academy/self-supervised-anatomical-embedding-v2)
- Note the conflict: the repo lists NIH-LN as the training data for the released `SAM.pth`, while the paper used DeepLesion + NIH-LN + ChestCT. — [GitHub](https://github.com/alibaba-damo-academy/self-supervised-anatomical-embedding-v2) vs [arXiv HTML v3](https://arxiv.org/html/2012.02383v3)

**VoCo / "Large-Scale 3D Medical" (Wu et al., Luffy03)**
- Checkpoints on HF `Luffy503/VoCo` (`https://huggingface.co/Luffy503/VoCo/resolve/main/<name>.pt`):

  | Model | Params | Type |
  |---|---|---|
  | VoCo_B_SSL_head | 53M (feature_size 48) | self-supervised |
  | VoCo_L_SSL_head | 206M (feature_size 96) | self-supervised |
  | VoCo_H_SSL_head | 818M (feature_size 192) | self-supervised |
  | VoComni_B / L / H | 72M / 290M / 1.2B | omni-supervised |
  | VoComni_nnunet | 31M | omni-supervised |

  — [GitHub Large-Scale-Medical](https://github.com/Luffy03/Large-Scale-Medical)
- Pretraining data: PreCT-160K (160k CT volumes, ~42M slices). VoComni uses 20k volumes with pseudo-labels for 20 organ and tumour classes. — [GitHub](https://github.com/Luffy03/Large-Scale-Medical)
- Framework: MONAI `SwinUNETR(use_v2=True)`, PyTorch 2.1.1. The example uses a 96³ ROI. Licence Apache-2.0. It reports 50+ downstream tasks, including registration, but the README gives no low-data numbers. — [GitHub](https://github.com/Luffy03/Large-Scale-Medical)

**SuPreM (Li et al., AbdomenAtlas; MrGiovanni)**
- Supervised pretraining on AbdomenAtlas 1.1 (9,262 CT, per-voxel labels for 25 organs plus pseudo-labels for 7 tumour types). The released SuPreM checkpoints are listed as trained on 2,100 CT:
  - U-Net 19.08M (`supervised_suprem_unet_2100.pth`)
  - SegResNet 4.70M (`supervised_suprem_segresnet_2100.pth`)
  - Swin UNETR 62.19M (`supervised_suprem_swinunetr_2100.pth`)
  - URL pattern `https://huggingface.co/MrGiovanni/SuPreM/resolve/main/<file>`
  - [GitHub SuPreM](https://github.com/MrGiovanni/SuPreM)
- The same repo re-hosts older baselines in one place:
  - Models Genesis U-Net: 19.08M, 623 CT.
  - Med3D (MedicalNet) residual U-Net: 85.75M, 1,638 CT. The filename says 1,623.
  - DoDNet: 17.29M, 920 CT.
  - UniMiSS tiny and small: CT and MRI, 5,022 volumes.
  - Universal Model (U-Net and Swin UNETR): 2,100 CT.
  - Tang et al. Swin UNETR SSL: 5,050 CT.
  - Valanarasu et al. Swin UNETR: 50,000 CT/MRI.
  - [GitHub SuPreM](https://github.com/MrGiovanni/SuPreM)
- The README says "Paper content is covered by patents pending". The licence text was not reproduced on the page. The code is modified from MONAI. — [GitHub SuPreM](https://github.com/MrGiovanni/SuPreM)

**STU-Net (Huang et al., 2023; uni-medical)**
- Sizes, all at a 128³ crop:

  | Model | Params | FLOPs |
  |---|---|---|
  | S | 14.6M | 0.13T |
  | B | 58.26M | 0.51T |
  | L | 440.30M | 3.81T |
  | H | 1457.33M | 12.60T |

  — [GitHub STU-Net](https://github.com/uni-medical/STU-Net)
- Supervised pretraining on TotalSegmentator (1,204 CT, 104 structures) for 4,000 epochs. Built on nnU-Net (bundles nnUNet 1.7.1 and 2.2). Weights on Baidu and Google Drive. Licence Apache-2.0. Paper dated April 2023. — [GitHub STU-Net](https://github.com/uni-medical/STU-Net)

**CT-FM (Pai et al., project-lighter)**
- SegResEncoder of about 77M params, pretrained on 148,000 CT scans. It uses a modified SimCLR with intra-sample contrastive learning (patches within the same scan), 500 epochs, with the epoch-449 checkpoint selected. HF org `project-lighter`. Licence MIT. Compatible with SegResNetDS and Auto3DSeg. — [GitHub CT-FM](https://github.com/project-lighter/CT-FM)
- Note the conflict: the nnFoundation paper lists CTFM at 87M params, vs about 77M in the CT-FM README. — [arXiv 2609.26924](https://arxiv.org/html/2609.26924v1) vs [GitHub CT-FM](https://github.com/project-lighter/CT-FM)

**FMCIB: Foundation model for cancer imaging biomarkers (Pai et al., Nature Machine Intelligence 2024)**
- SimCLR-based (topic tag). PyPI package `foundation-cancer-image-biomarker`. Licence MIT. 164 commits. The landing page does not state architecture, input size, spacing or weights URL. — [GitHub AIM-Harvard](https://github.com/AIM-Harvard/foundation-cancer-image-biomarker)

**Merlin (Stanford MIMI; Nature 2026)**
- Image encoder: inflated 3D (I3D) ResNet152, trained with paired CT and radiology reports (vision-language). Weights on HF `stanfordmimi/Merlin`, with extra report-generation, disease-prediction and nnU-Net files. Code and weights MIT. The training dataset needs a DUA. — [HF stanfordmimi/Merlin](https://huggingface.co/stanfordmimi/Merlin), [arXiv 2406.06512](https://arxiv.org/html/2406.06512v2)
- 271M params according to the nnFoundation comparison table. — [arXiv 2609.26924](https://arxiv.org/html/2609.26924v1)

**SAM-Med3D / SAM-Med3D-turbo (uni-medical)**
- Promptable 3D segmentation model trained on SA-Med3D-140K ("143K 3D masks and 245 categories"). Turbo weights are `sam_med3d_turbo.pth` on HF `blueyo0/SAM-Med3D`. Licence Apache-2.0. The README does not state architecture details or preprocessing. — [GitHub SAM-Med3D](https://github.com/uni-medical/SAM-Med3D)

**MIS-FM, VISTA3D, CURIA, SwinUNETR**
- These appear only as baselines in the nnFoundation table: MIS-FM 142M, VISTA3D 218M, CURIA 86M, SwinUNETR 102M. — [arXiv 2609.26924](https://arxiv.org/html/2609.26924v1)

### Inferences
**Ranking for dense BL-to-FU matching (inference, not benchmarked by any source for this task):**
1. **SAM / UAE-S.** Its training objective is the matching task itself. Its native spacing is 2 mm, which equals the current data. It was evaluated on follow-up lesion matching (91.1% in-box on DeepLesion) and its L2-normalised 128-d local embeddings mirror the current design. Costs: an obsolete mmdet 2.x / mmcv 1.4 / PyTorch 1.9 stack (port the ResNet-18+FPN to plain PyTorch, or run in a separate conda env), anisotropic 32×96×96 training crops, and no HU window stated. Two uses:
   - a zero-shot baseline: take the BL-centre embedding, then argmax its similarity in the FU patch;
   - a frozen feature channel concatenated with the learned features.
2. **nnFoundationCNN.** Newest, largest CT+MRI pretraining, and the authors say the CNN "dominates spatially localized tasks". MAE features are reconstruction-oriented and not trained to be invariant across timepoints, so expect to fine-tune rather than freeze. Licence CC-BY-SA.
3. **CT-FM.** Its intra-sample patch contrast is closer to a local discriminative objective than MAE is, and it is CT-only (148k). MIT licence, MONAI-compatible.
4. **VoCo-B, SuPreM U-Net / SegResNet, STU-Net-S.** Solid CT encoders with segmentation-oriented features. Mostly abdominal (SuPreM) or whole-body TotalSegmentator (STU-Net). STU-Net-S (14.6M) and SuPreM SegResNet (4.7M) are the cheapest to try.
5. **Merlin, FMCIB, SAM-Med3D, ViT-based models.** Global or semantic objectives with coarse token grids (ViT patch 8–16 voxels). Poor fit for 4 mm voxel-wise matching on a 64³ crop.
6. **Models Genesis and MedicalNet / Med3D.** 2019-era (flag as old). Models Genesis is a 19M U-Net trained on 623 CT. Usable but superseded.

**Likely dataset (side note):** the description fits the autoPET/CT IV Task 2 longitudinal whole-body CT dataset (Tübingen/Lübeck, Hering, Küstner et al.) better than LesionLocator's release.
- autoPET/CT IV Task 2: 300 patients with baseline and follow-up CT, lesion masks, lesion centres and propagated FU centres. Hosted on FDAT, DOI 10.57754/FDAT.qwsry-7t837, March 2025. — [arXiv 2509.00613](https://arxiv.org/html/2509.00613v1)
- LesionLocator's release is a synthetic-follow-up dataset (~5,200 images, ~700 GB, CC BY-NC-SA 4.0). — [search summary of LesionLocator/DKFZ record](https://inrepo02.dkfz.de/record/299862), [LesionLocator arXiv](https://arxiv.org/html/2502.20985v1)
- A separate longitudinal whole-body CT paper annotates lesion evolution such as persistence, regression, merging and new lesions. — [Uni Lübeck record](https://research.uni-luebeck.de/en/publications/a-longitudinal-whole-body-ct-dataset-with-manually-annotated-tumo/)
- Inference: the project's 224 train patients plus val/test patients may come to about 285, which would equal 300 minus the 15 excluded edge cases reported in 2509.00613. Unverified.
- Rokuss (LesionLocator, nnFoundation co-author) also co-authored the autoPET IV paper, so the DKFZ and nnFoundation link is real but indirect.

### Gaps
- SAM: numeric strides of the global and local FPN levels were not found (the paper refers to Fig. 4), and neither was the HU window. UAE-S architecture and spacing details were not checked (arXiv paper not fetched).
- VoCo, SuPreM, STU-Net, CT-FM: expected spacing and HU window were not on the READMEs fetched. From general knowledge (unverified this session): the MONAI SwinUNETR family (VoCo, SuPreM) usually uses 1.5 mm isotropic and HU [-175, 250] scaled to [0, 1]. STU-Net follows nnU-Net CT normalisation (percentile clip, then global z-score) at TotalSegmentator's 1.5 mm. Verify in each repo's transforms before use.
- FMCIB: architecture, patch size, spacing and weights URL were not verified. From memory it is a 3D ResNet50 trained on ~11k DeepLesion lesion-centred 50³ patches at 1 mm, but this is unconfirmed and should be checked in the docs or the paper.
- CT-CLIP / CT-RATE, MIS-FM (volume fusion) weights, TotalSegmentator nnU-Net weights as a standalone encoder: not researched (search budget). The SuPreM licence text was also not read.
- SAM-Med3D architecture (ViT patch size, 128³ at 1.5 mm) was not verified this session.

## Q3. Which models give dense features at about 4 mm on 128 mm crops, and which expect different spacing or normalisation?

### Takeaway
On a 2 mm, 64³ input, any CNN encoder with a stride-2 stage gives a 32³ grid of 4 mm features: nnFoundationCNN, STU-Net, SuPreM U-Net/SegResNet, CT-FM, SAM's finest FPN level. ViTs with patch size 8 or more give 16 mm tokens or coarser. Only SAM natively expects 2 mm. The others expect native spacing (nnFoundation), about 1–1.5 mm (most MONAI and nnU-Net CT models), and z-score or soft-tissue-window intensities, not the current [-1000, 400] → [0, 1].

### Cited Findings
- SAM: CT is resampled to isotropic 2 mm, and local embeddings come from the finest FPN level, L2-normalised, 128-d. — [arXiv HTML v3](https://arxiv.org/html/2012.02383v3)
- nnFoundation: no resampling during pretraining, z-score normalisation, 192³ patches. Downstream classification at 1 mm. — [arXiv HTML](https://arxiv.org/html/2609.26924v1), [HF card](https://huggingface.co/MIC-DKFZ/nnFoundationCNN)
- nnFoundationViT uses 8³ patch tokens. — [nnssl README](https://github.com/MIC-DKFZ/nnssl)
- STU-Net was trained at a 128³ crop. — [GitHub](https://github.com/uni-medical/STU-Net)
- VoCo example uses a 96³ ROI. — [GitHub](https://github.com/Luffy03/Large-Scale-Medical)

### Inferences
- With 64³ input, five downsamplings reach a 2³ bottleneck. That is fine for ResEnc, SwinUNETR (needs divisibility by 32) and SegResNet, though the deepest stages carry almost no spatial information.
- Practical recipe to match a pretrained model's expected input:
  1. Undo the scaling: `HU = x*1400 - 1000`.
  2. Apply the model's own window or z-score.
  3. Optionally resample the 128 mm cube to the model's spacing (1.5 mm gives about 85³, so pad or crop to 96³). Read features out at the stage whose stride × spacing ≈ 4 mm, or trilinearly resample them to 32³.
- The current [-1000, 400] clip discards >400 HU. Every model above saw bone and contrast >400 HU in pretraining. Re-extracting patches with a wider clip (e.g. [-1024, 3071]) is cheap and removes this shift.

### Gaps
- No source measured feature quality as a function of readout stride for correspondence. The best stage per model must be tested empirically, for example with a zero-shot nearest-neighbour matching error on val.

## Q4. Compute: parameters and V100 feasibility at 64³, batch about 16 pairs (32 volumes)

### Takeaway
Parameter counts (sources above):

| Model | Params |
|---|---|
| SuPreM SegResNet | 4.7M |
| STU-Net-S | 14.6M |
| SuPreM / Models Genesis U-Net | 19M |
| VoCo-B | 53M |
| SuPreM SwinUNETR | 62M |
| CT-FM | 77–87M |
| nnFoundationCNN | 102M |
| Merlin | 271M |
| VoCo-L | 206M |
| nnFoundationViT | 674M |
| STU-Net-L | 440M |
| VoCo-H | 818M |
| STU-Net-H | 1.46B |

Encoders up to about 100M should fit on a V100 at 64³ with batch 32 volumes using AMP (fp16) and gradient checkpointing, or by freezing the encoder. Multi-hundred-million-parameter models are realistic only frozen or with small batches.

### Cited Findings
- STU-Net FLOPs per 128³ crop: S 0.13T, B 0.51T, L 3.81T, H 12.60T. — [GitHub STU-Net](https://github.com/uni-medical/STU-Net)
- nnFoundationCNN checkpoint is 410 MB (102M params). — [HF](https://huggingface.co/MIC-DKFZ/nnFoundationCNN)
- SAM inference takes 0.23 s per chest CT case (landmark task). — [arXiv 2012.02383](https://arxiv.org/abs/2012.02383)

### Inferences
- A 64³ input is 1/8 of a 128³ crop. STU-Net-S therefore costs about 16 GFLOPs per volume and STU-Net-B about 64 GFLOPs, scaling linearly with voxels for CNNs.
- nnFoundationCNN (ResEnc-L-like): the stride-1 stage at 64³×32 ch is about 8.4M activations per conv output. Memory is dominated by the first two stages. Fine-tuning with batch 32 volumes in fp16 is plausibly 15–30 GB, borderline on a 16 GB V100 and fine on 32 GB. Freezing the encoder and caching features removes most of the cost, but caching is not compatible with on-the-fly augmentation. These numbers are rough estimates, not measured.
- V100 has no bf16. Use fp16 AMP with a grad scaler. MAE-pretrained weights are sometimes fp16-unstable, so check for NaNs.
- With ~2.2k real training lesions, full fine-tuning of 100M+ params risks overfitting. Use frozen or low-LR encoder layers with a small trainable head (the current correlation readout).

### Gaps
- No source gives measured memory for any of these models at 64³ and batch 32. Measure with `torch.cuda.max_memory_allocated` before committing. Which V100 variant (16 or 32 GB) is on the cluster is unknown.

## Q5. Evidence that pretraining helps with small labelled datasets

### Takeaway
The strongest documented small-data gains are segmentation Dice gains (nnFoundation: about +3 Dice at 10 training cases; +1.9 Dice over nnU-Net on average) and one-shot landmark matching (SAM beats supervised models trained on 50 labelled images on X-ray). None of the sources benchmark longitudinal lesion localisation from paired patches. OpenMind shows MAE pretraining gives only a small average gain over a well-trained from-scratch CNN.

### Cited Findings
- nnFoundation:
  - About +3 Dice with 10 training cases.
  - +3.0 Dice over CTFM and +1.9 over nnU-Net across 34 segmentation datasets.
  - Dynamic adaptation gives +1.2 Dice and +1.2 mAP. MSD Pancreas: scratch 56.5 vs adapted pretrained 67.0.
  - Keeps 99.16% of full-schedule performance at 15% of the compute.
  - OOD Dice 74.9 vs 70.4 from scratch.
  - [arXiv HTML](https://arxiv.org/html/2609.26924v1)
- SAM: with one labelled template image it "surpasses supervised methods trained on 50 labeled images" on two X-ray datasets. Follow-up lesion matching accuracy is 91% on CT. — [arXiv 2012.02383](https://arxiv.org/abs/2012.02383)
- OpenMind: ResEnc-L MAE 70.91 vs scratch 70.47 average segmentation score. — [arXiv 2412.17041](https://arxiv.org/pdf/2412.17041)
- SuPreM claims supervised pretraining "strikes as a preferred choice" over self-supervised pretraining (figure only, no numbers on the page). — [GitHub SuPreM](https://github.com/MrGiovanni/SuPreM)
- VoCo README reports 50+ downstream tasks but no low-data numbers. — [GitHub](https://github.com/Luffy03/Large-Scale-Medical)

### Inferences
- The project's 2,159 lesions (plus 8,610 augmentations) is "small" for a 100M encoder but large compared with the 10-case segmentation regimes where nnFoundation shows its biggest gains. Expect modest gains unless the pretrained features carry correspondence-specific invariances, as SAM/UAE does.
- The current error is 5.58 mm mean / 3.49 mm median on a 4 mm grid. Part of the remaining error is sub-cell. A pretrained encoder mainly helps the large-error tail (wrong-match failures), not the sub-voxel refinement. Analyse the val error distribution before and after.
- Cheapest informative experiment: zero-shot SAM/UAE-S matching on val (BL-centre embedding, argmax in FU) compared with the 12.47 mm zero baseline and the 5.58 mm current model. It needs no training and directly tests whether pretrained anatomical embeddings fit this data.

### Gaps
- No published result on autoPET IV longitudinal CT or LesionLocator data was found for frozen foundation-model features in lesion tracking. A Deep Lesion Tracker (DLT) comparison exists only on DeepLesion (82.1% vs SAM's 91.1%).
