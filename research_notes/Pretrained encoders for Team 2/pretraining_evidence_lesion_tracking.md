# Pretrained / self-supervised / foundation features for lesion tracking and dense CT matching

Scope: evidence on whether pretrained encoders help lesion tracking, lesion correspondence and dense anatomical matching in longitudinal CT. The goal is to judge whether Team 2 should swap its 0.12 M-param from-scratch Siamese CNN (val mean 5.58 mm, median 3.49 mm with flip TTA; zero baseline 12.47 mm) for a pretrained encoder.

Important correction to the project context: our data is most likely **not** the DKFZ LesionLocator data. It matches the public **autoPET/CT IV "Longitudinal-CT" dataset** from Universitätsklinikum Tübingen (UKT). That dataset has the same `topology_class` labels we see (UNCHANGED/MERGING, plus DISAPPEARING/NEWLYAPPEARING). DKFZ (Rokuss, Kirchhoff et al.) built on top of this dataset, so the DKFZ work is still the closest prior art.

## Q1. LesionLocator and related DKFZ work: method, encoders, accuracy, weights, zero-shot use

### Takeaway
LesionLocator does not track with a learned similarity search. It warps the baseline prompt with a registration network initialised from uniGradICON, then segments with a ResEnc U-Net pretrained on 47 public segmentation datasets. On its own 60-patient melanoma CT set it reports a 3.1–5.1 mm centre error. Weights are public on Zenodo. It needs full-resolution whole scans, not our 64³ × 2 mm patches. The DKFZ autoPET IV follow-up (LongiSeg) on our likely dataset shows that pretraining helps segmentation Dice by about 6 points over scratch. That paper reports no mm localisation numbers.

### Cited Findings
- **Dataset identity.** The Tübingen "Longitudinal-CT (autoPET/CT IV)" record (DOI 10.57754/FDAT.75kj1-64747) contains whole-body CT of melanoma patients. The page gives two counts: "300" in the introduction and "333" elsewhere. Each patient has a baseline and one or more follow-ups. Per-lesion CSV fields include `cog_bl`, `cog_fu`, `cog_propagated`, `cog_backpropagated`, `topology_class` ∈ {DISAPPEARING, MERGING, NEWLYAPPEARING, UNCHANGED}, `merged_into`, `use_for_challenge` and `linking_unclear`. All data comes from a single site (UKT). The licence is CC BY-NC 4.0. — [FDAT record](https://fdat.uni-tuebingen.de/records/75kj1-64747)
- **LesionLocator tracking mechanism.** A "prompt propagation module" estimates a deformation field between the baseline and follow-up scans and warps the point, box or mask prompt onto the follow-up. The module is a GradICON-based registration network initialised from uniGradICON weights. Tracking is autoregressive: each predicted mask becomes the next prompt. — [LesionLocator, arXiv 2502.20985](https://arxiv.org/html/2502.20985)
- **LesionLocator encoder.** The segmentation backbone is a residual-encoder U-Net. It was pretrained on 47 public 3D segmentation datasets (18,035 images) via nnU-Net/MultiTalent, then fine-tuned on 5,227 images with 22,580 annotated lesions. This is supervised multi-dataset pretraining, not SSL or a foundation model. — [LesionLocator](https://arxiv.org/html/2502.20985)
- **LesionLocator real longitudinal data.** 60 malignant-melanoma patients, 159 scans, about 5 lesions per scan. The set is intra-institutional; the paper names neither the institution nor a public release. The synthetic longitudinal set has 2,728 scans, each with one synthesised timepoint. — [LesionLocator](https://arxiv.org/html/2502.20985)
- **LesionLocator tracking numbers** (5-fold CV on the 60-patient melanoma set, Table 2):

  | Method | Prompt | MED (mm) | CPM@25 (%) |
  |---|---|---|---|
  | "Yan et al." point tracker | Point | 6.92 | n/a |
  | Hering et al. (nnU-Net + Elastix) | Point | 4.45 | 82.50 |
  | Reg. + ULS | Point | 5.81 | 79.81 |
  | Ours | Point | 5.09 | 83.62 |
  | Ours + Synth. | Point | 3.89 | 85.40 |
  | Ours + Synth. | Box | 3.25 | 83.95 |
  | Ours + Synth. | Prev. mask | 3.12 | 85.96 |

  The text reports "a retrieval rate of 86%" and a centre distance of "just 3mm" for the best setting. — [LesionLocator](https://arxiv.org/html/2502.20985)
- **LesionLocator inputs and patch sizes.** The propagation network takes both whole images resampled to [175,175,175]. Segmentation uses [224,224,160] patches at 0.8 × 0.8 × 1 mm. — [LesionLocator](https://arxiv.org/html/2502.20985)
- **Weights and code.** The checkpoint is on Zenodo (records/15174217). Code is GPL-3.0; the synthetic dataset is CC BY-NC-SA 4.0. The README states no licence for the weights. Tracking CLI inputs are the baseline image (`.nii.gz`), a baseline prompt (mask, or point/box JSON) and one or more follow-up images. Every image is resampled to 1.0 × 0.8 × 0.8 mm. The README warns that large scans can run out of RAM. — [GitHub MIC-DKFZ/LesionLocator](https://github.com/MIC-DKFZ/LesionLocator)
- **DKFZ on autoPET IV Task 2 (Kirchhoff, Rokuss, Isensee, Maier-Hein; arXiv 2509.00613).** The method extends LongiSeg (ResEncL U-Net). The challenge provides "propagated center locations in the follow-up for each individual lesion". Alignment uses these given centres, with random training shifts of up to 4 voxels. Pretraining used the LesionLocator synthetic longitudinal dataset (real CT of 2,625 patients plus anatomy-informed augmentation). Results from 5-fold CV, fold 4 excluded: — [arXiv 2509.00613](https://arxiv.org/html/2509.00613v1)

  | Setting | Dice |
  |---|---|
  | Longitudinal + Mask + Point, from scratch | 55.78 |
  | Pretrained cross-sectional + Mask | 62.27 |
  | Pretrained on synthetic longitudinal data | 62.89 |
  | Final | 63.71 |

  The abstract claims "up to 6 Dice points" over models trained from scratch. The longitudinal-over-cross-sectional gain after pretraining is only 0.6 Dice. The paper reports no centre-distance or detection numbers. 15 patients were excluded because prompt points lay at the scan edge. — [arXiv 2509.00613](https://arxiv.org/html/2509.00613v1)
- **Code for the autoPET variant.** github.com/MIC-DKFZ/LongiSeg (autoPET branch). — [search summary of arXiv 2509.00613](https://arxiv.org/html/2509.00613v1)

### Inferences
- The challenge's `cog_propagated` field is effectively what our jittered window centre simulates. In the real challenge, the FU prompt comes from a propagation step, probably registration, not from Gaussian jitter.
- An honest "registration-only" baseline on our data would be the error of `cog_propagated` versus `cog_fu`. Computing it on our train/val lesions is cheap if the raw CSVs are available. It would tell us whether our 5.58 mm beats what the organisers' propagation already gives.
- LesionLocator as a zero-shot baseline is feasible but awkward:
  - It needs whole scans at 0.8 mm. We would have to run it on the original autoPET IV volumes and then map predicted mask centroids back into our patch frame.
  - It solves a different problem: it propagates the whole BL prompt via registration, not from a jittered window.
  - Possible data leakage, unchecked: the LesionLocator pretraining and fine-tuning pool, and its "intra-institutional" melanoma set, may include UKT/autoPET scans.
- Extracting only the ResEnc encoder from LesionLocator or nnU-Net/MultiTalent is possible. That encoder was trained for segmentation at 0.8–1 mm, about 30–100 M parameters. That is very different from our 2 mm, 64³, 0.12 M-param setup. Expect a resampling/scale mismatch and a far bigger model.

### Gaps
- Which "Yan et al." point tracker (6.92 mm) is compared in LesionLocator? It is likely SAM-based (Yan et al. 2022) but not verified; the second part of the HTML was not read.
- No published mm localisation numbers exist for autoPET IV Task 2 (DKFZ report or leaderboard). I found no leaderboard page.
- I did not verify whether the LesionLocator training data overlaps with autoPET IV / UKT CT.
- I did not verify the LesionLocator weights' licence or VRAM needs.

## Q2. SAM (Self-supervised Anatomical Embedding) and successors; DLT ablations

### Takeaway
On the DeepLesion tracking benchmark (DLS), purely self-supervised anatomical embeddings trained without any lesion labels beat the supervised Deep Lesion Tracker. SAM reaches 6.1 mm MED and 86.0% CPM@10mm, against 7.0 mm and 78.9% for DLT. SAM++ improves this to 5.4 mm and 88.3%. This is the strongest published evidence that pretrained dense CT features help lesion correspondence. DLT itself used no pretrained encoder. Its "anatomy signal encoder" is a learned branch, and removing it costs about 2.4 mm.

### Cited Findings
- **DLS test set (480 test pairs; 3,008/403/480 train/val/test)**, as reprinted in SAM++ Table 1. — [SAM++, arXiv 2306.13988](https://arxiv.org/pdf/2306.13988); DLT rows match [DLT, arXiv 2012.04872](https://arxiv.org/html/2012.04872v2)

  | Method | Uses lesion labels? | CPM@10mm (%) | CPM@Radius (%) | MED (mm) |
  |---|---|---|---|---|
  | Affine | no | 48.33 | 65.21 | 11.2 ± 9.9 |
  | VoxelMorph | no | 49.90 | 65.59 | 10.9 ± 10.9 |
  | SiamRPN++ | yes | 68.85 | 80.31 | 8.3 ± 9.2 |
  | LENS-LesaNet | yes | 70.00 | 84.58 | 7.8 ± 10.3 |
  | DEEDS (registration) | no | 71.88 | 85.52 | 7.4 ± 8.1 |
  | DLT-Mix | yes | 78.65 | 88.75 | 7.1 ± 9.2 |
  | DLT | yes | 78.85 | 86.88 | 7.0 ± 8.9 |
  | TransT | yes | 79.59 | 88.99 | 7.6 ± 7.9 |
  | **SAM (self-supervised, zero lesion labels)** | no | **86.04** | **95.00** | **6.1 ± 6.7** |
  | TLT | yes | 87.37 | 95.32 | 6.1 ± 6.7 |
  | **SAM++ (self-supervised)** | no | **88.33** | **96.35** | **5.4 ± 6.0** |

  SAM++ states that DLT, TransT and TLT "utilize lesion annotations during training", while "VoxelMorph, DEEDS, and SAM do not use any lesion information". SAM/SAM++ use a 3D ResNet18 + FPN at 2 mm isotropic with 128-d embeddings. — [SAM++](https://arxiv.org/pdf/2306.13988)
- **DLT-SSL** is DLT trained only with its own self-supervised objective. It scores 71.04 CPM@10mm and 8.8 ± 10.5 mm MED, worse than supervised DLT (7.0 mm). — [DLT](https://arxiv.org/html/2012.04872v2)
- **DLT ablation** (supplementary Table 6, MED in mm, valid/test). — [DLT](https://arxiv.org/html/2012.04872v2)
  - Final configuration: 6.69 / 6.98.
  - Without the anatomy signal encoder (ASE): 8.23 / 9.34.
  - Without the global kernel K_g: 8.77 / 9.29.
  - Feature dim 32: 7.23 / 8.72. Feature dim 128: 7.15 / 7.91.
  - Gaussian signal size 2r: 7.56 / 7.51. Size 8r: 8.40 / 8.81.

  The DLT paper itself reports no SAM initialisation or pretraining ablation.
- **DLT external clinical set:** 536 lesions from 100 longitudinal studies of 86 patients. — [DLT](https://arxiv.org/html/2012.04872v2)
- **SAM++ ablation on the full DLS (3,883 pairs, no DLS training used).** — [SAM++](https://arxiv.org/pdf/2306.13988)
  - SAM: CPM@Radius 93.44, MED 5.8 ± 5.0 mm.
  - SAM + Semantic: 93.99 / 5.6 mm.
  - SAM + Structural inference: 94.44 / 5.3 mm.
  - SAM++: 95.45 / 5.1 ± 3.8 mm.
- **Original SAM paper (Yan et al.).** — [SAM, arXiv 2012.02383](https://arxiv.org/html/2012.02383v3)
  - DeepLesion lesion-matching accuracy (Table III): SAM 91.1%, DLT 82.1%, LesaNet 82.1%, lesion embedding 80.7%.
  - Chest CT 19-landmark MRE, in pixels not mm: SAM 4.3 vs DEEDS 4.6 vs VoxelMorph 7.3.
  - Inference takes 0.23 s.
- **SAM coarse/fine ablation (Table V, lesion matching accuracy).** — [SAM](https://arxiv.org/html/2012.02383v3)
  - Full SAM: 91.1%.
  - Without coarse-to-fine: 84.1%.
  - Global embedding only: 52.5%.
  - Local embedding only: 90.1%.

  The authors note the global embedding "can provide a rough localization, but it is not locally accurate", while the local embedding alone can produce "a large maximum error".
- **SAMv2 (Bai et al., arXiv 2311.15111).** It adds semantic and cross-modality embeddings. Reported lesion-tracking improvement over SAM is "from 81.87% to 84.06%"; the snippet does not say which metric or dataset. CT–MRI registration error: 5.1 ± 2.2 mm vs C2FViT 13.4 ± 11.6 mm. Training code, inference code and trained models are reported as released. — [SAMv2 abstract/search summary](https://arxiv.org/html/2311.15111v1)

### Inferences
- The gain from pretrained anatomical embeddings on DLS is about 0.9 mm MED (7.0 to 6.1) and +7 pts CPM@10mm, versus a supervised tracker trained from scratch. SAM++ adds about 0.7 mm more.
  - SAM was pretrained on thousands of unlabelled CTs. DLS has about 3k training pairs, similar to our ~2.2k lesions.
  - The gain comes with much lower error variance (σ 6.7 vs 8.9 mm). That is exactly the tail behaviour Team 2 needs.
- Coarse/global context mainly removes gross misses. Without coarse-to-fine, matching accuracy drops 91.1 to 84.1%, and local-only has "large maximum error". This maps directly to our failure mode: confident-but-wrong peaks among similar-looking neighbours.
- SAM/SAM++ run at 2 mm isotropic with a ResNet18 backbone, the same resolution as our patches. Their public weights could be applied frozen to our 64³ BL/FU patches with no resampling. This makes SAM the cheapest high-value pretrained-feature experiment:
  1. Cosine-match the BL-centre embedding against the FU embedding map.
  2. Use the result zero-shot, or fuse it with our CNN's similarity map.
- Caveat: SAM's global embedding exploits whole-volume context. Inside a 128 mm patch, the coarse level sees less anatomy, so part of its disambiguation benefit may be lost.

### Gaps
- SAMv2 per-metric DLS table (CPM@10mm, MED) was not extracted.
- I found no published "UAE" or "Alpha" anatomical-embedding results with DLS mm numbers in this pass.
- No paper reports SAM/DLT accuracy stratified by lesion size or by neighbouring-lesion density.
- I did not verify that the SAM GitHub weights (alibaba-damo-academy / mirror rifqi2708/self-supervised-anatomical-embedding-v2) load for patch inputs.

## Q3. DINO/DINOv2 and other foundation features for registration, correspondence and lesion matching

### Takeaway
Generic foundation features (DINOv2, applied slice-wise) are usable zero-shot for CT registration, but they do not clearly beat handcrafted MIND or registration models. Domain-specific 3D DINO-style pretraining beats DINOv2 by about 5 Dice on abdominal CT. I found no 2024–2026 CT lesion tracker built on DINOv2/CLIP with mm numbers. The only DINOv2 lesion-tracking work found is an MRI abstract.

### Cited Findings
- **DINO-Reg (MICCAI 2024)** is a training-free deformable registration method that uses DINOv2 features. DINO-Reg-Eco distils DINOv2 into a 3D CNN and cuts encoding time by 99%. — [DINO-Reg MICCAI 2024](https://papers.miccai.org/miccai-2024/paper/2230_paper.pdf); [DINO-Reg-Eco, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12496134/)
- **Kats & Heinrich 2026 (arXiv 2603.14086).** A PRIMUS ViT is pretrained DINOv2-style on 1,000 NAKO MRI + 1,228 TotalSegmentator CT, frozen, and fed via PCA into ConvexAdam. — [arXiv 2603.14086](https://arxiv.org/html/2603.14086)

  | Method | AbdomenCTCT Dice | NAKO MRI Dice |
  |---|---|---|
  | Initial | 25.31 | 29.57 |
  | MIND (handcrafted) | 37.08 | 50.55 |
  | DINOv2 (frozen, slice-wise) | 34.73 | 54.95 |
  | TotalRegistrator (out-of-domain) | 36.2 | – |
  | uniGradICON (out-of-domain) | 34.1 | – |
  | Ours (domain DINO, 4³ tokens) | 39.76 | 57.74 |

  Also: 4 × 4 × 4 tokens beat coarser 8³/12³. Inference uses about 5 GB VRAM vs about 35 GB for DINOv2. The paper reports no TRE.
- **DINOv3 benchmark (2025).** Feature-based registration is on par with VoxelMorph quantitatively; DINOv3 features give smoother warps in some cases. — [arXiv 2509.06467](https://arxiv.org/pdf/2509.06467)
- **VoxCor (2026).** A training-free triplanar frozen 2D ViT plus a closed-form weighted-PLS projection gives voxel correspondences. It is "competitive with handcrafted descriptors and learned 3D features" for registration. The abstract gives no mm numbers. — [arXiv 2605.13798](https://arxiv.org/abs/2605.13798)
- **ISMRM 2024 (GE HealthCare).** DINOv2 features localise a user-marked lesion in later serial MRI scans of oncology patients. It is MRI, not CT, and the result is an abstract. — [ISMRM 2024 #1951](https://archive.ismrm.org/2024/1951.html)
- **uniGradICON in LesionLocator.** Registration-pretrained weights initialise the prompt-propagation module. LesionLocator reports no ablation that isolates this initialisation. — [LesionLocator](https://arxiv.org/html/2502.20985)

### Inferences
- For our 64³ × 2 mm patches, 2D foundation models are a poor fit:
  - DINOv2 at 14-px patches on 64-px slices gives about 4 × 4 tokens per slice, which is far too coarse.
  - They need slice-wise inference and upsampling.
  - The registration evidence says they only reach MIND-level quality.
- CT-native 3D dense features (SAM/SAM++, or a domain-DINO PRIMUS) are the relevant candidates.

### Gaps
- No DINO-Reg TRE/Dice numbers were extracted.
- No multiGradICON numbers were extracted.
- No CLIP-based CT lesion tracker with published mm numbers was found for 2024–2026. The search was not exhaustive (Google Scholar "cited by" not run).
- Frozen-vs-fine-tuned comparisons specifically for lesion tracking were not found. All the foundation-feature registration results above are frozen/zero-shot.

## Q4. Does pretraining help most for small data and tail cases? Negative results

### Takeaway
For 3D CT/MRI dense tasks, SSL pretraining gives modest average gains for CNNs: about +2.5 Dice for the best method (MAE) on the OpenMind benchmark. Several SSL methods give zero or negative gains. Longer from-scratch training closes much of the gap. Gains are larger for transformers. For lesion correspondence specifically (DLS), SSL anatomical embeddings clearly beat supervised-from-scratch trackers, especially in error variance. No paper isolates small or tiny lesions.

### Cited Findings
- **OpenMind (Wald et al., arXiv 2412.17041).** Pretraining used 114k brain MRI; downstream were 12 segmentation datasets. Average Dice for ResEnc-L fine-tuned 150 epochs: — [OpenMind HTML](https://arxiv.org/html/2412.17041v2)

  | Pretraining | Avg Dice | Gain vs scratch |
  |---|---|---|
  | Scratch | 68.44 | – |
  | MAE | 70.91 | +2.47 |
  | S3D | 70.36 | +1.92 |
  | MG | 70.30 | +1.86 |
  | SimCLR | 69.44 | +1.00 |
  | VF | 69.29 | +0.85 |
  | VoCo | 68.50 | +0.06 |
  | SwinUNETR | 67.92 | −0.52 |

  - VoCo, SwinUNETR and SimCLR (for transformers) "do not, or just barely exceed, a from-scratch baseline".
  - "Individual datasets can be negatively impacted by any pre-training".
  - Scratch trained 1,000 epochs reaches 70.47, beating MG/S3D at 150 epochs.
  - Primus-M (transformer) gains +6.80 from MAE (63.62 to 70.42) but still trails the CNN.
  - Results "are inherently noisy". There are no small or few-shot experiments.
- **Wald et al., CVPR 2025 ("Revisiting MAE pre-training for 3D medical image segmentation").** Properly configured CNN MAE (Spark3D) gives "almost +3 DSC points" over nnU-Net across 11 test datasets and beats other SSL methods. — [CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Wald_Revisiting_MAE_Pre-training_for_3D_Medical_Image_Segmentation_CVPR_2025_paper.pdf)
- **DKFZ autoPET IV.** Pretraining (cross-sectional or synthetic-longitudinal) raised Dice from 55.78 to 62.27–62.89 on our likely dataset (about 285 patients). The specific longitudinal pretraining adds only +0.6 over cross-sectional pretraining. — [arXiv 2509.00613](https://arxiv.org/html/2509.00613v1)
- **LesionLocator synthetic longitudinal pretraining** cut point-prompt MED from 5.09 to 3.89 mm on 60 patients. — [LesionLocator](https://arxiv.org/html/2502.20985)
- **Generic-registration negative result.** On AbdomenCTCT, handcrafted MIND (37.08 Dice) beat frozen DINOv2 (34.73) and out-of-domain uniGradICON (34.1). — [arXiv 2603.14086](https://arxiv.org/html/2603.14086)
- **Supervised vs SSL on DLS.** SAM (no labels, 6.1 ± 6.7 mm) beat supervised DLT (7.0 ± 8.9 mm). — [SAM++](https://arxiv.org/pdf/2306.13988)

### Inferences
- Evidence suggests pretraining helps most where the downstream labelled set is small relative to anatomical variability, as with DLS (~3k pairs) and LesionLocator (60 patients). Our ~2.2k lesions / 224 patients sit in that regime.
- The improvement shows mostly as fewer gross errors: smaller σ and higher CPM. It does not show as a better median. This matches our tail-dominated mean error.
- The tiny-lesion tail (r < 5 mm) is unlikely to be fixed by 2 mm-grid embeddings, since SAM features also live at 2 mm. Our 4 mm correlation grid is a bigger limit there.
- Huge lesions (r > 20 mm) and multi-lesion patients are where global anatomical context (SAM coarse level) should help most. This is inference only; I found no stratified evidence.
- Segmentation-SSL gains (+2–3 Dice) do not translate directly into localisation mm. They are weak evidence for swapping in a generic MAE encoder.

### Gaps
- No study reports pretrained-vs-scratch tracking error stratified by lesion size or neighbour similarity.
- No learning-curve study on lesion tracking (performance vs number of training pairs) was found.
- The full Wald CVPR 2025 tables on small datasets were not extracted.

## Q5. Fair comparison of our 5.58 mm to published trackers

### Takeaway
Published numbers range roughly 3–7 mm MED, but setups differ greatly. Our 12.47 mm zero baseline (window centre) is similar to affine registration on DLS (11.2 mm), so the DLS table is the closest analogue. On it, our 5.58 mm sits between SAM (6.1) and SAM++ (5.4) and below DLT (7.0). The comparison is not apples-to-apples.

### Cited Findings
- DLS reference MEDs:
  - Affine 11.2 mm.
  - DEEDS 7.4 mm.
  - DLT 7.0 mm.
  - SAM 6.1 mm.
  - TLT 6.1 mm.
  - SAM++ 5.4 mm.

  — [SAM++](https://arxiv.org/pdf/2306.13988); [DLT](https://arxiv.org/html/2012.04872v2)
- Melanoma whole-body CT (60 patients, 5-fold CV) MEDs:
  - "Yan et al." point tracker 6.92 mm.
  - Registration + ULS 5.81 mm.
  - Hering et al. (nnU-Net + Elastix) 4.45 mm.
  - LesionLocator 3.12–3.89 mm.

  — [LesionLocator](https://arxiv.org/html/2502.20985)
- CPM definition: a match is correct if the centre distance is below the threshold. The thresholds are 10 mm and the lesion radius. — [SAM++](https://arxiv.org/pdf/2306.13988)

### Inferences
Differences that make the comparison unequal:
1. **Search space.** DLS and LesionLocator search whole volumes or registration-initialised regions. Our window is already within ±24 mm, so wrong-organ distractors far away are excluded. This makes our task easier.
2. **Ground truth.** DLS centres come from 2D RECIST key slices, so z errors are noisy (MED_Z is the largest component in most rows). Our centres are 3D mask centroids (`cog_fu`). LesionLocator also uses mask-based centres.
3. **Resolution.** We work at 2 mm with a 4 mm correlation grid. SAM/SAM++ and DEEDS work at 2 mm; LesionLocator at 0.8–1 mm.
4. **Population.** Our data is melanoma (UKT), the same disease as LesionLocator's set. DLS is mixed NIH lesions.

Suggested fair reporting:
- Report mean, median, CPM@10mm and CPM@radius together.
- Add two in-house references on our patches:
  - the provided `cog_propagated` error, if available;
  - frozen SAM/SAM++ embedding matching.

### Gaps
- No published method uses our exact "jittered 128 mm window" protocol, so there is no direct comparator.
- `cog_propagated` error statistics for autoPET IV were not found in any publication.
