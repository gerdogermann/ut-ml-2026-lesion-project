# Reducing the error tail of Team 2's lesion tracker: large, merged, growing and look-alike lesions, far offsets, confidence

Scope: Team 2's best model, run P3-rand (job 3472221): val mean 4.44 mm, median 2.54 mm with flip TTA. The notes build on
`research_notes/Pretrained encoders for Team 2/worst_case_error_review.md`, which gives the failure taxonomy, review tooling
and test-set rules. That material is not repeated here.

Labels used below:
- **[data]** marks my own analysis of repo files. The script joins `runs/3472221/worst/errors.csv` (207 val lesions, TTA
  error `err_tta`) with the per-patient CSVs in `data/lesion_csv/`. It ran from the session scratchpad and is not committed.
- **[lit]** marks published evidence.
- **[inference]** marks my own reasoning.

## Q0. Where does the tail come from? (own quantitative analysis of run P3-rand val errors)

### Takeaway
The biggest tail driver is **lesion growth**, not size, merging or offset. The 49 val lesions whose volume grew more than 4×
(`volume_fu/volume_bl > 4`) average 8.35 mm and carry **44.5% of the total error**. The other 158 lesions average 3.22 mm.
Patient e2c0be2456 is mostly this growth effect: 76% of its lesions grew more than 4×. Within that patient, growing
lesions average 7.98 mm and non-growing ones 2.84 mm. Large lesions (r > 20 mm) look bad in mm, but every one of the 12
predictions lands within one FU radius of the target. Within the ±24 mm training range, the starting offset does not
predict the error.

### Cited Findings (all [data], source: [errors.csv](../../runs/3472221/worst/errors.csv) + [lesion CSVs](../../data/lesion_csv/))
- Overall: n = 207, mean 4.44 mm, median 2.54 mm, 11.6% of lesions over 10 mm. The top 20 carry 37.2% of the total error and the top 10 carry 23.2%.
- **Growth (volume ratio FU/BL):**

  | Volume ratio | n | Mean (mm) | Median (mm) | Share of total error |
  |---|---|---|---|---|
  | ≤ 0.25 | 32 | 3.70 | 2.17 | 12.9% |
  | 0.25–0.5 | 21 | 2.84 | — | — |
  | 0.5–1 | 55 | 3.39 | — | — |
  | 1–2 | 33 | 2.70 | — | — |
  | 2–4 | 17 | 3.29 | — | — |
  | **> 4** | **49** | **8.35** | **6.21** | **44.5%** |

  Leaving out the growing lesions lowers the mean to 3.22 mm. Growth hurts at any FU size:
  - growing lesions with r_FU ≤ 10 mm: n = 15, mean 7.33 mm;
  - growing lesions with r_FU 10–20 mm: n = 27, mean 7.56 mm;
  - non-growing lesions with r_FU 10–20 mm: n = 22, mean 3.72 mm.
- **FU radius:**

  | r_FU | n | Mean (mm) | Median (mm) | Over 10 mm | Mean err / r_FU | Share with err < r_FU |
  |---|---|---|---|---|---|---|
  | ≤ 5 mm | 82 | 2.94 | — | — | 0.91 | 76% |
  | 5–10 mm | 64 | 3.95 | — | — | 0.52 | 89% |
  | 10–20 mm | 49 | 5.84 | — | — | 0.45 | 94% |
  | > 20 mm | 12 | 11.59 | 10.50 | 67% | 0.40 | **100%** |

  The mean "error beyond the lesion radius", max(0, err − r_FU), is only 0.68 mm over all 207 lesions.
- **Topology:** the 8 MERGING val lesions average 10.75 mm (9.4% of the total error). The 199 UNCHANGED lesions average
  4.18 mm. Leaving out MERGING or r > 20 mm lesions gives 3.99 mm.
- **Merged lesions have a shared target.** In all 38 merge groups across the 300 patients, every member BL lesion has the
  *same* `cog_fu`, the centre of the merged FU mass. Group sizes run from 2 to 24 BL lesions; 24 groups have 2 members.
  No MERGING row has `linking_unclear` set. So the target is already the merged-mask centre of gravity (COG), not "this BL
  lesion's own part" (the wiki wording in `Team2-Model.md` §1 is misleading here). Example: `3988c7f88e_20` is a BL lesion
  of r ≈ 2.9 mm that merged into an FU mass of r = 32.6 mm (volume ratio 1430). Its error is 30.8 mm, still under the FU
  radius.
- **Patient:** e2c0be2456 carries 36.7% of the total error (n = 50, mean 6.75 mm). Leaving it out gives 3.70 mm. Its
  growing lesions (n = 38) average 7.98 mm and its non-growing ones 2.84 mm, the same as the 3.04 mm of non-growing,
  non-large lesions overall.
- **Lesion type:**
  - Soft tissue / skin: n = 53, mean 6.99 mm, 40.4% of the total error.
  - Liver: n = 42, mean 5.20 mm.
  - Lung: n = 55, mean 2.34 mm.
  - Lymph node: n = 43, mean 3.61 mm.
- **Starting offset (synthetic jitter):** Spearman ρ(err, |t|) = −0.05. The 20–30 mm |t| bin averages 5.93 mm (n = 21).
  Only the 6 lesions with an axis offset of 20–25 mm, near the ±24 mm clip, are clearly worse (mean 9.02 mm).
- **Confidence:**

  | Peak softmax | n | Mean (mm) |
  |---|---|---|
  | ≤ 0.1 | 13 | 10.23 |
  | 0.1–0.2 | 25 | 8.97 |
  | 0.2–0.3 | 33 | 4.74 |
  | 0.3–0.5 | 65 | 3.78 |
  | > 0.5 | 71 | 2.24 |

  Spearman ρ(err, peak) = −0.57.
- **Readouts barely differ, also in the tail:**

  | Readout | All 207 (mm) | Top 20 (mm) |
  |---|---|---|
  | plain | 4.57 | 17.11 |
  | local | 4.62 | 18.22 |
  | TTA | 4.44 | 17.08 |
  | TTA-local | 4.39 | 17.73 |

  TTA-local beats TTA by more than 2 mm on 5 lesions and loses by more than 2 mm on 9. A per-lesion oracle of
  min(TTA, TTA-local) reaches 4.19 mm, so readout choice has at most about 0.25 mm of headroom.
- **Persistence across runs:**
  - 9 lesions are over 10 mm in all 7 trained runs (6, 6b, 6c, P1-rand, P3, P3-rand, P4); 17 are over 10 mm in at least 5.
  - A per-lesion oracle "best of the 7 runs" gives 2.97 mm.
  - The plain-prediction mean of the 3 two-scale runs (P3, P3-rand, P4) gives 4.27 mm overall. On the top-20 lesions it
    gives 14.47 mm vs 17.08 mm for single-run TTA.
- **Ensemble spread is a better failure detector than peak probability.** Spread = mean distance of the 3 two-scale runs'
  predictions from their mean. AUROC for flagging err > 10 mm:

  | Score | AUROC |
  |---|---|
  | **ensemble spread** | **0.92** |
  | 1 − peak | 0.84 |
  | r_FU | 0.79 |
  | volume ratio | 0.64 |

  Spearman ρ(err, spread) = 0.68.
- **Selective prediction by peak probability:**

  | Lesions kept | Mean (mm) |
  |---|---|
  | 100% | 4.44 |
  | 95% | 4.16 |
  | 90% | 3.78 |
  | 80% | 3.27 |

  But falling back to "no move" (zero offset) on synthetic val makes things worse: 4.97 mm at τ = 0.2 (38 lesions). The
  model beats the window centre on most low-confidence lesions too. Only 27 of 207 predictions are worse than not moving.

### Inferences
- [inference] The tail is mostly an **appearance-change problem**, not a search problem: the BL template does not look
  like the grown FU lesion. The mask-pooled and 20 mm templates come from a small BL lesion. When the FU lesion is 5–1000×
  larger, its inside looks different (necrosis, confluence), and the matched point can sit anywhere on the larger mass.
- [inference] For r > 20 mm and MERGING lesions, part of the "error" is target definition. A prediction within the
  lesion that misses the COG by 10–15 mm is not a tracking failure clinically. The ≥ 0.40·r errors still show that the
  model does not find the mass centre.
- [inference] Within the trained range, far offsets do not drive the synthetic-val tail. They do drive the joint tail
  (11.5% over 20 mm), where offsets exceed ±24 mm. These are two separate problems that need separate fixes.
- [inference] Since readouts barely differ on the top 20, the heatmaps of the worst lesions are mostly confidently wrong
  or broad, not cleanly bimodal. Multi-hypothesis readout alone will not fix the tail.

### Gaps
- Voxel spacing is not in the repo, and `cog_*` are voxel indices. So I could not compute in mm whether wrong predictions
  land on *another* lesion (a look-alike swap) or how far merge members sit from the merged COG. The headers are on the
  cluster.
- The per-lesion joint predictions (`runs/joint-restart-3467988/validation/predictions.csv`) are cluster-only, so the joint
  tail by starting offset could not be analysed.
- The ensemble-spread AUROC is in-sample: P3-rand is both one of the 3 runs and the evaluated run, and val picked the epochs.
  The real AUROC is likely somewhat lower.
- "err < r_FU" uses an equal-volume sphere radius. For irregular or merged masks this is not the same as "inside the mask".

## Q1. Large and growing lesions: is the COG target well defined, and should the target be mask-based?

### Takeaway
Published trackers that segment the FU lesion and score centre distance (LesionLocator, autoPET IV Task 2, LongiSeg) avoid
part of the problem: a mask is defined however big or irregular the lesion is, and its centroid follows from it. For Team 2
the cheapest way to "become mask-based" fits the current model: a soft-argmax over a heatmap spread *uniformly over the FU
mask* returns exactly the mask COG. So pushing the heatmap to cover the mask, rather than to peak at one cell, is
consistent with the COG target. Add radius-normalised metrics so large lesions are judged fairly.

### Cited Findings
- [lit] LesionLocator propagates the prompt with a pretrained multi-step deformable registration network. It then
  segments the FU lesion with a promptable ResEnc U-Net, and reuses the predicted mask autoregressively as the next
  prompt: "the generated mask Ŷt from the previous scan is utilized as an input prompt for the next examination". It
  evaluates with CPM@25, mean Euclidean centre distance (MED) and Dice. — [Rokuss et al., LesionLocator, CVPR 2025](https://arxiv.org/html/2502.20985)
- [lit] LesionLocator states that "accurate distinction of closely located or merging lesions remains a challenge", fixable
  with "an additional click". On the melanoma data it reports an 86% retrieval rate and 3 mm MED for retrieved lesions
  (from the earlier review). — [LesionLocator](https://arxiv.org/html/2502.20985)
- [lit] autoPET/CT IV Task 2, on the same Longitudinal-CT data, is framed as "longitudinal promptable segmentation". The
  organisers provide "lesion centers and propagated center locations in the follow-up". Scoring is Dice, FN volume and FP
  volume, not centre distance. The LongiSeg-based entry reaches Dice 63.7 in cross-validation. It excluded 15 of 300
  patients "due to points being at the edge of scans". — [Promptable Longitudinal Lesion Segmentation in Whole-Body CT, arXiv 2509.00613](https://arxiv.org/html/2509.00613v1)
- [lit] The same entry notes that, because prompts exist in both scans, "prompt propagation [is] unnecessary". It cuts
  patches around the provided centres and, in training, places "the center at a random location in the inner half of
  each patch". — [arXiv 2509.00613](https://arxiv.org/html/2509.00613v1)
- [lit] Metrics-pitfalls guidance: centre-distance criteria behave differently for small and large structures (pitfall
  P2.2), so size-stratified reporting is advised. — [Maier-Hein/Reinke et al., Nat Methods 2024](https://pubmed.ncbi.nlm.nih.gov/38347140/)
- [data] For r > 20 mm, mean err / r_FU = 0.40 and 100% of predictions are within r_FU. For growing lesions (volume ratio > 4)
  the mean is 8.35 mm vs 3.22 mm for the rest (Q0).

### Inferences
- [inference] **Mask-shaped heatmap target.** The current mask term −log Σ_{v∈mask} p(v) is satisfied by a single peak
  anywhere inside the mask. Combined with the distance term, it rewards a peak near the COG, which a template match cannot
  locate on a big mass. A cheap change is a cross-entropy (or KL) between p and a uniform distribution over the max-pooled
  FU mask, only for r_FU > ~10 mm. Its soft-argmax is then the mask COG by construction, the network only has to "paint the
  lesion", and segmenting a big lesion is easier than locating its centre. Gate it by radius so small lesions keep the
  sharp peak that helps the median. Run 6's wide Gaussian heatmap loss hurt the median, which is why it should be gated.
- [inference] **FU-mask head (full mask-based route).** Add a small decoder that predicts the FU lesion mask in the
  window; the model already has the FU mask as a label. Output Δ as the COG of the predicted mask, or of the connected
  component under the heatmap peak. This is what LesionLocator, LongiSeg and nnU-Net follow-up segmentation do. It costs
  a decoder (~1–2 M parameters, fits on a V100), but it is a bigger change and its value depends on segmenting grown and
  confluent masses well. Try the gated uniform-mask target first.
- [inference] **Growth-aware templates and data.**
  - (a) Oversample train pairs with volume ratio > 4 or MERGING, or weight their loss ×2–3. The ratio is known from the
    CSV at train time.
  - (b) Widen the independent FU scale augmentation beyond 0.9–1.1. This rescales the whole patch, not the lesion, so it
    only partly mimics growth.
  - (c) Add templates at more than one scale, e.g. a 7³ (28 mm) stage-1 block or a stage-2 template pooled over a dilated
    BL mask, so a grown lesion still matches some template.
  - (d) For merged and grown targets the BL appearance is a weak cue for the *centre*. Context (the 8 mm stage) is what
    locates "the mass that this lesion became".
- [inference] **Report** CPM@r_FU (err < r_FU), the mean of max(0, err − r_FU), and strata by volume ratio and topology
  next to the mm mean. This shows that the "large-lesion" errors are mostly inside the lesion.

### Gaps
- ULS23 (universal lesion segmentation) and an nnU-Net FU-segmentation baseline were not fetched in this session, so there
  are no numbers here on centroid error from segmentation for large lesions.
- I found no published comparison of "COG regression" against "segment, then take the centroid" for lesion tracking.

## Q2. Merged lesions: how do datasets and trackers handle merge/split, and should these be excluded, down-weighted or retargeted?

### Takeaway
The Joskowicz group's graph methods treat merge and split as first-class events: lesions are vertices, overlap-based
pairings are edges, and connected components define change classes. They do not force one-to-one matching. Hungarian
pipelines force one-to-one and fail on exactly these cases. In Longitudinal-CT the target for every merge member is
already the merged-mask COG. So the label is consistent, but it is a "where is the mass this joined" target that local
appearance cannot pin down. Recommendation: keep MERGING in training, possibly down-weighted for small members of large
merges. Report results with and without MERGING. Do not count them as look-alike failures.

### Cited Findings
- [lit] Di Veroli et al.: "lesions are vertices and edges are lesion pairings computed by overlap-based lesion matching".
  Seven individual change classes and five patterns are "directly computed from the graph properties and its connected
  components". Lung (83 CTs, 19 patients) and liver (77 CECTs, 18 patients): change-class accuracy 98% and 85%,
  pattern accuracy 96% and 76%. A reviewer notes "the approach needs a very good pairwise registration" and 4 tunable
  parameters. — [Di Veroli et al., MICCAI 2023](https://conferences.miccai.org/2023/papers/309-Paper0455.html)
- [lit] Rochman, Szeskin et al.: lesion-change classification "formalized as bipartite graph matching where lesion pairings
  are computed by adaptive overlap-based lesion matching". Six change types come from connected components. On 208 CT
  pairs from 57 patients (4,600 lesions, 1,713 matchings), matching accuracy was 99.7% (lung) and 95.0% (liver). The
  analysis "helped the radiologist detect 48 missed lesions and 8 spurious lesions" in the ground truth. — [Rochman et al., IJCARS 2023](https://www.semanticscholar.org/paper/10df4864967734abf1f428601c97f2bbc95a536a)
- [lit] A registration + Hungarian pipeline (rigid SimpleITK registration, centroid-distance cost, 40/50 mm gating) failed
  on 2 of 25 patients. Its errors came from targets "in close proximity [that] should have been merged and labeled as a
  single target". One-to-one assignment cannot represent merges. — [Mukherjee et al., arXiv 2404.16544](https://arxiv.org/html/2404.16544v1)
- [lit] Longitudinal-CT: correspondence, merging and splitting "were established manually by expert radiologists".
  Topology classes are UNCHANGED, DISAPPEARING, MERGING and NEWLYAPPEARING; there is no SPLITTING class. — [Gatidis et al., Sci Data 2026](https://www.nature.com/articles/s41597-026-07466-y)
- [lit] autoPET IV Task 2 entry: "Predictions are merged according to the ground truths and metrics are calculated for
  each merged lesion pair", i.e. scored at the merged-lesion level. — [arXiv 2509.00613](https://arxiv.org/html/2509.00613v1)
- [lit] Hering et al. (earlier version of the same Tübingen cohort) note that lesions "can split or merge" while their
  pipeline assumes 0 or 1 correspondence. — [Hering et al., MIDL 2021](https://proceedings.mlr.press/v143/hering21a/hering21a.pdf)
- [data] All 38 merge groups share one `cog_fu` per group (0 groups with more than one). The 8 val MERGING lesions average
  10.75 mm. The two worst sit in the same merge, 3988c7f88e lesions 20 and 44 (r_FU 32.6 mm, errors 30.8 and 17.3 mm).
  The 4-member group 39059724f7 (r_FU 51.6 mm) has errors of only 4.9–10.4 mm (Q0).

### Inferences
- [inference] **Do not exclude** MERGING from training. The target is consistent (the merged COG), and Team 1's real
  pipeline will meet merges. **Do report** "all" and "excluding MERGING" (3.99–4.18 mm), since the official task scores
  every eligible lesion.
- [inference] **Down-weight** only where the target is ill-posed from the inputs: tiny BL members (r_BL < ~5 mm) of large
  merges (volume ratio > ~20), e.g. a loss weight of 0.3. Alternatively make these pairs use the gated uniform-mask target
  from Q1. A heatmap that covers the merged mass gives its COG by construction, which is exactly the label.
- [inference] Patient-level post-processing can use the merge structure: predictions of BL lesions that land in the same
  FU blob could be snapped to a common point (their mean). This needs a merge detector at test time, i.e. the FU lesion
  segmentation that Team 2 does not have. It is out of scope unless the FU-mask head (Q1) is built.

### Gaps
- I found no paper that reports centre-distance metrics *per merge member* the way this task does. Published work scores
  merges at the group or graph level.
- The full texts of the Di Veroli MedIA 2024 extension ([link](https://pubmed.ncbi.nlm.nih.gov/39029156/)) and of Szeskin's
  SimU-Net work were not read.

## Q3. Look-alike lesions in patients with many metastases: joint assignment, neighbour consistency, registration priors

### Takeaway
The literature's fixes for many similar lesions are (a) overlap or distance-based global matching on top of a deformable
registration (graph and bipartite matching, Hungarian with gating), and (b) neighbour-motion consistency, which predicts
where a lesion should be from how its neighbours moved. Both need all lesions of a patient at once. Team 2 can apply (b)
at inference without FU segmentations. But my data analysis suggests the e2c0be2456 failure is mostly growth (Q0), not
swaps. Measure swaps first (they need voxel spacing) before building assignment machinery.

### Cited Findings
- [lit] Auto-MBT (colorectal liver metastases) uses neighbour context, "predicting expected lesion positions from
  neighboring lesion movements". It reports errors "most often occur with clustered lesions…". Overlap-only tracking
  drops to matched F1 0.716 for more than 10 lesions per scan, vs 0.984 for the multi-cue method (summarised in the earlier
  review). — [Karunanayake et al., Eur Radiol Exp 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13369105/)
- [lit] Hungarian matching on centroid distance after rigid registration, with distance gating (40 mm target, 50 mm
  non-target), is a simple global assignment baseline. It needs FU lesion candidates. — [Mukherjee et al., arXiv 2404.16544](https://arxiv.org/html/2404.16544v1)
- [lit] Bipartite and graph matching on segmented lesions reaches 95–99.7% matching accuracy (Rochman et al.) and 85–98%
  change-class accuracy (Di Veroli et al.). Both require lesion segmentations in both scans plus registration. — [Rochman et al.](https://www.semanticscholar.org/paper/10df4864967734abf1f428601c97f2bbc95a536a); [Di Veroli et al.](https://conferences.miccai.org/2023/papers/309-Paper0455.html)
- [lit] DLT adds an anatomical-signal encoder that "prevents lesions being matched with visually similar but spurious
  regions" (from the earlier review). — [Cai et al., DLT, CVPR 2021](https://ieeexplore.ieee.org/document/9578122)
- [lit] Hering et al.: "lesions that are very close to each other could be wrongly assigned". — [Hering et al., MIDL 2021](https://proceedings.mlr.press/v143/hering21a/hering21a.pdf)
- [data] In e2c0be2456 (50 val lesions), non-growing lesions average 2.84 mm, as good as the rest of val. Growing lesions
  (38 of 50) average 7.98 mm. Soft tissue / skin lesions, mostly this patient, carry 40% of the total val error.

### Inferences
- [inference] **Neighbour-displacement prior (cheap, needs no FU masks).**
  1. For each BL lesion i, compute the predicted FU position.
  2. Take the robust (median) displacement of its k nearest BL neighbours (e.g. within 60 mm, k ≤ 5), using only
     neighbours with high confidence (low ensemble spread).
  3. If lesion i is low-confidence and its displacement disagrees with that median by more than ~10 mm, re-score its
     heatmap with a Gaussian prior centred on the neighbour-implied position (σ ≈ 8–10 mm), and take the argmax plus a
     local soft-argmax.

  This is Auto-MBT's idea in point form. It uses the heatmaps already computed and needs no retraining. Run 6c's centre
  prior tied because it was centred on the *window centre*. A neighbour-implied centre carries new information.
- [inference] **Exclusivity (Hungarian) over heatmap peaks.** Extract the top-k peaks per lesion in patient coordinates
  and solve an assignment that penalises two BL lesions claiming the same FU peak. Merges are the exception: there two BL
  lesions *should* share a target. So only use a soft penalty, and only between lesions whose BL masks are far apart.
  Implement this only if the swap measurement below shows swaps are common.
- [inference] **Measure first:** on the cluster, convert `cog_fu` to mm with the image headers. For each val lesion with
  err > 10 mm, check whether the prediction lies within r_FU of *another* lesion's `cog_fu`. This splits the e2c tail into
  "swap" and "growth/centre" cases.

### Gaps
- No swap statistics, because spacing is not in the repo (Q0 gaps).
- The TLT (transformer lesion tracker) full text was not fetched. No published method was found that does joint
  assignment *on top of per-lesion heatmaps* in CT.

## Q4. Far starting offsets: re-centring, larger search, registration fallback, confidence-gated hybrids

### Takeaway
On synthetic val (offset ≤ ±24 mm) the offset barely matters. The joint tail (11.5% of lesions over 20 mm) comes from
Team 1 offsets beyond the training range. The fixes, cheapest first:
1. Train on Team 1's real residuals, or on a wider, heavy-tailed jitter.
2. A second pass re-centred on the first prediction, which costs one more forward pass.
3. Gate the second pass, or a fallback, on confidence.

Falling back to "no move" is harmful on synthetic val (4.97 vs 4.44 mm at τ = 0.2), so a fallback is only useful when the
alternative estimate is independent of the window centre: `cog_propagated` or a deformable registration.

### Cited Findings
- [lit] Joint report: Team 1 alone leaves 13.9% of val lesions over 20 mm off, and the joint pipeline 11.5%. Team 1
  per-axis SDs are (7.1, 8.2, 11.7) mm, against 8 mm training jitter. — [Joint experiment results](../../reports/Joint%20experiment%20results.md)
- [lit] Same report: `propagated` (dataset `cog_propagated`) has a median of 5.1 mm after refinement, but some lesions lack
  propagation and fall back to a BL position ~1 m away. 10.6% are outside the window. — [Joint experiment results](../../reports/Joint%20experiment%20results.md)
- [lit] LesionLocator is iterative: it propagates the mask through registration and re-segments, autoregressively, from
  scan to scan. — [LesionLocator](https://arxiv.org/html/2502.20985)
- [lit] An autoPET IV entry trains with the centre at "a random location in the inner half of each patch", i.e. uniform
  offsets up to ±¼ of the patch, wider than a Gaussian. — [arXiv 2509.00613](https://arxiv.org/html/2509.00613v1)
- [lit] uniGradICON, a registration foundation model, works zero-shot. On COPDGene lung CT, mTRE drops from 23.36 mm
  (initial) to 2.26 mm, or 1.40 mm with instance optimisation; SyN gets 1.79 mm. It beat SyN zero-shot on out-of-distribution
  tasks. Code: [github.com/uncbiag/uniGradICON](https://github.com/uncbiag/uniGradICON). — [Tian et al., MICCAI 2024](https://arxiv.org/html/2403.05780v1)
- [lit] Hering et al. (same cohort, earlier version): registration-propagated centres were off by 3.79 / 3.16 / 4.49 mm on
  average (x, y, z), comparable to Team 2's median. — [Hering et al., MIDL 2021](https://proceedings.mlr.press/v143/hering21a/hering21a.pdf)
- [data] Spearman ρ(err, |t|) = −0.05. Lesions with an axis offset of 20–25 mm average 9.02 mm (n = 6). The zero-fallback
  rule worsens the mean at every τ tested: 4.48, 4.65, 4.97 and 6.09 mm at τ = 0.1, 0.15, 0.2 and 0.3.

### Inferences
- [inference] **Two-pass re-centring.** Predict Δ₁, re-cut the 64³ FU window at the centre + Δ₁ (needs the raw FU scan or a
  larger stored cube), predict Δ₂, output Δ₁ + Δ₂. A lesion that starts 30–40 mm off, near the edge of the trained range,
  is then re-scored from ~5–10 mm off, where the model is accurate. It needs no retraining and doubles inference. Gate it
  on low confidence or on |Δ₁| > 16 mm to limit drift. Check on joint val that pass 2 never makes the median worse.
- [inference] **Wider training offsets.** Sample the jitter from Team 1's train residuals, or from a mixture of
  0.8·N(0, 8 mm) and 0.2·Uniform(±48 mm), with `fu_big` enlarged to cover it. 88³ only covers ±24 mm, so the preprocessing
  must be re-run with a bigger cube (e.g. 112³ for ±48 mm; storage grows ~2×).
- [inference] **Confidence-gated hybrid for the joint pipeline.** When ensemble spread is high *and* `cog_propagated`
  exists *and* it disagrees with Team 1's centre by more than ~20 mm, re-run Team 2 with the window centred on
  `cog_propagated` and keep the more confident result. A deformable registration (uniGradICON) is the heavier option: it
  fits a V100, but needs whole-scan preprocessing per patient pair.

### Gaps
- No per-lesion joint data locally, so I could not test whether far-offset lesions are also low-confidence (this decides
  whether gating works).
- uniGradICON accuracy on whole-body oncologic follow-up CT (long intervals, tumour growth) was not found. Its benchmarks
  are lung, knee, brain and abdomen datasets.

## Q5. Confidence and calibration: what should trigger fallbacks or a second pass?

### Takeaway
The published evidence says simple scores are hard to beat. In classification, softmax response was the best overall
confidence score in a large benchmark. In medical segmentation, the pairwise agreement between ensemble members ranked
first. My data agrees: the agreement of the 3 two-scale runs flags lesions over 10 mm with AUROC 0.92, against 0.84 for
peak softmax. Team 2 already has the 3 checkpoints, so this costs only inference.

### Cited Findings
- [lit] "The revelation of a simple softmax response baseline as the overall best performing method underlines the drastic
  shortcomings of current evaluation" (image classification, many failure sources). — [Jaeger et al., ICLR 2023](https://arxiv.org/abs/2211.15259)
- [lit] For medical segmentation, "the pairwise Dice … between ensemble predictions consistently performs best among all
  compared methods". The authors recommend "Ensemble + pairwise DSC" as "simple and consistently ranked first". AURC is
  the main metric. — [Zenk et al., MedIA 2025](https://www.sciencedirect.com/science/article/pii/S1361841524003177)
- [lit] In Auto-MBT, accepted matches had higher confidence than rejected ones (median 0.72 vs 0.54). Accuracy fell from
  0.99 at confidence ≥ 0.8 to 0.92 below 0.6 (from the earlier review). — [Karunanayake et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC13369105/)
- [data] AUROC for err > 10 mm: ensemble spread 0.92, 1 − peak 0.84. Rejecting the 10% of lesions with the lowest peak
  gives a mean of 3.78 mm on the remaining 90%; the rejected lesions average 10.2 mm (Q0).

### Inferences
- [inference] Use **ensemble spread (mm)** as the primary confidence score for gating, with peak probability as a fallback
  for single-model use. Report a risk–coverage curve or AURC on val, and the mean at 90% coverage, as a "triage" result.
- [inference] Choose the gate action by what is available. On synthetic val there is nothing better than the model, so
  the only value is flagging. In the joint pipeline, possible actions are the second pass (Q4), the neighbour prior (Q3)
  or the `cog_propagated` re-run. Tune τ on val *for the joint setting*, not on synthetic jitter.
- [inference] The 3-run plain ensemble already gives 4.27 mm. Add TTA to each member (24 forward passes per lesion, still
  well under a second on a V100) for the final predictions. On the top 20 the ensemble is ~2.6 mm better than single-run
  TTA.

### Gaps
- No published failure-detection benchmark for point or landmark *regression* in 3D CT was found. The evidence above is
  for classification and segmentation.
- The ensemble AUROC is in-sample (Q0 gaps). Recompute it on test once the configuration is frozen.

## Q6. Multimodal heatmaps: global soft-argmax bias and argmax + local readouts

### Takeaway
Landmark-localisation work shows that global soft-argmax is pulled by background mass and by second modes. A local
soft-argmax around the argmax fixes this when the argmax is right. In Team 2's data the readouts tie overall (TTA 4.44 vs
TTA-local 4.39 mm) and also on the top 20. Run 9 showed that a local readout sometimes locks onto the wrong peak. The tail
is not mainly a readout problem; multi-hypothesis readouts are only worth it when combined with an external cue (Q3 or Q4)
to choose between peaks.

### Cited Findings
- [lit] Bulat et al.: "applying soft-argmax globally leads to semantically unstructured outputs that hurt the performance".
  Their local soft-argmax (find the max, crop k × k, soft-argmax inside) gives NME 2.04 on 300W vs 3.19 for global
  soft-argmax and 2.32 for L2 heatmap regression. The best window (5 × 5 px) matches the Gaussian size: "as the window size
  increases the amount of noise and background pixels also increases and hence the accuracy decreases". — [Bulat et al., Subpixel Heatmap Regression, arXiv 2111.02360](https://arxiv.org/html/2111.02360)
- [lit] Recent landmark work drops soft-argmax entirely in favour of structured heatmap objectives (only the snippet was
  read). — [Yang et al., ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Yang_Heatmap_Regression_without_Soft-Argmax_for_Facial_Landmark_Detection_ICCV_2025_paper.pdf)
- [data] Plain / local / TTA / TTA-local means: 4.57 / 4.62 / 4.44 / 4.39 mm; top 20: 17.11 / 18.22 / 17.08 / 17.73 mm.
  The per-lesion oracle of min(TTA, TTA-local) is 4.19 mm (Q0). Run 9 (row 9 of [Experiments](../../docs/wiki/Experiments.md)):
  local soft-argmax added 0.26 mm "because it sometimes locks onto the wrong peak".

### Inferences
- [inference] Keep global soft-argmax with TTA as the default. If a local readout is used, make the window lesion-sized
  (Bulat's window ≈ target width), e.g. ±max(8 mm, r_BL) in place of a fixed ±8 mm, and apply it only when the top two
  peaks are close (< ~12 mm). There a mode average is harmless and the local readout only removes background pull.
- [inference] For truly bimodal maps (two peaks more than 15 mm apart, second peak at least 0.5× the first), output both
  hypotheses. Let the neighbour-displacement prior (Q3) or a second pass (Q4) pick one. On its own, choosing the argmax
  mode is a coin flip, as run 9 showed.
- [inference] A **training-side** fix for large lesions is more promising than readout tweaks: the gated uniform-mask
  heatmap target (Q1) makes soft-argmax's averaging behaviour *correct* for large lesions, not a bias.

### Gaps
- Integral regression (Sun et al. 2018) and DSNT (Nibali et al. 2018) were not re-fetched here. Their known point that
  soft-argmax is biased toward the map centre when mass is spread is from background knowledge, not cited.
- I did not log the heatmap bimodality (second-peak ratio and distance) per lesion. `heatmaps.npz` holds only the top N
  lesions and is on the cluster.

## Q7. Concrete recommendations, ranked by expected tail gain per effort (one V100, student team)

### Takeaway
Grown lesions carry 44.5% of the val error, so attack growth first. Then cut the joint far-offset tail with wider
offsets and a second pass, and use the free 3-run ensemble for both accuracy and confidence. Report radius-normalised and
topology-stratified metrics so ill-posed targets (huge and merged lesions) are visible, not hidden in the mean.

### Cited Findings
- [data] Growth: 49 lesions with volume ratio > 4, mean 8.35 mm, 44.5% of the error. MERGING: 8 lesions, 10.75 mm. r > 20 mm:
  12 lesions, all within r_FU. e2c0be2456: 36.7% of the error, of which growing lesions average 7.98 mm. Ensemble spread
  AUROC 0.92 (Q0).
- [lit] Mask-based tracking and segmentation scoring ([LesionLocator](https://arxiv.org/html/2502.20985),
  [arXiv 2509.00613](https://arxiv.org/html/2509.00613v1)); neighbour-motion priors ([Karunanayake et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC13369105/));
  ensemble-agreement failure detection ([Zenk et al.](https://www.sciencedirect.com/science/article/pii/S1361841524003177)); local
  soft-argmax ([Bulat et al.](https://arxiv.org/html/2111.02360)).

### Inferences
1. **No training, do now.**
   - (a) Make the 3-run (P3, P3-rand, P4) ensemble with TTA the final readout.
   - (b) Add strata to `worst_cases.py`: volume ratio, topology, r_FU, CPM@r_FU, max(0, err − r), ensemble spread. Report
     "all / excluding MERGING / excluding e2c0be2456".
   - (c) Compute the swap statistic on the cluster (Q3).
2. **Cheap training changes aimed at growth, 1 run each (~1.5 h at 25 epochs).**
   - (a) Oversample, or weight ×2–3, train pairs with volume ratio > 4 or MERGING.
   - (b) Gated uniform-over-FU-mask heatmap loss for r_FU > 10 mm (Q1).
   - (c) Extra larger template (7³ at 4 mm, or a stage-2 template pooled over a dilated BL mask).

   Judge each on the "volume ratio > 4" stratum with a patient-cluster bootstrap CI. The overall mean is too noisy for
   effects of about 0.3 mm.
3. **Joint far-offset tail.**
   - (a) Train with Team 1 residuals or a heavy-tailed jitter (re-cut `fu_big` to cover ±48 mm).
   - (b) A two-pass re-centred inference gated on spread or |Δ₁|.
   - (c) Optionally, re-run on `cog_propagated` when it disagrees with Team 1 by more than 20 mm. Evaluate only on joint val.
4. **Patient-level post-processing, only if the swaps from 1(c) are common.** A neighbour-displacement prior to re-score
   low-confidence heatmaps (Q3), then soft exclusivity over the top-k peaks.
5. **Do not prioritise:** readout-only changes (≤ 0.25 mm oracle headroom), zero-offset fallback (harmful on val), excluding
   MERGING from training (the target is consistent), or a full FU-segmentation decoder until step 2(b) shows whether
   mask-shaped targets help.

### Gaps
- All numbers come from one val split (28 patients, one patient holding 24% of lesions), and val was also used to pick the
  epochs. Stratum effects with n < 10 (MERGING n = 8, r > 20 mm n = 12) are hypotheses, not findings.
- The real Team 1 offset distribution per lesion was not analysed. The effect of the far-offset fixes is not quantified here.
