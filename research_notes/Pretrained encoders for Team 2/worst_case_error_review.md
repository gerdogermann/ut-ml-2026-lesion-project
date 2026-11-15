# Worst-case error review for Team 2 (FU lesion localisation): val vs test, failure taxonomy, checklist

Project context used (read from the repo, not modified): `scripts/worst_cases.py` writes `worst/errors.csv` (every lesion, worst first: error, target and prediction vectors, `fu_radius_mm`, `peak_prob`, `has_propagated`, alternative readouts) and `heatmaps.npz` (32³ probability maps of the top N), and prints per-patient means. `data/README.md`: Team 2 FU windows are centred on the true FU lesion plus a fixed Gaussian jitter (sd 8 mm, clipped ±24 mm); MERGING lesions use the merged-into FU mask; `warn_mask_cog_far` flags an FU mask centre of mass > 4 mm from the target; `linking_unclear` lesions are dropped. `docs/wiki/Teams.md`: "Use the test split only for final numbers." Split: train 2,159 lesions / 224 patients, val 207 / 28, test 286.

## Q1. Is it sound to review the TEST-set worst cases? When is it legitimate and when does it become test-set overfitting?

### Takeaway
Looking at test errors is legitimate and expected (CLAIM 2024 item 39), but only after the model, readout and every hyperparameter are frozen and the test number is recorded. From then on the review is descriptive, post-hoc analysis. Anything you change because of it can't be scored again on the same test set and reported as the headline number. Before the freeze, do all worst-case review on val.

### Cited Findings
- CLAIM 2024 item 39, "Failure analysis of incorrect results": "Provide information to help understand incorrect results… Consider presenting examples of incorrectly classified cases to help readers better understand the strengths and limitations of the algorithm. Provide sufficient detail to frame incorrect results in the appropriate medical context." It sits in the Results / Model Performance section, next to the reported test performance. — [CLAIM 2024, Tejani et al., Radiol AI](https://pmc.ncbi.nlm.nih.gov/articles/PMC11304031/)
- CLAIM 2024 asks authors to say "internal testing dataset" for the held-out subset, and to use "tuning" or "model optimization" in place of the ambiguous "validation". Authors must specify how data were partitioned for training, model optimization and testing. — [CLAIM 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11304031/)
- Dwork et al.: classical holdout guarantees assume "a fixed procedure to be performed, selected before the data are examined." Once a model is modified based on holdout feedback, "they create a dependency between the model and the holdout data that invalidates the assumption of the classic holdout method", and leaderboard scores keep improving while true performance does not. — [Dwork et al., Science 2015](https://www.science.org/doi/10.1126/science.aaa9375); [Hardt, Google Research blog](https://research.google/blog/the-reusable-holdout-preserving-validity-in-adaptive-data-analysis/)
- The reusable-holdout fix is to put an algorithm (based on differential privacy) between the analyst and the holdout, so the holdout can be queried many times. The plain alternative is "simply to discourage" adaptivity. — [Hardt, Google Research blog](https://research.google/blog/the-reusable-holdout-preserving-validity-in-adaptive-data-analysis/)
- Recht et al. built new CIFAR-10 and ImageNet test sets and found accuracy drops of 3–15% (CIFAR-10) and 11–14% (ImageNet). However, "accuracy gains on the original test sets translate to larger gains on the new test sets", and the drops "are not caused by adaptivity" but by distribution shift to slightly harder images. — [Recht et al., ICML 2019](https://arxiv.org/abs/1902.10811)
- A medical tracking paper that does it properly: the test cohort "was reserved for evaluation of the final model performance". The pipeline was "locked" before external testing, and representative failure cases are then described in a supplement. — [Karunanayake et al., Eur Radiol Exp 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13369105/)

### Inferences
- Recht's "little adaptive overfitting" result comes from test sets of 10k–50k images. Team 2's test set is 286 lesions from about 30 patients (official split: 30 test patients, [Gatidis et al., Sci Data 2026](https://www.nature.com/articles/s41597-026-07466-y)), and val has 28 patients with one patient holding 24% of lesions. With so few independent units, each test-informed decision moves the estimate much more. So Recht's reassurance should not be carried over to this setting.
- The concrete leakage path here: looking at test worst cases and then changing, for example, the local soft-argmax width, the centre prior σ, TTA on/off, or a filter rule for MERGING / tiny lesions. All of these are cheap knobs that could be tuned to a few test patients.
- Acceptable after the freeze: (1) record the frozen test numbers in `Experiments.md` with commit hash and checkpoint; (2) run `worst_cases.py --split test`; (3) review and tag the cases; (4) report the review as "post-hoc failure analysis on the internal test set, model frozen". If the review shows a fix, apply it, but judge it on val. Report the original frozen test number as primary. Any re-scored test number must be labelled "test-informed, optimistic".
- Test label errors: if the review finds wrong `cog_fu` or wrong links in test, report metrics both with the original labels and with the corrected labels. Apply the correction rule to all test lesions, not only to those where the model failed. Fixing only failed cases biases the result toward the model.

### Gaps
- I didn't fetch the TRIPOD+AI text on post-hoc test analyses. The search only confirmed that it defines "evaluation or test data" and has a fairness item (item 14) on subgroups ([TRIPOD+AI, BMJ 2024](https://www.bmj.com/content/385/bmj-2023-078378); [TRIPOD+AI expanded checklist](https://www.tripod-statement.org/wp-content/uploads/2024/04/TRIPODAI-Supplement.pdf)).
- Not fetched: Roelofs et al. 2019 (Kaggle meta-analysis of holdout reuse) and Northcutt et al. 2021 (label errors in test sets). Both would be relevant citations.

## Q2. Best practice for qualitative failure analysis in medical imaging AI (stratification, hidden stratification, label review)

### Takeaway
Combine three things. Pre-defined strata, scored on all cases ("schema completion"). An audit of the worst cases to find strata you didn't think of ("error auditing"). Optionally, clustering. A top-20 list alone doesn't measure anything: every pattern it shows has to be checked as a stratum metric over the whole split.

### Cited Findings
- Hidden stratification means "the data contains unrecognized subsets of cases which may affect model training, model performance, and most importantly the clinical outcomes". It can cause "performance differences of up to 20% on clinically important subsets". — [Oakden-Rayner et al., 2020](https://arxiv.org/html/1909.12475v2)
- The paper gives three methods. Schema completion: subclasses are prescribed in advance and labelled on test data, but it is "limited by the understanding of the schema author". Error auditing: the auditor "examines model outputs for unexpected regularities"; it is labour-efficient but "critically dependent on the ability of the auditor" and weak on "low-prevalence, high discordance subclasses". Algorithmic measurement: for example k-means, which "still requires human review". — [Oakden-Rayner et al.](https://arxiv.org/html/1909.12475v2)
- Worked example of error auditing: a radiologist reviewing false negatives saw that pneumothoraces without chest drains were over-represented. Measured over the whole set, AUC was 0.94 with a drain vs 0.77 without, against 0.87 overall. — [Oakden-Rayner et al.](https://arxiv.org/html/1909.12475v2)
- A lesion-tracking study stratified results a priori by lesion size (< 1 cm, ≤ 3 cm) and by lesion count per scan (≤ 5, 6–10, > 10), "thresholds commonly used in oncology practice". It flagged strata with n < 5 or n ≤ 10 rather than over-reading them. — [Karunanayake et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC13369105/)
- Metrics pitfalls (Maier-Hein/Reinke et al.): averaging over all images when one patient contributes many hides per-patient differences. Their figure shows aggregating per patient revealing what simple aggregation hid. They recommend plotting raw values as jittered dots over box plots. Their pitfall P2.2 says small structures make localisation criteria such as Center Distance behave differently. — [Understanding metric-related pitfalls, Nat Methods 2024](https://pubmed.ncbi.nlm.nih.gov/38347140/)

### Inferences
- For Team 2, the "schema" can mostly be computed from the CSV and npz, with no new labelling. Use these strata: `lesion_type` (anatomical site), `topology_class` (UNCHANGED vs MERGING), FU radius bins (< 5, 5–10, 10–20, > 20 mm), volume ratio FU/BL (shrinkage), jitter magnitude bins, `has_propagated`, mask-centroid vs `cog_fu` distance (> 4 mm), `peak_prob` bins, and patient.
- The val top-20 observations (8/20 r < 5 mm, 3/20 r > 20 mm, all peak prob < 0.43, two patients dominating) are hypotheses at this point. Check each by computing mean, median and CPM@10 per stratum over all 207 val lesions, and report each stratum's n.

### Gaps
- Contrast phase and scan interval are not in the CSV fields described by the dataset paper ([Gatidis et al.](https://www.nature.com/articles/s41597-026-07466-y)), so these two strata likely can't be computed without DICOM headers. Unverified.

## Q3. Label noise in longitudinal lesion datasets: correspondences, merging/splitting, centre definitions

### Takeaway
The project's data is Longitudinal-CT (Tübingen melanoma, 300 patients; also used for autoPET IV). One radiologist linked the lesions by hand, a two-radiologist consensus reviewed them, and automatic QC warnings ran. No published per-lesion link-error rate was found. Expect a small but non-zero rate of wrong or ambiguous links and centre definitions, concentrated in MERGING lesions, large irregular lesions and dense clusters. These are exactly the cases that show up in a worst-case list.

### Cited Findings
- Longitudinal-CT: 300 melanoma patients, 600 CT studies, 7,182 segmented lesions (4,079 BL, 3,103 FU), and an official split of 240/30/30 patients. The CSV has `cog_bl`, `cog_fu`, `lesion_type`, `topology_class` (UNCHANGED, DISAPPEARING, MERGING, NEWLYAPPEARING), `volume_bl` and `volume_fu`. — [Gatidis et al., Sci Data 2026](https://www.nature.com/articles/s41597-026-07466-y)
- How the annotation was done: one radiologist segmented first. Then, "after a temporal delay to avoid recall bias", two radiologists did a consensus review. The tool showed BL and FU side by side, with slice synchronisation from a deformable registration. Correspondence, merging and splitting "were established manually by expert radiologists based on spatial proximity, anatomical context, and visual assessment". — [Gatidis et al.](https://www.nature.com/articles/s41597-026-07466-y)
- QC warnings fired on "(1) missing connectivity, i.e. a lesion mask consists of multiple connected components; (2) missing tracking consistency, i.e. tracked lesions are not at similar relative positions based on the deformable registration result". — [Gatidis et al.](https://www.nature.com/articles/s41597-026-07466-y)
- Splitting is defined as "a single baseline lesion corresponding to multiple distinct lesions at follow-up". However, the listed topology classes contain no SPLITTING value. — [Gatidis et al.](https://www.nature.com/articles/s41597-026-07466-y)
- An earlier version of the same Tübingen cohort (206 pairs) reported an average absolute offset of 3.79, 3.16 and 4.49 mm (x, y, z) between the registration-propagated BL centre and the manual FU centre. The authors note lesions "can split or merge" while their pipeline assumes 0 or 1 correspondence, and that very close lesions "could be wrongly assigned". — [Hering et al., MIDL 2021](https://proceedings.mlr.press/v143/hering21a/hering21a.pdf)
- Reader variability on lesion-level annotation in another tracking cohort (colorectal liver metastases): inter-reader agreement 86.5% at BL and 86.3% at FU (κ ≈ 0.76). Individual radiologists scored matched-lesion F1 of 0.93–0.95 against the consensus. In blinded adjudication, 3 of 254 and 11 of 256 links were flagged as mis-linked for two readers. — [Karunanayake et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC13369105/)
- DLT's benchmark has 3,891 lesion pairs mined from DeepLesion. DeepLesion annotations are RECIST bookmarks on single axial slices (32,735 lesions in 32,120 slices), not 3D masks. Centres from DeepLesion are therefore 2D-slice based. — [Cai et al., DLT, CVPR 2021](https://ieeexplore.ieee.org/document/9578122); [DeepLesion](https://pmc.ncbi.nlm.nih.gov/articles/PMC6052252/)

### Inferences
- MERGING is a target-definition risk. The FU mask is the merged lesion, while `cog_fu` may be the centre of the merged blob or of where this lesion's part sits. Either way, one FU point is shared by several BL lesions, and the model sees only one BL lesion. These cases should be tagged and reported separately, not "fixed" by the model.
- `warn_mask_cog_far` (> 4 mm) is the most direct label-noise probe available. Cross-tab it against error in val. If high-error cases are enriched for a mask/COG mismatch, part of the 5.58 mm mean is label noise, not model error.
- The ~3–4.5 mm per-axis registration offsets from Hering et al. give a sense of the inherent ambiguity of an "FU centre". Team 2's val median of 3.49 mm is already of the same order. Improving the median further will be hard to tell apart from label noise, while the tail is where real gains are.
- Large lesions (r > 20 mm) have an ill-defined centre: the COG of an irregular or necrotic mass moves with segmentation choices. A 5–10 mm error there may not be clinically a miss. Report error normalised by FU radius (CPM@radius style) next to raw mm.

### Gaps
- No source states how `cog_fu` is computed (mask centre of mass vs other), what `linking_unclear` exactly means, or how `cog_propagated` was produced. The autoPET longitudinal-CT page ([autopet.org](https://www.autopet.org/longitudinalct.html)) lists the fields, but the scrape returned no content.
- No published rate of wrong links or wrong centres for Longitudinal-CT was found.

## Q4. Typical failure modes of lesion trackers and the fixes applied

### Takeaway
Across DLT, the Hering pipeline, LesionLocator and Auto-MBT, the same failures come up again and again: similar nearby lesions, merging or splitting lesions, small lesions, poor registration or large deformation, lesions hard to tell from surrounding tissue, and enhancement or morphology change. The usual fixes are anatomical or context priors, multi-cue matching with neighbour consistency, confidence scores for triage, and human re-prompting.

### Cited Findings
- LesionLocator: "accurate distinction of closely located or merging lesions remains a challenge, however, re-identification with an additional click can enable further reliable tracking". Competing models "struggle with lesion retrieval, failing to correctly identify vanishing lesions or suffer from mismatches". Retrieval rate was 86%, with a 3 mm mean centre distance for retrieved lesions on the melanoma dataset. — [Rokuss et al., LesionLocator, CVPR 2025](https://arxiv.org/html/2502.20985)
- Hering et al. failure reasons: registration "not accurate enough and therefore a wrong or no lesion was selected"; lesions "hard to distinguish from surrounding tissue or… untypical shape"; a lesion that split after therapy where only one part was selected; "lesions that are very close to each other could be wrongly assigned". In some of these cases "it is also difficult for a radiologist". — [Hering et al., MIDL 2021](https://proceedings.mlr.press/v143/hering21a/hering21a.pdf)
- DLT adds an anatomical signal encoder that "prevents lesions being matched with visually similar but spurious regions", plus self-supervised training on unpaired images. Mean centre error was 7 mm on the DeepLesion benchmark. — [Cai et al., DLT](https://ieeexplore.ieee.org/document/9578122)
- Auto-MBT: "Errors most often occur with clustered lesions, lesions near the liver boundary, and hypervascular lesions with variable enhancement or interval morphologic change". Alignment also fails "under extreme deformation, including postoperative change, large confluent lesions". Fixes: neighbour context ("predicting expected lesion positions from neighboring lesion movements"), mask dilation and size-adaptive priors for small lesions, and per-lesion confidence. Confidence was higher for accepted than rejected matches (median 0.72 vs 0.54). Accuracy fell from 0.99 (confidence ≥ 0.8) to 0.92 (< 0.6). — [Karunanayake et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC13369105/)
- Older overlap-based tracking "struggle[s] when lesion counts are high, lesions are small, morphologies change, or registration is imperfect". For > 10 lesions/scan, matched F1 was 0.716 for deformable + overlap vs 0.984 for the multi-cue method. — [Karunanayake et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC13369105/)

### Inferences
- Team 2's two dominant val patients (50 and 26 lesions) are the "high lesion count" regime in which every paper above reports the most failures. Multiple similar metastases or nodes inside the 128 mm window are the likely main cause. This can be checked by counting other FU lesions within 64 mm, which needs the full FU label map and not only the npz's single-instance mask.
- All top-20 val misses have peak softmax < 0.43. This matches the Auto-MBT finding that low confidence marks hard cases. Report the error vs `peak_prob` curve, plus mean error with the lowest-confidence 5–10% flagged, as a triage result instead of hiding these cases.
- A low peak can mean either a flat map (lesion invisible or tiny) or two peaks (a distractor). Plain soft-argmax on a two-peak map lands between the peaks, which is the worst outcome. The existing local soft-argmax and TTA readouts address this, and the review should record which of the two shapes each worst case shows.

### Gaps
- I couldn't fetch the DLT and TLT full-text failure sections (IEEE paywall; the arXiv abs scrape failed). Their specific failure images are not summarised here.

## Q5. Reporting uncertainty with 207 val lesions in 28 patients

### Takeaway
Use the patient as the resampling unit. Report the lesion-level mean with a patient-cluster bootstrap 95% CI, the per-patient mean (already printed by `worst_cases.py`), and a leave-one-patient-out sensitivity check for e2c0be2456. Don't report run-to-run differences of a few tenths of a mm as real without that CI.

### Cited Findings
- Karunanayake et al. used "patient-cluster bootstrap inference… (patients/scan pairs as the resampling unit, 10,000 bootstrap replicates)". Their reason: resampling patients "avoids treating multiple lesions from the same patient as independent observations, preventing high lesion-burden patients from artificially narrowing confidence intervals or inflating significance". They used Benjamini–Hochberg for multiple subgroup comparisons. — [Karunanayake et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC13369105/)
- Metrics pitfalls: ignoring the hierarchical structure (many images from one patient) is a named pitfall, and per-patient aggregation can reveal what pooled averaging hides. — [Maier-Hein/Reinke et al., Nat Methods 2024](https://pubmed.ncbi.nlm.nih.gov/38347140/)
- CLAIM 2024's data-partition item asks authors to state whether partitions differ systematically. — [CLAIM 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11304031/)

### Inferences
- With 28 patients, one of them holding 24% of lesions, the CI on the val mean will be wide, likely around ±1 mm or more given a heavy tail. That is the same size as the gaps between runs 6, 6b and 6c (5.63 vs 5.58 mm, and a "tie"). Those runs should be called indistinguishable unless the paired bootstrap of the difference excludes 0. This estimate is not computed here; run it.
- Minimal implementation, which fits into `worst_cases.py` after `df` is built (paired version: resample the same patient ids for both runs' error columns):
  ```python
  pid = df.lesion.str.split("_").str[0]; g = df.groupby(pid).err_mm
  sums, cnts = g.sum().to_numpy(), g.size().to_numpy()
  rng = np.random.default_rng(0); idx = rng.integers(0, len(sums), (10000, len(sums)))
  boot = sums[idx].sum(1) / cnts[idx].sum(1)          # lesion-weighted mean per resample
  lo, hi = np.percentile(boot, [2.5, 97.5])
  ```
- Report the tail separately from the mean: median, 90th percentile, CPM@10 mm, and the count of errors > 10 mm or > 20 mm. A few gross misses dominate the mean.

### Gaps
- No source gives a minimum number of patients for a stable stratum estimate. Use the Karunanayake convention of flagging strata with n < 5 or n ≤ 10.

## Q6. Practical review tooling: visual layout and per-case metadata

### Takeaway
Use one static figure per case, with orthogonal slices through three points (BL centre, true FU centre, predicted FU point), a heatmap overlay and mask contours. Pair it with a tag column in `errors.csv` filled in from a fixed taxonomy. Review the top 20 plus about 10 random median-error cases as controls.

### Cited Findings
- Auto-MBT's review interface has "zoomed snapshots of the maximal-diameter slice" and "a three-dimensional locator for spatial context", along with synchronised 3D overlays and axial slices for split/merge events. — [Karunanayake et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC13369105/)
- Longitudinal-CT annotators worked with BL and FU side by side, with registration-synchronised slices. That is the same view radiologists used to create the labels. — [Gatidis et al.](https://www.nature.com/articles/s41597-026-07466-y)
- Use jittered raw-value plots, not bare box plots, to show clusters. — [Maier-Hein/Reinke et al.](https://pubmed.ncbi.nlm.nih.gov/38347140/)

### Inferences
- **Figure layout per case** (matplotlib, from the npz and `heatmaps.npz`): a 3 × 4 grid.
  - Rows are axial, coronal and sagittal.
  - Column 1 is the BL patch through the BL centre, with the BL mask contour.
  - Column 2 is the FU patch through the true FU point, with the FU mask contour, a true-point marker and the window centre.
  - Column 3 is the FU patch through the predicted point, with a pred marker.
  - Column 4 is FU plus the upsampled heatmap (alpha) through the heatmap argmax.
  - Title: lesion id, error mm, error vector, r_FU, peak prob, topology, |jitter|, mask-vs-COG distance.
  - Use the stored [-1000, 400] HU scaling. Optionally add a narrow soft-tissue window (W400/L40) for small lesions.
- **Per-case metadata to log** (extra `errors.csv` columns): patient; `lesion_type`; `topology_class`; `volume_bl`, `volume_fu` and their ratio; FU radius; |jitter| and its largest component (distance to the ±24 mm / window-edge regime); mask-centroid vs `cog_fu` distance; `has_propagated`; peak prob; second-peak prob and its distance from the first peak (bimodality); error split into xy vs z; whether the FU patch contains −1024 padding (scan edge); number of FU lesions of the patient (dense-disease proxy).
- **Failure taxonomy (one primary tag per case, plus optional secondary):**
  1. LABEL-COG: target point off the visible lesion, or mask centroid vs COG > 4 mm.
  2. LABEL-LINK: the FU lesion at the target does not look like the BL lesion (possible wrong link).
  3. MERGE: MERGING lesion; FU target shared with other BL lesions.
  4. SPLIT/CONFLUENT: FU shows several fragments or a confluent mass.
  5. TINY: r < 5 mm (≤ 2.5 voxels at 2 mm), lesion barely visible at patch resolution.
  6. SHRUNK/VANISHING: strong volume decrease, lesion nearly gone after therapy.
  7. HUGE/IRREGULAR: r > 20 mm, centre ill-defined; check error / radius.
  8. DISTRACTOR: a similar lesion, node or vessel in the window, and the heatmap peak sits on it.
  9. DEFORMATION: large anatomical change (breathing, organ shift, post-op), with the right anatomy moved.
  10. APPEARANCE: contrast phase or enhancement change, lesion conspicuous in BL but not in FU.
  11. EDGE/FOV: true point near the window edge or in padded region (scan coverage).
  12. READOUT: heatmap peak is correct but soft-argmax is pulled off by a broad or bimodal map (check whether `err_local` / `err_tta_local` is much lower than `err_mm`).
- **Checklist for each case:**
  - (a) Is the true FU point on the lesion visible in FU?
  - (b) Does the FU lesion plausibly match the BL lesion?
  - (c) Is it MERGING or split?
  - (d) What size and volume change?
  - (e) Is the heatmap unimodal or bimodal, and what is under the peak?
  - (f) Is the prediction on another lesion or structure?
  - (g) Does local or TTA readout fix it?
  - (h) Is it near the window or scan edge?
  - (i) Would a radiologist also find it ambiguous?
- To reduce auditor bias, assign the tag from the images before looking at the error number where possible. Include the random controls so that "looks hard" is calibrated.

### Gaps
- Contrast phase and scan interval are not available from the described CSV fields. Counting neighbouring lesions needs the full FU instance label map, not the single-instance npz mask.

## Q7. Direct recommendation for Team 2

### Takeaway
Keep worst-case review on val (and train, for label-noise hunting) for all decisions. Run the identical review on test exactly once, after the final numbers are frozen and logged, and present it as post-hoc failure analysis per CLAIM item 39. Never feed it back into a reported test number.

### Cited Findings
- The legitimacy and the risk rest on CLAIM item 39 ([CLAIM 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11304031/)) and on the holdout-independence assumption ([Dwork et al.](https://www.science.org/doi/10.1126/science.aaa9375)). The patient-cluster bootstrap comes from [Karunanayake et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC13369105/), and the stratification approach from [Oakden-Rayner et al.](https://arxiv.org/html/1909.12475v2).

### Inferences
- **Now (val):**
  1. Add the metadata columns and a patient-cluster bootstrap CI to `worst_cases.py`.
  2. Compute stratum metrics over all 207 val lesions for size, topology, mask-COG flag, peak-prob bin and patient.
  3. Render and tag the top 20 plus 10 random controls.
  4. Report the val mean with and without e2c0be2456 and the per-patient mean.
  5. If LABEL-COG or MERGE cases dominate the tail, report those as label/definition issues rather than chasing them with model changes.
  6. Fixes should target the confirmed categories: DISTRACTOR or READOUT via local soft-argmax or TTA; TINY or HUGE via radius-normalised reporting.
- **For label noise,** it is legitimate (and leak-free) to also run the review on train, since a wrong train label costs training signal.
- **Final:**
  1. Freeze the checkpoint and flags, run test once, and log it.
  2. Then run `worst_cases.py --split test` and tag with the same taxonomy.
  3. Report the stratum table and 3–4 example figures as post-hoc.
  4. If the test review suggests a fix, mention it as future work, or evaluate it on val only.

### Gaps
- Whether the course or instructor counts post-freeze test review as acceptable is a local policy question. Check with the instructor if it isn't clear.
