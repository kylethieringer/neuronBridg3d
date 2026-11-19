# nb3d benchmark, November 2026

**Status: draft.** MANC and BANC results are final (BANC rerun with the brain-image fix finished 2026-11-17; BANC
brain-score run finished 2026-11-18).

## Summary

On MANC neurons with independent evidence for a driver line (tier 1), nb3d puts the correct split-GAL4 line at
**median rank 3**. NeuronBridge puts the same line at median rank 15. nb3d puts the correct line in the top 5 for 63% of these
neurons (NeuronBridge: 32%), and ranks it higher than NeuronBridge for 182 of 247 neurons (worse for 47).

On BANC (exploratory ground truth), the tier-1 result holds but by less: median rank 3 vs NeuronBridge's 5, with the correct
line in the top 20 for 93% of neurons vs 76%. Tier 2 is about even. A separate BANC brain score is worse than the
VNC score on its own (median rank 5 vs 3), but often finds the correct line when the VNC score misses it. It is
reported beside the rank, not used for it.

The main limit is the candidate list, not the scoring. nb3d only re-ranks NeuronBridge's top 200 lines. For 21% of
tier-1 neurons, NeuronBridge ranks the correct line below 200, so nb3d cannot find it.

## What was measured

For each neuron with a published driver line, both methods rank candidate lines, and we record where the correct line
lands.

- **NeuronBridge:** lines ordered by their best colour-depth (CDM) `normalizedScore` over the line's images. This
  is the order NeuronBridge's own site shows.
- **nb3d:** the same lines, NeuronBridge's top 200, re-ranked by 3D shape. Each line gets the best score of up to 3 of its
  images. The score is the mean of forward NBLAST (neuron → image: "is the cell in the image") and reverse NBLAST
  (image → neuron: "is the image only the cell"). The EM neuron is also scored mirrored left-to-right, and the
  better side is kept. Images are scored from the aligned 3D stack where one exists (**3D**), otherwise from the
  colour-depth image (**2.5D**).
- Only split-GAL4 libraries are searched: FlyLight Split-GAL4 Drivers and Omnibus Broad. This is the app's default.
- The extra-expression penalty is off. It does not affect these numbers.

### Ground truth (`data/groundtruth.csv`)

| Tier | Meaning | Sources | MANC neurons |
|---|---|---|---|
| 1 | Independent evidence (headline) | Cheong et al. 2024 supp. 3 (motor neurons); Zung et al. 2025, A-quality DNs | 324 |
| 2 | Partly circular: matches made partly with CDM search | Cheong et al. 2024 supp. 1, confidence ≥ 4; Stürner et al. 2025 | 626 |
| 3 | Derived from CDM search | Ehrhardt et al. 2025, confidence 3 | 24 |

Tiers 2 and 3 can favour NeuronBridge, because the published matches were partly found with the same CDM search.

## MANC results

The comparison below uses the same neurons for both methods: those whose correct line is in NeuronBridge's top 200, so
nb3d had a chance to rank it. Top-k values are fractions of these neurons.

| Tier | Neurons (of total) | Method | Top 1 | Top 5 | Top 20 | Median rank | nb3d better / worse / tied |
|---|---|---|---|---|---|---|---|
| 1 | 247 of 324 | NeuronBridge | 0.11 | 0.32 | 0.53 | 15 | |
| | | **nb3d** | **0.24** | **0.63** | **0.88** | **3** | 182 / 47 / 18 |
| 2 | 182 of 626 | NeuronBridge | 0.10 | 0.34 | 0.63 | 12 | |
| | | **nb3d** | **0.17** | **0.46** | **0.76** | **6** | 92 / 74 / 16 |
| 3 | 14 of 24 | NeuronBridge | 0.00 | 0.14 | 0.36 | 80 | |
| | | **nb3d** | **0.21** | **0.50** | **0.93** | **6** | 11 / 3 / 0 |

### How often the correct line can be found at all

| Tier | Correct line anywhere in NeuronBridge's list | In NeuronBridge's top 200 (nb3d's ceiling) |
|---|---|---|
| 1 | 97% | 76% |
| 2 | 32% | 29% |
| 3 | 75% | 58% |

Tier 2's low numbers are mostly not a search failure. For 357 of its 626 neurons, the published line is a
Gen1 GAL4 line (R… or VT…), and the benchmark doesn't search those libraries. Among tier-2 neurons whose published line is an SS
split line, NeuronBridge finds it in 199 of 267 (75%).

### Results by image type (tier 1)

| Correct line scored from | Neurons | Top 5 | Top 20 | Median rank |
|---|---|---|---|---|
| 3D stack | 165 | 0.59 | 0.88 | 4 |
| 2.5D colour-depth image | 82 | 0.70 | 0.89 | 3 |

Full tables, including rows for every tier: `uv run python -c "import pandas as pd; from nb3d.benchmark.run import report;
print(report(pd.read_csv('results/benchmark-manc-tier1-split.csv')))"`.

> In those `report()` tables, the median is taken only over neurons where that method found the line. NeuronBridge's
> median therefore includes ranks above 200 that nb3d never sees (tier 1: 41 vs 3). Use the same-neuron table above
> to compare the two methods.

## How the scoring got here

These are tier-1 nb3d results on all 324 neurons. Top 20 is a fraction of all 324 neurons, and the median is over found lines.

| Version | Change | Top 5 | Top 20 | Median rank |
|---|---|---|---|---|
| Forward NBLAST only | — | 0.29 | 0.59 | 9 |
| Mean of forward and reverse | Stops broad images from outscoring sparse ones | 0.48 | 0.65 | 4 |
| + adaptive 2.5D threshold (current, a7f22bf) | Removes tissue background from colour-depth images | 0.48 | 0.67 | 3 |

The mean score had its biggest effect on tier 2, where the median rank went from 25 to 10 over found lines. Before that, broad Omnibus
images that contain the cell (plus much else) beat the sparse correct line. The adaptive threshold replaced a
fixed brightness cut-off of 20, which kept tissue background on dim colour-depth images. Against 3D points from the same images,
it raised agreement from F1 0.77 to 0.98 (300 images). Earlier, the 3D extraction got the same kind of fix: iterated Otsu
with a 0.1% voxel floor (83ee891, 46cc00c). See `docs/spike-findings.md`.

## Are 2.5D and 3D scores comparable?

After the adaptive threshold, neurons whose correct line is 2.5D improved a lot, while those whose correct line is 3D slipped by 1–2
ranks. That raised the worry that 2.5D scores run higher. We checked on 56 MANC tier-1/2 neurons by scoring every candidate
image that has a 3D stack twice: from the stack, and from its colour-depth image (`research/bias_25d_vs_3d.py`).

- **No inflation.** On the same image, the two scores agree closely (r = 0.94). Among the top-scoring images,
  2.5D is slightly *lower* (median −0.03), mainly because it misses part of the cell (forward −0.04).
- Images with only a 2.5D version do score higher on average (median 0.12 vs 0.09). But lines without stacks are about 3×
  more likely to be correct lines, so those higher scores look earned.
- Alternative rules changed little and not significantly (sign test p ≈ 0.2). Scoring every image as 2.5D: median
  rank 6 → 4, 20 better / 12 worse. Calibrating 2.5D scores onto the 3D scale: 5 better / 1 worse. The scoring was left
  unchanged.

The slip for neurons with a 3D correct line is better explained by 2.5D lines now being extracted properly and competing
on equal terms. Some of those competing lines may label the same cell (next section).

## Caveats

- **Ground truth is incomplete.** A "wrong" line ranked above the correct one may also label the cell. For example,
  SS01566 for MANC body 24965 looks like a genuine match. So these numbers understate both methods, and probably
  nb3d more, since it tends to rank sparse, well-matching lines higher.
- **Recall ceiling.** nb3d can only re-order NeuronBridge's top 200. Raising the cut-off costs time roughly
  in proportion.
- **Tiers 2 and 3 are partly circular** (see above). Treat them as supporting evidence, not as the headline.
- Each line is scored from at most 3 images (its best by NeuronBridge score). A line whose best-matching sample is
  further down its image list is under-scored.

## BANC (exploratory)

BANC ground truth comes from cross-dataset cell-type matches (a MANC/FAFB cell type with a known line, mapped to BANC),
so these results are exploratory only. NeuronBridge's BANC data is v626, and newer BANC IDs are translated through the
public BANC table. 694 neurons were run, each counted once under its best tier (tier 1: 320, tier 2: 374). One tier-2
neuron (720575941628231138) was skipped because its line's 3D stack returns 404 from the FlyLight S3 bucket, leaving 693.

Same-neuron comparison, as for MANC (neurons whose correct line is in NeuronBridge's top 200):

| Tier | Neurons (of total) | Method | Top 1 | Top 5 | Top 20 | Median rank | nb3d better / worse / tied |
|---|---|---|---|---|---|---|---|
| 1 | 267 of 320 | NeuronBridge | 0.23 | 0.53 | 0.76 | 5 | |
| | | **nb3d** | **0.27** | **0.70** | **0.93** | **3** | 144 / 87 / 36 |
| 2 | 101 of 373 | NeuronBridge | **0.26** | **0.50** | **0.79** | **6** | |
| | | nb3d | 0.15 | 0.41 | 0.74 | 8 | 43 / 41 / 17 |

On tier 1, nb3d beats NeuronBridge on BANC too, although by less than on MANC, because NeuronBridge already does well
on BANC (median 5, vs 15 on MANC). On tier 2 the two are about even: nb3d does better on as many neurons as it does worse, but
NeuronBridge has more top-1 hits. Tier 2 is partly circular (see the ground-truth table), so this tie is weak evidence.

| Tier | Correct line anywhere in NeuronBridge's list | In NeuronBridge's top 200 (nb3d's ceiling) |
|---|---|---|
| 1 | 98% | 83% |
| 2 | 30% | 27% |

As on MANC, tier 2's low recall comes mostly from Gen1 GAL4 lines that the benchmark doesn't search: 240 of its 373
neurons have no split line in their ground truth. Among tier-2 neurons with an SS line, NeuronBridge finds it in
113 of 133 (85%).

### Results by image type

| Tier | Correct line scored from | Neurons | nb3d top 5 | nb3d top 20 | nb3d median | NeuronBridge median |
|---|---|---|---|---|---|---|
| 1 | 3D stack | 165 | 0.64 | 0.88 | 3 | 5 |
| 1 | 2.5D colour-depth image | 102 | 0.80 | 0.99 | 2 | 4.5 |
| 2 | 3D stack | 56 | 0.39 | 0.75 | 8.5 | 7.5 |
| 2 | 2.5D colour-depth image | 44 | 0.43 | 0.75 | 6.5 | 3 |

### Brain-image fix (48b228a)

The first BANC run lost to NeuronBridge (tier 1 median 7 vs 5, 105 better / 138 worse; tier 2 median 22.5 vs 6). The cause
was a bug, not the scoring method. For BANC neurons with a brain EM image, NeuronBridge's brain colour-depth hits (in
JRC2018_Unisex_20x_HR) were pooled into the candidates. nb3d then scored them against the neuron's VNC skeleton using VNC
geometry, so the scores were meaningless. Lines matched through brain images are now scored using the VNC image from the
same fly (same slide, objective and channel). A line with no such image is ranked last. One tier-2 line ended up there
(SS29909: NeuronBridge rank 8, nb3d rank 166). MANC has no brain hits, so its results are unchanged.

| Tier | nb3d median: before → after | nb3d top 20: before → after | nb3d better / worse: before → after |
|---|---|---|---|
| 1 | 7 → 3 | 0.79 → 0.93 | 105 / 138 → 144 / 87 |
| 2 | 22.5 → 8 | 0.48 → 0.74 | 10 / 62 → 43 / 41 |

The previous outputs are kept as `results/*banc*-brainbug.*`.

### Remaining BANC losses

Most of the largest remaining losses are tier-2 lines that NeuronBridge ranks first and nb3d scores from their
colour-depth image: SS02542 (rank 1 → 115), SS02383 (1 → 73), SS02256 (1 → 67) and SS01592 (1 → 59). Two tier-2
neurons with SS00731 drop from 19 → 121 and 13 → 101, and one tier-1 neuron drops from 5 → 113 (SS41027). These cases
have not been examined. Many BANC ground-truth neurons are descending neurons (DNs). Their skeletons used here cover
only the VNC (more than 90% of BANC DN skeleton nodes lie inside the VNC template), so brain morphology plays no part in
the VNC score that sets the rank. A separate brain score is now reported beside it (next section).

### Brain score (d9e334d)

For BANC neurons with a brain EM image, nb3d also scores each line against the neuron's brain morphology, and reports
this beside the VNC score without changing the rank. EM brain points come from NeuronBridge's EM brain colour-depth
image, because the brain SWC file is the same VNC-frame skeleton. LM points come from the line's 20x brain images in
JRC2018_Unisex_20x_HR, from the 3D stack where one exists. The score is the same mean of forward and reverse NBLAST,
including the mirrored check. Run: `results/benchmark-banc-tierall-split-brain.csv` (2026-11-18). The VNC ranks in
that run are identical to the VNC-only run above.

The comparison is head to head on the same lines. For each neuron, only lines with a brain score are kept (median 175
lines per neuron), and these are ranked once by VNC score and once by brain score. It covers the 301 of the 368
neurons with a ranked correct line whose correct line also has a brain score. "Better of the two" takes, for each neuron, whichever rank is
higher. This is an upper bound that assumes something could pick the right score each time, not a method.

| Tier | Neurons | Ranked by | Top 1 | Top 5 | Top 20 | Median rank | Brain better / worse / tied |
|---|---|---|---|---|---|---|---|
| 1 | 205 | **VNC score** | **0.32** | **0.80** | **0.94** | **2** | |
| | | Brain score | 0.20 | 0.65 | 0.89 | 4 | 58 / 110 / 37 |
| | | Better of the two | 0.41 | 0.90 | 0.98 | 2 | |
| 2 | 96 | **VNC score** | **0.16** | **0.43** | **0.80** | **7** | |
| | | Brain score | 0.11 | 0.40 | 0.64 | 8 | 38 / 50 / 8 |
| | | Better of the two | 0.21 | 0.58 | 0.90 | 4 | |

The brain score is worse than the VNC score on its own, but the two are complementary. The brain score helps where the
VNC score fails:

| VNC rank of correct line | Neurons | Brain better | Brain worse |
|---|---|---|---|
| 1 | 81 | 0 | 52 |
| 2–5 | 124 | 34 | 74 |
| 6–20 | 64 | 41 | 23 |
| 21 or worse | 32 | 21 | 11 |

Examples of rescues: VNC rank 134 → brain 2 (SS29909), 124 → 14 (R80H02/SS02393) and 52 → 2 (R28E01/SS01058/SS96100).
Combining the two scores into one rank was deliberately left out. The "better of the two" row shows the room for it,
but a rule for choosing between the scores would need its own validation.

**Why the brain score loses.** 10 of the 15 largest brain losses are neurons whose correct line is SS02111 (rank 2–29 by
VNC, 84–127 by brain). Its 20x brain images contain much more than this cell. Only 5–24% of their signal lies within
10 µm of the EM neuron, and the reverse NBLAST score is −0.43. Split lines are usually selected for being sparse in the
VNC, not the brain, so broad brain expression is likely common. The reverse half of the score penalises it, as
intended. Neurons whose correct line is R28A08/SS53394/SS68367 also lose (VNC 60–67 → brain 136–146), but they were
already VNC failures. These have not been examined further.

**Caveat: VNC scores of zero.** The SS02111 neurons barely enter the VNC (81–133 EM points, vs a median of 293 over
all BANC neurons). On the one examined, every candidate's VNC score clamps to 0. Its VNC rank (2) is therefore just
NeuronBridge's order, used as a tie-break, not an nb3d measurement. The benchmark files don't record scores, so how many
BANC neurons this affects is unknown. It does not reverse the BANC result: on the 56 neurons with the smallest VNC
arbors (under 150 EM points), nb3d still beats NeuronBridge (median 7 vs 9, 29 better / 16 worse).

An earlier 40-neuron experiment (`results/benchmark-banc-brain-experiment*.csv`) found two bugs before this run. The
point cache mixed brain stacks decoded with VNC geometry with correct ones. The colour bar drawn inside brain
colour-depth images was also read as signal. Both are fixed in d9e334d.

## Reproducing

```powershell
uv run nb3d benchmark --dataset manc --tier 1     # writes results/benchmark-manc-tier1-split.csv (resumable)
uv run nb3d benchmark --dataset manc --tier 2
uv run nb3d benchmark --dataset manc --tier 3
uv run nb3d benchmark --dataset banc
uv run python research/bias_25d_vs_3d.py --n 60   # 2.5D vs 3D check, then bias_25d_vs_3d_report.py
```

Earlier versions' outputs are kept for comparison: `results/*-split-forward.csv` (forward NBLAST only) and
`results/*-split-mean-cdm20.csv` (mean score, fixed threshold 20). The first 30 tier-1 neurons before the 3D threshold fix
are in `results/*-otsu-30.csv`.
