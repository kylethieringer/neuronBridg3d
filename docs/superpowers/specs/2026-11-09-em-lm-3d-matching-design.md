# neuronBridg3d — 3D re-ranking of driver lines for MANC/BANC neurons

Date: 2026-11-09 · Status: draft for review · Owner: personal tool (one user, one Windows PC)

## Intent and why

When starting from an EM neuron in a VNC connectome (MANC or BANC) and looking
for a GAL4 / split-GAL4 driver line that labels it, NeuronBridge's ranked hits
"just do not match very well." The user wants a personal tool that produces a
better-ranked list of candidate lines and lets them **verify candidates by eye
in an interactive 3D view** of the EM cells overlaid on each line's LM signal.

Why NeuronBridge falls short for VNC (verified in the Janelia code, see
"Background"): matching is 2D colour-depth-MIP only (no PPPM for MANC/BANC);
only the top 300 lines by a crude pixel-overlap count ever get a shape score;
extra LM expression is penalised only in a 20–60 px ring; and scores from
separately normalised library runs are merged into one list.

## Goals

1. **Input:** a MANC or BANC body ID, a list of body IDs, or a **cell type**
   (resolving to a group of morphologically similar cells, e.g. both sides).
2. **Output:** a ranked list of candidate driver lines, scored by **3D shape
   similarity** between the EM cell(s) and the line's LM signal in a common
   template space.
3. **Extra-expression penalty** as a **user toggle, off by default**: when on,
   lines whose labelling extends far beyond the EM cells rank lower.
4. **Interactive 3D inspection** (see UI) so the user can judge each candidate.
5. **Measured improvement:** on a published MANC ground-truth set, the new
   ranking places the correct line higher than NeuronBridge's ranking
   (top‑1 / top‑5 / top‑20 hit rate, plus median rank of the correct line).

## Non-goals

- Not a contribution to Janelia's NeuronBridge / colormipsearch / precompute
  repos; we don't modify or redeploy their pipeline.
- Not a hosted/shared service; no accounts, no multi-user.
- Not brain datasets (FlyWire/hemibrain/maleCNS) in v1 — MANC and BANC only.
- Not custom-uploaded LM images in v1 — only published FlyLight lines that
  NeuronBridge already matches against.
- No side-by-side multi-viewer comparison in v1 (layout "C" from mockups —
  deferred).

## Anti-goals (would be a failure even if it "works")

- Reporting an improvement measured on test pairs that were themselves found
  with NeuronBridge/CDM, without separating them out (circular benchmark).
- A 3D view that silently misregisters EM and LM (wrong template, unmirrored,
  wrong units) — the user would draw wrong conclusions. Registration must be
  visibly checkable (VNC outline/template shown).
- Hiding the original NeuronBridge rank — the user must be able to see both.
- Penalty applied without the user asking for it.

## Background (from recon on 2026-11-09)

**NeuronBridge internals** (read-only clones of JaneliaSciComp repos):
- Scoring lives in `colormipsearch` (Java): pixel match
  (`PixelMatchColorDepthSearchAlgorithm`), shape/gradient score
  (`Shape2DMatchColorDepthSearchAlgorithm`, `GradientAreaGapUtils`). The web
  repo `neuronbridge` only displays precomputed results.
- Current data release **v3_10_0**; VNC template **JRC2018_VNC_Unisex_40x_DS**.
  Libraries: `FlyEM_MANC_v1.2.1` (~24k bodies) and `FlyWire_BANC_v626`
  (~23k VNC CDMs, added 2026-05), matched against Split-GAL4, Gen1 MCFO,
  Annotator MCFO and Omnibus Broad.
- Public precomputed results: `s3://janelia-neuronbridge-data-prod/v3_10_0/metadata/`
  — `by_body/<name>.json` → MIP ids; `cdsresults/<mipId>.json` → matches with
  `normalizedScore`, `matchingPixels`, `mirrored`, image file refs; schemas in
  `schemas/`. Python client: `neuronbridge-python`. Images in the
  `janelia-flylight-color-depth` bucket. Export keeps ≤5000 matches per EM MIP.

**Ground truth** (tables downloaded to `research/groundtruth-sources/`,
counting scripts in `research/groundtruth-scripts/`):

| Source | Dataset | Mapping | ~Size | NeuronBridge circularity |
|---|---|---|---|---|
| Cheong et al. 2024 eLife, supp 1 (DNs) | MANC | bodyId→line, confidence 1–5 | 383 bodies, 107 lines | partly circular |
| Cheong 2024, supp 3 (motor/efferent) | MANC | bodyId→line | 86 bodies, 45 lines | mostly independent |
| Ehrhardt, Whitehead et al. 2025 eLife, Table 4 | MANC | line→bodyId, conf 1–3 | 196 rows, 118 lines | substantially circular (used CDM) |
| Stürner et al. 2025 Nature | MANC (via type) | DN type→line | 628 bodies, 209 types | partly circular |
| Zung, Namiki et al. 2025 eLife | MANC (via type) | line→DN type, quality A/B/C | 238 bodies at A-quality | mostly independent |
| BANC v888 metadata | BANC | type columns only, no lines | ~500–600 DNs + 86 MNs via type join | transitive; cross-dataset type risk |

Counts are per source and not yet de-duplicated. Estimated MANC set: ~600–700
bodies, ~250–400 high-confidence.

## Decisions made with the user

- Personal tool for the user's own use.
- 3D matching as the core improvement.
- Extra-expression penalty is a **toggle the user controls**.
- Input can be a single ID or a **cell type / group of similar cells**.
- Output is a list of candidates the user **inspects in interactive 3D**
  (rotate, zoom) to verify.
- Datasets: **MANC and BANC**.
- UI layout: **ranked list + one large 3D viewer** as the main view, with a
  **card-grid view** (snapshot per line, click to open in 3D) as an alternative
  view. Approved from mockups ("looks good for now").
- Project lives at `D:\neuronBridg3d`.
- **End goal: find clean, sparse driver lines usable for experiments.**
  Split-GAL4 lines are prioritised (default library filter and ranking
  preference); the extra-expression penalty is judged on a line's **full
  expression pattern** (split-GAL4 / Gen1 GAL4 expression images), not on
  single MCFO channels.
- **Fallback when aligned 3D LM stacks aren't available for a library:** use a
  **2.5D approximation** built from the colour-depth MIPs (hue → z per pixel).
  These results are labelled 2.5D in the UI and reported separately in the
  benchmark. No in-house registration of raw stacks.
- **Compute/disk budget: light.** No whole-library preprocessing (no TB
  downloads / multi-day compute). Each query 3D-scores only NeuronBridge's
  **top ~200 candidate lines** for the query cells. Accepted trade-off: a
  correct line ranked very low by NeuronBridge stays missed; the benchmark
  reports this recall ceiling so it's visible.

## Claude's calls (user may override)

- **Re-rank, don't re-search.** Candidates come from NeuronBridge's published
  matches for the query cells (union across the group's MIPs), collapsed to
  **lines**, top ~200 lines kept (number configurable). The benchmark reports
  the **candidate-recall ceiling** at that cutoff and at the full export.
- **Unit ranked = driver line.** A line's shape score is its best-matching
  image (any library/sample) for the query cells; per-image detail is shown on
  expand. The penalty, when on, uses the line's full-expression image.
- **NeuronBridge baseline for the benchmark:** per line, best
  `normalizedScore` over its images and over the query group's MIPs, merged
  across libraries as NeuronBridge's own UI does. Headline metric = tier 1
  (independent evidence); other tiers and 2.5D-only lines reported alongside.
  No hard go/no-go threshold; the user judges from the numbers.
- **Cell-type queries** resolve to **all annotated members** of the type by
  default, with UI filters for side and segment (T1/T2/T3/abdominal) so the
  user can narrow the group; the penalty uses the current (filtered) group.
- **BANC IDs:** NeuronBridge indexes `FlyWire_BANC_v626`. v1 accepts v626 body
  IDs and BANC cell types (version-independent); mapping newer BANC root IDs to
  v626 is a later addition.
- **Group queries:** score each line against every cell in the group and
  aggregate (default: best cell per line, also show per-cell scores); a line
  labelling both left/right homologues should not be penalised for the other
  side. Mirroring handled explicitly.
- **3D score basis:** NBLAST-style comparison of the EM skeleton (transformed
  to JRC2018 VNC Unisex) against a point cloud / dotprops derived from the
  line's aligned 3D LM stack. Exact LM point extraction (threshold, local
  maxima, skeletonisation) left to the builder, tuned on the benchmark.
- **Penalty when on:** fraction (or volume) of thresholded LM signal farther
  than a set distance from any EM cell in the group, combined with the shape
  score; weight user-adjustable with a sensible default.
- **Language/stack:** Python (navis / flybrains-style tooling for skeletons,
  transforms and NBLAST). Browser-based UI served locally from Python.
- **Runs locally** on the user's Windows PC, opened in their browser; downloaded
  data cached on disk.
- **Line verdicts:** user can mark ✓ good / ✗ bad / ? unsure per line+query;
  saved locally (CSV/JSON) for reuse and possible future test-set growth.
- Each candidate shows: new 3D rank & score, original NeuronBridge rank &
  `normalizedScore`, library, links to FlyLight and NeuronBridge pages.
- **Benchmark tiers** reported separately: (1) independent evidence — Cheong
  supp 3 motor neurons, Zung A-quality DNs; (2) partly circular — Cheong supp 1
  conf ≥4, Stürner; (3) CDM-derived — Ehrhardt conf 3. BANC results are
  reported as exploratory only (transitive type matches).

## UI (approved wireframe, details left to builder)

- Top bar: query box (body ID(s) or cell type), dataset selector (MANC/BANC),
  library filter (split-GAL4, Gen1 GAL4/MCFO…), penalty checkbox (+ weight),
  Search.
- Main view: ranked list (left) · 3D viewer (right). EM cells in one colour
  (magenta), LM signal in another (green), VNC template outline. Controls:
  rotate/zoom, show/hide individual cells, LM threshold slider.
- Alternate view: grid of candidate cards with a static overlay snapshot +
  scores; click → 3D viewer.

## Constraints

- Windows 11; user has no Python install yet (only the Microsoft Store stub);
  Node.js is installed. Setup must be straightforward for one person.
- All data sources are public; no Janelia credentials assumed.
- LM 3D stacks are large; download lazily and cache. A single query should
  feel interactive after first download (exact budget left to builder).

## Left to the builder

- Where to get **aligned 3D LM stacks** (or per-line 3D segmentations) for the
  candidate lines in JRC2018 VNC Unisex space, and in what format. **Not yet
  verified that these are publicly downloadable for all libraries — spike it
  first** and record which libraries fall back to 2.5D.
- EM→JRC2018 VNC transforms for MANC and BANC (existing published bridging
  registrations expected; verify BANC).
- 3D viewer technology and the web framework.
- LM point-cloud extraction, NBLAST variant, score normalisation, penalty
  formula and defaults — tuned on the benchmark.
- De-duplication of ground truth across sources and the exact test split.
- Caching layout and disk budget.
