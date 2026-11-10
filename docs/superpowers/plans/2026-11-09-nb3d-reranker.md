# neuronBridg3d 3D Re-ranker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local Python tool that takes a MANC/BANC body ID or cell type, re-ranks NeuronBridge's candidate driver lines by 3D (or 2.5D) shape similarity with an optional extra-expression penalty, shows them in a browser with an interactive 3D overlay viewer, and benchmarks the new ranking against published ground truth.

**Architecture:** A Python package `nb3d` with one module per concern: cached HTTP fetch, NeuronBridge metadata client, query resolution, EM skeleton loading, LM point extraction (3D from H5J stacks, 2.5D from colour-depth MIPs), scoring, ranking, verdict storage, a benchmark runner, and a FastAPI server serving a static three.js front end. Everything reads public NeuronBridge data from `https://janelia-neuronbridge-data-prod.s3.us-east-1.amazonaws.com` and caches it on disk.

**Tech Stack:** Python 3.12 (managed with `uv`), numpy, pandas, pyarrow, openpyxl, requests, h5py, navis (dotprops + NBLAST), Pillow, FastAPI + uvicorn, pytest; front end: plain HTML/JS with three.js 0.160.0 from cdn.jsdelivr.net.

**Spec:** `docs/superpowers/specs/2026-11-09-em-lm-3d-matching-design.md`

## Global Constraints

- Datasets: MANC (`FlyEM_MANC_v1.2.1`) and BANC (`FlyWire_BANC_v626`) only; NeuronBridge data release `v3_10_0`; alignment space `JRC2018_VNC_Unisex_40x_DS`.
- Runs locally on Windows 11; UI opened in the user's browser at `http://127.0.0.1:8765`; no hosting, no accounts.
- Light compute: no whole-library preprocessing. Per query, 3D-score only NeuronBridge's top `top_n=200` candidate **lines** (configurable).
- Ranked unit = driver line (`publishedName`). A line's shape score = its best image for the query group.
- Extra-expression penalty: **off by default**, user toggle + weight; computed on the line's full-expression image (Split-GAL4 library first), never on MCFO channels.
- When no aligned 3D stack is available for an image, fall back to 2.5D from the CDM; every score carries `source: "3d" | "2.5d"`, shown in the UI and reported separately in the benchmark.
- Always show the original NeuronBridge rank and `normalizedScore` beside the new rank.
- Default library filter: Split-GAL4 first (split-GAL4 lines preferred).
- Benchmark tiers reported separately; headline = tier 1 (independent evidence). BANC results labelled exploratory.
- BANC input: v626 body IDs or cell types; other BANC ID versions produce a clear error naming v626.
- Cache directory `cache/` (git-ignored); all downloads written atomically (temp file + rename).

## Review Focus

1. **A BANC v888/current root ID or unknown body ID typed in** → clear message "not found in NeuronBridge FlyWire_BANC_v626; use a v626 ID or a cell type", not a stack trace. (Task 3 test `test_unknown_body_id_message`.)
2. **The correct line labels the contralateral homologue only** → mirrored scoring still finds it and the result is flagged `mirrored=True`. (Task 6 test `test_mirrored_match_scores_high`.)
3. **A download is interrupted** → no truncated file left in `cache/`; the next call re-downloads. (Task 2 test `test_interrupted_download_leaves_no_file`.)
4. **Cell type typed in different case or with no members after side/segment filtering** → case-insensitive match; empty group gives a clear message, not an empty silent result. (Task 3 tests `test_cell_type_case_insensitive`, `test_filter_to_empty_group_message`.)
5. **A line has no full-expression image while the penalty is on** → line keeps its unpenalised score and shows `penalty: "n/a"` rather than being dropped or scored 0. (Task 7 test `test_penalty_missing_expression_image`.)

---

## Phase A — Spike and checkpoint

### Task 1: Environment + data-availability spike

The spec's biggest risk: whether aligned 3D LM stacks and aligned EM skeletons are downloadable. The NeuronBridge data model (`neuronbridge-python/neuronbridge/model.py`) declares `Files.VisuallyLosslessStack` ("H5J 3D image stack of all channels", LMImage) and `Files.AlignedBodySWC` (EM body skeleton in the alignment space). This task verifies them for our libraries and records every fact later tasks need.

**Files:**
- Create: `pyproject.toml`, `src/nb3d/__init__.py`, `spike/probe_availability.py`, `docs/spike-findings.md`
- Modify: `.gitignore` (add `cache/`, `.venv/` — already present; confirm)

**Interfaces:**
- Produces: `docs/spike-findings.md` with exactly these headings, each filled with observed values (not guesses), which later tasks read:
  - `## EM skeletons` — is `AlignedBodySWC` present for MANC and for BANC v626 images? URL pattern, units (µm vs voxels), coordinate ranges.
  - `## 3D LM stacks` — per library (Split-GAL4, Gen1 MCFO, Annotator MCFO, Omnibus Broad): fraction of 20 sampled `LMImage`s with a reachable `VisuallyLosslessStack`; file size; how to decode H5J in Python (which package / ffmpeg route works on Windows); voxel size and dimensions; which channel corresponds to `LMImage.channel`.
  - `## CDM geometry` — CDM pixel size in µm, image width/height, the hue→z lookup used by NeuronBridge (cite file in JaneliaSciComp/colormipsearch, e.g. `GradientAreaGapUtils.java`), z range in µm, template x-midline in µm (for mirroring).
  - `## Full-expression images` — for a split-GAL4 line, which `LMImage`(s) represent the line's whole expression pattern vs MCFO channels, and how to tell them apart from metadata (`libraryName`, `objective`, `channel`, `annotations`).
  - `## Cell-type resolution` — how to map a cell type to NeuronBridge EM images for MANC and BANC v626 (e.g. `EMImage.neuronType`, an index file under `metadata/`, or public metadata tables), and where side/segment (soma side, T1/T2/T3/abdominal) come from.
  - `## Download budget` — measured MB per candidate image in 3D and 2.5D, and the projected GB per `top_n=200` query.

- [ ] **Step 1: Install tooling**

Run (PowerShell): `scoop install uv; uv python install 3.12`
Expected: `uv --version` prints a version; `uv python find 3.12` prints a path.

- [ ] **Step 2: Create `pyproject.toml`** with project name `nb3d`, `requires-python = ">=3.12,<3.13"`, src layout, console script `nb3d = "nb3d.cli:main"`, dependencies listed in Tech Stack, dev dependency `pytest`. Run `uv sync`.
Expected: `uv run python -c "import navis, fastapi, h5py"` exits 0.

- [ ] **Step 3: Write `spike/probe_availability.py`** that, for 3 MANC body IDs from `research/groundtruth-sources/cheong2024/supp3.csv` and 3 BANC v626 body IDs, fetches `by_body` and `cdsresults` JSON via the URL scheme in `neuronbridge-python/neuronbridge/client.py`, then for 20 matched LM images per library checks each file key with an HTTP HEAD and prints a table. Spike code: no tests, not imported by the package.

- [ ] **Step 4: Run it and decode one H5J stack and one CDM by hand**, recording all six headings in `docs/spike-findings.md`. If H5J decoding needs ffmpeg, record the exact install command (`scoop install ffmpeg`) and Python call.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock src/nb3d/__init__.py spike/ docs/spike-findings.md
git commit -m "spike: verify NeuronBridge 3D/2.5D data availability"
```

### Checkpoint (stop and report to the user)

Compare `docs/spike-findings.md` against this plan. Continue to Phase B unchanged only if: EM aligned SWCs exist for both datasets, 2.5D decoding works for every library, and the projected budget for a 200-line query is under ~20 GB. Otherwise report the gap and the proposed plan change to the user and wait. Specifically flag: (a) libraries with no 3D stacks (expected: they go 2.5D — fine), (b) no aligned BANC SWCs (needs a transform task), (c) no way to identify full-expression images (penalty task needs a redesign).

---

## Phase B — Core pipeline

### Task 2: Cached fetch + NeuronBridge client

**Files:**
- Create: `src/nb3d/cache.py`, `src/nb3d/nbclient.py`, `tests/test_cache.py`, `tests/test_nbclient.py`, `tests/fixtures/` (one real `by_body` JSON and one trimmed `cdsresults` JSON saved during the spike)

**Interfaces:**
- Produces:
  - `fetch(url: str, cache_dir: Path = CACHE_DIR) -> Path` — returns local path; downloads once; atomic write.
  - `@dataclass EmImage: id: str; dataset: Literal["manc","banc"]; body_id: int; neuron_type: str | None; files: dict[str,str]`
  - `@dataclass LmHit: image_id: str; line: str; library: str; slide_code: str; channel: int | None; normalized_score: float; matching_pixels: int; mirrored: bool; files: dict[str,str]`
  - `get_em_images(dataset: str, body_id: int) -> list[EmImage]` — raises `BodyNotFound(dataset, body_id)`.
  - `get_cds_hits(em: EmImage) -> list[LmHit]` — ordered as in the file.
  - `file_url(files: dict[str,str], key: str) -> str | None` — resolves store prefixes as in `client.py:_get_files_url`.

- [ ] **Step 1: Write failing tests**: `test_fetch_downloads_once` (monkeypatched `requests.get` called once for two `fetch` calls), `test_interrupted_download_leaves_no_file` (fake response raising mid-stream → no file at the target path, exception propagates), `test_get_cds_hits_parses_fixture` (fixture → first hit has expected `line`, `normalized_score`, `mirrored` values copied from the fixture), `test_body_not_found` (404 → `BodyNotFound`).
- [ ] **Step 2: Run** `uv run pytest tests/test_cache.py tests/test_nbclient.py -v` → FAIL (import errors).
- [ ] **Step 3: Implement** `cache.py` (`requests` streaming into `*.part`, then `os.replace`; cache key = URL path under `cache/`) and `nbclient.py` (URLs exactly as spike recorded; plain dict parsing, no dependency on `neuronbridge-python`).
- [ ] **Step 4: Run** the same command → PASS.
- [ ] **Step 5: Commit** `feat: cached fetch and NeuronBridge client`.

### Task 3: Query resolution (body IDs, cell types, filters)

**Files:**
- Create: `src/nb3d/query.py`, `tests/test_query.py`, `tests/fixtures/celltypes_sample.csv`

**Interfaces:**
- Consumes: `get_em_images` (Task 2); the cell-type source from spike `## Cell-type resolution`.
- Produces:
  - `@dataclass Cell: dataset: str; body_id: int; cell_type: str | None; side: Literal["L","R"] | None; segment: str | None; em_images: list[EmImage]`
  - `@dataclass QueryGroup: dataset: str; label: str; cells: list[Cell]`
  - `resolve(dataset: str, query: str, side: str | None = None, segment: str | None = None) -> QueryGroup` — `query` is a body ID, comma-separated body IDs, or a cell type. All annotated members of a type by default.
  - `class QueryError(ValueError)` with user-facing messages.

- [ ] **Step 1: Failing tests**: `test_single_body_id`, `test_comma_separated_ids`, `test_cell_type_case_insensitive` ("dng11" resolves same as "DNg11"), `test_side_and_segment_filter`, `test_filter_to_empty_group_message` (message contains "no cells" and the filter values), `test_unknown_body_id_message` (BANC unknown ID → message contains "FlyWire_BANC_v626" and "cell type").
- [ ] **Step 2: Run** `uv run pytest tests/test_query.py -v` → FAIL.
- [ ] **Step 3: Implement** with the cell-type table loaded once per dataset and cached under `cache/`.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: resolve body IDs and cell types into query groups`.

### Task 4: EM skeletons

**Files:**
- Create: `src/nb3d/em.py`, `tests/test_em.py`, `tests/fixtures/em_sample.swc` (from spike)

**Interfaces:**
- Consumes: `fetch`, `file_url`, `Cell`.
- Produces:
  - `load_em_points(cell: Cell) -> np.ndarray` — (N,3) float, µm, template space, from `AlignedBodySWC`.
  - `mirror_x(xyz: np.ndarray) -> np.ndarray` — reflect about `MIDLINE_X_UM` (constant from spike `## CDM geometry`).

- [ ] **Step 1: Failing tests**: `test_load_em_points_units` (fixture SWC → all points within template bounds recorded in the spike, in µm), `test_mirror_twice_is_identity`, `test_mirror_moves_left_to_right` (point at midline−10 → midline+10).
- [ ] **Step 2: Run** `uv run pytest tests/test_em.py -v` → FAIL.
- [ ] **Step 3: Implement** (SWC parsed with `navis.read_swc`; apply the unit scale recorded in the spike).
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: load aligned EM skeletons`.

### Task 5: LM point extraction (3D and 2.5D)

**Files:**
- Create: `src/nb3d/lm.py`, `src/nb3d/cdm25d.py`, `tests/test_lm.py`, `tests/fixtures/cdm_sample.png`, `tests/fixtures/stack_sample.h5j` (smallest real stack found in spike; if >20 MB, a synthetic H5 with the same layout instead)

**Interfaces:**
- Consumes: `fetch`, `file_url`, `LmHit`.
- Produces:
  - `@dataclass LmPoints: xyz: np.ndarray; source: Literal["3d","2.5d"]`
  - `cdm_to_points(rgb: np.ndarray, threshold: int = 20) -> np.ndarray` — pixels with max(R,G,B) > threshold → (x_um, y_um, z_um) using the hue→z lookup ported from colormipsearch (cite source file in a comment).
  - `stack_to_points(volume: np.ndarray, voxel_um: tuple[float,float,float], threshold: float | None = None, max_points: int = 20000) -> np.ndarray` — threshold defaults to Otsu; random subsample to `max_points` with a fixed seed.
  - `load_lm_points(hit: LmHit, prefer_3d: bool = True, threshold: float | None = None) -> LmPoints` — 3D from `VisuallyLosslessStack` channel `hit.channel` if available, else 2.5D from the CDM.

- [ ] **Step 1: Failing tests**: `test_cdm_hue_maps_to_expected_z` (synthetic pure-colour pixels at the LUT's first, middle and last entries → z ≈ 0, mid, max within one slice), `test_cdm_threshold_drops_background`, `test_stack_to_points_finds_bright_blob` (synthetic volume with a 5³ bright cube → points' centroid within 1 voxel of the cube centre, in µm), `test_stack_subsample_deterministic`, `test_load_lm_points_falls_back_to_25d` (hit without `VisuallyLosslessStack` → `source == "2.5d"`).
- [ ] **Step 2: Run** `uv run pytest tests/test_lm.py -v` → FAIL.
- [ ] **Step 3: Implement** using the H5J decode route recorded in the spike.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: extract LM point clouds in 3D and 2.5D`.

### Task 6: Shape score and penalty

**Files:**
- Create: `src/nb3d/score.py`, `tests/test_score.py`

**Interfaces:**
- Consumes: `LmPoints`, `mirror_x`.
- Produces:
  - `@dataclass ShapeScore: value: float; mirrored: bool`
  - `shape_score(em_xyz: np.ndarray, lm: LmPoints) -> ShapeScore` — forward NBLAST (EM query → LM target) with `navis.make_dotprops(k=5)` and navis' default FlyCircuit scoring matrix, normalised to the EM self-score; computed for EM and `mirror_x(EM)`, best kept.
  - `extra_expression_fraction(group_xyz: np.ndarray, expr_xyz: np.ndarray, radius_um: float = 15.0) -> float` — fraction of expression points farther than `radius_um` from any group point (KD-tree; group includes mirrored copies).
  - `combine(shape: float, extra: float | None, weight: float = 0.5) -> float` — `shape * (1 - weight * extra)`; `extra is None` → `shape`.

- [ ] **Step 1: Failing tests**: `test_identical_cloud_scores_near_one` (EM points vs same points → value > 0.9), `test_unrelated_cloud_scores_low` (< 0.2), `test_mirrored_match_scores_high` (LM = mirror of EM → value > 0.9 and `mirrored is True`), `test_extra_fraction_zero_when_inside`, `test_extra_fraction_counts_far_points` (half of points 50 µm away → 0.5 ± 0.01), `test_combine_values` (`combine(0.8, 0.5, 0.5) == 0.6`; `combine(0.8, None) == 0.8`).
- [ ] **Step 2: Run** `uv run pytest tests/test_score.py -v` → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: 3D shape score and extra-expression penalty`.

### Task 7: Candidate collection and re-ranking

**Files:**
- Create: `src/nb3d/rank.py`, `tests/test_rank.py`

**Interfaces:**
- Consumes: `QueryGroup`, `get_cds_hits`, `load_em_points`, `load_lm_points`, `shape_score`, `extra_expression_fraction`, `combine`; full-expression image rule from spike `## Full-expression images`.
- Produces:
  - `@dataclass Candidate: line: str; nb_rank: int; nb_score: float; hits: list[LmHit]` — `hits` = up to 3 best images by `normalized_score`.
  - `collect_candidates(group: QueryGroup, top_n: int = 200, libraries: list[str] | None = None) -> list[Candidate]` — union of hits over all group cells' EM images, collapsed to lines by best `normalized_score`, ordered and `nb_rank`ed by that score (this is also the benchmark baseline), Split-GAL4 library selected by default.
  - `@dataclass RankedLine: line: str; rank: int; score: float; shape: float; source: str; mirrored: bool; best_cell: int; best_image_id: str; per_cell: dict[int,float]; penalty: float | None | Literal["n/a"]; nb_rank: int; nb_score: float`
  - `rerank(group: QueryGroup, candidates: list[Candidate], penalty: bool = False, weight: float = 0.5, progress: Callable[[int,int],None] | None = None) -> list[RankedLine]` — shape = max over cells × images; penalty uses the line's full-expression image vs all group cells' points.
  - `find_full_expression_image(line: str, dataset: str) -> LmHit | None`

- [ ] **Step 1: Failing tests** (fakes for loaders, no network): `test_collect_collapses_images_to_lines`, `test_collect_top_n_cuts_lines_not_images`, `test_nb_rank_is_baseline_order`, `test_rerank_orders_by_score`, `test_group_uses_best_cell` (line matching only cell B still scores high), `test_penalty_off_by_default` (`penalty is None` on all results), `test_penalty_lowers_broad_line` (broad line drops below sparse line when on), `test_penalty_missing_expression_image` (→ `penalty == "n/a"`, score unchanged), `test_source_reported`.
- [ ] **Step 2: Run** `uv run pytest tests/test_rank.py -v` → FAIL.
- [ ] **Step 3: Implement** (load each image once per call; loaders injected via module-level functions so tests can monkeypatch).
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: collect NeuronBridge candidates and re-rank in 3D`.

---

## Phase C — Benchmark

### Task 8: Ground-truth table

**Files:**
- Create: `src/nb3d/benchmark/groundtruth.py`, `tests/test_groundtruth.py`, output `data/groundtruth.csv` (committed)

**Interfaces:**
- Produces: `build_groundtruth(src: Path = Path("research/groundtruth-sources")) -> pd.DataFrame` with columns `dataset, body_id, cell_type, line, tier, source, confidence`, de-duplicated on `(dataset, body_id, line)` keeping the lowest tier. Line names normalised to NeuronBridge `publishedName` form (e.g. `SS01557`; slide-code suffixes stripped).
  - Tier 1 (independent): Cheong `supp3.csv` (all rows with a line); Zung A-quality rows joined to MANC bodies by type via Stürner `Supplemental_file7_MANC_DNs.tsv`.
  - Tier 2 (partly circular): Cheong `supp1.csv` confidence ≥ 4; Stürner SF4 joined by type via SF7.
  - Tier 3 (CDM-derived): Ehrhardt `table4.xlsx` confidence 3.
  - BANC rows: tier-1/2 types joined through `banc_888_meta.feather` `manc_cell_type`, `dataset="banc"`, tier suffixed `-banc` (exploratory). Column names come from `research/groundtruth-scripts/*.py`.

- [ ] **Step 1: Failing tests**: `test_columns_and_dtypes`, `test_no_duplicate_pairs`, `test_tier_precedence` (pair present in tiers 1 and 3 → tier 1), `test_line_name_normalised` (`"SS01557-20160318_..."` → `"SS01557"`), `test_counts_in_expected_range` (MANC distinct body_ids between 300 and 900).
- [ ] **Step 2: Run** `uv run pytest tests/test_groundtruth.py -v` → FAIL.
- [ ] **Step 3: Implement**; write `data/groundtruth.csv` via `uv run python -m nb3d.benchmark.groundtruth`.
- [ ] **Step 4: Run** → PASS; print per-tier counts.
- [ ] **Step 5: Commit** `feat: build de-duplicated ground-truth table with evidence tiers`.

### Task 9: Benchmark runner

**Files:**
- Create: `src/nb3d/benchmark/run.py`, `tests/test_benchmark.py`

**Interfaces:**
- Consumes: `build_groundtruth` output, `resolve`, `collect_candidates`, `rerank`.
- Produces:
  - `rank_of_correct(ranked_lines: list[str], true_lines: set[str]) -> int | None` — 1-based best rank of any true line; `None` if absent.
  - `summarize(ranks: list[int | None]) -> dict` — keys `n, top1, top5, top20, median_rank, recall_ceiling` (fraction not `None`); top-k as fractions of `n`.
  - CLI `nb3d benchmark --dataset manc --tier 1 [--limit N] [--top-n 200]` → writes `results/benchmark-<date>.csv` (one row per body: baseline rank, new rank, source) and prints a table: tier × {NeuronBridge, nb3d, nb3d 3D-only, nb3d 2.5D-only} × metrics, plus recall ceiling at `top_n` and at the full export.

- [ ] **Step 1: Failing tests**: `test_rank_of_correct_first_match`, `test_rank_of_correct_absent`, `test_summarize_values` (ranks `[1, 3, None, 25]` → top1 0.25, top5 0.5, top20 0.5, recall_ceiling 0.75, median over found = 3).
- [ ] **Step 2: Run** `uv run pytest tests/test_benchmark.py -v` → FAIL.
- [ ] **Step 3: Implement** (one query per body ID; resumable — skip bodies already in today's CSV).
- [ ] **Step 4: Run** tests → PASS; then `uv run nb3d benchmark --dataset manc --tier 1 --limit 10` → table printed with 10 rows counted.
- [ ] **Step 5: Commit** `feat: benchmark runner comparing NeuronBridge and nb3d rankings`.

---

## Phase D — App

### Task 10: Verdicts + API server

**Files:**
- Create: `src/nb3d/verdicts.py`, `src/nb3d/server.py`, `src/nb3d/cli.py`, `tests/test_verdicts.py`, `tests/test_server.py`

**Interfaces:**
- Consumes: `resolve`, `collect_candidates`, `rerank`, `load_em_points`, `load_lm_points`, `find_full_expression_image`.
- Produces:
  - `save_verdict(dataset: str, query: str, line: str, verdict: Literal["good","bad","unsure"], note: str = "") -> None` / `load_verdicts() -> pd.DataFrame` — append-only `data/verdicts.csv` with timestamp; latest wins on load.
  - `GET /api/search?dataset=&query=&side=&segment=&penalty=false&weight=0.5&top_n=200&libraries=` → `{group: {label, cells:[{body_id, side, segment}]}, results: [RankedLine as dict + verdict + links {flylight, neuronbridge} + thumbnail_url]}`; `QueryError` → HTTP 400 `{error: message}`.
  - `GET /api/geometry?dataset=&body_ids=&image_id=&mirror=&threshold=` → `{em: {<body_id>: [[x,y,z],...]}, lm: [[x,y,z],...], lm_source, template_outline: [[x,y,z],...]}` (points subsampled to ≤ 20000 each).
  - `POST /api/verdict` JSON `{dataset, query, line, verdict, note}`.
  - `nb3d serve [--port 8765]` starts uvicorn on 127.0.0.1 and opens the browser; serves `src/nb3d/web/` at `/`.

- [ ] **Step 1: Failing tests**: `test_verdict_latest_wins`, `test_search_returns_both_ranks` (FastAPI `TestClient`, `rerank` monkeypatched → each result has `rank` and `nb_rank`), `test_search_query_error_400`, `test_penalty_param_defaults_false`.
- [ ] **Step 2: Run** `uv run pytest tests/test_verdicts.py tests/test_server.py -v` → FAIL.
- [ ] **Step 3: Implement.** FlyLight link: `https://splitgal4.janelia.org/cgi-bin/view_splitgal4_imagery.cgi?line=<line>`; NeuronBridge link: `https://neuronbridge.janelia.org/search?q=<line>`. Template outline: thresholded CDM-mask of the template if the spike found one, else the bounding box.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: API server, verdict storage and CLI`.

### Task 11: Browser UI (list + 3D viewer, card grid)

**Files:**
- Create: `src/nb3d/web/index.html`, `src/nb3d/web/app.js`, `src/nb3d/web/viewer.js`, `src/nb3d/web/style.css`

**Interfaces:**
- Consumes: the three API endpoints from Task 10.
- Produces: the approved layout (spec "UI").
  - Top bar: query box, dataset select (MANC/BANC), side + segment filters, library multi-select (Split-GAL4 checked by default), "Penalise extra expression" checkbox (unchecked by default) + weight slider (0–1, default 0.5), Search; progress text while scoring.
  - List view: rows with new rank, line, score, `3D`/`2.5D` badge, NeuronBridge rank + score, mirrored ↔ marker, penalty value, ✓/✗/? buttons, FlyLight/NeuronBridge links. Selecting a row loads the viewer.
  - Viewer (`viewer.js`, three.js 0.160.0 via `https://cdn.jsdelivr.net/npm/three@0.160.0/`, OrbitControls): EM cells magenta (`#d63fd6`), LM green (`#33cc33`), template outline grey; per-cell visibility checkboxes; LM threshold slider (re-requests `/api/geometry` with `threshold`), point-size and opacity sliders; mirror toggle.
  - Grid view toggle: cards with NeuronBridge CDM thumbnail, rank, line, score, badge; click → list view with that line selected.

- [ ] **Step 1: Implement** the four files.
- [ ] **Step 2: Manual verification:** `uv run nb3d serve`, search MANC with one tier-1 body ID from `data/groundtruth.csv`; confirm: list shows both ranks; viewer rotates/zooms; EM and LM visibly overlap for the true line; penalty toggle changes order; verdict buttons write `data/verdicts.csv`; grid view shows thumbnails; a bad ID shows the Task 3 message. Record a screenshot in `docs/screenshots/`.
- [ ] **Step 3: Commit** `feat: browser UI with ranked list, 3D viewer and card grid`.

### Task 12: README and first benchmark report

**Files:**
- Create: `README.md`, `docs/benchmark-2026-11.md`

- [ ] **Step 1:** README: install (`scoop install uv` [+ `ffmpeg` if spike required], `uv sync`), `nb3d serve`, `nb3d benchmark`, what 3D vs 2.5D means, penalty meaning, known limits (top-200 recall ceiling, BANC v626 IDs only, BANC benchmark exploratory).
- [ ] **Step 2:** Run `uv run nb3d benchmark --dataset manc --tier 1` (then tiers 2, 3, and `--dataset banc`), paste tables into `docs/benchmark-2026-11.md` with a short plain-language reading: does nb3d beat NeuronBridge on tier 1, and what is the recall ceiling.
- [ ] **Step 3: Commit** `docs: README and first benchmark report`.
