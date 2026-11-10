# neuronBridg3d

A personal tool for finding **clean, sparse driver lines** that label a given neuron in the
MANC or BANC VNC connectomes. It takes NeuronBridge's precomputed candidate lines for your
neuron(s), **re-ranks them by 3D shape similarity**, and shows each candidate in an
interactive 3D overlay so you can judge the match yourself.

- Input: a body ID, several body IDs, or a cell type (all cells of that type; filter by side, and for BANC by segment).
- Output: ranked driver lines, with NeuronBridge's original rank shown beside the new rank.
- Optional **extra-expression penalty** (off by default): ranks down lines whose full expression
  pattern extends well beyond your cells.

Design: `docs/superpowers/specs/2026-11-09-em-lm-3d-matching-design.md` ·
Data facts: `docs/spike-findings.md`

## Install (Windows)

```powershell
scoop install uv          # Python project manager
cd D:\neuronBridg3d
uv sync                   # installs Python 3.12 deps into .venv
```

No other software is needed. 3D image stacks are decoded with the `av` package.

## Use

```powershell
uv run nb3d serve         # opens http://127.0.0.1:8765 in your browser
```

1. Type a body ID (`10123`), several (`10123, 10011`), or a cell type (`hg1 MN`). Pick MANC or BANC.
2. Choose which line collections to search (Split-GAL4 is on by default) and, if you want,
   tick **Penalise extra expression** and set its weight.
3. Search. The first search of a neuron downloads images (up to a few GB, a few minutes);
   everything is cached in `cache/`, so repeat searches are fast.
4. Click a line to see it in 3D: **magenta** = EM cells, **green** = the matched LM image,
   **orange** = the line's full expression (tick "Full expression"). Drag to rotate, scroll to zoom.
5. Mark lines ✓ / ✗ / ? — verdicts are saved to `data/verdicts.csv`.

### What the labels mean

- **3D** — scored against the line's aligned 3D image stack.
- **2.5D** — no 3D stack was available, so depth was decoded from NeuronBridge's colour-depth
  image (accurate to about one slice, but cells hidden behind others are lost).
- **↔** — the match was found on the mirrored side.
- **NB #** — the line's rank in NeuronBridge for the same cells.
- **Penalty** — fraction of the line's full expression lying more than 15 µm from your cells
  (or their mirror image). `n/a` means no full-expression image was found; the line is not penalised.

## Benchmark

```powershell
uv run nb3d benchmark --dataset manc --tier 1            # independent evidence (headline)
uv run nb3d benchmark --dataset manc --tier 2            # partly circular
uv run nb3d benchmark --dataset manc --tier 3            # CDM-derived
uv run nb3d benchmark --dataset banc                     # exploratory (via cell-type matching)
uv run nb3d benchmark --dataset manc --tier 1 --limit 10 # quick check
```

Results are appended to `results/benchmark-*.csv` (re-running resumes where it stopped).
Ground truth (`data/groundtruth.csv`) is rebuilt with `uv run python -m nb3d.benchmark.groundtruth`
from the published tables in `research/groundtruth-sources/`.
Each neuron takes about 1–3 minutes cold, so a full tier-1 run (324 MANC neurons) takes hours.

## Known limits

- Only NeuronBridge's **top 200 candidate lines** are re-scored (configurable). A correct line
  NeuronBridge ranks lower is missed; the benchmark reports how often that happens ("recall").
- BANC searches use NeuronBridge's BANC **v626**; newer BANC IDs are translated through the
  public BANC table. BANC benchmark results rely on cross-dataset cell-type matches and are exploratory.
- MANC has no segment annotation in the public table, so the segment filter is BANC-only.
- Gen1 GAL4 (R…/VT…) full-expression images exist only as colour-depth images, so their penalty is 2.5D.
