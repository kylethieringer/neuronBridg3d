# neuronBridg3d

A tool for finding **clean, sparse driver lines** that label a given neuron in the
MANC or BANC VNC connectomes. It takes NeuronBridge's precomputed candidate lines for your
neuron(s), **re-ranks them by 3D shape similarity**, and shows each candidate in an
interactive 3D overlay so you can judge the match yourself.

- Input: a body ID, several body IDs, or a cell type (all cells of that type; filter by side, and for BANC by segment).
- Output: ranked driver lines, with NeuronBridge's original rank shown beside the new rank.
- Optional **extra-expression penalty** (off by default): ranks down lines whose full expression
  pattern extends well beyond your cells.

Data facts (URLs, image formats, geometry): `docs/spike-findings.md`

> **Testing release.** This is an early version shared for testing. Please report problems,
> surprising rankings, or confusing parts of the interface as
> [GitHub issues](https://github.com/kylethieringer/neuronBridg3d/issues), including the
> dataset and body ID / cell type you searched.

## Install

You need [uv](https://docs.astral.sh/uv/getting-started/installation/) (a Python project manager)
and git. uv downloads the right Python (3.12) by itself; nothing else is needed (3D image stacks are
decoded with the `av` package). Works on Windows, macOS and Linux.

1. Install uv:
   - Windows (PowerShell): `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
   - macOS / Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Get the code and install its dependencies:

   ```sh
   git clone https://github.com/kylethieringer/neuronBridg3d.git
   cd neuronBridg3d
   uv sync                   # installs dependencies into .venv
   ```

To update later: `git pull` then `uv sync`.

### Disk space and the cache

All data comes from public sources (NeuronBridge, FlyLight, the MANC/BANC public tables); no login or
API key is needed. Downloaded images and extracted point clouds are cached so repeat searches are
fast. The first search of a neuron downloads up to a few GB; the cache grows with every new neuron
searched (tens of GB after many searches). It lives in `cache/` inside the project folder unless
you point it elsewhere with the `NB3D_CACHE` environment variable, e.g.

```sh
# macOS / Linux
export NB3D_CACHE=/path/with/space/nb3d-cache
# Windows (PowerShell)
$env:NB3D_CACHE = "E:\nb3d-cache"
```

Deleting the cache folder is always safe; files are downloaded again when needed.

## Use

```sh
uv run nb3d serve         # opens http://127.0.0.1:8765 in your browser
```

The app runs only on your own computer (it is not reachable from other machines). Stop it with Ctrl+C.
Use `--port 8800` if port 8765 is taken, `--no-browser` to not open a browser tab.

1. Type a body ID (`10123`), several (`10123, 10011`), or a cell type (`hg1 MN`). Pick MANC or BANC.
2. Choose which line collections to search (Split-GAL4 is on by default) and, if you want,
   tick **Penalise extra expression** and set its weight.
3. Search. The first search of a neuron downloads images (up to a few GB, a few minutes);
   results are cached (see above), so repeat searches are fast. Raw 3D stacks (`.h5j`, ~8 MB each)
   are not kept, only the point clouds extracted from them, so moving the threshold slider
   re-downloads the stack it needs.
4. Click a line to see it in 3D: **magenta** = EM cells, **green** = the matched LM image,
   **orange** = the line's full expression (tick "Full expression"). Drag to rotate, scroll to zoom.
5. Mark lines ✓ / ✗ / ? — verdicts are saved to `data/verdicts.csv` on your computer only
   (the file is not tracked by git). If you are testing, sharing this file helps.

### What the labels mean

- **3D** — scored against the line's aligned 3D image stack.
- **2.5D** — no 3D stack was available, so depth was decoded from NeuronBridge's colour-depth
  image (accurate to about one slice, but cells hidden behind others are lost).
- **Score / Fwd / Rev** — the rank uses the mean of forward NBLAST (EM → LM: is the whole cell
  in the image?) and reverse NBLAST (LM → EM: is the image only the cell?).
- **Brain / B fwd / B rev** (BANC only) — the same score against the cell's brain part and the
  line's 20x brain images. It is shown for reference and does not affect the rank. It is most useful when
  the VNC score is weak; lines with broad brain expression score low here even when correct.
  Tick **Brain** in the viewer to see the brain image.
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
Ground truth (`data/groundtruth.csv`) was compiled from published driver-line tables (Cheong et al. 2024,
Stürner et al. 2025, Zung et al. 2025, Ehrhardt et al. 2025; see `docs/benchmark-2026-11.md`).
Each neuron takes about 1–3 minutes cold, so a full tier-1 run (324 MANC neurons) takes hours.

Results (November 2026; full write-up in `docs/benchmark-2026-11.md`). These compare the same neurons, those
whose correct line is in NeuronBridge's top 200:

| Dataset, tier | Neurons | Median rank: NeuronBridge → nb3d | Top 20: NeuronBridge → nb3d |
|---|---|---|---|
| MANC, tier 1 (headline) | 247 | 15 → **3** | 53% → **88%** |
| MANC, tier 2 | 182 | 12 → **6** | 63% → **76%** |
| BANC, tier 1 (exploratory) | 267 | 5 → **3** | 76% → **93%** |
| BANC, tier 2 (exploratory) | 101 | **6** → 8 | **79%** → 74% |

## Known limits

- Only NeuronBridge's **top 200 candidate lines** are re-scored (configurable). A correct line
  NeuronBridge ranks lower is missed; the benchmark reports how often that happens ("recall").
- BANC searches use NeuronBridge's BANC **v626**; newer BANC IDs are translated through the
  public BANC table. BANC benchmark results rely on cross-dataset cell-type matches and are exploratory.
- MANC has no segment annotation in the public table, so the segment filter is BANC-only.
- Cells that barely enter the VNC (e.g. some BANC brain neurons) can score 0 against every line; their
  order then falls back to NeuronBridge's.
- Gen1 GAL4 (R…/VT…) full-expression images exist only as colour-depth images, so their penalty is 2.5D.
- Each line is scored only on the images NeuronBridge matched best (its top hits per line). When those
  are noisy older samples, a correct line can rank lower than a cleaner sample of it would.
- The BANC **Brain** score is for reference only and does not affect the rank; lines with broad brain
  expression score low there even when correct.
