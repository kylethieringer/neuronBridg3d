# Spike findings — NeuronBridge data availability (2026-11-09)

All facts below were observed directly against the public buckets with the
probe scripts in `spike/` (`explore1.py`–`explore9.py`). Data release: `v3_10_0`
(`https://janelia-neuronbridge-data-prod.s3.us-east-1.amazonaws.com/current.txt`).

## URLs and lookup

- Config: `{BASE}/v3_10_0/config.json`. VNC store key `fl:open_data:vnc`.
  File-key prefixes (VNC store):
  - `CDM`, `CDMInput`, `CDMMatch`, `AlignedBodySWC`, `AlignedBodyOBJ` → `https://s3.amazonaws.com/janelia-flylight-color-depth/`
  - `CDMThumbnail` → `https://s3.amazonaws.com/janelia-flylight-color-depth-thumbnails/`
  - `VisuallyLosslessStack`, `Gal4Expression` → `https://s3.amazonaws.com/janelia-flylight-imagery/`
  - `CDSResults` → `https://s3.amazonaws.com/janelia-neuronbridge-data-prod/v3_10_0/metadata/cdsresults/`
- `metadata/by_body/<bare id>.json` → `{"results": [EMImage...]}`. **One file holds
  every dataset's image for that number** (e.g. `10123` returns FlyEM_VNC_v0.5,
  FlyEM_MANC_v1.2.1, male-cns brain + VNC). Filter on `libraryName`:
  MANC = `FlyEM_MANC_v1.2.1`, BANC = `FlyWire_BANC_v626`.
- `metadata/by_line/<line>.json` → all `LMImage`s of a line (Brain and VNC; filter `anatomicalArea == "VNC"`).
- `metadata/cdsresults/<imageId>.json` → `{"inputImage":..., "results":[CDSMatch...]}`,
  sorted by `normalizedScore` descending. Example MANC 10123: 2212 hits, 912 distinct lines.
  LM libraries seen: `FlyLight Gen1 MCFO`, `FlyLight Split-GAL4 Omnibus Broad`,
  `FlyLight Split-GAL4 Drivers`, `FlyLight Annotator Gen1 MCFO`.
- The bucket is listable (S3 `list-type=2`): `metadata/` has only `by_body/`, `by_line/`,
  `cdsresults/`, `pppmresults/` — no by-type index.
- Baseline sanity check: for MANC 10123 (hg1 MN; Cheong lines SS32023, SS47161, SS48311)
  NeuronBridge line ranks are 32, 55 and 153.

## EM skeletons

- `AlignedBodySWC` is present for MANC (`.../FlyEM_MANC_v1.2.1/SWC/<id>.swc`) and BANC
  (`.../FlyWire_BANC_v626/SWC/<root_626>.swc`).
- Units: **microns, in JRC2018_VNC_Unisex template space** (same frame as the LM stacks).
  Verified by projecting onto each body's CDM (after cropping the 90-px label strip):
  MANC 10123 SWC x 132.1–262.6, y 131.5–274.8 vs CDM x 131.4–263.8, y 126.8–275.3.
- BANC skeletons can extend outside the template (e.g. x up to 326 µm for
  720575941542412805, nerve fibres); points outside the template volume should be kept
  for display but are simply unmatched by LM.
- EMImage metadata: MANC has `neuronType`, `neuronInstance` (e.g. `MNwm03_ADMN_L`);
  BANC has `annotations` (e.g. `["...", "IN11B016", "11B"]`).

## 3D LM stacks

Sample of 20 matched images per library (MANC 10123's hits), HTTP HEAD:

| Library | `VisuallyLosslessStack` listed & reachable | mean MB |
|---|---|---|
| Split-GAL4 Omnibus Broad | 19/20 | 8.7 |
| Split-GAL4 Drivers | 17/20 (63x tiles often lack a stack) | 7.4 |
| Gen1 MCFO | 20/20 | 11.3 |
| Annotator Gen1 MCFO | 11/20 | 11.1 |

Across all 2212 hits: 2040 (92%) list a stack. Images without one → 2.5D.

- Format: H5J (HDF5). Root attrs `alignment_space=JRC2018_VNC_Unisex_40x_DS`, `unit=micron`,
  `image_size=[573,1119,219]`, `voxel_size=[0.461122, 0.461122, 0.7]`, `channel_spec` e.g.
  `sssr` (s = signal, r = reference). Group `Channels` attrs `width=573, height=1119,
  frames=219, pad_right=3, pad_bottom=1`; datasets `Channel_0..Channel_N` each an HEVC
  bitstream (`gray12le`).
- **Decode route (works on Windows, no separate ffmpeg):** Python package `av` (PyAV);
  `av.open(io.BytesIO(ds[()].tobytes()))`, decode frames with
  `to_ndarray(format="gray16le")`, stack to (z, y, x), crop to `[:, :height, :width]`.
  Frame count can be 218 instead of 219 → pad/crop z to 219.
  Gen1 MCFO stacks (file name says `JRC2018_VNC_Unisex-aligned`) carry the same
  `40x_DS` grid attributes.
- **Channel mapping:** `LMImage.channel = N` (the `CDM_N` suffix) ↔ dataset `Channel_{N-1}`.
  Verified on SS41075: Channel_0 MIP IoU with CDM_1 mask 0.45, other channels ≤ 0.11.
- Timing: 3.6 MB stack downloaded in 0.9 s; one channel decoded in 0.4 s.

## CDM geometry

- CDM PNG is 573 × 1209: **top 90 rows are a label strip**; rows 90..1208 map to
  y = 0..1118 of the template. Pixel size 0.461122 µm (x, y).
- Template extents: x 0–264.2 µm, y 0–516.0 µm, z 0–153.3 µm (219 slices × 0.7 µm).
- **Mirror midline:** x = 573 × 0.461122 / 2 = **132.12 µm**.
- Hue→z: the 256-entry colour LUT in JaneliaSciComp/colormipsearch
  `colormipsearch-api/src/main/java/org/janelia/colormipsearch/cds/GradientAreaGapUtils.java`
  (`findSliceNumberInLUT`, copy in `spike/ref/`). Entry 0 = (127,0,255) = front,
  entry 255 = (255,0,200) = back. Decode: normalise each pixel's RGB by its max channel,
  nearest LUT entry (normalised the same way) → index i; z_slice = i / 255 × 218.
  **Validated against the 3D stack (SS41075 Channel_0 argmax-z): r = 0.995, median error
  ≈ 1 slice** (`explore7.py`). Use pixels with max(R,G,B) > 20 (NeuronBridge's mask/target
  threshold); 60 used in the validation.

## Full-expression images

- **Split-GAL4 lines:** `by_line/<line>.json` images with `libraryName == "FlyLight Split-GAL4 Drivers"`,
  `anatomicalArea == "VNC"`, `objective == "20x"`. These are whole-VNC expression samples
  (often several slides and channels 1–3). Rule: prefer `channel == 1` 20x VNC images with a
  `VisuallyLosslessStack`; fall back to that image's CDM (2.5D). 63x images are tiles — skip.
- **Gen1 GAL4 lines (R…, VT…):** MCFO `LMImage`s carry `Gal4Expression`, a **CDM PNG** of the
  parent GAL4's full expression (e.g. `Gen1/CDM/VT046828/...-CDM_1.png`) → 2.5D only.
- MCFO channels are never used for the penalty.

## Cell-type resolution

- No by-type index in NeuronBridge. Use the public metadata tables (anonymous GCS):
  - MANC: `https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/manc_121/manc_121_meta.feather`
    (23,650 rows): `manc_121_id`, `cell_type`, `side` (`left`/`right`), `region`, … —
    **no segment/neuromere column.**
  - BANC: `.../compiled_data/banc_888/banc_888_meta.feather` (188,508 rows; also in
    `research/groundtruth-sources/banc/`): `root_626` (**the NeuronBridge ID**), `root_888`,
    `cell_type`, `manc_cell_type`, `side`, `region` (`ventral_nerve_cord` 31,688 rows, all with
    `root_626`), `neuromere` (T1/T2/T3/A1…).
- Bonus: the BANC table maps v888/v850/v890 roots to `root_626`, so typed newer BANC IDs can
  be translated instead of rejected.

## Download budget

- Per candidate line, up to 3 images: 3D ≈ 7–11 MB each, 2.5D (CDM) ≈ 0.6–0.8 MB each.
- 200-line query, all 3D: ≈ 200 × 3 × 9 MB ≈ **5.4 GB worst case** (cached; neighbouring
  queries share many lines). All 2.5D: ≈ 0.4 GB.
- Time: ≈ 1.3 s per image download+decode serial → ~13 min serial for 600 images; with
  8 parallel downloads, a few minutes for a cold query.

## LM signal threshold (added 2026-11-10)

A single Otsu threshold on a 20x VNC stack often separates empty space from **tissue
autofluorescence**, not tissue from labelled neurons: in 48% of 107 true-line images (33 MANC
tier-1 bodies) more than 5% of voxels were above it (typically 20–30%, a solid VNC silhouette;
see `docs/threshold-otsu-vs-signal.png`). `lm.signal_threshold` re-applies Otsu to the voxels
above the previous threshold while more than 2% remain (max 4 rounds). On the same images it
keeps a median 0.10% of voxels (90th percentile 0.30%, none above 5%) and lowers the median
extra-expression fraction of true lines from 0.59 to 0.25; median EM coverage within 5 µm drops
from 0.89 (inflated by tissue) to 0.67. Alternatives tried: median+k·MAD of non-zero voxels
(fails, 27–31% above), Otsu capped at the 99/99.5/99.8th percentile (works, but fixes the
signal fraction regardless of how broad the line is).
