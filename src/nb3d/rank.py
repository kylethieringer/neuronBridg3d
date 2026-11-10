"""Collect NeuronBridge candidate lines for a query group and re-rank them by 3D shape."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

from .em import load_em_points
from .lm import LmPoints, load_lm_points
from .nbclient import LmHit, get_cds_hits, get_line_images
from .query import QueryGroup
from .score import combine, extra_expression_fraction, shape_score

SPLIT_DRIVERS = "FlyLight Split-GAL4 Drivers"
SPLIT_LIBRARIES = [SPLIT_DRIVERS, "FlyLight Split-GAL4 Omnibus Broad"]
ALL_LIBRARIES = SPLIT_LIBRARIES + ["FlyLight Gen1 MCFO", "FlyLight Annotator Gen1 MCFO"]
IMAGES_PER_LINE = 3
WORKERS = 8


@dataclass
class Candidate:
    line: str
    nb_rank: int
    nb_score: float
    hits: list[LmHit]


@dataclass
class RankedLine:
    line: str
    rank: int
    score: float
    shape: float
    source: str
    mirrored: bool
    best_cell: int
    best_image_id: str
    per_cell: dict[int, float]
    penalty: float | None | Literal["n/a"]
    nb_rank: int
    nb_score: float
    library: str = ""
    hits: list[LmHit] = field(default_factory=list)


def collect_candidates(group: QueryGroup, top_n: int = 200, libraries: list[str] | None = None) -> list[Candidate]:
    """Union of NeuronBridge hits over the group's EM images, collapsed to lines by best
    normalizedScore. The order (and nb_rank) is the NeuronBridge baseline."""
    libs = set(libraries or SPLIT_LIBRARIES)
    by_line: dict[str, list[LmHit]] = {}
    for cell in group.cells:
        for em in cell.em_images:
            for h in get_cds_hits(em):
                if h.library in libs:
                    by_line.setdefault(h.line, []).append(h)
    ordered = []
    for line, hs in by_line.items():
        seen, best = set(), []
        for h in sorted(hs, key=lambda h: -h.normalized_score):
            if h.image_id not in seen:
                seen.add(h.image_id)
                best.append(h)
        ordered.append((line, best))
    ordered.sort(key=lambda lb: -lb[1][0].normalized_score)
    return [Candidate(line, i + 1, hs[0].normalized_score, hs[:IMAGES_PER_LINE])
            for i, (line, hs) in enumerate(ordered[:top_n])]


def find_full_expression_image(line: str, dataset: str) -> LmHit | None:
    """The line's whole-VNC expression image (docs/spike-findings.md, "Full-expression images"):
    Split-GAL4 Drivers 20x VNC, channel 1 with a 3D stack preferred; else a Gen1 GAL4 expression CDM."""
    ims = get_line_images(line)

    def as_hit(im: dict, files: dict) -> LmHit:
        return LmHit(image_id=im["id"], line=line, library=im["libraryName"], slide_code=im.get("slideCode", ""),
                     channel=im.get("channel"), normalized_score=0.0, matching_pixels=0, mirrored=False,
                     files=files, objective=im.get("objective"))

    split = [im for im in ims if im.get("libraryName") == SPLIT_DRIVERS and im.get("objective") == "20x"]
    if split:
        split.sort(key=lambda im: (im.get("channel") != 1, "VisuallyLosslessStack" not in im["files"]))
        return as_hit(split[0], split[0]["files"])
    for im in ims:
        expr = im["files"].get("Gal4Expression")
        if expr:
            return as_hit(im, {"store": im["files"].get("store", ""), "Gal4Expression": expr})
    return None


def rerank(group: QueryGroup, candidates: list[Candidate], penalty: bool = False, weight: float = 0.5,
           progress: Callable[[int, int], None] | None = None) -> list[RankedLine]:
    em_pts = {c.body_id: load_em_points(c) for c in group.cells}
    group_xyz = np.vstack(list(em_pts.values()))

    def score_line(cand: Candidate) -> RankedLine:
        per_cell: dict[int, float] = {}
        best = (-1.0, None, None, False)  # value, hit, source, mirrored
        best_cell = group.cells[0].body_id
        for h in cand.hits:
            lm: LmPoints = load_lm_points(h)
            for body, xyz in em_pts.items():
                s = shape_score(xyz, lm)
                per_cell[body] = max(per_cell.get(body, 0.0), s.value)
                if s.value > best[0]:
                    best, best_cell = (s.value, h, lm.source, s.mirrored), body
        shape = max(best[0], 0.0)
        pen: float | None | Literal["n/a"] = None
        if penalty:
            expr = find_full_expression_image(cand.line, group.dataset)
            pen = "n/a" if expr is None else extra_expression_fraction(group_xyz, load_lm_points(expr).xyz)
        hit = best[1] or cand.hits[0]
        return RankedLine(
            line=cand.line, rank=0,
            score=combine(shape, pen if isinstance(pen, float) else None, weight),
            shape=shape, source=best[2] or ("3d" if "VisuallyLosslessStack" in hit.files else "2.5d"),
            mirrored=best[3], best_cell=best_cell, best_image_id=hit.image_id, per_cell=per_cell,
            penalty=pen, nb_rank=cand.nb_rank, nb_score=cand.nb_score, library=hit.library, hits=cand.hits,
        )

    results: list[RankedLine] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for i, r in enumerate(pool.map(score_line, candidates), 1):
            results.append(r)
            if progress:
                progress(i, len(candidates))
    results.sort(key=lambda r: (-r.score, r.nb_rank))
    for i, r in enumerate(results, 1):
        r.rank = i
    return results
