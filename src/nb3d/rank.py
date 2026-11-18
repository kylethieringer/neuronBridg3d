"""Collect NeuronBridge candidate lines for a query group and re-rank them by 3D shape."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Callable, Literal

import numpy as np

from .em import BRAIN, VNC, load_em_brain_points, load_em_points
from .lm import LmPoints, load_lm_points
from .nbclient import LmHit, get_cds_hits, get_line_images
from .query import QueryGroup
from .score import combine, extra_expression_fraction, shape_score

SPLIT_DRIVERS = "FlyLight Split-GAL4 Drivers"
OMNIBUS_BROAD = "FlyLight Split-GAL4 Omnibus Broad"
SPLIT_LIBRARIES = [SPLIT_DRIVERS, OMNIBUS_BROAD]
ALL_LIBRARIES = SPLIT_LIBRARIES + ["FlyLight Gen1 MCFO", "FlyLight Annotator Gen1 MCFO"]
IMAGES_PER_LINE = 3
WORKERS = 8


@dataclass
class Candidate:
    line: str
    nb_rank: int
    nb_score: float
    hits: list[LmHit]  # VNC images, scored for the ranking
    brain_hits: list[LmHit] = field(default_factory=list)  # 20x brain images, scored separately (BANC)


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
    forward: float = 0.0  # best EM -> LM score over the line's images (high for any line containing the cell)
    reverse: float = 0.0  # best LM -> EM score over the line's images (high for lines labelling little else)
    # Brain-template score against the EM cell's brain points, reported beside the VNC score and not used for
    # the rank. None when the line has no 20x brain hit or the EM cells have no brain image (MANC). Not clamped
    # at 0, so negative scores keep their order.
    brain: float | None = None
    brain_forward: float | None = None
    brain_reverse: float | None = None
    brain_image_id: str = ""
    brain_mirrored: bool = False
    brain_source: str | None = None


def _nb_lines(group: QueryGroup, libraries: list[str] | None) -> list[tuple[str, list[LmHit]]]:
    """NeuronBridge baseline: hits over all the group's EM images (VNC and, for BANC, brain), collapsed to
    lines and ordered by best normalizedScore. Each line's hits are de-duplicated, best first."""
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
    return ordered


def nb_order(group: QueryGroup, libraries: list[str] | None = None) -> list[str]:
    """All lines in NeuronBridge's order (the benchmark baseline), without preparing them for scoring."""
    return [line for line, _ in _nb_lines(group, libraries)]


def _vnc_hits(line: str, hits: list[LmHit]) -> list[LmHit]:
    """Hits that can be scored against a VNC skeleton, best first. A brain-image hit (BANC bodies have a brain
    EM image too) is replaced by the VNC image of the same fly, objective and channel, which shows the same
    neuron's VNC part; it is dropped if the line has no such image."""
    vnc_ims: list[dict] | None = None
    out: dict[str, LmHit] = {}
    for h in hits:
        if h.anatomical_area != "VNC":
            if vnc_ims is None:
                vnc_ims = get_line_images(line)
            im = next((im for im in vnc_ims if im.get("slideCode") == h.slide_code
                       and im.get("objective") == h.objective and im.get("channel") == h.channel), None)
            if im is None:
                continue
            h = replace(h, image_id=im["id"], library=im["libraryName"], files=im["files"],
                        match_files={}, anatomical_area="VNC")
        out.setdefault(h.image_id, h)
    return list(out.values())


def collect_candidates(group: QueryGroup, top_n: int = 200, libraries: list[str] | None = None) -> list[Candidate]:
    """NeuronBridge's top ``top_n`` lines (order and nb_rank = the baseline), each with up to
    IMAGES_PER_LINE VNC images to score. A line can end up with none (brain-only match, no VNC image)."""
    return [Candidate(line, i + 1, hs[0].normalized_score, _vnc_hits(line, hs)[:IMAGES_PER_LINE], _brain_hits(hs))
            for i, (line, hs) in enumerate(_nb_lines(group, libraries)[:top_n])]


def _brain_hits(hits: list[LmHit]) -> list[LmHit]:
    """The line's 20x brain hits (already de-duplicated, best first), up to IMAGES_PER_LINE."""
    return [h for h in hits if h.anatomical_area == "Brain" and h.objective == "20x"][:IMAGES_PER_LINE]


def find_full_expression_images(line: str, dataset: str) -> list[LmHit]:
    """The line's whole-VNC expression images (docs/spike-findings.md, "Full-expression images"):
    every 20x VNC image from Split-GAL4 Drivers or Omnibus Broad (all samples and channels, so the
    penalty can take the median and one noisy sample does not dominate), preferred display image first; else a Gen1 GAL4 expression CDM."""
    ims = get_line_images(line)

    def as_hit(im: dict, files: dict) -> LmHit:
        return LmHit(image_id=im["id"], line=line, library=im["libraryName"], slide_code=im.get("slideCode", ""),
                     channel=im.get("channel"), normalized_score=0.0, matching_pixels=0, mirrored=False,
                     files=files, objective=im.get("objective"))

    split = [as_hit(im, im["files"]) for im in ims if im.get("libraryName") in (SPLIT_DRIVERS, OMNIBUS_BROAD)
             and im.get("objective") == "20x" and im.get("anatomicalArea", "VNC") == "VNC"]
    # Preferred first (the viewer shows exprs[0]): Split-GAL4 Drivers, channel 1, with a 3D stack.
    split.sort(key=lambda h: (h.library != SPLIT_DRIVERS, h.channel != 1, "VisuallyLosslessStack" not in h.files))
    if split:
        return split
    for im in ims:
        expr = im["files"].get("Gal4Expression")
        if expr:
            return [as_hit(im, {"store": im["files"].get("store", ""), "Gal4Expression": expr})]
    return []


def rerank(group: QueryGroup, candidates: list[Candidate], penalty: bool = False, weight: float = 0.5,
           progress: Callable[[int, int], None] | None = None) -> list[RankedLine]:
    em_pts = {c.body_id: load_em_points(c) for c in group.cells}
    group_xyz = np.vstack(list(em_pts.values()))
    em_brain = {c.body_id: p for c in group.cells if (p := load_em_brain_points(c)) is not None}

    def score_brain(cand: Candidate) -> dict:
        """Best brain-template score over the line's brain images and the group's cells."""
        if not em_brain or not cand.brain_hits:
            return {}
        best, fwd, rev = None, float("-inf"), float("-inf")
        for h in cand.brain_hits:
            lm = load_lm_points(h)
            for xyz in em_brain.values():
                s = shape_score(xyz, lm, BRAIN)
                fwd, rev = max(fwd, s.forward), max(rev, s.reverse)
                if best is None or s.value > best[0]:
                    best = (s.value, h, lm.source, s.mirrored)
        return dict(brain=best[0], brain_forward=fwd, brain_reverse=rev, brain_image_id=best[1].image_id,
                    brain_mirrored=best[3], brain_source=best[2])

    def score_line(cand: Candidate) -> RankedLine:
        per_cell: dict[int, float] = {}
        best = (-1.0, None, None, False)  # value, hit, source, mirrored
        best_cell = group.cells[0].body_id
        forward = reverse = float("-inf")  # stays -inf only if the line has no images
        for h in cand.hits:
            lm: LmPoints = load_lm_points(h)
            for body, xyz in em_pts.items():
                s = shape_score(xyz, lm, VNC)
                per_cell[body] = max(per_cell.get(body, 0.0), s.value)
                forward, reverse = max(forward, s.forward), max(reverse, s.reverse)
                if s.value > best[0]:
                    best, best_cell = (s.value, h, lm.source, s.mirrored), body
        shape = max(best[0], 0.0)
        pen: float | None | Literal["n/a"] = None
        if penalty:
            exprs = find_full_expression_images(cand.line, group.dataset)
            pen = "n/a" if not exprs else float(np.median(
                [extra_expression_fraction(group_xyz, load_lm_points(e).xyz) for e in exprs]))
        hit = best[1] or (cand.hits[0] if cand.hits else None)
        if hit is None:  # no VNC image to score (brain-only match): keep the line, ranked by NeuronBridge order at the bottom
            source = "none"
        else:
            source = best[2] or ("3d" if "VisuallyLosslessStack" in hit.files else "2.5d")
        return RankedLine(
            line=cand.line, rank=0,
            score=combine(shape, pen if isinstance(pen, float) else None, weight),
            shape=shape, source=source,
            mirrored=best[3], best_cell=best_cell, best_image_id=hit.image_id if hit else "", per_cell=per_cell,
            penalty=pen, nb_rank=cand.nb_rank, nb_score=cand.nb_score, library=hit.library if hit else "",
            hits=cand.hits,
            forward=forward if cand.hits else 0.0, reverse=reverse if cand.hits else 0.0,
            **score_brain(cand),
        )

    results: list[RankedLine] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for i, r in enumerate(pool.map(score_line, candidates), 1):
            results.append(r)
            if progress:
                progress(i, len(candidates))
    results.sort(key=lambda r: (r.source == "none", -r.score, r.nb_rank))
    for i, r in enumerate(results, 1):
        r.rank = i
    return results
