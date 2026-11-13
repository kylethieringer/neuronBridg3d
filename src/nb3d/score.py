"""3D shape similarity (mean of EM -> LM and LM -> EM NBLAST) and the optional extra-expression penalty."""
from __future__ import annotations

from dataclasses import dataclass

import navis
import numpy as np
from scipy.spatial import cKDTree

from .em import mirror_x
from .lm import LmPoints

K = 5  # neighbours for dotprops tangents


@dataclass
class ShapeScore:
    value: float
    mirrored: bool


def _dotprops(xyz: np.ndarray) -> navis.Dotprops:
    dp = navis.make_dotprops(np.asarray(xyz, dtype=float), k=K)
    dp.units = "1 micron"
    return dp


def _forward_nblast(q: navis.Dotprops, t: navis.Dotprops) -> float:
    res = navis.nblast(q, t, scores="forward", normalized=True, progress=False, n_cores=1)
    return float(np.asarray(res)[0, 0])


def shape_score(em_xyz: np.ndarray, lm: LmPoints) -> ShapeScore:
    """Mean of forward (EM -> LM) and reverse (LM -> EM) NBLAST, each normalised to its query's self-score.
    The reverse term penalises LM signal away from the cell, so broad images no longer outscore sparse
    ones. Scored for the EM cell and its mirror image; the better one is kept."""
    if len(lm.xyz) <= K or len(em_xyz) <= K:
        return ShapeScore(0.0, False)
    lm_dp = _dotprops(lm.xyz)

    def mean_score(em_dp: navis.Dotprops) -> float:
        return (_forward_nblast(em_dp, lm_dp) + _forward_nblast(lm_dp, em_dp)) / 2

    direct = mean_score(_dotprops(em_xyz))
    mirrored = mean_score(_dotprops(mirror_x(em_xyz)))
    if mirrored > direct:
        return ShapeScore(mirrored, True)
    return ShapeScore(direct, False)


def extra_expression_fraction(group_xyz: np.ndarray, expr_xyz: np.ndarray, radius_um: float = 15.0) -> float:
    """Fraction of expression points farther than ``radius_um`` from every group point
    (the group's mirror image counts as part of the group)."""
    if len(expr_xyz) == 0:
        return 0.0
    ref = np.vstack([group_xyz, mirror_x(group_xyz)])
    d, _ = cKDTree(ref).query(expr_xyz, k=1)
    return float((d > radius_um).mean())


def combine(shape: float, extra: float | None, weight: float = 0.5) -> float:
    if extra is None:
        return shape
    return shape * (1 - weight * extra)
