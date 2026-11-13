import numpy as np
import pytest

from nb3d import score
from nb3d.em import mirror_x
from nb3d.lm import LmPoints


def curve(offset=(0.0, 0.0, 0.0), n=400):
    """A smooth 3D curve on the left half of the template (x < midline), microns."""
    t = np.linspace(0, 1, n)
    xyz = np.column_stack([60 + 40 * t, 150 + 120 * t, 60 + 20 * np.sin(6 * t)])
    return xyz + np.asarray(offset)


def test_identical_cloud_scores_near_one():
    em = curve()
    s = score.shape_score(em, LmPoints(em.copy(), "3d"))
    assert s.value > 0.9
    assert s.mirrored is False


def test_unrelated_cloud_scores_low():
    em = curve()
    rng = np.random.default_rng(0)
    other = rng.uniform([150, 300, 0], [260, 500, 150], size=(2000, 3))
    assert score.shape_score(em, LmPoints(other, "3d")).value < 0.2


def test_mirrored_match_scores_high():
    em = curve()
    s = score.shape_score(em, LmPoints(mirror_x(em), "3d"))
    assert s.value > 0.9
    assert s.mirrored is True


def test_extra_cells_lower_shape_score():
    # Mean of EM -> LM and LM -> EM: an image that also labels other cells scores below a clean one,
    # so broad lines no longer outrank sparse ones (tier-2 benchmark, 2026-11-13).
    em = curve()
    clean = score.shape_score(em, LmPoints(em.copy(), "3d")).value
    broad = score.shape_score(em, LmPoints(np.vstack([em, curve(offset=(0, 150, 0))]), "3d")).value
    assert broad < clean - 0.2


def test_partial_cover_scores_below_full_cover():
    # The reverse direction must not reward an image that shows only part of the cell.
    em = curve()
    full = score.shape_score(em, LmPoints(em.copy(), "3d")).value
    half = score.shape_score(em, LmPoints(em[:200].copy(), "3d")).value
    assert half < full


def test_mirror_choice_uses_combined_score(monkeypatch):
    calls = {}

    def fake(q, t):
        key = (q.tag, t.tag)
        return calls[key]

    calls.update({("em", "lm"): 0.9, ("lm", "em"): -0.5, ("em_m", "lm"): 0.6, ("lm", "em_m"): 0.4})
    tags = iter(["lm", "em", "em_m"])

    class Dp:
        def __init__(self, tag):
            self.tag = tag

    monkeypatch.setattr(score, "_dotprops", lambda xyz: Dp(next(tags)))
    monkeypatch.setattr(score, "_forward_nblast", fake)
    s = score.shape_score(curve(), LmPoints(curve(), "3d"))
    assert s.mirrored is True and s.value == pytest.approx(0.5)  # (0.6+0.4)/2 beats (0.9-0.5)/2


def test_extra_fraction_zero_when_inside():
    grp = curve()
    assert score.extra_expression_fraction(grp, grp + 1.0) == 0.0


def test_extra_fraction_counts_far_points():
    grp = curve()
    expr = np.vstack([grp, grp + np.array([0, 0, 50.0])])
    assert score.extra_expression_fraction(grp, expr) == pytest.approx(0.5, abs=0.01)


def test_extra_fraction_ignores_mirror_side():
    # Expression of the contralateral homologue is not "extra".
    grp = curve()
    expr = np.vstack([grp, mirror_x(grp)])
    assert score.extra_expression_fraction(grp, expr) == pytest.approx(0.0, abs=0.01)


def test_combine_values():
    assert score.combine(0.8, 0.5, 0.5) == pytest.approx(0.6)
    assert score.combine(0.8, None) == pytest.approx(0.8)
    assert score.combine(0.8, 0.0, 1.0) == pytest.approx(0.8)
