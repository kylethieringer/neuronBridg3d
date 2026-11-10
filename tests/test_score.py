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


def test_extra_cells_barely_lower_shape_score():
    # Forward (EM -> LM) score: extra labelled cells elsewhere don't hide a good match.
    em = curve()
    extra = curve(offset=(0, 150, 0))
    s = score.shape_score(em, LmPoints(np.vstack([em, extra]), "3d"))
    assert s.value > 0.85


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
