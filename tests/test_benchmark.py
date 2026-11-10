import pytest

from nb3d.benchmark import run
from nb3d.rank import Candidate, RankedLine


def test_rank_of_correct_first_match():
    assert run.rank_of_correct(["A", "B", "C", "D"], {"C", "B"}) == 2


def test_rank_of_correct_absent():
    assert run.rank_of_correct(["A", "B"], {"Z"}) is None


def test_summarize_values():
    s = run.summarize([1, 3, None, 25])
    assert s["n"] == 4
    assert s["top1"] == pytest.approx(0.25)
    assert s["top5"] == pytest.approx(0.5)
    assert s["top20"] == pytest.approx(0.5)
    assert s["recall_ceiling"] == pytest.approx(0.75)
    assert s["median_rank"] == 3


def test_summarize_empty():
    s = run.summarize([])
    assert s["n"] == 0 and s["median_rank"] is None


def _ranked(line, rank, nb_rank, source="3d"):
    return RankedLine(line=line, rank=rank, score=1.0 / rank, shape=0, source=source, mirrored=False,
                      best_cell=1, best_image_id="i", per_cell={}, penalty=None, nb_rank=nb_rank, nb_score=0)


def test_evaluate_body(monkeypatch):
    full = [Candidate(l, i + 1, 100 - i, []) for i, l in enumerate(["A", "B", "C", "D", "E"])]
    monkeypatch.setattr(run, "resolve", lambda ds, q: "group")
    monkeypatch.setattr(run, "collect_candidates", lambda g, top_n=200, libraries=None: full[:top_n])
    monkeypatch.setattr(run, "rerank", lambda g, c, **kw: [_ranked("C", 1, 3, "2.5d"), _ranked("A", 2, 1), _ranked("B", 3, 2)])
    row = run.evaluate_body("manc", 7, {"C", "E"}, top_n=3)
    assert row["nb_rank"] == 3          # best NeuronBridge rank of a true line (full list)
    assert row["nb3d_rank"] == 1
    assert row["source"] == "2.5d"
    assert row["in_top_n"] is True
    assert row["n_candidates"] == 3


def test_evaluate_body_true_line_beyond_cutoff(monkeypatch):
    full = [Candidate(l, i + 1, 100 - i, []) for i, l in enumerate(["A", "B", "C", "D", "E"])]
    monkeypatch.setattr(run, "resolve", lambda ds, q: "group")
    monkeypatch.setattr(run, "collect_candidates", lambda g, top_n=200, libraries=None: full[:top_n])
    monkeypatch.setattr(run, "rerank", lambda g, c, **kw: [_ranked("A", 1, 1), _ranked("B", 2, 2)])
    row = run.evaluate_body("manc", 7, {"E"}, top_n=2)
    assert row["nb_rank"] == 5 and row["nb3d_rank"] is None and row["in_top_n"] is False
