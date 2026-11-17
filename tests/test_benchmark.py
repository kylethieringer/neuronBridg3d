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
    monkeypatch.setattr(run, "nb_order", lambda g, libraries=None: [c.line for c in full])
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
    monkeypatch.setattr(run, "nb_order", lambda g, libraries=None: [c.line for c in full])
    monkeypatch.setattr(run, "rerank", lambda g, c, **kw: [_ranked("A", 1, 1), _ranked("B", 2, 2)])
    row = run.evaluate_body("manc", 7, {"E"}, top_n=2)
    assert row["nb_rank"] == 5 and row["nb3d_rank"] is None and row["in_top_n"] is False


def _fake_gt(tmp_path, monkeypatch, bodies=(1, 2, 3)):
    import pandas as pd
    gt = tmp_path / "gt.csv"
    pd.DataFrame({"dataset": "manc", "body_id": list(bodies), "line": "SS1", "tier": 1}).to_csv(gt, index=False)
    monkeypatch.setattr(run, "GROUNDTRUTH", gt)
    monkeypatch.setattr(run, "RESULTS", tmp_path)


def _ok_row(ds, body, lines, top_n, libs):
    return {"dataset": ds, "body_id": body, "true_lines": "SS1", "nb_rank": 1, "nb3d_rank": 1,
            "source": "3d", "in_top_n": True, "n_candidates": 1}


def test_default_output_name_has_no_date(tmp_path, monkeypatch):
    _fake_gt(tmp_path, monkeypatch)
    monkeypatch.setattr(run, "evaluate_body", _ok_row)
    run.run_benchmark("manc", 1)
    run.run_benchmark("manc", 1, libraries=run.ALL_LIBRARIES)
    assert (tmp_path / "benchmark-manc-tier1-split.csv").exists()
    assert (tmp_path / "benchmark-manc-tier1-all.csv").exists()


def test_resume_skips_done_bodies(tmp_path, monkeypatch):
    _fake_gt(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(run, "evaluate_body", lambda *a: calls.append(a[1]) or _ok_row(*a))
    out = tmp_path / "x.csv"
    run.run_benchmark("manc", 1, limit=2, out=out)
    df = run.run_benchmark("manc", 1, out=out)
    assert calls == [1, 2, 3] and len(df) == 3


def test_error_rows_keep_columns_aligned(tmp_path, monkeypatch):
    from nb3d.query import QueryError
    _fake_gt(tmp_path, monkeypatch)

    def ev(ds, body, *a):
        if body == 2:
            raise QueryError("nope")
        return _ok_row(ds, body, *a)
    monkeypatch.setattr(run, "evaluate_body", ev)
    df = run.run_benchmark("manc", 1, out=tmp_path / "x.csv")
    assert list(df.columns) == run.COLUMNS
    assert df.set_index("body_id").loc[2, "error"] == "nope"
    assert df.set_index("body_id").loc[3, "nb_rank"] == 1


def test_transient_error_skips_body_without_recording(tmp_path, monkeypatch):
    _fake_gt(tmp_path, monkeypatch)

    def ev(ds, body, *a):
        if body == 2:
            raise ConnectionError("network blip")
        return _ok_row(ds, body, *a)
    monkeypatch.setattr(run, "evaluate_body", ev)
    df = run.run_benchmark("manc", 1, out=tmp_path / "x.csv")
    assert sorted(df.body_id) == [1, 3]


def test_cli_out_flag(tmp_path, monkeypatch):
    from nb3d import cli
    seen = {}
    monkeypatch.setattr(run, "run_benchmark", lambda *a, out=None, **k: seen.setdefault("out", out) and run.pd.DataFrame())
    monkeypatch.setattr(run, "report", lambda df, n: "")
    cli.main(["benchmark", "--out", str(tmp_path / "o.csv")])
    assert str(seen["out"]) == str(tmp_path / "o.csv")


def test_lookup_error_recorded_not_fatal(tmp_path, monkeypatch):
    from nb3d.em import NoSkeleton
    _fake_gt(tmp_path, monkeypatch)

    def ev(ds, body, *a):
        if body == 1:
            raise NoSkeleton("no swc")
        return _ok_row(ds, body, *a)
    monkeypatch.setattr(run, "evaluate_body", ev)
    df = run.run_benchmark("manc", 1, out=tmp_path / "x.csv")
    assert len(df) == 3 and df.set_index("body_id").loc[1, "error"] == "no swc"


def test_cli_workers_flag(monkeypatch):
    from nb3d import cli, rank
    monkeypatch.setattr(rank, "WORKERS", 8)
    monkeypatch.setattr(run, "run_benchmark", lambda *a, **k: run.pd.DataFrame())
    monkeypatch.setattr(run, "report", lambda df, n: "")
    cli.main(["benchmark", "--workers", "3"])
    assert rank.WORKERS == 3
