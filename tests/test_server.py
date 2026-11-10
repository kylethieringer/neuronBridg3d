import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from nb3d import server
from nb3d.lm import LmPoints
from nb3d.nbclient import EmImage, LmHit
from nb3d.query import Cell, QueryError, QueryGroup
from nb3d.rank import Candidate, RankedLine


def _hit(i, line):
    return LmHit(image_id=i, line=line, library="FlyLight Split-GAL4 Drivers", slide_code="s", channel=1,
                 normalized_score=1, matching_pixels=1, mirrored=False,
                 files={"store": "fl:open_data:vnc", "CDM": f"{i}.png", "CDMThumbnail": f"{i}.jpg"})


@pytest.fixture
def client(monkeypatch, tmp_path):
    group = QueryGroup("manc", "T", [Cell("manc", 1, "T", "L", None,
                                          [EmImage(id="e", dataset="manc", body_id=1, neuron_type="T", files={})])])
    calls = {}

    def fake_resolve(ds, q, side=None, segment=None):
        if q == "bad":
            raise QueryError("cell type 'bad' not found in MANC annotations")
        return group

    def fake_rerank(g, c, penalty=False, weight=0.5, progress=None):
        calls["penalty"] = penalty
        if progress:
            progress(1, 1)
        h = _hit("img1", "SS1")
        return [RankedLine(line="SS1", rank=1, score=0.9, shape=0.9, source="3d", mirrored=False, best_cell=1,
                           best_image_id="img1", per_cell={1: 0.9}, penalty=None, nb_rank=7, nb_score=123.0,
                           library=h.library, hits=[h])]

    monkeypatch.setattr(server, "resolve", fake_resolve)
    monkeypatch.setattr(server, "collect_candidates", lambda g, top_n=200, libraries=None: [Candidate("SS1", 7, 123.0, [_hit("img1", "SS1")])])
    monkeypatch.setattr(server, "rerank", fake_rerank)
    monkeypatch.setattr(server, "load_em_points", lambda cell: np.array([[1.0, 2, 3], [4, 5, 6]]))
    monkeypatch.setattr(server, "load_lm_points", lambda h, threshold=None: LmPoints(np.array([[7.0, 8, 9]]), "3d"))
    monkeypatch.setattr(server.verdicts, "VERDICTS", tmp_path / "v.csv")
    server.JOBS.clear()
    c = TestClient(server.app)
    c.calls = calls
    return c


def _search(c, **params):
    for _ in range(100):
        r = c.get("/api/search", params={"dataset": "manc", **params})
        if r.status_code != 200 or r.json()["status"] == "done":
            return r
        time.sleep(0.02)
    raise AssertionError("search never finished")


def test_search_returns_both_ranks(client):
    r = _search(client, query="T")
    assert r.status_code == 200
    res = r.json()["results"][0]
    assert res["rank"] == 1 and res["nb_rank"] == 7 and res["nb_score"] == 123.0
    assert res["source"] == "3d"
    assert res["links"]["flylight"].endswith("line=SS1")
    assert res["links"]["neuronbridge"].endswith("SS1")
    assert res["thumbnail_url"].endswith("img1.jpg")
    assert r.json()["group"]["cells"] == [{"body_id": "1", "side": "L", "segment": None, "cell_type": "T"}]


def test_search_query_error_400(client):
    r = client.get("/api/search", params={"dataset": "manc", "query": "bad"})
    assert r.status_code == 400
    assert "not found" in r.json()["error"]


def test_penalty_param_defaults_false(client):
    _search(client, query="T")
    assert client.calls["penalty"] is False
    _search(client, query="T", penalty="true")
    assert client.calls["penalty"] is True


def test_geometry_after_search(client):
    _search(client, query="T")
    r = client.get("/api/geometry", params={"dataset": "manc", "body_ids": "1", "image_id": "img1"})
    assert r.status_code == 200
    g = r.json()
    assert g["em"]["1"] == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert g["lm"] == [[7.0, 8.0, 9.0]] and g["lm_source"] == "3d"
    assert len(g["template_outline"]) == 8


def test_geometry_unknown_image_404(client):
    r = client.get("/api/geometry", params={"dataset": "manc", "body_ids": "1", "image_id": "nope"})
    assert r.status_code == 404


def test_verdict_roundtrip(client):
    r = client.post("/api/verdict", json={"dataset": "manc", "query": "T", "line": "SS1", "verdict": "good", "note": ""})
    assert r.status_code == 200
    res = _search(client, query="T").json()["results"][0]
    assert res["verdict"] == "good"


def test_static_files_revalidate():
    from fastapi.testclient import TestClient
    from nb3d.server import app
    r = TestClient(app).get("/style.css")
    assert r.status_code == 200 and r.headers["cache-control"] == "no-cache"
