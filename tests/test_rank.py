import numpy as np
import pytest

from nb3d import rank
from nb3d.lm import LmPoints
from nb3d.nbclient import EmImage, LmHit
from nb3d.query import Cell, QueryGroup
from nb3d.score import ShapeScore

SPLIT = "FlyLight Split-GAL4 Drivers"
MCFO = "FlyLight Gen1 MCFO"


def hit(image_id, line, score, library=SPLIT, stack=True):
    files = {"store": "fl:open_data:vnc", "CDM": f"{image_id}.png"}
    if stack:
        files["VisuallyLosslessStack"] = f"{image_id}.h5j"
    return LmHit(image_id=image_id, line=line, library=library, slide_code="s", channel=1,
                 normalized_score=score, matching_pixels=1, mirrored=False, files=files)


def group(*body_ids):
    cells = [Cell("manc", b, "T", "L", None,
                  [EmImage(id=f"em{b}", dataset="manc", body_id=b, neuron_type="T", files={})])
             for b in body_ids]
    return QueryGroup("manc", "T", cells)


@pytest.fixture
def fakes(monkeypatch):
    """hits per EM image; shape per (body_id, image_id); LM/expression points per image id."""
    state = {"hits": {}, "shape": {}, "points": {}, "expr": {}}
    monkeypatch.setattr(rank, "get_cds_hits", lambda em: state["hits"].get(em.id, []))
    monkeypatch.setattr(rank, "load_em_points", lambda cell: np.array([[float(cell.body_id), 0, 0]]))

    def fake_lm(h, prefer_3d=True, threshold=None):
        src = "3d" if "VisuallyLosslessStack" in h.files else "2.5d"
        return LmPoints(state["points"].get(h.image_id, np.zeros((10, 3))), src)

    monkeypatch.setattr(rank, "load_lm_points", fake_lm)
    monkeypatch.setattr(rank, "shape_score",
                        lambda em_xyz, lm_pts: state["shape"].get(state["_cur"](em_xyz, lm_pts), ShapeScore(0.0, False)))
    # map (em body, lm points identity) -> key
    state["_cur"] = lambda em_xyz, lm_pts: (int(em_xyz[0, 0]), state["_ids"].get(id(lm_pts.xyz)))
    state["_ids"] = {}

    orig_fake = fake_lm

    def tracking_lm(h, prefer_3d=True, threshold=None):
        p = orig_fake(h, prefer_3d, threshold)
        state["_ids"][id(p.xyz)] = h.image_id
        return p

    monkeypatch.setattr(rank, "load_lm_points", tracking_lm)
    monkeypatch.setattr(rank, "find_full_expression_image", lambda line, ds: state["expr"].get(line))
    return state


def test_collect_collapses_images_to_lines(fakes):
    fakes["hits"]["em1"] = [hit("a1", "SS1", 90), hit("b1", "SS2", 80), hit("a2", "SS1", 70)]
    c = rank.collect_candidates(group(1))
    assert [x.line for x in c] == ["SS1", "SS2"]
    assert [h.image_id for h in c[0].hits] == ["a1", "a2"]
    assert c[0].nb_score == 90


def test_collect_keeps_at_most_three_images_per_line(fakes):
    fakes["hits"]["em1"] = [hit(f"a{i}", "SS1", 100 - i) for i in range(6)]
    c = rank.collect_candidates(group(1))
    assert [h.image_id for h in c[0].hits] == ["a0", "a1", "a2"]


def test_collect_top_n_cuts_lines_not_images(fakes):
    fakes["hits"]["em1"] = [hit(f"x{i}", f"SS{i // 2}", 100 - i) for i in range(10)]  # 5 lines, 2 images each
    c = rank.collect_candidates(group(1), top_n=3)
    assert [x.line for x in c] == ["SS0", "SS1", "SS2"]
    assert all(len(x.hits) == 2 for x in c)


def test_nb_rank_is_baseline_order_across_group(fakes):
    fakes["hits"]["em1"] = [hit("a", "SS1", 50)]
    fakes["hits"]["em2"] = [hit("b", "SS2", 70), hit("c", "SS1", 10)]
    c = rank.collect_candidates(group(1, 2))
    assert [(x.line, x.nb_rank, x.nb_score) for x in c] == [("SS2", 1, 70), ("SS1", 2, 50)]


def test_collect_default_libraries_split_only(fakes):
    fakes["hits"]["em1"] = [hit("m", "R1", 99, library=MCFO), hit("s", "SS1", 10)]
    assert [x.line for x in rank.collect_candidates(group(1))] == ["SS1"]
    assert [x.line for x in rank.collect_candidates(group(1), libraries=rank.ALL_LIBRARIES)] == ["R1", "SS1"]


def test_rerank_orders_by_score(fakes):
    fakes["hits"]["em1"] = [hit("a", "SS1", 90), hit("b", "SS2", 80)]
    fakes["shape"][(1, "a")] = ShapeScore(0.2, False)
    fakes["shape"][(1, "b")] = ShapeScore(0.7, True)
    g = group(1)
    r = rank.rerank(g, rank.collect_candidates(g))
    assert [(x.line, x.rank, x.nb_rank) for x in r] == [("SS2", 1, 2), ("SS1", 2, 1)]
    assert r[0].shape == pytest.approx(0.7) and r[0].mirrored is True and r[0].best_image_id == "b"


def test_group_uses_best_cell(fakes):
    fakes["hits"]["em1"] = [hit("a", "SS1", 90)]
    fakes["shape"][(1, "a")] = ShapeScore(0.1, False)
    fakes["shape"][(2, "a")] = ShapeScore(0.8, False)
    g = group(1, 2)
    r = rank.rerank(g, rank.collect_candidates(g))
    assert r[0].score == pytest.approx(0.8)
    assert r[0].best_cell == 2
    assert r[0].per_cell == {1: pytest.approx(0.1), 2: pytest.approx(0.8)}


def test_penalty_off_by_default(fakes):
    fakes["hits"]["em1"] = [hit("a", "SS1", 90)]
    g = group(1)
    assert all(x.penalty is None for x in rank.rerank(g, rank.collect_candidates(g)))


def test_penalty_lowers_broad_line(fakes):
    fakes["hits"]["em1"] = [hit("a", "BROAD", 90), hit("b", "SPARSE", 80)]
    fakes["shape"][(1, "a")] = ShapeScore(0.8, False)
    fakes["shape"][(1, "b")] = ShapeScore(0.7, False)
    em = np.array([[1.0, 0, 0]])
    fakes["expr"]["BROAD"] = hit("ea", "BROAD", 0)
    fakes["expr"]["SPARSE"] = hit("eb", "SPARSE", 0)
    fakes["points"]["ea"] = np.vstack([em, np.full((9, 3), 200.0)])  # 90% far away
    fakes["points"]["eb"] = np.repeat(em, 10, axis=0)                 # all on the cell
    g = group(1)
    c = rank.collect_candidates(g)
    assert [x.line for x in rank.rerank(g, c)] == ["BROAD", "SPARSE"]
    on = rank.rerank(g, c, penalty=True, weight=0.5)
    assert [x.line for x in on] == ["SPARSE", "BROAD"]
    assert on[1].penalty == pytest.approx(0.9)
    assert on[1].score == pytest.approx(0.8 * (1 - 0.5 * 0.9))


def test_penalty_missing_expression_image(fakes):
    fakes["hits"]["em1"] = [hit("a", "SS1", 90)]
    fakes["shape"][(1, "a")] = ShapeScore(0.6, False)
    g = group(1)
    r = rank.rerank(g, rank.collect_candidates(g), penalty=True)
    assert r[0].penalty == "n/a"
    assert r[0].score == pytest.approx(0.6)


def test_source_reported(fakes):
    fakes["hits"]["em1"] = [hit("a", "SS1", 90, stack=False), hit("b", "SS2", 80)]
    fakes["shape"][(1, "a")] = ShapeScore(0.5, False)
    fakes["shape"][(1, "b")] = ShapeScore(0.4, False)
    g = group(1)
    r = {x.line: x.source for x in rank.rerank(g, rank.collect_candidates(g))}
    assert r == {"SS1": "2.5d", "SS2": "3d"}


def test_progress_called(fakes):
    fakes["hits"]["em1"] = [hit("a", "SS1", 90), hit("b", "SS2", 80)]
    calls = []
    g = group(1)
    rank.rerank(g, rank.collect_candidates(g), progress=lambda done, total: calls.append((done, total)))
    assert calls[-1] == (2, 2)


def test_full_expression_prefers_split_20x_channel1(monkeypatch):
    ims = [
        {"id": "63", "libraryName": SPLIT, "anatomicalArea": "VNC", "objective": "63x", "channel": 1,
         "publishedName": "SS1", "slideCode": "s", "files": {"store": "v", "CDM": "a.png"}},
        {"id": "c2", "libraryName": SPLIT, "anatomicalArea": "VNC", "objective": "20x", "channel": 2,
         "publishedName": "SS1", "slideCode": "s", "files": {"store": "v", "CDM": "b.png", "VisuallyLosslessStack": "b.h5j"}},
        {"id": "c1", "libraryName": SPLIT, "anatomicalArea": "VNC", "objective": "20x", "channel": 1,
         "publishedName": "SS1", "slideCode": "s", "files": {"store": "v", "CDM": "c.png", "VisuallyLosslessStack": "c.h5j"}},
    ]
    monkeypatch.setattr(rank, "get_line_images", lambda line: ims)
    assert rank.find_full_expression_image("SS1", "manc").image_id == "c1"


def test_full_expression_gen1_uses_gal4_expression(monkeypatch):
    ims = [{"id": "m", "libraryName": MCFO, "anatomicalArea": "VNC", "objective": "40x", "channel": 2,
            "publishedName": "R1", "slideCode": "s",
            "files": {"store": "v", "CDM": "m.png", "VisuallyLosslessStack": "m.h5j", "Gal4Expression": "Gen1/CDM/R1/x.png"}}]
    monkeypatch.setattr(rank, "get_line_images", lambda line: ims)
    h = rank.find_full_expression_image("R1", "manc")
    assert h.files == {"store": "v", "Gal4Expression": "Gen1/CDM/R1/x.png"}


def test_full_expression_none(monkeypatch):
    monkeypatch.setattr(rank, "get_line_images", lambda line: [])
    assert rank.find_full_expression_image("SS9", "manc") is None


def _im(id_, lib, area="VNC", obj="20x", ch=1, stack=True):
    files = {"store": "v", "CDM": f"{id_}.png"}
    if stack:
        files["VisuallyLosslessStack"] = f"{id_}.h5j"
    return {"id": id_, "libraryName": lib, "anatomicalArea": area, "objective": obj, "channel": ch,
            "publishedName": "SS1", "slideCode": "s", "files": files}


def test_full_expression_uses_omnibus_broad_20x(monkeypatch):
    monkeypatch.setattr(rank, "get_line_images", lambda line: [_im("o", rank.OMNIBUS_BROAD)])
    assert rank.find_full_expression_image("SS1", "manc").image_id == "o"


def test_full_expression_prefers_split_drivers_over_omnibus(monkeypatch):
    ims = [_im("o", rank.OMNIBUS_BROAD), _im("d", SPLIT)]
    monkeypatch.setattr(rank, "get_line_images", lambda line: ims)
    assert rank.find_full_expression_image("SS1", "manc").image_id == "d"


def test_full_expression_skips_brain_images(monkeypatch):
    ims = [_im("b", SPLIT, area="Brain"), _im("o", rank.OMNIBUS_BROAD, ch=2)]
    monkeypatch.setattr(rank, "get_line_images", lambda line: ims)
    assert rank.find_full_expression_image("SS1", "manc").image_id == "o"
