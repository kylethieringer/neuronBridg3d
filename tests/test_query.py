import pandas as pd
import pytest

from nb3d import nbclient, query

# Normalised meta tables as returned by query.load_meta (see query.py docstring).
MANC = pd.DataFrame({
    "body_id": [10123, 10107, 10013, 20001],
    "cell_type": ["hg1 MN", "DNg11", "DNg11", "IN03A001"],
    "side": ["L", "R", "L", "L"],
    "segment": [None, None, None, None],
})
BANC = pd.DataFrame({
    "body_id": [720575941542412805, 720575941667143601, 720575941000000003],
    "cell_type": ["IN11B016", "IN11B016", "DNg11"],
    "manc_cell_type": ["IN11B016", "IN11B016", "DNg11"],
    "side": ["L", "R", "R"],
    "segment": ["T1", "T2", None],
    "alt_ids": [[720575941999999999], [], []],
})


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setattr(query, "load_meta", lambda ds: MANC if ds == "manc" else BANC)
    known = set(MANC.body_id) | set(BANC.body_id)

    def fake_em(dataset, body_id):
        if body_id not in known:
            raise nbclient.BodyNotFound(dataset, body_id)
        return [nbclient.EmImage(id=f"im{body_id}", dataset=dataset, body_id=body_id,
                                 neuron_type=None, files={})]

    monkeypatch.setattr(query, "get_em_images", fake_em)


def test_single_body_id(offline):
    g = query.resolve("manc", "10123")
    assert [c.body_id for c in g.cells] == [10123]
    assert g.cells[0].cell_type == "hg1 MN"
    assert g.cells[0].side == "L"
    assert g.cells[0].em_images[0].id == "im10123"


def test_comma_separated_ids(offline):
    g = query.resolve("manc", "10123, 10107")
    assert [c.body_id for c in g.cells] == [10123, 10107]


def test_cell_type_case_insensitive(offline):
    a = query.resolve("manc", "DNg11")
    b = query.resolve("manc", "dng11")
    assert sorted(c.body_id for c in a.cells) == [10013, 10107]
    assert sorted(c.body_id for c in b.cells) == [10013, 10107]
    assert a.label == "DNg11"


def test_side_and_segment_filter(offline):
    assert [c.body_id for c in query.resolve("manc", "DNg11", side="R").cells] == [10107]
    assert [c.body_id for c in query.resolve("banc", "IN11B016", segment="T2").cells] == [720575941667143601]


def test_manc_segment_filter_unavailable(offline):
    with pytest.raises(query.QueryError) as e:
        query.resolve("manc", "DNg11", segment="T1")
    assert "segment" in str(e.value) and "MANC" in str(e.value)


def test_filter_to_empty_group_message(offline):
    with pytest.raises(query.QueryError) as e:
        query.resolve("banc", "IN11B016", side="L", segment="T2")
    msg = str(e.value)
    assert "no cells" in msg and "side=L" in msg and "segment=T2" in msg


def test_unknown_body_id_message(offline):
    with pytest.raises(query.QueryError) as e:
        query.resolve("banc", "720575941111111111")
    msg = str(e.value)
    assert "FlyWire_BANC_v626" in msg and "cell type" in msg


def test_newer_banc_id_translated_to_v626(offline):
    g = query.resolve("banc", "720575941999999999")
    assert [c.body_id for c in g.cells] == [720575941542412805]


def test_unknown_cell_type_message(offline):
    with pytest.raises(query.QueryError) as e:
        query.resolve("manc", "NotAType")
    assert "NotAType" in str(e.value)


def test_cells_missing_from_neuronbridge_are_reported(offline, monkeypatch):
    extra = pd.concat([MANC, pd.DataFrame({"body_id": [55555], "cell_type": ["DNg11"],
                                            "side": ["R"], "segment": [None]})])
    monkeypatch.setattr(query, "load_meta", lambda ds: extra)
    g = query.resolve("manc", "DNg11")
    assert sorted(c.body_id for c in g.cells) == [10013, 10107]
    assert g.missing == [55555]
