import pandas as pd
import pytest

from nb3d.benchmark import groundtruth as gt


@pytest.fixture(scope="module")
def table():
    return gt.build_groundtruth()


def test_columns_and_dtypes(table):
    assert list(table.columns) == ["dataset", "body_id", "cell_type", "line", "tier", "source", "confidence"]
    assert table["body_id"].dtype == "int64"
    assert table["tier"].dtype == "int64"
    assert set(table["dataset"]) == {"manc", "banc"}
    assert set(table["tier"]) <= {1, 2, 3}


def test_no_duplicate_pairs(table):
    assert not table.duplicated(["dataset", "body_id", "line"]).any()


def test_tier_precedence():
    rows = pd.DataFrame({
        "dataset": ["manc", "manc"], "body_id": [1, 1], "cell_type": ["X", "X"], "line": ["SS1", "SS1"],
        "tier": [3, 1], "source": ["ehrhardt2025", "cheong2024_supp3"], "confidence": ["3", "5"],
    })
    out = gt.dedupe(rows)
    assert len(out) == 1 and out.iloc[0]["tier"] == 1 and out.iloc[0]["source"] == "cheong2024_supp3"


@pytest.mark.parametrize("raw,expected", [
    ("SS01557-20160318_31_A1-f", "SS01557"),
    ("R24D12-20180921_65_J3-f", "R24D12"),
    (" SS32023 ", "SS32023"),
    ("14E09", "R14E09"),
    ("VT046828", "VT046828"),
    ("", None),
])
def test_line_name_normalised(raw, expected):
    assert gt.normalise_line(raw) == expected


def test_counts_in_expected_range(table):
    manc = table[table.dataset == "manc"]
    assert 300 <= manc.body_id.nunique() <= 900
    assert (manc.tier == 1).sum() > 50


def test_known_pair_present(table):
    # Cheong supp3: hg1 MN body 10123 <-> SS32023, independent evidence
    row = table[(table.dataset == "manc") & (table.body_id == 10123) & (table.line == "SS32023")]
    assert len(row) == 1 and row.iloc[0]["tier"] == 1


def test_banc_rows_use_root_626(table):
    banc = table[table.dataset == "banc"]
    assert len(banc) > 0
    assert banc.body_id.min() > 10**17  # BANC root IDs are 18 digits
