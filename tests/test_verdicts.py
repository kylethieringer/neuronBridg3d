from nb3d import verdicts


def test_verdict_latest_wins(tmp_path, monkeypatch):
    monkeypatch.setattr(verdicts, "VERDICTS", tmp_path / "verdicts.csv")
    verdicts.save_verdict("manc", "hg1 MN", "SS32023", "unsure")
    verdicts.save_verdict("manc", "hg1 MN", "SS47161", "bad", note="labels too much")
    verdicts.save_verdict("manc", "hg1 MN", "SS32023", "good")
    df = verdicts.load_verdicts()
    assert len(df) == 2
    got = dict(zip(df.line, df.verdict))
    assert got == {"SS32023": "good", "SS47161": "bad"}
    assert df[df.line == "SS47161"].note.iloc[0] == "labels too much"


def test_verdict_rejects_unknown_value(tmp_path, monkeypatch):
    import pytest
    monkeypatch.setattr(verdicts, "VERDICTS", tmp_path / "verdicts.csv")
    with pytest.raises(ValueError):
        verdicts.save_verdict("manc", "q", "SS1", "maybe")


def test_load_verdicts_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(verdicts, "VERDICTS", tmp_path / "none.csv")
    assert len(verdicts.load_verdicts()) == 0
