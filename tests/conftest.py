import pytest

from nb3d import lm


@pytest.fixture(autouse=True)
def _isolated_points_cache(tmp_path, monkeypatch):
    """Keep the derived LM point cache out of the repo's cache/ during tests."""
    monkeypatch.setattr(lm, "POINTS_DIR", tmp_path / "points")
