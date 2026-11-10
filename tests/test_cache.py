import pytest

from nb3d import cache


class FakeResponse:
    def __init__(self, chunks, status=200, fail_after=None):
        self.chunks = chunks
        self.status_code = status
        self.fail_after = fail_after

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"{self.status_code}")

    def iter_content(self, chunk_size=None):
        for i, c in enumerate(self.chunks):
            if self.fail_after is not None and i >= self.fail_after:
                raise ConnectionError("connection dropped")
            yield c

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_fetch_downloads_once(tmp_path, monkeypatch):
    calls = []

    def fake_get(url, stream=True, timeout=None):
        calls.append(url)
        return FakeResponse([b"hello ", b"world"])

    monkeypatch.setattr(cache.requests, "get", fake_get)
    url = "https://example.org/a/b/file.json"
    p1 = cache.fetch(url, cache_dir=tmp_path)
    p2 = cache.fetch(url, cache_dir=tmp_path)
    assert p1 == p2
    assert p1.read_bytes() == b"hello world"
    assert len(calls) == 1


def test_interrupted_download_leaves_no_file(tmp_path, monkeypatch):
    def fake_get(url, stream=True, timeout=None):
        return FakeResponse([b"part1", b"part2"], fail_after=1)

    monkeypatch.setattr(cache.requests, "get", fake_get)
    url = "https://example.org/x/stack.h5j"
    with pytest.raises(ConnectionError):
        cache.fetch(url, cache_dir=tmp_path)
    assert not cache.cache_path(url, tmp_path).exists()
    assert not any(p.is_file() for p in tmp_path.rglob("*"))
