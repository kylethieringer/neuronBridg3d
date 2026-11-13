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


def test_fetch_transient_deletes_download_and_keeps_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(cache.requests, "get", lambda url, stream=True, timeout=None: FakeResponse([b"stack"]))
    url = "https://example.org/x/stack.h5j"
    with cache.fetch_transient(url, cache_dir=tmp_path) as p:
        assert p.read_bytes() == b"stack"
    assert not p.exists()
    assert not cache.cache_path(url, tmp_path).exists()
    assert not any(q.is_file() for q in tmp_path.rglob("*"))


def test_fetch_transient_uses_existing_cached_copy(tmp_path, monkeypatch):
    url = "https://example.org/x/stack.h5j"
    cached = cache.cache_path(url, tmp_path)
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"old")
    monkeypatch.setattr(cache.requests, "get", lambda *a, **k: pytest.fail("downloaded despite cached copy"))
    with cache.fetch_transient(url, cache_dir=tmp_path) as p:
        assert p == cached and p.read_bytes() == b"old"
    assert cached.exists()


def test_fetch_transient_failed_download_leaves_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr(cache.requests, "get",
                        lambda url, stream=True, timeout=None: FakeResponse([b"a", b"b"], fail_after=1))
    with pytest.raises(ConnectionError):
        with cache.fetch_transient("https://example.org/x/stack.h5j", cache_dir=tmp_path):
            pass
    assert not any(q.is_file() for q in tmp_path.rglob("*"))
