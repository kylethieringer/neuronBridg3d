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


def test_fetch_of_same_url_while_another_download_is_in_progress(tmp_path, monkeypatch):
    # Parallel workers can request the same image: worker A is mid-download when worker B
    # fetches the same URL and finishes first. Both must get the complete file.
    import threading
    a_mid_download, b_finished = threading.Event(), threading.Event()

    class SlowResponse(FakeResponse):
        def iter_content(self, chunk_size=None):
            yield b"hello "
            a_mid_download.set()
            b_finished.wait(5)
            yield b"world"

    responses = iter([SlowResponse([]), FakeResponse([b"hello ", b"world"])])
    monkeypatch.setattr(cache.requests, "get", lambda url, stream=True, timeout=None: next(responses))
    url = "https://example.org/a/image.png"
    a_result, a_error = [], []

    def worker_a():
        try:
            a_result.append(cache.fetch(url, cache_dir=tmp_path))
        except Exception as e:           # noqa: BLE001 - asserted below
            a_error.append(e)

    a = threading.Thread(target=worker_a)
    a.start()
    assert a_mid_download.wait(5)
    try:
        b_path = cache.fetch(url, cache_dir=tmp_path)   # worker B
    finally:
        b_finished.set()
        a.join(10)
    assert a_error == []
    assert a_result == [b_path]
    assert b_path.read_bytes() == b"hello world"
    assert [p.name for p in tmp_path.rglob("*") if p.is_file()] == ["image.png"]


LONG_URL = ("https://s3.amazonaws.com/janelia-flylight-color-depth/JRC2018_VNC_Unisex_40x_DS/"
            "FlyLight_Split-GAL4_Omnibus_Broad/"
            "SS64893-20200828_19_H3-Split_GAL4-f-63x-vnc-JRC2018_VNC_Unisex_40x_DS-CDM_3.png")


def _dir_of_length(base, n):
    """A folder under ``base`` whose absolute path is exactly ``n`` characters (a deep tester checkout)."""
    pad = n - len(str(base.absolute())) - 1
    assert pad > 0
    return base / ("d" * pad)


def test_cache_path_stays_under_windows_path_limit(tmp_path):
    # Windows refuses paths over 260 characters unless long paths are enabled; a tester's cache
    # can sit in a deep folder. Leave room for the temp-file suffix used while downloading.
    deep = _dir_of_length(tmp_path, 150)
    p = cache.cache_path(LONG_URL, deep)
    assert len(str(p)) <= cache.MAX_PATH
    assert p.suffix == ".png"
    assert p == cache.cache_path(LONG_URL, deep)                      # deterministic
    assert p != cache.cache_path(LONG_URL.replace("CDM_3", "CDM_2"), deep)


def test_short_cache_paths_keep_the_url_layout(tmp_path):
    # Existing caches stay valid: paths that fit are unchanged.
    url = "https://example.org/a/b/image.png"
    assert cache.cache_path(url, tmp_path) == tmp_path / "example.org" / "a" / "b" / "image.png"


def test_fetch_and_fetch_transient_work_in_a_deep_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(cache.requests, "get", lambda url, stream=True, timeout=None: FakeResponse([b"img"]))
    deep = _dir_of_length(tmp_path, 150)
    assert cache.fetch(LONG_URL, cache_dir=deep).read_bytes() == b"img"
    with cache.fetch_transient(LONG_URL.replace(".png", ".h5j"), cache_dir=deep) as p:
        assert p.read_bytes() == b"img" and p.suffix == ".h5j"
