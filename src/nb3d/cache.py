"""Download-once disk cache for public HTTP resources."""
from __future__ import annotations

import hashlib
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from urllib.parse import unquote, urlparse

import requests

CACHE_DIR = Path(os.environ.get("NB3D_CACHE", Path(__file__).resolve().parents[2] / "cache"))


MAX_PATH = 240  # Windows refuses paths over 260 chars (long paths off by default); room for ".xxxxxxxx.part"


def cache_path(url: str, cache_dir: Path = CACHE_DIR) -> Path:
    """Local path for a URL: <cache_dir>/<host>/<url path>, or, when that would be too long for
    Windows, <cache_dir>/<host>/_long/<hash of url><extension>."""
    u = urlparse(url)
    rel = unquote(u.path).lstrip("/")
    p = Path(cache_dir) / u.netloc / rel
    if len(str(p.absolute())) <= MAX_PATH:
        return p
    digest = hashlib.sha1(url.encode()).hexdigest()[:20]
    return Path(cache_dir) / u.netloc / "_long" / f"{digest}{p.suffix}"


def fetch(url: str, cache_dir: Path = CACHE_DIR) -> Path:
    """Return a local copy of ``url``, downloading it once. Writes are atomic.

    Safe when parallel workers fetch the same URL: each download goes to its own temp file, and if
    another worker has already put the file in place (or holds it open, which blocks replacing on
    Windows), that copy is used."""
    target = cache_path(url, cache_dir)
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=target.parent, prefix=target.name + ".", suffix=".part")
    os.close(fd)
    part = Path(name)
    try:
        _download(url, part)
        try:
            os.replace(part, target)
        except PermissionError:
            if not target.exists():
                raise
    finally:
        part.unlink(missing_ok=True)
    return target


@contextmanager
def fetch_transient(url: str, cache_dir: Path = CACHE_DIR) -> Iterator[Path]:
    """Yield a local copy of ``url`` without keeping it (for large files used once, like H5J stacks):
    an already-cached copy is used as-is, otherwise the file goes to a private temp file deleted on exit."""
    cached = cache_path(url, cache_dir)
    if cached.exists():
        yield cached
        return
    tmp_dir = Path(cache_dir) / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=tmp_dir, suffix="-" + cached.name)
    os.close(fd)
    tmp = Path(name)
    try:
        _download(url, tmp)
        yield tmp
    finally:
        tmp.unlink(missing_ok=True)


def _download(url: str, dest: Path) -> None:
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
