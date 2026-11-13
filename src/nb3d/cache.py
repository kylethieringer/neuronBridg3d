"""Download-once disk cache for public HTTP resources."""
from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from urllib.parse import unquote, urlparse

import requests

CACHE_DIR = Path(os.environ.get("NB3D_CACHE", Path(__file__).resolve().parents[2] / "cache"))


def cache_path(url: str, cache_dir: Path = CACHE_DIR) -> Path:
    """Local path for a URL: <cache_dir>/<host>/<url path>."""
    u = urlparse(url)
    rel = unquote(u.path).lstrip("/")
    return Path(cache_dir) / u.netloc / rel


def fetch(url: str, cache_dir: Path = CACHE_DIR) -> Path:
    """Return a local copy of ``url``, downloading it once. Writes are atomic."""
    target = cache_path(url, cache_dir)
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    try:
        _download(url, part)
        os.replace(part, target)
    finally:
        if part.exists():
            part.unlink()
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
