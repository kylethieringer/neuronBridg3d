"""Download-once disk cache for public HTTP resources."""
from __future__ import annotations

import os
from pathlib import Path
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
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(part, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        os.replace(part, target)
    finally:
        if part.exists():
            part.unlink()
    return target
