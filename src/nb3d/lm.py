"""LM point clouds in template microns: 3D from H5J stacks, 2.5D fallback from CDMs.

H5J decode route, channel mapping and geometry: docs/spike-findings.md ("3D LM stacks").
"""
from __future__ import annotations

import hashlib
import io
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import av
import h5py
import numpy as np
from PIL import Image

from .cache import CACHE_DIR, fetch
from .cdm25d import cdm_to_points
from .em import TEMPLATE_PX, VOXEL_UM
from .nbclient import LmHit, file_url

MAX_POINTS = 20000
# Derived 3D point clouds, so a cached stack is decoded once rather than once per query.
POINTS_DIR = CACHE_DIR / "points"
POINTS_VERSION = 1  # bump when stack_to_points / decode_h5j change behaviour


@dataclass
class LmPoints:
    xyz: np.ndarray
    source: Literal["3d", "2.5d"]


def decode_h5j(path: Path, channel: int) -> np.ndarray:
    """Decode 1-based ``channel`` (CDM_N suffix) of an H5J stack -> (z, y, x) uint16, z fixed to 219."""
    with h5py.File(path, "r") as f:
        grp = f["Channels"]
        data = grp[f"Channel_{channel - 1}"][()].tobytes()
        h, w = int(grp.attrs["height"][0]), int(grp.attrs["width"][0])
    with av.open(io.BytesIO(data)) as c:
        frames = [fr.to_ndarray(format="gray16le")[:h, :w] for fr in c.decode(video=0)]
    vol = np.stack(frames)
    nz = TEMPLATE_PX[2]
    if vol.shape[0] < nz:
        vol = np.concatenate([vol, np.zeros((nz - vol.shape[0], h, w), vol.dtype)])
    return vol[:nz]


def _otsu(values: np.ndarray, bins: int = 256) -> float:
    hist, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers) / np.maximum(w0, 1)
    m1 = ((hist * centers).sum() - np.cumsum(hist * centers)) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(centers[np.argmax(between)])


def stack_to_points(volume: np.ndarray, voxel_um: tuple[float, float, float],
                    threshold: float | None = None, max_points: int = MAX_POINTS) -> np.ndarray:
    """Voxels above threshold (Otsu by default) -> (N, 3) microns (x, y, z), subsampled deterministically."""
    if threshold is None:
        threshold = _otsu(volume.ravel()[::7])
    zs, ys, xs = np.nonzero(volume > threshold)
    pts = np.column_stack([xs * voxel_um[0], ys * voxel_um[1], zs * voxel_um[2]])
    if len(pts) > max_points:
        keep = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        pts = pts[np.sort(keep)]
    return pts


def _stack_points(stack_url: str, channel: int, threshold: float | None) -> np.ndarray:
    key = f"v{POINTS_VERSION}|{stack_url}|{channel}|{threshold}|{MAX_POINTS}"
    path = POINTS_DIR / f"{hashlib.sha1(key.encode()).hexdigest()}.npy"
    if path.exists():
        return np.load(path)
    pts = stack_to_points(decode_h5j(fetch(stack_url), channel), VOXEL_UM, threshold)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}-{threading.get_ident()}.part")
    with open(tmp, "wb") as f:
        np.save(f, pts)
    os.replace(tmp, path)
    return pts


def load_lm_points(hit: LmHit, prefer_3d: bool = True, threshold: float | None = None) -> LmPoints:
    stack_url = file_url(hit.files, "VisuallyLosslessStack")
    if prefer_3d and stack_url:
        return LmPoints(_stack_points(stack_url, hit.channel or 1, threshold), "3d")
    cdm_url = file_url(hit.files, "CDM") or file_url(hit.files, "Gal4Expression")
    rgb = np.asarray(Image.open(fetch(cdm_url)).convert("RGB"))
    pts = cdm_to_points(rgb, threshold=int(threshold) if threshold is not None else 20)
    if len(pts) > MAX_POINTS:
        pts = pts[np.sort(np.random.default_rng(0).choice(len(pts), MAX_POINTS, replace=False))]
    return LmPoints(pts, "2.5d")
