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

from .cache import CACHE_DIR, fetch, fetch_transient
from .cdm25d import cdm_to_points, crop_label
from .em import TEMPLATE_PX, VOXEL_UM
from .nbclient import LmHit, file_url

MAX_POINTS = 20000
# Derived 3D point clouds, so a stack is downloaded and decoded once rather than once per query.
# The stacks themselves are not kept (about 8 MB each; hundreds of GB over a benchmark).
POINTS_DIR = CACHE_DIR / "points"
POINTS_VERSION = 3  # bump when stack_to_points / decode_h5j change behaviour


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


SIGNAL_MAX_FRACTION = 0.02  # above this, a threshold is taken to include tissue background
SIGNAL_MAX_ROUNDS = 4
SIGNAL_MIN_FRACTION = 0.001  # floor: keep at least the brightest 0.1% of voxels
# CDMs are projections, so a neuron covers a larger share of pixels than of voxels. Chosen by agreement
# with the 3D points of the same image (300 images: F1 0.77 at the old fixed threshold 20 -> 0.98).
CDM_SIGNAL_MAX_FRACTION = 0.10


def signal_threshold(volume: np.ndarray, max_fraction: float = SIGNAL_MAX_FRACTION) -> float:
    """Otsu, re-applied to the voxels above the previous threshold while more than
    ``max_fraction`` of voxels remain above it.

    A single Otsu often separates empty space from tissue autofluorescence instead of
    tissue from labelled neurons (~20-30% of voxels above it). Iterating can overshoot onto the
    brightest structure alone (e.g. a motor axon in a nerve, dropping its dendrites), so at least
    SIGNAL_MIN_FRACTION of voxels is always kept. See docs/spike-findings.md."""
    s = volume.ravel()[::7]
    t = _otsu(s)
    for _ in range(SIGNAL_MAX_ROUNDS):
        above = s[s > t]
        if len(above) <= max_fraction * len(s) or above.min() == above.max():
            break
        t_next = _otsu(above)
        if not (above > t_next).any():
            break
        t = t_next
    return float(min(t, np.quantile(s, 1 - SIGNAL_MIN_FRACTION)))


def cdm_threshold(rgb: np.ndarray) -> float:
    """Brightness threshold for a CDM: ``signal_threshold`` on max(R, G, B), allowing more signal
    than in a stack. Replaces a fixed threshold of 20, which kept tissue background on dim images."""
    return signal_threshold(crop_label(np.asarray(rgb))[..., :3].max(axis=2), CDM_SIGNAL_MAX_FRACTION)


def stack_to_points(volume: np.ndarray, voxel_um: tuple[float, float, float],
                    threshold: float | None = None, max_points: int = MAX_POINTS) -> np.ndarray:
    """Voxels above threshold (``signal_threshold`` by default) -> (N, 3) microns (x, y, z),
    subsampled deterministically."""
    if threshold is None:
        threshold = signal_threshold(volume)
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
    with fetch_transient(stack_url) as stack:
        vol = decode_h5j(stack, channel)
    pts = stack_to_points(vol, VOXEL_UM, threshold)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}-{threading.get_ident()}.part")
    with open(tmp, "wb") as f:
        np.save(f, pts)
    os.replace(tmp, path)
    return pts


def load_lm_points(hit: LmHit, prefer_3d: bool = True, threshold: float | None = None) -> LmPoints:
    if hit.anatomical_area != "VNC":
        raise ValueError(f"image {hit.image_id} is a {hit.anatomical_area} image; only VNC-template images can be decoded")
    stack_url = file_url(hit.files, "VisuallyLosslessStack")
    if prefer_3d and stack_url:
        return LmPoints(_stack_points(stack_url, hit.channel or 1, threshold), "3d")
    cdm_url = file_url(hit.files, "CDM") or file_url(hit.files, "Gal4Expression")
    rgb = np.asarray(Image.open(fetch(cdm_url)).convert("RGB"))
    pts = cdm_to_points(rgb, threshold=int(threshold if threshold is not None else cdm_threshold(rgb)))
    if len(pts) > MAX_POINTS:
        pts = pts[np.sort(np.random.default_rng(0).choice(len(pts), MAX_POINTS, replace=False))]
    return LmPoints(pts, "2.5d")
