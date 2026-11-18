"""Minimal client for NeuronBridge's public precomputed metadata (release v3_10_0).

URL scheme and file-key prefixes: see docs/spike-findings.md, "URLs and lookup".
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import requests

from .cache import fetch

VERSION = "v3_10_0"
META_URL = f"https://janelia-neuronbridge-data-prod.s3.us-east-1.amazonaws.com/{VERSION}/metadata"

LIBRARIES = {"manc": "FlyEM_MANC_v1.2.1", "banc": "FlyWire_BANC_v626"}

_COLOR_DEPTH = "https://s3.amazonaws.com/janelia-flylight-color-depth/"
_IMAGERY = "https://s3.amazonaws.com/janelia-flylight-imagery/"
PREFIXES = {
    "CDM": _COLOR_DEPTH,
    "CDMInput": _COLOR_DEPTH,
    "CDMMatch": _COLOR_DEPTH,
    "AlignedBodySWC": _COLOR_DEPTH,
    "AlignedBodyOBJ": _COLOR_DEPTH,
    "CDMThumbnail": "https://s3.amazonaws.com/janelia-flylight-color-depth-thumbnails/",
    "VisuallyLosslessStack": _IMAGERY,
    "Gal4Expression": _IMAGERY,
    "CDSResults": f"https://s3.amazonaws.com/janelia-neuronbridge-data-prod/{VERSION}/metadata/cdsresults/",
}


class NotFound(Exception):
    """A metadata URL does not exist."""


class BodyNotFound(LookupError):
    def __init__(self, dataset: str, body_id: int):
        self.dataset, self.body_id = dataset, body_id
        super().__init__(f"body {body_id} not found in NeuronBridge {LIBRARIES.get(dataset, dataset)}")


@dataclass
class EmImage:
    id: str
    dataset: str
    body_id: int
    neuron_type: str | None
    files: dict[str, str]
    neuron_instance: str | None = None
    annotations: list[str] = field(default_factory=list)
    anatomical_area: str = "VNC"  # BANC bodies also have a "Brain" image (JRC2018_Unisex_20x_HR)


@dataclass
class LmHit:
    image_id: str
    line: str
    library: str
    slide_code: str
    channel: int | None
    normalized_score: float
    matching_pixels: int
    mirrored: bool
    files: dict[str, str]
    objective: str | None = None
    match_files: dict[str, str] = field(default_factory=dict)
    anatomical_area: str = "VNC"  # "Brain" images are in JRC2018_Unisex_20x_HR, not the VNC template


def fetch_json(url: str) -> dict:
    """Fetch (cached) JSON; raises NotFound on HTTP 403/404 (S3 returns 403 for missing keys)."""
    try:
        return json.loads(fetch(url).read_text(encoding="utf-8"))
    except requests.HTTPError as e:
        code = getattr(e.response, "status_code", None)
        if code in (403, 404) or str(e).startswith(("403", "404")):
            raise NotFound(url) from e
        raise


def file_url(files: dict[str, str], key: str) -> str | None:
    path = files.get(key)
    if not path:
        return None
    return PREFIXES[key] + path


def get_em_images(dataset: str, body_id: int) -> list[EmImage]:
    lib = LIBRARIES[dataset]
    try:
        data = fetch_json(f"{META_URL}/by_body/{body_id}.json")
    except NotFound:
        raise BodyNotFound(dataset, body_id) from None
    out = [
        EmImage(
            id=im["id"],
            dataset=dataset,
            body_id=int(body_id),
            neuron_type=im.get("neuronType"),
            files=im["files"],
            neuron_instance=im.get("neuronInstance"),
            annotations=im.get("annotations") or [],
            anatomical_area=im.get("anatomicalArea", "VNC"),
        )
        for im in data.get("results", [])
        if im.get("libraryName") == lib
    ]
    if not out:
        raise BodyNotFound(dataset, body_id)
    return out


def get_cds_hits(em: EmImage) -> list[LmHit]:
    url = file_url(em.files, "CDSResults")
    if url is None:
        return []
    data = fetch_json(url)
    hits = []
    for r in data.get("results", []):
        im = r["image"]
        hits.append(LmHit(
            image_id=im["id"],
            line=im["publishedName"],
            library=im["libraryName"],
            slide_code=im.get("slideCode", ""),
            channel=im.get("channel"),
            normalized_score=float(r["normalizedScore"]),
            matching_pixels=int(r["matchingPixels"]),
            mirrored=bool(r["mirrored"]),
            files=im["files"],
            objective=im.get("objective"),
            match_files=r.get("files", {}),
            anatomical_area=im.get("anatomicalArea", "VNC"),
        ))
    return hits


def get_line_images(line: str) -> list[dict]:
    """Raw VNC LMImage dicts for a line (by_line index)."""
    try:
        data = fetch_json(f"{META_URL}/by_line/{line}.json")
    except NotFound:
        return []
    return [im for im in data.get("results", []) if im.get("anatomicalArea") == "VNC"]
