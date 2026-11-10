"""Turn user input (body IDs or a cell type) into a group of cells with NeuronBridge EM images.

Cell-type, side and segment come from public metadata tables (docs/spike-findings.md,
"Cell-type resolution"). ``load_meta`` returns a normalised DataFrame with columns
``body_id`` (the NeuronBridge ID; BANC = root_626), ``cell_type``, ``side`` ("L"/"R"/None),
``segment`` (BANC neuromere, MANC None) and, for BANC, ``manc_cell_type`` and ``alt_ids``
(other BANC root IDs that map to this root_626).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

import pandas as pd

from .cache import fetch
from .nbclient import BodyNotFound, EmImage, LIBRARIES, get_em_images

GCS = "https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data"
META_URLS = {
    "manc": f"{GCS}/manc_121/manc_121_meta.feather",
    "banc": f"{GCS}/banc_888/banc_888_meta.feather",
}
_SIDE = {"left": "L", "right": "R", "l": "L", "r": "R"}
_BANC_ALT_COLS = ["root_850", "root_888", "root_890", "root_id", "banc_888_id"]


class QueryError(ValueError):
    """User-facing problem with a query."""


@dataclass
class Cell:
    dataset: str
    body_id: int
    cell_type: str | None
    side: str | None
    segment: str | None
    em_images: list[EmImage]


@dataclass
class QueryGroup:
    dataset: str
    label: str
    cells: list[Cell]
    missing: list[int] = field(default_factory=list)


def _to_int(v) -> int | None:
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def _norm_side(v) -> str | None:
    return _SIDE.get(str(v).strip().lower()) if isinstance(v, str) else None


@lru_cache(maxsize=None)
def load_meta(dataset: str) -> pd.DataFrame:
    raw = pd.read_feather(fetch(META_URLS[dataset]))
    if dataset == "manc":
        df = pd.DataFrame({
            "body_id": raw["manc_121_id"].map(_to_int),
            "cell_type": raw["cell_type"],
            "side": raw["side"].map(_norm_side),
            "segment": None,
        })
    else:
        raw = raw[raw["root_626"].notna()]
        body = raw["root_626"].map(_to_int)
        alts = [
            sorted({a for a in (_to_int(r[c]) for c in _BANC_ALT_COLS) if a is not None and a != b})
            for b, (_, r) in zip(body, raw[_BANC_ALT_COLS].iterrows())
        ]
        df = pd.DataFrame({
            "body_id": body.values,
            "cell_type": raw["cell_type"].values,
            "manc_cell_type": raw["manc_cell_type"].values,
            "side": raw["side"].map(_norm_side).values,
            "segment": raw["neuromere"].where(raw["neuromere"].notna(), None).values,
            "alt_ids": alts,
        })
    return df[df["body_id"].notna()].reset_index(drop=True)


def _rows_for_ids(meta: pd.DataFrame, ids: list[int], dataset: str) -> pd.DataFrame:
    rows = []
    for i in ids:
        hit = meta[meta["body_id"] == i]
        if hit.empty and "alt_ids" in meta:
            hit = meta[meta["alt_ids"].map(lambda a: i in a)]
        if hit.empty:
            # Not annotated: still allow lookup in NeuronBridge without type/side.
            hit = pd.DataFrame({"body_id": [i], "cell_type": [None], "side": [None], "segment": [None]})
        rows.append(hit.iloc[[0]])
    return pd.concat(rows, ignore_index=True)


def _rows_for_type(meta: pd.DataFrame, cell_type: str, dataset: str) -> tuple[pd.DataFrame, str]:
    key = cell_type.strip().lower()
    for col in ("cell_type", "manc_cell_type"):
        if col not in meta:
            continue
        m = meta[meta[col].astype(str).str.lower() == key]
        if not m.empty:
            return m, str(m[col].iloc[0])
    raise QueryError(f"cell type '{cell_type}' not found in {dataset.upper()} annotations")


def resolve(dataset: str, query: str, side: str | None = None, segment: str | None = None) -> QueryGroup:
    dataset = dataset.lower()
    if dataset not in LIBRARIES:
        raise QueryError(f"unknown dataset '{dataset}'; use manc or banc")
    if segment and dataset == "manc":
        raise QueryError("segment filter is not available for MANC (no segment annotation); use side only")
    text = query.strip()
    if not text:
        raise QueryError("enter a body ID or a cell type")
    meta = load_meta(dataset)

    tokens = [t for t in re.split(r"[,\s]+", text) if t]
    if all(t.isdigit() for t in tokens):
        rows = _rows_for_ids(meta, [int(t) for t in tokens], dataset)
        label = ", ".join(tokens)
    else:
        rows, label = _rows_for_type(meta, text, dataset)

    if side:
        rows = rows[rows["side"] == side.upper()]
    if segment:
        rows = rows[rows["segment"].astype(str).str.upper() == segment.upper()]
    if rows.empty:
        filters = ", ".join(f for f in (f"side={side}" if side else "", f"segment={segment}" if segment else "") if f)
        raise QueryError(f"no cells of '{label}' match the filters ({filters})")

    cells, missing = [], []
    for r in rows.itertuples(index=False):
        bid = int(r.body_id)
        try:
            ims = get_em_images(dataset, bid)
        except BodyNotFound:
            missing.append(bid)
            continue
        cells.append(Cell(dataset, bid, r.cell_type if isinstance(r.cell_type, str) else None,
                          r.side, r.segment if isinstance(r.segment, str) else None, ims))
    if not cells:
        ids = ", ".join(str(m) for m in missing)
        raise QueryError(
            f"{ids}: not found in NeuronBridge {LIBRARIES[dataset]}. "
            + ("Use a v626 or newer BANC body ID, or a cell type." if dataset == "banc"
               else "Use a MANC v1.2.1 body ID or a cell type.")
        )
    return QueryGroup(dataset, label, cells, missing)
