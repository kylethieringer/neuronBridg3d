"""Local FastAPI app: search (as a polled background job), geometry for the 3D viewer, verdicts."""
from __future__ import annotations

import threading
import webbrowser
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import verdicts
from .em import TEMPLATE_UM, load_em_points
from .lm import load_lm_points
from .nbclient import LmHit, file_url
from .query import QueryError, resolve
from .rank import ALL_LIBRARIES, SPLIT_LIBRARIES, collect_candidates, find_full_expression_image, rerank

WEB = Path(__file__).parent / "web"
MAX_POINTS = 20000

app = FastAPI(title="neuronBridg3d")
JOBS: dict[tuple, dict] = {}
GROUPS: dict[tuple, object] = {}
HITS: dict[str, LmHit] = {}
_LOCK = threading.Lock()


def _libraries(spec: str | None) -> list[str]:
    if not spec or spec == "split":
        return SPLIT_LIBRARIES
    if spec == "all":
        return ALL_LIBRARIES
    return [s for s in spec.split(",") if s]


def _subsample(xyz: np.ndarray) -> list[list[float]]:
    if len(xyz) > MAX_POINTS:
        xyz = xyz[np.sort(np.random.default_rng(0).choice(len(xyz), MAX_POINTS, replace=False))]
    return np.round(np.asarray(xyz, dtype=float), 2).tolist()


def _result_dict(r, dataset: str, query: str, verdict_map: dict) -> dict:
    best = next((h for h in r.hits if h.image_id == r.best_image_id), r.hits[0] if r.hits else None)
    return {
        "line": r.line, "rank": r.rank, "score": r.score, "shape": r.shape, "source": r.source,
        "mirrored": r.mirrored, "best_cell": str(r.best_cell), "best_image_id": r.best_image_id,
        "per_cell": {str(k): v for k, v in r.per_cell.items()}, "penalty": r.penalty,
        "nb_rank": r.nb_rank, "nb_score": r.nb_score, "library": r.library,
        "images": [{"image_id": h.image_id, "library": h.library, "slide_code": h.slide_code,
                    "channel": h.channel, "nb_score": h.normalized_score} for h in r.hits],
        "thumbnail_url": file_url(best.files, "CDMThumbnail") if best else None,
        "links": {"flylight": f"https://splitgal4.janelia.org/cgi-bin/view_splitgal4_imagery.cgi?line={r.line}",
                  "neuronbridge": f"https://neuronbridge.janelia.org/search?q={r.line}"},
        "verdict": verdict_map.get((dataset, query, r.line)),
    }


def _run_job(key: tuple, group, top_n: int, libs: list[str], penalty: bool, weight: float) -> None:
    job = JOBS[key]
    try:
        cands = collect_candidates(group, top_n=top_n, libraries=libs)
        for c in cands:
            for h in c.hits:
                HITS[h.image_id] = h
        job["total"] = len(cands)

        def progress(done, total):
            job["done"], job["total"] = done, total

        job["ranked"] = rerank(group, cands, penalty=penalty, weight=weight, progress=progress)
        job["status"] = "done"
    except Exception as e:  # surfaced to the UI
        job["status"], job["error"] = "error", f"{type(e).__name__}: {e}"


@app.get("/api/search")
def search(dataset: str, query: str = "", side: str | None = None, segment: str | None = None,
           penalty: bool = False, weight: float = 0.5, top_n: int = 200, libraries: str | None = None):
    side, segment = side or None, segment or None
    gkey = (dataset, query.strip(), side, segment)
    try:
        group = GROUPS.get(gkey) or resolve(dataset, query, side=side, segment=segment)
    except QueryError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    GROUPS[gkey] = group
    libs = _libraries(libraries)
    key = gkey + (penalty, round(weight, 3), top_n, tuple(libs))
    with _LOCK:
        if key not in JOBS:
            JOBS[key] = {"status": "running", "done": 0, "total": 0}
            threading.Thread(target=_run_job, args=(key, group, top_n, libs, penalty, weight), daemon=True).start()
    job = JOBS[key]
    out = {
        "status": job["status"], "done": job["done"], "total": job["total"],
        "group": {"label": group.label, "missing": [str(m) for m in group.missing],
                  "cells": [{"body_id": str(c.body_id), "side": c.side, "segment": c.segment,
                             "cell_type": c.cell_type} for c in group.cells]},
    }
    if job["status"] == "error":
        return JSONResponse({**out, "error": job["error"]}, status_code=500)
    if job["status"] == "done":
        vd = verdicts.load_verdicts()
        vmap = {(d, q, l): v for d, q, l, v in zip(vd.dataset, vd["query"], vd.line, vd.verdict)}
        out["results"] = [_result_dict(r, dataset, query.strip(), vmap) for r in job["ranked"]]
    return out


@app.get("/api/geometry")
def geometry(dataset: str, body_ids: str, image_id: str, threshold: float | None = None,
             expression_line: str | None = None):
    hit = HITS.get(image_id)
    if hit is None:
        return JSONResponse({"error": f"unknown image {image_id}; run a search first"}, status_code=404)
    ids = [int(b) for b in body_ids.split(",") if b.strip()]
    cells = {c.body_id: c for g in GROUPS.values() if g.dataset == dataset for c in g.cells}
    em = {str(b): _subsample(load_em_points(cells[b])) for b in ids if b in cells}
    lm = load_lm_points(hit, threshold=threshold)
    x, y, z = TEMPLATE_UM
    out = {"em": em, "lm": _subsample(lm.xyz), "lm_source": lm.source,
           "template_outline": [[a, b, c] for a in (0, x) for b in (0, y) for c in (0, z)]}
    if expression_line:
        expr = find_full_expression_image(expression_line, dataset)
        if expr is not None:
            e = load_lm_points(expr)
            out["expression"], out["expression_source"] = _subsample(e.xyz), e.source
    return out


class Verdict(BaseModel):
    dataset: str
    query: str
    line: str
    verdict: str
    note: str = ""


@app.post("/api/verdict")
def post_verdict(v: Verdict):
    try:
        verdicts.save_verdict(v.dataset, v.query.strip(), v.line, v.verdict, v.note)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return {"ok": True}


if WEB.exists():
    app.mount("/", StaticFiles(directory=WEB, html=True), name="web")


def serve(port: int = 8765, open_browser: bool = True) -> None:
    import uvicorn
    url = f"http://127.0.0.1:{port}"
    if open_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    print(f"neuronBridg3d running at {url}  (Ctrl+C to stop)")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
