"""Is the 2.5D (colour-depth) shape score on the same scale as the 3D (stack) score?

For a seeded sample of benchmark bodies, every candidate image that HAS a 3D stack is scored twice
against the body: from its stack (3D) and from its CDM (2.5D, as if no stack existed). Images without
a stack are scored 2.5D only, as in the app. One row per (body, image) -> results/bias-25d-vs-3d.csv.

    uv run python research/bias_25d_vs_3d.py --n 40         # resumable
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from nb3d.benchmark.run import RESULTS
from nb3d.em import load_em_points
from nb3d.lm import load_lm_points
from nb3d.query import resolve
from nb3d.rank import collect_candidates
from nb3d.score import shape_score

OUT = RESULTS / "bias-25d-vs-3d.csv"
COLS = ["body_id", "tier", "gt_source", "line", "nb_rank", "image_id", "is_true", "has_stack"] + [
    f"{t}_{m}" for t in ("3d", "25d") for m in ("value", "fwd", "rev", "npts")]


def sample_bodies(n: int) -> pd.DataFrame:
    """Half with a 3D-scored correct line, half 2.5D, from MANC tiers 1-2 (correct line in the top 200)."""
    df = pd.concat([pd.read_csv(RESULTS / f"benchmark-manc-tier{t}-split.csv") for t in (1, 2)])
    df = df[df.source.notna()]
    parts = [g.sample(min(len(g), n // 2), random_state=0) for _, g in df.groupby("source")]
    return pd.concat(parts)[["body_id", "true_lines", "source", "nb3d_rank", "tier"]]


def score_body(body_id: int, true_lines: set[str]) -> list[dict]:
    group = resolve("manc", str(body_id))
    em = {c.body_id: load_em_points(c) for c in group.cells}
    cands = collect_candidates(group, top_n=200)

    def one(args):
        cand, h = args
        has_stack = "VisuallyLosslessStack" in h.files
        row = {"body_id": body_id, "line": cand.line, "nb_rank": cand.nb_rank, "image_id": h.image_id,
               "is_true": cand.line in true_lines, "has_stack": has_stack}
        kinds = [("3d", True), ("25d", False)] if has_stack else [("25d", False)]
        for tag, prefer in kinds:
            lm = load_lm_points(h, prefer_3d=prefer)
            s = max((shape_score(xyz, lm) for xyz in em.values()), key=lambda s: s.value)
            row |= {f"{tag}_value": s.value, f"{tag}_fwd": s.forward, f"{tag}_rev": s.reverse,
                    f"{tag}_npts": len(lm.xyz)}
        return row

    jobs = [(c, h) for c in cands for h in c.hits]
    with ThreadPoolExecutor(8) as pool:
        return list(pool.map(one, jobs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()
    bodies = sample_bodies(args.n)
    done = set(pd.read_csv(OUT).body_id) if OUT.exists() else set()
    for i, rec in enumerate(bodies.itertuples(), 1):
        if rec.body_id in done:
            continue
        try:
            rows = score_body(int(rec.body_id), set(rec.true_lines.split(",")))
        except Exception as e:  # noqa: BLE001 - keep going; a resume retries the body
            print(f"[{i}/{len(bodies)}] {rec.body_id}: skipped ({type(e).__name__}: {e})", flush=True)
            continue
        out = pd.DataFrame(rows).assign(gt_source=rec.source, tier=rec.tier).reindex(columns=COLS)
        out.to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
        print(f"[{i}/{len(bodies)}] {rec.body_id}: {len(rows)} images", flush=True)


if __name__ == "__main__":
    main()
