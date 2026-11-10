"""Compare NeuronBridge's ranking with nb3d's on the ground-truth table."""
from __future__ import annotations

import statistics
from pathlib import Path

import pandas as pd

from ..query import QueryError, resolve
from ..rank import ALL_LIBRARIES, SPLIT_LIBRARIES, collect_candidates, rerank
from .groundtruth import OUT as GROUNDTRUTH

RESULTS = Path(__file__).resolve().parents[3] / "results"
FULL = 10**9
COLUMNS = ["dataset", "body_id", "true_lines", "nb_rank", "nb3d_rank", "source",
           "in_top_n", "n_candidates", "tier", "error"]
# Network/IO failures: skip the body without recording it, so a later resume retries it.
TRANSIENT = (OSError, ConnectionError, TimeoutError)


def rank_of_correct(ranked_lines: list[str], true_lines: set[str]) -> int | None:
    for i, line in enumerate(ranked_lines, 1):
        if line in true_lines:
            return i
    return None


def summarize(ranks: list[int | None]) -> dict:
    n = len(ranks)
    found = sorted(r for r in ranks if r is not None)

    def frac(k):
        return sum(1 for r in found if r <= k) / n if n else 0.0

    return {
        "n": n, "top1": frac(1), "top5": frac(5), "top20": frac(20),
        "median_rank": statistics.median(found) if found else None,
        "recall_ceiling": len(found) / n if n else 0.0,
    }


def evaluate_body(dataset: str, body_id: int, true_lines: set[str], top_n: int = 200,
                  libraries: list[str] | None = None) -> dict:
    group = resolve(dataset, str(body_id))
    libs = libraries or SPLIT_LIBRARIES
    full = collect_candidates(group, top_n=FULL, libraries=libs)
    cands = full[:top_n]
    ranked = rerank(group, cands)
    new_rank = rank_of_correct([r.line for r in ranked], true_lines)
    best_true = next((r for r in ranked if r.line in true_lines), None)
    return {
        "dataset": dataset, "body_id": body_id, "true_lines": ",".join(sorted(true_lines)),
        "nb_rank": rank_of_correct([c.line for c in full], true_lines),
        "nb3d_rank": new_rank,
        "source": best_true.source if best_true else None,
        "in_top_n": rank_of_correct([c.line for c in cands], true_lines) is not None,
        "n_candidates": len(cands),
    }


def _libs_tag(libraries: list[str]) -> str:
    if list(libraries) == SPLIT_LIBRARIES:
        return "split"
    if list(libraries) == ALL_LIBRARIES:
        return "all"
    return "custom"


def run_benchmark(dataset: str = "manc", tier: int | None = None, limit: int | None = None,
                  top_n: int = 200, libraries: list[str] | None = None, out: Path | None = None) -> pd.DataFrame:
    gt = pd.read_csv(GROUNDTRUTH)
    gt = gt[gt.dataset == dataset]
    if tier is not None:
        gt = gt[gt.tier == tier]
    bodies = gt.groupby("body_id").agg(lines=("line", set), tier=("tier", "min"))
    if limit:
        bodies = bodies.head(limit)
    libraries = libraries or SPLIT_LIBRARIES
    out = Path(out) if out else RESULTS / f"benchmark-{dataset}-tier{tier or 'all'}-{_libs_tag(libraries)}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = pd.read_csv(out) if out.exists() else pd.DataFrame()
    done_ids = set(done.body_id) if len(done) else set()
    for i, (body, rec) in enumerate(bodies.iterrows(), 1):
        if body in done_ids:
            continue
        try:
            row = evaluate_body(dataset, int(body), rec["lines"], top_n, libraries)
        except TRANSIENT as e:
            print(f"[{i}/{len(bodies)}] {body}: skipped ({type(e).__name__}: {e}); rerun to retry", flush=True)
            continue
        except (QueryError, LookupError) as e:
            row = {"dataset": dataset, "body_id": int(body), "true_lines": ",".join(sorted(rec["lines"])),
                   "nb_rank": None, "nb3d_rank": None, "source": None, "in_top_n": False,
                   "n_candidates": 0, "error": str(e)}
        row["tier"] = int(rec["tier"])
        pd.DataFrame([row]).reindex(columns=COLUMNS).to_csv(out, mode="a", header=not out.exists(), index=False)
        print(f"[{i}/{len(bodies)}] {body}: NB {row['nb_rank']} -> nb3d {row['nb3d_rank']}", flush=True)
    return pd.read_csv(out)


def report(df: pd.DataFrame, top_n: int = 200) -> str:
    def r(col, sub):
        return [None if pd.isna(x) else int(x) for x in sub[col]]

    df = df[df.get("error").isna()] if "error" in df else df
    lines = [f"{'tier':>4} {'method':<14} {'n':>4} {'top1':>6} {'top5':>6} {'top20':>6} {'median':>7} {'recall':>7}"]
    for tier, sub in [("all", df)] + [(str(t), d) for t, d in df.groupby("tier")]:
        nb = r("nb_rank", sub)
        rows = [
            ("NeuronBridge", nb),
            (f"NB@{top_n}", [x if x is not None and x <= top_n else None for x in nb]),
            ("nb3d", r("nb3d_rank", sub)),
            ("nb3d 3D", r("nb3d_rank", sub[sub.source == "3d"])),
            ("nb3d 2.5D", r("nb3d_rank", sub[sub.source == "2.5d"])),
        ]
        for name, ranks in rows:
            s = summarize(ranks)
            med = "-" if s["median_rank"] is None else f"{s['median_rank']:.0f}"
            lines.append(f"{tier:>4} {name:<14} {s['n']:>4} {s['top1']:>6.2f} {s['top5']:>6.2f} "
                         f"{s['top20']:>6.2f} {med:>7} {s['recall_ceiling']:>7.2f}")
    return "\n".join(lines)


__all__ = ["rank_of_correct", "summarize", "evaluate_body", "run_benchmark", "report",
           "SPLIT_LIBRARIES", "ALL_LIBRARIES"]
