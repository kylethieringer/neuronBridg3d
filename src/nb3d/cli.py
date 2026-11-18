"""Command line: ``nb3d benchmark`` and ``nb3d serve``."""
from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="nb3d")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("benchmark", help="compare NeuronBridge and nb3d rankings on ground truth")
    b.add_argument("--dataset", default="manc", choices=["manc", "banc"])
    b.add_argument("--tier", type=int, choices=[1, 2, 3])
    b.add_argument("--limit", type=int)
    b.add_argument("--top-n", type=int, default=200)
    b.add_argument("--libraries", choices=["all", "split"], default="split")
    b.add_argument("--workers", type=int, help="parallel image workers (each holds a ~280 MB stack; default 8)")
    b.add_argument("--bodies", help="comma-separated body IDs: run only these ground-truth bodies")
    b.add_argument("--out", help="results CSV (default results/benchmark-<dataset>-tier<N>-<libraries>.csv); "
                   "bodies already in it are skipped")

    s = sub.add_parser("serve", help="start the local browser app")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--no-browser", action="store_true")

    a = p.parse_args(argv)
    if a.cmd == "benchmark":
        from .benchmark import run
        if a.workers:
            from . import rank
            rank.WORKERS = a.workers
        libs = run.ALL_LIBRARIES if a.libraries == "all" else run.SPLIT_LIBRARIES
        ids = [int(x) for x in a.bodies.split(",") if x.strip()] if a.bodies else None
        df = run.run_benchmark(a.dataset, a.tier, a.limit, a.top_n, libs, out=a.out, body_ids=ids)
        if a.limit:
            df = df.head(a.limit)
        print(run.report(df, a.top_n))
    elif a.cmd == "serve":
        from .server import serve
        serve(port=a.port, open_browser=not a.no_browser)


if __name__ == "__main__":
    main()
