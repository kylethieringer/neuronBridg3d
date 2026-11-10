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
    b.add_argument("--libraries", choices=["all", "split"], default="all")

    s = sub.add_parser("serve", help="start the local browser app")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--no-browser", action="store_true")

    a = p.parse_args(argv)
    if a.cmd == "benchmark":
        from .benchmark import run
        libs = run.ALL_LIBRARIES if a.libraries == "all" else run.SPLIT_LIBRARIES
        df = run.run_benchmark(a.dataset, a.tier, a.limit, a.top_n, libs)
        if a.limit:
            df = df.head(a.limit)
        print(run.report(df, a.top_n))
    elif a.cmd == "serve":
        from .server import serve
        serve(port=a.port, open_browser=not a.no_browser)


if __name__ == "__main__":
    main()
