"""User verdicts on candidate lines, stored append-only in data/verdicts.csv (latest wins)."""
from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

import pandas as pd

VERDICTS = Path(__file__).resolve().parents[2] / "data" / "verdicts.csv"
FIELDS = ["timestamp", "dataset", "query", "line", "verdict", "note"]
ALLOWED = {"good", "bad", "unsure"}


def save_verdict(dataset: str, query: str, line: str, verdict: str, note: str = "") -> None:
    if verdict not in ALLOWED:
        raise ValueError(f"verdict must be one of {sorted(ALLOWED)}")
    VERDICTS.parent.mkdir(parents=True, exist_ok=True)
    new = not VERDICTS.exists()
    with open(VERDICTS, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({"timestamp": dt.datetime.now().isoformat(timespec="seconds"), "dataset": dataset,
                    "query": query, "line": line, "verdict": verdict, "note": note})


def load_verdicts() -> pd.DataFrame:
    if not VERDICTS.exists():
        return pd.DataFrame(columns=FIELDS)
    df = pd.read_csv(VERDICTS, dtype=str, keep_default_na=False)
    return df.drop_duplicates(["dataset", "query", "line"], keep="last").reset_index(drop=True)
