"""Build the de-duplicated (EM body <-> driver line) ground-truth table with evidence tiers.

Tier 1 = independent evidence, 2 = partly circular (NeuronBridge used to find candidates),
3 = CDM-derived (same algorithm as NeuronBridge). See the design doc, "Benchmark tiers".
BANC rows are derived from MANC tier-1/2 types through banc_888_meta.manc_cell_type and are
exploratory only.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import openpyxl
import pandas as pd

SRC = Path(__file__).resolve().parents[3] / "research" / "groundtruth-sources"
OUT = Path(__file__).resolve().parents[3] / "data" / "groundtruth.csv"
COLUMNS = ["dataset", "body_id", "cell_type", "line", "tier", "source", "confidence"]


def normalise_line(raw: str | None) -> str | None:
    """'SS01557-20160318_..' -> 'SS01557'; bare Janelia GMR plate '14E09' -> 'R14E09'."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    s = re.split(r"[-\s]", s)[0]
    if re.fullmatch(r"\d{1,3}[A-H]\d{2}", s):
        s = "R" + s
    return s


def _split_lines(text: str) -> list[str]:
    return [l for l in (normalise_line(t) for t in re.split(r"[,;/]", text or "")) if l]


def _ids(text) -> list[int]:
    return [int(x) for x in re.findall(r"\d+", str(text or ""))]


def _rows(dataset, body_id, cell_type, lines, tier, source, confidence):
    return [dict(dataset=dataset, body_id=int(body_id), cell_type=cell_type, line=l, tier=tier,
                 source=source, confidence=str(confidence)) for l in lines]


def _cheong(src: Path) -> list[dict]:
    out = []
    for r in csv.DictReader(open(src / "cheong2024/supp3.csv", encoding="latin-1")):
        if r["line_match"].strip() and r["bodyid"].strip().isdigit():
            out += _rows("manc", r["bodyid"], r["type"], _split_lines(r["line_match"]), 1,
                         "cheong2024_supp3", r["match_certainty(1-5)"])
    for r in csv.DictReader(open(src / "cheong2024/supp1.csv", encoding="latin-1")):
        conf = r["confidence_1_5"].strip()
        if r["LM_match_slide_code"].strip() and conf.isdigit() and int(conf) >= 4 and r["bodyid"].strip().isdigit():
            out += _rows("manc", r["bodyid"], r["type"], _split_lines(r["LM_match_slide_code"]), 2,
                         "cheong2024_supp1", conf)
    return out


def _manc_dns(src: Path) -> dict[str, list[tuple[int, str]]]:
    """DN name (type, systematic type or synonym) -> [(MANC body id, MANC type)] from Stürner SF7."""
    idx: dict[str, list[tuple[int, str]]] = {}
    for r in csv.DictReader(open(src / "sturner2025/Supplemental_file7_MANC_DNs.tsv", encoding="utf-8"), delimiter="\t"):
        if not r["bodyid"].strip().isdigit():
            continue
        names = {r["type"], r["systematic_type"], *re.split(r"[,;]\s*", r["synonyms"] or "")}
        for n in {n.strip() for n in names if n and n.strip()}:
            idx.setdefault(n, []).append((int(r["bodyid"]), r["type"]))
    return idx


def _sturner_zung(src: Path) -> list[dict]:
    dns = _manc_dns(src)
    out = []
    for r in csv.DictReader(open(src / "sturner2025/Supplemental_file4_DN_identification.tsv", encoding="utf-8"), delimiter="\t"):
        lines = _split_lines(r["LM_stack_id"])
        for body, mtype in dns.get(r["dn_type"].strip(), []):
            out += _rows("manc", body, mtype, lines, 2, "sturner2025", "")
    for r in csv.DictReader(open(src / "zung2025/zung_file02.tsv", encoding="utf-8"), delimiter="\t"):
        lines = _split_lines(r["A"])
        for body, mtype in dns.get(r["Type"].strip(), []):
            out += _rows("manc", body, mtype, lines, 1, "zung2025_A", "A")
    return out


def _ehrhardt(src: Path) -> list[dict]:
    ws = openpyxl.load_workbook(src / "ehrhardt2025/table4.xlsx", read_only=True).worksheets[0]
    out = []
    for r in list(ws.iter_rows(values_only=True))[1:]:
        if not r[0] or r[4] != 3:
            continue
        lines = _split_lines(str(r[5] or ""))
        for body in sorted(set(_ids(r[1]) + _ids(r[2]))):
            out += _rows("manc", body, r[3], lines, 3, "ehrhardt2025", 3)
    return out


def _banc(src: Path, manc: pd.DataFrame) -> list[dict]:
    meta = pd.read_feather(src / "banc/banc_888_meta.feather", columns=["root_626", "manc_cell_type"])
    meta = meta[meta.root_626.notna() & meta.manc_cell_type.notna()]
    lines_by_type = (manc[manc.tier <= 2].groupby("cell_type")
                     .apply(lambda d: d.sort_values("tier").drop_duplicates("line")[["line", "tier", "source"]]
                            .to_dict("records"), include_groups=False))
    out = []
    for root, mtype in meta.itertuples(index=False):
        for rec in lines_by_type.get(mtype, []):
            out.append(dict(dataset="banc", body_id=int(root), cell_type=mtype, line=rec["line"],
                            tier=rec["tier"], source=f"{rec['source']}+banc_manc_cell_type", confidence=""))
    return out


def dedupe(rows: pd.DataFrame) -> pd.DataFrame:
    return (rows.sort_values(["tier", "source"])
            .drop_duplicates(["dataset", "body_id", "line"], keep="first")
            .reset_index(drop=True))


def build_groundtruth(src: Path = SRC) -> pd.DataFrame:
    manc = pd.DataFrame(_cheong(src) + _sturner_zung(src) + _ehrhardt(src), columns=COLUMNS)
    manc = dedupe(manc)
    banc = pd.DataFrame(_banc(src, manc), columns=COLUMNS)
    out = dedupe(pd.concat([manc, banc], ignore_index=True))
    out["body_id"] = out["body_id"].astype("int64")
    out["tier"] = out["tier"].astype("int64")
    return out.sort_values(["dataset", "tier", "body_id", "line"]).reset_index(drop=True)[COLUMNS]


if __name__ == "__main__":
    t = build_groundtruth()
    OUT.parent.mkdir(exist_ok=True)
    t.to_csv(OUT, index=False)
    print(t.groupby(["dataset", "tier"]).agg(pairs=("line", "size"), bodies=("body_id", "nunique"),
                                              lines=("line", "nunique")))
