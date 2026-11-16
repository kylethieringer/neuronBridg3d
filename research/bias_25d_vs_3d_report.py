"""Analyse results/bias-25d-vs-3d.csv (from bias_25d_vs_3d.py).

1. Paired comparison on images with both a stack and a CDM: 2.5D minus 3D, for the score and its
   forward / reverse halves, for correct-line vs other images and for the high-scoring images that decide ranks.
2. Rank of the correct line under alternative scoring rules (line score = best image, ties by NB rank):
   current   3D where a stack exists, else 2.5D (the app today)
   all_25d   2.5D for every image (one scale for all)
   max_both  best of 3D and 2.5D for stack images
   calib     current, with 2.5D-only scores mapped onto the 3D scale by a linear fit on paired images
             (fit leaving the body out)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from nb3d.benchmark.run import RESULTS, summarize

df = pd.read_csv(RESULTS / "bias-25d-vs-3d.csv")
print(f"{df.body_id.nunique()} bodies, {len(df)} images, {df.has_stack.mean():.0%} with a 3D stack\n")

# 1. paired differences ------------------------------------------------------------------------------
p = df[df.has_stack].copy()
p["hi"] = p.groupby("body_id")["3d_value"].rank(pct=True) >= 0.9  # top 10% of images per body
print("2.5D - 3D on the same image (median [IQR]; Wilcoxon p)")
for name, sub in [("all", p), ("top-10% images", p[p.hi]), ("correct line", p[p.is_true]),
                  ("other lines", p[~p.is_true])]:
    cells = []
    for m in ("value", "fwd", "rev"):
        d = sub[f"25d_{m}"] - sub[f"3d_{m}"]
        q1, q2, q3 = d.quantile([.25, .5, .75])
        cells.append(f"{m} {q2:+.3f} [{q1:+.3f},{q3:+.3f}] p={wilcoxon(d).pvalue:.0e}")
    print(f"  {name:<15} n={len(sub):>5}  " + "  ".join(cells))
print(f"  corr(3D, 2.5D score) = {p['3d_value'].corr(p['25d_value']):.3f}")
print(f"  points 3D median {p['3d_npts'].median():.0f}, 2.5D median {p['25d_npts'].median():.0f}\n")

# 2.5D-only images (no stack) vs 2.5D-of-stack images: are the populations comparable?
print("Score distribution by what the app uses (median, 90th pct):")
for name, s in [("3D (stack images)", p["3d_value"]), ("2.5D of stack images", p["25d_value"]),
                ("2.5D-only images", df.loc[~df.has_stack, "25d_value"])]:
    print(f"  {name:<22} {s.median():.3f}  {s.quantile(.9):.3f}")
print()

# 2. counterfactual ranks ----------------------------------------------------------------------------
def calibrated(d: pd.DataFrame) -> pd.Series:
    out = d["25d_value"].copy()
    for b in d.body_id.unique():
        train = p[p.body_id != b]
        slope, icpt = np.polyfit(train["25d_value"], train["3d_value"], 1)
        m = (d.body_id == b) & ~d.has_stack
        out[m] = icpt + slope * d.loc[m, "25d_value"]
    return out


rules = {
    "current": df["3d_value"].where(df.has_stack, df["25d_value"]),
    "all_25d": df["25d_value"],
    "max_both": df[["3d_value", "25d_value"]].max(axis=1),
    "calib": df["3d_value"].where(df.has_stack, calibrated(df)),
}
ranks = {}
for name, score in rules.items():
    lines = df.assign(s=score).groupby(["body_id", "line"]).agg(
        s=("s", "max"), nb=("nb_rank", "first"), true=("is_true", "first")).reset_index()
    lines = lines.sort_values(["body_id", "s", "nb"], ascending=[True, False, True])
    lines["rank"] = lines.groupby("body_id").cumcount() + 1
    ranks[name] = lines[lines.true].groupby("body_id")["rank"].min()
ranks = pd.DataFrame(ranks).join(df.groupby("body_id")[["gt_source", "tier"]].first())

print(f"Rank of the correct line ({len(ranks)} bodies; gt_source = how the app scores the correct line)")
print(f"{'group':<10} {'rule':<9} {'n':>3} {'top1':>5} {'top5':>5} {'top20':>5} {'median':>6}  vs current: better/worse")
for g, sub in [("all", ranks)] + list(ranks.groupby("gt_source")):
    for rule in rules:
        s = summarize(list(sub[rule]))
        bw = "" if rule == "current" else f"{(sub[rule] < sub.current).sum()}/{(sub[rule] > sub.current).sum()}"
        print(f"{g:<10} {rule:<9} {s['n']:>3} {s['top1']:>5.2f} {s['top5']:>5.2f} {s['top20']:>5.2f} "
              f"{s['median_rank']:>6.1f}  {bw}")
ranks.to_csv(RESULTS / "bias-25d-vs-3d-ranks.csv")
