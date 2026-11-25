# -*- coding: utf-8 -*-
# AI-assisted code: developed with Claude Code (Anthropic); reviewed and tested by the SOMAD authors.
"""Sensitivity of the Schall et al. (2024) benchmark to calls cut by the 15 s segment edges.

Re-scores the segment table of schall_benchmark.py (schall_segments.csv.gz: SOMAD's prediction per segment is
already in it; no scores/model needed) with the fraction of each partially included call that lies inside the
segment, computed from the annotation table with the same rules (site, Kerguelen 2015 downsweeps dropped).

Variants (all per site-year, mean +- SD over the 11 site-years, plus pooled):
  protocol            100% rule (as published): segment = call only if a call lies completely inside, else Noise
  drop_le50           segments whose only calls are <=50% inside are dropped (the edge cases); calls >50% but
                      <100% inside still make the segment Noise (as in the protocol)
  rule50              drop_le50 + a call >50% inside counts as inside (segment labelled with it), i.e. a 50%
                      inclusion rule instead of 100%; calls <=50% inside are ignored (segment dropped if they are
                      its only calls)
  partial_excluded    every Noise segment touching any part of a call is dropped (schall_benchmark.py's
                      'partial_noise_excluded')

    python schall_benchmark_partial.py --segments <bench>/schall_segments.csv.gz \
        --annotations bluefin_library_annotations.csv --out <bench>/partial50
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--segments", required=True)
ap.add_argument("--annotations", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--frac", type=float, default=0.5, help="inclusion fraction threshold (default 0.5)")
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

SEG = 15.0
CATS = ["ABZ", "20Hz20Plus", "DDswp"]
ALL = CATS + ["Noise"]
SITES = {"ElephantIsland2013Aural": "Elephant Island 2013", "ElephantIsland2014": "Elephant Island 2014",
         "kerguelen2005": "Kerguelen 2005", "kerguelen2014": "Kerguelen 2014", "kerguelen2015": "Kerguelen 2015",
         "BallenyIslands2015": "Balleny Islands 2015", "Greenwich64S2015": "Greenwich 2015",
         "MaudRise2014": "Maud Rise 2014", "RossSea2014": "Ross Sea 2014", "casey2014": "Casey 2014",
         "casey2017": "Casey 2017"}
ANN2CAT = {"Blue whale A call": "ABZ", "Blue whale B call": "ABZ", "Blue whale Z call": "ABZ",
           "Fin whale 20 Hz": "20Hz20Plus", "Blue whale D call": "DDswp", "Fin whale downsweep": "DDswp"}
rng = np.random.default_rng(0)


def site_of(name):
    return next((k for k in SITES if k in name), None)


ann = pd.read_csv(args.annotations)
ann["t0"] = pd.to_datetime(ann.start_datetime, utc=True)
ann["t1"] = pd.to_datetime(ann.end_datetime, utc=True)
ann["cat"] = ann.annotation.map(ANN2CAT)
ann = ann[ann.cat.notna()]
ann["site"] = ann.filename.map(site_of)
ann = ann[ann.site.notna()]
ann = ann[~((ann.site == "kerguelen2015") & (ann.annotation == "Fin whale downsweep"))]
ann_by_site = {s: g for s, g in ann.groupby("site")}

seg = pd.read_csv(args.segments)
print(len(seg), "segments")
max_in_frac = np.zeros(len(seg))          # largest inside-fraction among calls that are partial (0<f<1)
min_part_frac = np.ones(len(seg))         # smallest inside-fraction among partial calls (1 = none)
cats_gt = [""] * len(seg)                 # categories of calls with inside-fraction > frac (incl. complete)
has_le = np.zeros(len(seg), bool)         # any call with 0 < fraction <= frac
for stem, idx in seg.groupby("stem").indices.items():
    site = site_of(stem)
    t_file = pd.Timestamp(pd.to_datetime(stem[:19], format="%Y_%m_%d_%H_%M_%S"), tz="UTC")
    a = ann_by_site.get(site)
    t = seg.t.to_numpy()[idx]
    if a is None or a.empty:
        continue
    t_end = t.max() + SEG
    a = a[(a.t1 > t_file) & (a.t0 < t_file + pd.Timedelta(seconds=float(t_end)))]
    if a.empty:
        continue
    s = (a.t0 - t_file).dt.total_seconds().to_numpy()
    e = (a.t1 - t_file).dt.total_seconds().to_numpy()
    c = a.cat.to_numpy()
    dur = np.maximum(e - s, 1e-6)
    ov = np.clip(np.minimum(e[None, :], t[:, None] + SEG) - np.maximum(s[None, :], t[:, None]), 0, None)
    f = ov / dur[None, :]                                           # (segments, calls) fraction inside
    part = (f > 0) & (f < 1 - 1e-9)
    has_le[idx] = ((f > 0) & (f <= args.frac + 1e-9)).any(1)
    gt = f > args.frac + 1e-9
    for j, i in enumerate(idx):
        if gt[j].any():
            cats_gt[i] = "|".join(sorted(set(c[gt[j]])))
    fp = np.where(part, f, np.nan)
    with np.errstate(all="ignore"):
        max_in_frac[idx] = np.nan_to_num(np.nanmax(np.where(part, f, -1), 1).clip(0))
        min_part_frac[idx] = np.where(part.any(1), np.nanmin(np.where(part, f, 2), 1), 1)
seg["cats_gt"] = cats_gt
seg["has_le"] = has_le


def relabel_rule50(d):
    truth = []
    for cg, pred in zip(d.cats_gt, d.pred):
        if not cg:
            truth.append("Noise")
            continue
        cs = cg.split("|")
        truth.append(pred if pred in cs else cs[rng.integers(len(cs))])
    d = d.copy()
    d["truth"] = truth
    return d


variants = {
    "protocol": seg,
    "drop_le50": seg[~((seg.truth == "Noise") & seg.has_le & (seg.cats_gt == ""))],
    "rule50": relabel_rule50(seg[~((seg.cats_gt == "") & seg.has_le)]),
    "partial_excluded": seg[~seg.partial_only],
}


def metrics(d):
    cm = pd.crosstab(d.truth, d.pred).reindex(index=ALL, columns=ALL, fill_value=0)
    tot = cm.sum(axis=1)
    tcr = np.nanmean([cm.loc[k, k] / tot[k] if tot[k] else np.nan for k in CATS])
    nmr = cm.loc["Noise", CATS].sum() / tot["Noise"] if tot["Noise"] else np.nan
    cmr = np.nanmean([cm.loc[k, o] / tot[k] if tot[k] else np.nan for k in CATS for o in CATS if o != k])
    return dict(TCR=tcr, NMR=nmr, CMR=cmr, F=np.mean([tcr, 1 - nmr, 1 - nmr, 1 - cmr]), n_segments=len(d),
                **{f"n_{k}": int(tot[k]) for k in ALL}), cm


rows = []
for name, d0 in variants.items():
    for site, d in d0.groupby("site"):
        rows.append(dict(variant=name, site=site, **metrics(d)[0]))
    m, cm = metrics(d0)
    rows.append(dict(variant=name, site="pooled", **m))
    cm.to_csv(out / f"confusion_{name}.csv")
res = pd.DataFrame(rows)
res.to_csv(out / "metrics_per_site.csv", index=False)
ps = res[res.site != "pooled"]
summ = ps.groupby("variant", sort=False)[["TCR", "NMR", "CMR", "F"]].agg(["mean", "std"])
summ.to_csv(out / "metrics_summary.csv")
pd.set_option("display.width", 200)
print(summ.round(3).to_string())
print(res[res.site == "pooled"].round(3).to_string(index=False))
print("segments dropped vs protocol:", {k: len(seg) - len(v) for k, v in variants.items()})
r50 = variants["rule50"]
calls = r50[r50.cats_gt != ""]
print("rule50: call segments with several categories: %.1f%%" % (100 * calls.cats_gt.str.contains("|", regex=False).mean()))
r50[["site", "stem", "t", "truth", "pred", "cats_gt"]].to_csv(out / "segments_rule50.csv.gz", index=False)
