# -*- coding: utf-8 -*-
# AI-assisted code: developed with Claude Code (Anthropic); reviewed and tested by the SOMAD authors.
"""Schall et al. (2024) baleen-whale benchmark metrics for SOMAD scores (no model / audio needed).

    # validation split only (scores from predict_eval.py):
    python schall_benchmark.py --scores <eval>/scores_cqt.npz --thresholds <eval>/thresholds_cqt_continuous.csv \
        --annotations bluefin_library_annotations.csv --out <dir>
    # all BlueFinLibrary site-years (scores from run_inference.py on the library, one or more .nc files), with the
    # training split's time ranges removed:
    python schall_benchmark.py --scores scores_*.nc --thresholds ... --annotations ... --out <dir> \
        --exclude-files <split>/train_files.txt --exclude-audio-dir <training_data>/audio

Protocol (Schall et al. 2024, Remote Sens. Ecol. Conserv. 10:642-654, four-category case):
* categories ABZ (blue whale A, B, Z), 20Hz20Plus (fin whale 20 Hz pulse incl. overtone), DDswp (blue whale D call +
  fin whale 40 Hz downsweep), Noise;
* each recording is cut into 15 s segments with 12.5 s overlap (2.5 s hop);
* a segment gets a call label when an annotation lies completely inside it ("overlapped to 100%"), otherwise Noise;
  a segment containing several categories gets one of them (Schall et al.: random choice) -- here the one SOMAD
  predicted if it is among them (so a correct call is never counted as wrong), otherwise a seeded random one;
* metrics from the 4x4 confusion matrix (rows true, columns predicted):
    TCR = mean_k TP_k / total_k                         (k over the three call categories)
    NMR = (noise segments predicted as any call) / noise total
    CMR = mean over the six call->other-call cells of FPc / total_(true call category)
    F   = mean(TCR, 1 - NMR, 1 - NMR, 1 - CMR)          (noise false positives weighted twice)
  computed per site-year and reported as mean and SD over site-years (as Schall et al. over their 11 folds), plus
  pooled over all segments.
SOMAD's segment prediction: per category, the maximum logit over the segment's 0.5 s bins (DDswp: D call or
downsweep output, each against its own calibrated threshold; ABZ: the pooled ABZ output); the predicted category is
the one with the highest above-threshold logit, Noise if none is above threshold.
--exclude-files: segments overlapping (in absolute time, same site-year) any file of that list -- e.g. the training
split, whose BlueFin files are 5-min pieces of the library's hour files -- are dropped. Site-years with no excluded
file are "unseen" (never used in training: the analogue of Schall et al.'s blocked test).
At Kerguelen 2015 the downsweep table duplicates Kerguelen 2014's (removed from the annotations, class masked in
training), so DDswp there is D call only (truth and prediction).
The "partial_noise_excluded" variant drops Noise segments that contain only part of an annotation, which the 100%
rule otherwise counts as Noise.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--scores", nargs="+", required=True, help="scores_cqt.npz (predict_eval.py) or .nc (run_inference.py)")
ap.add_argument("--thresholds", required=True, help="thresholds_cqt_continuous.csv")
ap.add_argument("--annotations", required=True, help="bluefin_library_annotations.csv (absolute datetimes)")
ap.add_argument("--out", required=True)
ap.add_argument("--exclude-files", default=None, help="file-stem list whose time ranges are excluded (train split)")
ap.add_argument("--exclude-audio-dir", default=None,
                help="where the --exclude-files audio lives, to read their durations (default: 300 s each)")
ap.add_argument("--default-exclude-dur", type=float, default=300.0)
ap.add_argument("--tolerance", type=float, default=0.0,
                help="boundary tolerance (s), sensitivity analysis only: a call may stick out of a segment by up to "
                     "this much and still count as inside it (loose/offset annotation boxes), and SOMAD's prediction "
                     "uses only the bins at least this far from the segment edges (score spill-over). 0 = protocol")
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

SEG, HOP, LI = 15.0, 2.5, 0.5
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


def t_of(stem):
    return pd.Timestamp(pd.to_datetime(stem[:19], format="%Y_%m_%d_%H_%M_%S"), tz="UTC")


def iter_scores(paths):
    """yield (stem, scores (C, n_bins) with NaN/-1e4 where not covered, duration, labels)"""
    for path in paths:
        if path.endswith(".npz"):
            z = dict(np.load(path, allow_pickle=False))
            labels = json.loads(str(z["meta"]))["label_list"]
            off = 0
            for stem, n, dur in zip(z["stems"], z["n_bins"], z["durations"]):
                s = z["scores"][:, off:off + n].astype(np.float32).copy()
                s[:, ~z["covered"][off:off + n]] = np.nan
                off += n
                yield str(stem), s, float(dur), labels
        else:
            import xarray as xr
            with xr.open_dataset(path) as ds:
                labels = [str(x) for x in ds["label"].values]
                for i, stem in enumerate(ds["file"].values):
                    dur = float(ds["duration"].values[i])
                    n = int(dur / LI) + 1
                    yield str(stem), ds["scores"].isel(file=i).values[:, :n].astype(np.float32), dur, labels


thr_tab = pd.read_csv(args.thresholds).set_index("label")["best_threshold"]

ann = pd.read_csv(args.annotations)
ann["t0"] = pd.to_datetime(ann.start_datetime, utc=True)
ann["t1"] = pd.to_datetime(ann.end_datetime, utc=True)
ann["cat"] = ann.annotation.map(ANN2CAT)
ann = ann[ann.cat.notna()]
ann["site"] = ann.filename.map(site_of)
ann = ann[ann.site.notna()]

excl = {}
if args.exclude_files:
    import soundfile as sf
    for stem in (l.strip() for l in open(args.exclude_files) if l.strip()):
        site = site_of(stem)
        if site is None:
            continue
        dur = args.default_exclude_dur
        if args.exclude_audio_dir:
            p = os.path.join(args.exclude_audio_dir, stem + ".wav")
            if os.path.exists(p):
                dur = sf.info(p).duration
        t0 = t_of(stem)
        excl.setdefault(site, []).append((t0, t0 + pd.Timedelta(seconds=dur)))
    excl = {k: (np.array([a.value for a, _ in v]), np.array([b.value for _, b in v])) for k, v in excl.items()}
    print("excluded (training) files per site:", {SITES[k]: len(v[0]) for k, v in excl.items()})

rows, n_files = [], 0
for stem, scores, dur, labels in iter_scores(args.scores):
    site = site_of(stem)
    if site is None:
        continue
    n_files += 1
    li = {lb: k for k, lb in enumerate(labels)}
    thr = thr_tab.reindex(labels).to_numpy()
    scores = np.where(np.isnan(scores), -np.inf, scores)
    t_file = t_of(stem)
    a = ann[(ann.site == site) & (ann.t1 > t_file) & (ann.t0 < t_file + pd.Timedelta(seconds=dur))]
    s_rel = (a.t0 - t_file).dt.total_seconds().to_numpy()
    e_rel = (a.t1 - t_file).dt.total_seconds().to_numpy()
    cat = a.cat.to_numpy()
    if site == "kerguelen2015":   # downsweep table unusable there: DDswp = D call only
        keep = a.annotation.to_numpy() != "Fin whale downsweep"
        s_rel, e_rel, cat = s_rel[keep], e_rel[keep], cat[keep]
    dd = ["Blue whale D call"] + ([] if site == "kerguelen2015" else ["Fin whale downsweep"])
    outs = {"ABZ": ["Blue whale ABZ call"], "20Hz20Plus": ["Fin whale 20 Hz"], "DDswp": dd}
    ex0, ex1 = excl.get(site, (np.array([], dtype=np.int64), np.array([], dtype=np.int64)))
    for t in np.arange(0.0, dur - SEG + 1e-6, HOP):
        tol = args.tolerance
        b0, b1 = int(np.ceil((t + tol) / LI - 1e-9)), int(np.floor((t + SEG - tol) / LI + 1e-9))   # bin k = [k*LI, (k+1)*LI)
        seg = scores[:, b0:b1]
        if seg.shape[1] == 0 or not np.isfinite(seg).any():
            continue
        if len(ex0):
            g0 = (t_file + pd.Timedelta(seconds=float(t))).value
            g1 = g0 + int(SEG * 1e9)
            if ((ex0 < g1) & (ex1 > g0)).any():
                continue
        inside = (s_rel >= t - tol) & (e_rel <= t + SEG + tol)
        partial = (e_rel > t) & (s_rel < t + SEG) & ~inside
        true_set = sorted(set(cat[inside]))
        best, best_score = "Noise", -np.inf
        for c in CATS:
            for o in outs[c]:
                k = li[o]
                m = seg[k].max()
                if m > thr[k] and m > best_score:
                    best, best_score = c, m
        if not true_set:
            truth = "Noise"
        elif best in true_set:
            truth = best
        else:
            truth = true_set[rng.integers(len(true_set))]
        rows.append(dict(site=SITES[site], seen=(site in excl) if args.exclude_files else None, stem=stem, t=t, truth=truth, pred=best,
                         n_true=len(true_set), partial_only=(not true_set) and partial.any()))
seg_df = pd.DataFrame(rows)
seg_df.to_csv(out / "schall_segments.csv.gz", index=False)
print(f"{n_files} BlueFin files scored, {len(seg_df)} segments")
if seg_df.empty:
    raise SystemExit("no segments left to score (all excluded, or no BlueFin files in --scores)")


def metrics(d):
    cm = pd.crosstab(d.truth, d.pred).reindex(index=ALL, columns=ALL, fill_value=0)
    tot = cm.sum(axis=1)
    tcr = np.nanmean([cm.loc[c, c] / tot[c] if tot[c] else np.nan for c in CATS])
    nmr = cm.loc["Noise", CATS].sum() / tot["Noise"] if tot["Noise"] else np.nan
    cmr = np.nanmean([cm.loc[c, o] / tot[c] if tot[c] else np.nan for c in CATS for o in CATS if o != c])
    f = np.mean([tcr, 1 - nmr, 1 - nmr, 1 - cmr])
    return dict(TCR=tcr, NMR=nmr, CMR=cmr, F=f, n_segments=len(d), **{f"n_{c}": int(tot[c]) for c in ALL}), cm


res = []
for variant, d0 in (("protocol", seg_df), ("partial_noise_excluded", seg_df[~seg_df.partial_only])):
    for (site, seen), d in d0.groupby(["site", "seen"], dropna=False):
        m, _ = metrics(d)
        res.append(dict(variant=variant, site=site, seen=seen, **m))
    m, cm = metrics(d0)
    res.append(dict(variant=variant, site="pooled", seen=None, **m))
    cm.to_csv(out / f"schall_confusion_{variant}.csv")
res = pd.DataFrame(res)
summ = []
for variant, d in res[res.site != "pooled"].groupby("variant"):
    groups = [("all site-years", d)]
    if args.exclude_files:
        groups += [("unseen site-years", d[d.seen == False]),  # noqa: E712
                   ("seen site-years (val part)", d[d.seen == True])]  # noqa: E712
    for group, g in groups:
        if not len(g):
            continue
        for stat, fn in (("mean", "mean"), ("sd", "std")):
            summ.append(dict(variant=variant, group=group, n_sites=len(g), stat=stat,
                             **getattr(g[["TCR", "NMR", "CMR", "F"]], fn)().to_dict()))
summ = pd.DataFrame(summ)
res.to_csv(out / "schall_metrics_per_site.csv", index=False)
summ.to_csv(out / "schall_metrics_summary.csv", index=False)
pd.set_option("display.width", 220)
print(res.round(3).to_string(index=False))
print(summ.round(3).to_string(index=False))
print("multi-category segments:", int((seg_df.n_true > 1).sum()),
      "| noise segments with only a partial call:", int(seg_df.partial_only.sum()))
print("NOTE: site-years with very few segments of a category (e.g. Ross Sea 2014, one file) give noisy rates; the "
      "per-site table has the counts.")
