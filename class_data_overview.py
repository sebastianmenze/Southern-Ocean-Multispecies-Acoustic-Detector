# -*- coding: utf-8 -*-
# AI-assisted code: developed with Claude Code (Anthropic); reviewed and tested by the SOMAD authors.
"""v13: overview of classes x data sources x labelled bins x split.

  python class_data_overview.py --data-root /data --out-dir overview/
  python class_data_overview.py --aplose-csv gerlache.csv --aplose-csv bluefin.csv \
        --aplose-csv dclde.csv --soi-csv soi.csv --out-dir overview/     # annotations only

Counts UNIQUE positive 0.5 s bins (a bin touched by >= 1 annotation of the
class; overlapping annotations counted once) -- the label signal the model is
actually trained and scored on -- per class, per source and per split, with
the split computed exactly as training does (class_config.json roles).

If the audio folders exist they are used (the split then includes
un-annotated files). If none are found, files are taken from the annotation
CSVs only (the split is then computed over annotated files; "approximate").
"""
import os

import numpy as np
import pandas as pd

from somad_args import build_argparser, resolve_defaults
import common


def _annotations_only(args):
    frames = []
    for pth in args.aplose_csv:
        d = common.load_aplose_annotations(pth)
        d["vdir"] = d["dataset"]
        frames.append(d)
    if args.soi_csv and os.path.exists(args.soi_csv):
        d = common.load_soi_annotations(args.soi_csv)
        d["vdir"] = "2024_soi_train"
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["stem"] = df["file"].apply(lambda f: os.path.splitext(f)[0])
    ok = df["stem"].apply(lambda s: bool(common._TIME_PATTERN.search(s)))
    df = df[ok].reset_index(drop=True)
    idx = {s: f"/virtual/{v}/{s}.wav" for s, v in zip(df["stem"], df["vdir"])}
    # files whose rows are ALL unmapped (e.g. abiotic-only) still exist as audio
    # and are pure negatives: add them to the file list so file counts / the
    # split see them (their rows were dropped by the label maps).
    for pth in args.aplose_csv:
        raw = pd.read_csv(pth, usecols=lambda c: c in ("filename", "dataset"))
        vd = raw["dataset"].astype(str) if "dataset" in raw.columns else pd.Series("aplose", index=raw.index)
        for fn, v in zip(raw["filename"].astype(str), vd):
            st = os.path.splitext(fn)[0]
            if common._TIME_PATTERN.search(st):
                idx.setdefault(st, f"/virtual/{v}/{st}.wav")
    st = pd.to_datetime(df["stem"].apply(common.gettime), utc=True)
    df["t_1_s"] = (df["start_datetime"] - st).dt.total_seconds()
    df["t_2_s"] = (df["end_datetime"] - st).dt.total_seconds()
    plaus = (df["t_1_s"] > -60) & (df["t_2_s"] < 3600)
    return common.drop_out_of_scope_annotations(df[plaus].reset_index(drop=True), idx), idx


def main():
    p = build_argparser()
    p.add_argument("--out-dir", default="overview_v13")
    p.add_argument("--annotations-only", action="store_true",
                    help="ignore audio folders; build the file list from the annotation CSVs only")
    args = resolve_defaults(p.parse_args())
    L = list(common.BAND_LABELS["cqt"])
    band = {c["name"]: c["band"] for c in common.CLASS_CFG.classes}

    audio_dirs = common.resolve_audio_dirs(args, "cqt")
    have_audio = (not args.annotations_only) and any(
        os.path.isdir(d) and any(f.endswith(".wav") for f in os.listdir(d)) for d in audio_dirs)
    approx = not have_audio
    if have_audio:
        df, idx = common.build_dataset_table(args.aplose_csv, args.soi_csv, audio_dirs,
                                             bluefin_dir=getattr(args, "bluefin_dir", None),
                                             bluefin_sites=getattr(args, "bluefin_sites", None))
    else:
        df, idx = _annotations_only(args)
    train, val = common.get_split(args, idx, df, verbose=False)
    src = common.stem_sources(idx)
    tab = common.class_bin_table(df, train, val, idx, L)
    os.makedirs(args.out_dir, exist_ok=True)
    tab.to_csv(os.path.join(args.out_dir, "overview_class_source_split.csv"), index=False)

    # ---- per-class totals
    piv = tab.pivot_table(index="class", columns="split", values="positive_bins", aggfunc="sum", fill_value=0)
    for c in ("train", "val"):
        if c not in piv.columns:
            piv[c] = 0
    tot_rows = []
    for lb in L:
        tr = int(piv.loc[lb, "train"]) if lb in piv.index else 0
        va = int(piv.loc[lb, "val"]) if lb in piv.index else 0
        t = tab[tab["class"] == lb]
        tsrc = sorted(t.loc[(t["split"] == "train") & (t["positive_bins"] > 0), "source"].unique())
        vsrc = sorted(t.loc[(t["split"] == "val") & (t["positive_bins"] > 0), "source"].unique())
        masked_in = sorted({s for s in set(src.values())
                            if not common.CLASS_CFG.scope(s, [lb])[0]})
        tot_rows.append({"class": lb, "band": band[lb], "train_bins": tr, "val_bins": va, "total_bins": tr + va,
                         "val_%": round(100 * va / max(1, tr + va), 1),
                         "train_sources": ", ".join(tsrc), "val_sources": ", ".join(vsrc),
                         "masked_in_sources": ", ".join(masked_in),
                         "scored_in_selection": "yes" if va >= common.MIN_VAL_POSITIVE_BINS else "NO (too few val bins)"})
    totals = pd.DataFrame(tot_rows)
    totals.to_csv(os.path.join(args.out_dir, "overview_class_totals.csv"), index=False)

    # ---- per-source
    rows = []
    vs, ts = set(val), set(train)
    for s in sorted(set(src.values())):
        stems = [x for x in idx if src[x] == s]
        scope = common.CLASS_CFG.scope(s, L)
        rows.append({"source": s, "role": common.CLASS_CFG.split_role(s),
                     "train_files": sum(x in ts for x in stems), "val_files": sum(x in vs for x in stems),
                     "annotated_classes": ", ".join(lb for lb, ok in zip(L, scope) if ok) if not all(scope) else "all",
                     "masked_classes": ", ".join(lb for lb, ok in zip(L, scope) if not ok) or "-",
                     "sample_weight": common.CLASS_CFG.sample_weight(s),
                     "val_only_classes": ", ".join(sorted(common.CLASS_CFG.val_only_classes(s))) or "-"})
    srcs = pd.DataFrame(rows)
    srcs.to_csv(os.path.join(args.out_dir, "overview_sources.csv"), index=False)

    # ---- class x source matrix
    mat = tab.assign(v=tab["positive_bins"]).pivot_table(index="class", columns=["source", "split"],
                                                          values="v", aggfunc="sum", fill_value=0)

    def md(df_, floatfmt=None):
        cols = list(df_.columns)
        out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
        for _, r in df_.iterrows():
            out.append("| " + " | ".join(("" if (isinstance(v, float) and np.isnan(v)) else
                                          (f"{v:,}" if isinstance(v, (int, np.integer)) else str(v)))
                                         for v in r.tolist()) + " |")
        return "\n".join(out)

    lines = ["# SOMAD v13 -- classes x data sources x labelled bins",
             "",
             f"*Unique positive 0.5 s bins. Split rule: sources with `split: train_only` (DCLDE) never in validation; "
             f"files containing Southern Ocean Orca are forced into validation; remaining random-source files "
             f"split {int(round((1 - args.val_fraction) * 100))}/{int(round(args.val_fraction * 100))} "
             f"(seed {args.seed}).*",
             ("*Computed from annotated files only (no audio folders found) -- file counts and the random 70/30 "
              "draw are approximate; bin counts per source are exact.*" if approx else
              "*Computed on the audio folders found, un-annotated files included.*"),
             "", "## 1. Per class", "",
             md(totals[["class", "band", "train_bins", "val_bins", "total_bins", "val_%", "train_sources",
                        "val_sources", "scored_in_selection"]]),
             "", "## 2. Per source", "",
             md(srcs),
             "", "## 3. Class x source (positive bins, train / val)", ""]
    ignored_src = {s for s in set(src.values()) if common.CLASS_CFG.split_role(s) == "ignore"}
    cols = sorted({c[0] for c in mat.columns} - ignored_src)
    hdr = ["class"] + cols
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "|".join("---" for _ in hdr) + "|")
    for lb in L:
        cells = [lb]
        for s in cols:
            a = int(mat.loc[lb, (s, "train")]) if lb in mat.index and (s, "train") in mat.columns else 0
            b = int(mat.loc[lb, (s, "val")]) if lb in mat.index and (s, "val") in mat.columns else 0
            m = "" if common.CLASS_CFG.scope(s, [lb])[0] else " (masked)"
            cells.append(f"{a:,} / {b:,}" if (a or b) else ("-" + m if m else "0"))
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "`(masked)` = the class is not annotated in that source, so its frames are excluded from the "
                  "loss and metrics there (neither positive nor negative).",
              ("Ignored sources (role `ignore`, in neither split): " + ", ".join(sorted(ignored_src))) if ignored_src else "",
              ""]
    with open(os.path.join(args.out_dir, "overview.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwritten to {args.out_dir}")


if __name__ == "__main__":
    main()
