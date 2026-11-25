# -*- coding: utf-8 -*-
# AI-assisted code: developed with Claude Code (Anthropic); reviewed and tested by the SOMAD authors.
"""v19 self-tests for the CQT input path (CPU, synthetic data). Run:

    python tests_v17.py            # fast checks (~1-2 min on CPU)
    python tests_v17.py --smoke    # + one real training epoch of band="cqt" and inference from its checkpoint

Includes v13's tests for what v17 did not change (classes, scope, split,
masking, class weights, training-data sectioning), run on the cqt dataset.
"""
import copy
import datetime as dt
import os
import re
import sys
import tempfile
import traceback

import numpy as np
import pandas as pd
import soundfile as sf
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import class_config  # noqa: E402,F401
import common  # noqa: E402

RESULTS = []
CFG = common.BAND_CONFIG["cqt"]
SR = CFG["sample_rate"]


def check(name):
    def deco(fn):
        try:
            fn()
            RESULTS.append((name, True, ""))
            print(f"PASS  {name}")
        except Exception:
            RESULTS.append((name, False, traceback.format_exc()))
            print(f"FAIL  {name}\n{traceback.format_exc()}")
        return fn
    return deco


# ---- v13 tests, unchanged except band 'joint' -> 'cqt' ----

# The v13-era tests below check the scope / split / masking MECHANISMS against the v13 data policy
# (DCLDE train-only, Kerguelen D/downsweep masked, ...). v17/v18/v19 ship a different policy, so they run
# on a frozen copy of the v13 config; the shipped v19 config is restored before the v17/v19 tests.
V19_CFG = common.CLASS_CFG.raw
common.configure_classes(os.path.join(HERE, "tests_fixtures", "class_config_v13_policy.json"))

V12_LABELS = ["Fin whale 20 Hz", "Fin whale downsweep", "Blue whale AB call", "Blue whale D call",
              "Blue whale Z call", "Minke whale", "Humpback", "Leopard seal", "Weddell seal",
              "Crabeater seal", "Orca"]


SRW = "Southern right whale"
V13_LABELS = V12_LABELS[:6] + [SRW] + V12_LABELS[6:]      # new LF class appended to the LF block


@check("config: v12's 11 classes keep their order, + Southern right whale at the end of the LF block")
def _():
    assert common.LABELS == V13_LABELS, common.LABELS
    assert common.BAND_LABELS["cqt"] is common.LABELS
    for raw, cls in {"Humpback song": "Humpback", "Blue whale B": "Blue whale AB call", "Orca": "Orca",
                     "Blue whale D call": "Blue whale D call", "Minke whale": "Minke whale"}.items():
        assert common.APLOSE_LABEL_MAP[raw] == cls
    for raw, cls in {"orca?": "Orca", "bw_a": "Blue whale AB call", "humpack?": "Humpback"}.items():
        assert common.SOI_LABEL_MAP[raw] == cls
    assert common.UNCERTAIN_LABEL_MAP["Orca (uncertain)"] == "Orca"
    assert common.UNCERTAIN_LABEL_MAP["Humpback (uncertain)"] == "Humpback"


@check("scope: DCLDE sources only annotated for Orca/Humpback (VFPA: Orca only); old sources all classes")
def _():
    cfg, L = common.CLASS_CFG, common.LABELS
    assert cfg.source_of("2022_03_08_17_49_41_DFOWDLP_ST6249_x") == "dclde_wdlp"
    assert cfg.source_of("2017_07_18_04_30_01_VFPAONC_x") == "dclde_vfpa"
    sc = dict(zip(L, cfg.scope("dclde_wdlp", L)))
    assert [k for k, v in sc.items() if v] == ["Humpback", "Orca"]
    assert [k for k, v in zip(L, cfg.scope("dclde_vfpa", L)) if v] == ["Orca"]
    # Gerlache/hermine and SOI: every class annotated INCLUDING SRW (none present there -> valid negatives)
    assert all(cfg.scope("hermine_SN8059_aplose_subset_2h", L))
    assert all(cfg.scope("2024_soi_train", L)) and all(cfg.scope("2024_soi_val", L))
    # ... but DCLDE and BlueFin were never checked for SRW -> masked
    assert not cfg.scope("dclde_wdlp", L)[L.index(SRW)] and not cfg.scope("bluefin_casey2014", L)[L.index(SRW)]
    assert [k for k, v in zip(L, cfg.scope("srw_ei", L)) if v] == [SRW, "Humpback"]


@check("bluefin: D call + downsweep only from Elephant Island; Kerguelen used for its other classes; the rest ignored")
def _():
    cfg, L = common.CLASS_CFG, common.LABELS
    D, DS, MK = "Blue whale D call", "Fin whale downsweep", "Minke whale"
    def ann(src): return {k for k, v in zip(L, cfg.scope(src, L)) if v}
    lf6 = set(V12_LABELS[:6])
    assert ann("bluefin_ElephantIsland2013Aural") == lf6
    assert ann("bluefin_ElephantIsland2014") == lf6 - {MK}
    assert ann("bluefin_kerguelen2005") == lf6 - {D, DS}
    assert ann("bluefin_kerguelen2014") == lf6 - {D, DS, MK}
    assert ann("bluefin_kerguelen2015") == lf6 - {D, DS}
    for site in ("ElephantIsland2013Aural", "ElephantIsland2014", "kerguelen2005", "kerguelen2014", "kerguelen2015"):
        assert cfg.split_role(f"bluefin_{site}") == "random", site
    for site in ("BallenyIslands2015", "casey2014", "casey2017", "Greenwich64S2015", "MaudRise2014", "RossSea2014"):
        assert cfg.split_role(f"bluefin_{site}") == "ignore", site
    assert cfg.split_role("bluefin_library_aplose") == "ignore"        # a BlueFin site-year nobody listed
    srcs = ["hermine_SN8059_aplose_subset_2h", "2024_soi_train", "2024_soi_val"] +            [f"bluefin_{s}" for s in ("ElephantIsland2013Aural", "ElephantIsland2014", "kerguelen2005", "kerguelen2014",
                                     "kerguelen2015", "BallenyIslands2015", "casey2014", "casey2017",
                                     "Greenwich64S2015", "MaudRise2014", "RossSea2014")]
    used = [s for s in srcs if cfg.split_role(s) != "ignore"]
    base = ["hermine_SN8059_aplose_subset_2h", "2024_soi_train", "2024_soi_val",
            "bluefin_ElephantIsland2013Aural", "bluefin_ElephantIsland2014"]
    assert sorted(s for s in used if DS in ann(s)) == sorted(base)
    assert sorted(s for s in used if D in ann(s)) == sorted(base)
    stem = "2013_12_25_06_00_00_casey2014_200_2013-12-25_06-00-00"
    assert cfg.source_of(stem, "/data/bluefin_library_aplose/x.wav") == "bluefin_casey2014"
    # an SRW Elephant Island file from 2014 must NOT be taken for the bluefin ElephantIsland2014 site
    assert cfg.source_of("2014_01_15_08_10_34_SRWEI_ElephantIsland20140115_081034_AWI251_01_SV1008") == "srw_ei"


@check("bluefin: ignored files are in neither split; out-of-scope annotation rows (Kerguelen D/downsweep) are dropped")
def _():
    idx = {"2005_01_01_00_00_00_kerguelen2005_a": "/d/bluefin_library_aplose/a.wav",
           "2014_01_01_00_00_00_casey2014_b": "/d/bluefin_library_aplose/b.wav",
           "2013_01_01_00_00_00_ElephantIsland2013Aural_c": "/d/bluefin_library_aplose/c.wav",
           "hermine_SN8059_2024_03_17_00_00_05": "/d/hermine_SN8059_aplose_subset_2h/h.wav"}
    rows = []
    for st in idx:
        for lb in ("Blue whale D call", "Fin whale downsweep", "Fin whale 20 Hz"):
            rows.append(dict(stem=st, label=lb, t_1_s=1.0, t_2_s=2.0, mask_label=np.nan))
    df = pd.DataFrame(rows)
    out = common.drop_out_of_scope_annotations(df, idx)
    kg = out[out.stem.str.contains("kerguelen")]
    assert set(kg.label) == {"Fin whale 20 Hz"}                     # D call + downsweep dropped at Kerguelen
    ei = out[out.stem.str.contains("ElephantIsland")]
    assert set(ei.label) == {"Blue whale D call", "Fin whale downsweep", "Fin whale 20 Hz"}
    assert len(out[out.stem.str.startswith("hermine")]) == 3
    tr, va = common.split_files(idx, 0.5, 0, df_anno=out, verbose=False)
    assert "2014_01_01_00_00_00_casey2014_b" not in set(tr) | set(va)
    assert "2005_01_01_00_00_00_kerguelen2005_a" in set(tr) | set(va)


def _mk_index_and_anno():
    idx, rows = {}, []
    def add(stem, folder, labels=()):
        idx[stem] = f"/data/{folder}/{stem}.wav"
        for lb, t1, t2 in labels:
            rows.append(dict(stem=stem, label=lb, t_1_s=t1, t_2_s=t2, mask_label=np.nan))
    for i in range(60):
        add(f"hermine_SN8059_2024_03_17_{i:02d}_00_05", "hermine_SN8059_aplose_subset_2h",
            [("Orca", 5, 8)] if i in (3, 17) else ([("Humpback", 1, 4)] if i % 5 == 0 else []))
    for i in range(40):
        add(f"2024_02_12_{i:02d}_29_42_1_000060", "2024_soi_train", [("Orca", 2, 4)] if i == 7 else [])
    for i in range(30):
        add(f"2022_03_08_{i:02d}_49_41_DFOWDLP_ST6249_{i}", "audio", [("Orca", 10, 12)])
    for i in range(30):
        add(f"2013_10_11_{i:02d}_06_00_DFOCRP_x_{i}", "audio", [("Humpback", 10, 12)])
    return idx, pd.DataFrame(rows)


@check("split: DCLDE never in val; Southern Ocean orca files forced to val; ~30% of the random pool; deterministic")
def _():
    idx, df = _mk_index_and_anno()
    tr, va = common.split_files(idx, 0.30, seed=0, df_anno=df, verbose=False)
    assert not set(tr) & set(va)
    assert set(tr) | set(va) == set(idx)
    dclde = [s for s in idx if "_DFOWDLP_" in s or "_DFOCRP_" in s]
    assert all(s in set(tr) for s in dclde) and not any(s in set(va) for s in dclde)
    orca_files = {s for s, g in df[df.label == "Orca"].groupby("stem") if not ("DFO" in s)}
    assert orca_files <= set(va), "Southern Ocean orca files must be validation-only"
    pool = [s for s in idx if s not in dclde]
    frac = len([s for s in va]) / len(pool)
    assert abs(frac - 0.30) < 0.03, frac
    tr2, va2 = common.split_files(idx, 0.30, seed=0, df_anno=df, verbose=False)
    assert (tr, va) == (tr2, va2)
    # no orca annotation of a Southern Ocean source can be in training
    d = df[df.stem.isin(tr) & (df.label == "Orca")]
    assert all("DFOWDLP" in s for s in d.stem)


@check("loss: all-ones mask == unmasked loss; masked class gets exactly zero gradient")
def _():
    torch.manual_seed(0)
    x = torch.randn(4, 11, 30, requires_grad=True)
    y = (torch.rand(4, 11, 30) > 0.8).float()
    w = torch.rand(11) + 0.5
    crit = common.WeightedCombinedLoss(class_weights=w)
    a = crit(x, y)
    b = crit(x, y, torch.ones_like(y))
    assert torch.allclose(a, b, atol=1e-6), (a.item(), b.item())
    m = torch.ones_like(y)
    m[:, 3, :] = 0
    m[:, 7, 10:20] = 0
    loss = crit(x, y, m)
    loss.backward()
    assert x.grad[:, 3, :].abs().sum().item() == 0.0
    assert x.grad[:, 7, 10:20].abs().sum().item() == 0.0
    assert x.grad[:, 0, :].abs().sum().item() > 0


@check("metric: masked frames excluded from AP/F1; under-supported class -> NaN and ignored by macro")
def _():
    torch.manual_seed(1)
    N, C, T = 20, 3, 40
    y = (torch.rand(N, C, T) > 0.9).float()
    logits = torch.where(y > 0.5, torch.tensor(3.0), torch.tensor(-3.0)) + 0.1 * torch.randn(N, C, T)
    m = torch.ones_like(y)
    r = common.masked_class_metrics(logits, y, m, min_pos=5)
    assert r["macro_ap"] > 0.95
    bad = logits.clone()
    bad[:, 1, :] = -bad[:, 1, :]                 # make class 1 terrible ...
    r_bad = common.masked_class_metrics(bad, y, m, min_pos=5)
    m2 = m.clone()
    m2[:, 1, :] = 0                              # ... then mask it away: score must recover & class 1 is NaN
    r_masked = common.masked_class_metrics(bad, y, m2, min_pos=5)
    assert r_bad["macro_ap"] < r["macro_ap"] - 0.1
    assert np.isnan(r_masked["ap"][1]) and r_masked["macro_ap"] > 0.95
    y3 = y.clone(); y3[:, 2, :] = 0              # no positives for class 2
    r3 = common.masked_class_metrics(logits, y3, m, min_pos=5)
    assert np.isnan(r3["ap"][2]) and np.isfinite(r3["macro_ap"])


@check("add a class later: new class annotated only in its own source -> every other source masked for it")
def _():
    saved = copy.deepcopy(common.CLASS_CFG.raw)
    try:
        d = copy.deepcopy(saved)
        d["classes"].append({"name": "Sperm whale", "aplose_labels": ["Sperm whale"],
                             "soi_labels": [], "annotated_in": ["spermnet*"]})
        common.configure_classes(d)
        L = common.LABELS
        assert len(L) == 13 and L.index("Sperm whale") == len(L) - 1          # appended at the end, in config order
        assert common.APLOSE_LABEL_MAP["Sperm whale"] == "Sperm whale"
        cfg = common.CLASS_CFG
        sc_old = dict(zip(L, cfg.scope("hermine_SN8059_aplose_subset_2h", L)))
        sc_new = dict(zip(L, cfg.scope("spermnet_2025", L)))
        assert sc_old["Orca"] and not sc_old["Sperm whale"], "old sources must be masked for the NEW class"
        assert not dict(zip(L, cfg.scope("dclde_wdlp", L)))["Sperm whale"]
        assert sc_new["Sperm whale"]
        # old class order unchanged -> warm-start by name has nothing to scramble
        assert L[:12] == V13_LABELS
        # a bad config is rejected loudly
        bad = copy.deepcopy(d); bad["classes"].append({"name": "Orca"})
        try:
            class_config.load_class_config(bad); raise AssertionError("duplicate class accepted")
        except ValueError:
            pass
    finally:
        common.configure_classes(saved)
    assert len(common.LABELS) == 12


def _write_wav(path, seconds=130, sr=32000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    y = 0.01 * rng.standard_normal(len(t)) + 0.02 * np.sin(2 * np.pi * 2000 * t) * (np.sin(2 * np.pi * 0.2 * t) > 0)
    sf.write(path, y.astype("float32"), sr)


def _build_tiny_dataset(root):
    os.makedirs(root, exist_ok=True)
    herm = os.path.join(root, "hermine_SN8059_aplose_subset_2h")
    soit = os.path.join(root, "2024_soi_train")
    dcl = os.path.join(root, "somad_dclde_orca", "audio")
    for d in (herm, soit, dcl):
        os.makedirs(d, exist_ok=True)
    hs = [f"hermine_SN8059_2024_03_17_00_0{i}_05" for i in range(6)]
    for i, s in enumerate(hs):
        _write_wav(os.path.join(herm, s + ".wav"), seed=i)
    ss = [f"2024_02_12_21_2{i}_42_1_000060" for i in range(4)]
    for i, s in enumerate(ss):
        _write_wav(os.path.join(soit, s + ".wav"), seed=10 + i)
    ds = ["2022_03_08_17_49_41_DFOWDLP_ST6249_a", "2022_03_08_18_49_41_DFOWDLP_ST6249_b",
          "2013_10_11_00_06_00_DFOCRP_x_c"]
    for i, s in enumerate(ds):
        _write_wav(os.path.join(dcl, s + ".wav"), seed=20 + i)

    def iso(stem, sec):
        import re, datetime as dt
        m = re.search(r"\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}", stem)
        t0 = dt.datetime.strptime(m[0], "%Y_%m_%d_%H_%M_%S") + dt.timedelta(seconds=sec)
        return t0.strftime("%Y-%m-%dT%H:%M:%S.000+00:00")

    ap = []
    for i, s in enumerate(hs):
        ap.append(("hermine_SN8059_aplose_subset_2h", s, 10, 14, "Humpback squeaking", "BOX"))
        if i in (1, 4):
            ap.append(("hermine_SN8059_aplose_subset_2h", s, 40, 45, "Orca", "BOX"))
    rows = [dict(dataset=d, filename=s, start_datetime=iso(s, a), end_datetime=iso(s, b), annotation=lab, type=t)
            for d, s, a, b, lab, t in ap]
    pd.DataFrame(rows).to_csv(os.path.join(root, "Gerlache_Strait_SN8059_-_2024_-_subset_1h_results (3).csv"),
                              index=False)
    soi = [dict(audiofilename=s + ".wav", t1=iso(s, 20), t2=iso(s, 24), label=lab)
           for s, lab in zip(ss, ["fw_20hz", "orca", "leopard_seal", "orca?"])]
    pd.DataFrame(soi).to_csv(os.path.join(soit, "annotation_aural_subset_2024_correct.csv"), index=False)
    dc = []
    for s in ds:
        dc.append(dict(dataset="x", filename=s, start_datetime=iso(s, 30), end_datetime=iso(s, 33),
                       annotation="Orca", type="BOX"))
        dc.append(dict(dataset="x", filename=s, start_datetime=iso(s, 50), end_datetime=iso(s, 53),
                       annotation="Orca (uncertain)", type="BOX"))
        dc.append(dict(dataset="x", filename=s, start_datetime=iso(s, 70), end_datetime=iso(s, 74),
                       annotation="Humpback squeaking", type="BOX"))
    pd.DataFrame(dc).to_csv(os.path.join(root, "somad_dclde_orca", "dclde_orca_aplose.csv"), index=False)
    return hs, ss, ds


@check("dataset: returns (spec, labels, mask); out-of-scope classes and uncertain bins are masked; legacy 2-tuple intact")
def _():
    import somad_args as train_lf
    with tempfile.TemporaryDirectory() as root:
        hs, ss, ds = _build_tiny_dataset(root)
        args = train_lf.resolve_defaults(train_lf.build_argparser().parse_args(["--data-root", root]))
        df, idx = common.build_dataset_table(args.aplose_csv, args.soi_csv, common.resolve_audio_dirs(args, "cqt"))
        assert set(ds) <= set(idx) and set(hs) <= set(idx) and set(ss) <= set(idx)
        assert df["mask_label"].notna().sum() == 3 and (df["label"] == "Orca").sum() >= 5
        L = common.LABELS
        ds_ = common.PamSequenceDataset([ds[0]], idx, df, band="cqt", augment=False, return_mask=True)
        spec, labels, mask = ds_[0]
        assert mask.shape == labels.shape == (12, 121)
        for lb in L:
            c = L.index(lb)
            if lb in ("Orca", "Humpback"):
                assert mask[c].min() == 1.0 or lb == "Orca"
            else:
                assert mask[c].sum() == 0, f"{lb} must be fully masked for a DCLDE-WDLP file"
        o = L.index("Orca")
        # the window starting at 0 s covers 30-33 s (positive) and 50-53 s (uncertain -> masked, not positive)
        assert labels[o][60:67].sum() > 0 or labels[o].sum() > 0
        assert mask[o][100:107].sum() == 0 and labels[o][100:107].sum() == 0
        legacy = common.PamSequenceDataset([hs[0]], idx, df, band="cqt", augment=False)
        out = legacy[0]
        assert len(out) == 2
        full = common.PamSequenceDataset([hs[1]], idx, df, band="cqt", augment=False, return_mask=True)[0]
        assert full[2].min() == 1.0, "Gerlache/SOI (all classes incl. SRW annotated) must stay fully unmasked"
        assert common.PamSequenceDataset([ds[0]], idx, df, band="cqt", augment=False,
                                         return_mask=True)[0][2][L.index(SRW)].sum() == 0


@check("weights: scope-aware (DCLDE files do not dilute classes they are not annotated for)")
def _():
    import somad_args as train_lf
    with tempfile.TemporaryDirectory() as root:
        hs, ss, ds = _build_tiny_dataset(root)
        args = train_lf.resolve_defaults(train_lf.build_argparser().parse_args(["--data-root", root]))
        df, idx = common.build_dataset_table(args.aplose_csv, args.soi_csv, common.resolve_audio_dirs(args, "cqt"))
        L = common.LABELS
        w_h = common.compute_class_weights(df, hs, idx, L, 60.0).numpy()
        w_hd = common.compute_class_weights(df, hs + ds, idx, L, 60.0).numpy()
        i = L.index("Leopard seal")
        # leopard seal is masked for DCLDE files, so adding them must not change its weight relative to a
        # class that is annotated everywhere that has no new positives (Fin whale 20 Hz)
        j = L.index("Fin whale 20 Hz")
        assert abs(w_h[i] / w_h[j] - w_hd[i] / w_hd[j]) < 1e-6



@check("training_data_v13: 5-min sectioning, stem timestamps, annotation clipping/re-basing, SOMAD t_1_s exact")
def _():
    import datetime as dt
    try:
        import build_training_data_v13 as B
    except ImportError:
        print("      (build_training_data_v13.py not present -- skipped)")
        return
    assert B.plan_pieces(60) == [(0.0, 60.0)] and B.plan_pieces(300.2) == [(0.0, 300.2)]
    assert B.plan_pieces(330) == [(0.0, 330.0)]
    p = B.plan_pieces(3600.004)
    assert len(p) == 12 and p[-1][1] == 3600.004 and p[0] == (0.0, 300.0)       # 4 ms tail merged into the last section
    p = B.plan_pieces(2867.0)
    assert len(p) == 10 and p[-1] == (2700.0, 2867.0)                        # 167 s tail kept as its own section
    p = B.plan_pieces(620.0)
    assert p == [(0.0, 300.0), (300.0, 620.0)] or p == [(0.0, 300.0), (300.0, 600.0), (600.0, 620.0)] is False
    fs = dt.datetime(2005, 1, 1, 3, 0, 0)
    assert B.make_stem("2005_01_01_03_00_00_kerguelen2005_x", fs, 600, None, 2) == "2005_01_01_03_10_00_kerguelen2005_x_s02"
    assert B.make_stem("2024_02_12_21_29_42_1_000060", dt.datetime(2024, 2, 12, 21, 29, 42), 0, "SOI") ==         "2024_02_12_21_29_42_SOI_1_000060"
    # an annotation 590-615 s crosses the 600 s cut; one 10-20 s is inside the first section
    base = pd.Timestamp(fs, tz="UTC")
    rows = pd.DataFrame({"s": [590.0, 10.0], "e": [615.0, 20.0],
                         "start_datetime": [base + pd.Timedelta(seconds=590), base + pd.Timedelta(seconds=10)],
                         "end_datetime": [base + pd.Timedelta(seconds=615), base + pd.Timedelta(seconds=20)],
                         "annotation": ["a", "b"]})
    first = B.clip_rows(rows, fs, 300.0, 600.0)
    second = B.clip_rows(rows, fs, 600.0, 900.0)
    assert list(first.annotation) == ["a"] and first.start_time.iloc[0] == 290.0 and first.end_time.iloc[0] == 300.0
    assert list(second.annotation) == ["a"] and second.start_time.iloc[0] == 0.0 and second.end_time.iloc[0] == 15.0
    # SOMAD derives t = start_datetime - stem timestamp: must equal start_time in every section
    for a_, piece in ((300.0, first), (600.0, second)):
        stem = B.make_stem("2005_01_01_03_00_00_k", fs, a_, None, 1)
        t_1 = (piece.start_datetime.iloc[0] - pd.Timestamp(common.gettime(stem), tz="UTC")).total_seconds()
        assert abs(t_1 - piece.start_time.iloc[0]) < 1e-6, (t_1, piece.start_time.iloc[0])
    # single-source rules for the flat folder
    cfg = common.CLASS_CFG
    assert cfg.source_of("2024_02_12_21_29_42_SOI_1_000060") == "2024_soi_train"
    assert cfg.source_of("hermine_SN8059_2024_03_17_00_00_05") == "hermine_SN8059_aplose_subset_2h"
    assert cfg.source_of("2005_01_01_03_10_00_kerguelen2005_x_s02") == "bluefin_kerguelen2005"


# ---- v17: CQT input path ----

common.configure_classes(V19_CFG)    # back to the shipped v19 config


def _frontend():
    return common.CQTFrontend(CFG).eval()


@check("cqt_kernels == cqt-pytorch's CQT windows (with gamma patched into get_bandwidths); CQT output identical")
def _():
    try:
        import cqt_pytorch.cqt as cq
    except ImportError:
        print("      (cqt-pytorch not installed -- skipped)")
        return
    octs, bpo, gamma = CFG["cqt_octaves"], CFG["cqt_bins_per_octave"], CFG["cqt_gamma_hz"]
    block = int((CFG["snippet_length_s"] + 2 * CFG["cqt_pad_s"]) * SR)
    orig = cq.get_bandwidths

    def bw(num_octaves, num_bins_per_octave, sample_rate, frequencies):
        b = orig(num_octaves, num_bins_per_octave, sample_rate, frequencies).clone()
        K = num_octaves * num_bins_per_octave
        b[:K] += gamma
        b[K + 1:] += gamma
        return b
    cq.get_bandwidths = bw
    try:
        ref = cq.CQT(octs, bpo, SR, block_length=block)
    finally:
        cq.get_bandwidths = orig
    freqs, idx, win = common.cqt_kernels(octs, bpo, SR, block, gamma)
    assert torch.equal(idx, ref.windows_range_idx), "window index mismatch"
    assert torch.allclose(win, ref.windows), "window mismatch"
    assert abs(freqs[0].item() - SR / 2 / 2 ** octs) < 1e-6 and len(freqs) == octs * bpo
    x = torch.randn(1, 1, block, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        want = ref.encode(x).abs()[0, 0] ** 2
        crops = torch.fft.fft(x[0])[:, idx] * win
        c = torch.fft.ifft(crops)[0]
        got = c.real ** 2 + c.imag ** 2
    assert torch.allclose(got, want, rtol=1e-4, atol=1e-6 * want.max().item())


@check("frontend: (B,1,500,600) in [0,1]; a 20 Hz pulse at window time 30 s lands at the 20 Hz row / time bin 300")
def _():
    fe = _frontend()
    n = fe.block
    t = np.arange(n) / SR
    rng = np.random.default_rng(0)
    y = 0.001 * rng.standard_normal(n)
    t_pulse = CFG["cqt_pad_s"] + 30.0                      # 30 s into the 60 s window
    env = np.exp(-0.5 * ((t - t_pulse) / 0.3) ** 2)
    y += env * np.sin(2 * np.pi * 20.0 * t)
    x = torch.from_numpy(y.astype("float32"))[None, None]
    out = fe(torch.cat([x, x]))
    assert out.shape == (2, 1, CFG["freq_bins"], CFG["time_bins"]), out.shape
    assert out.min() >= 0 and out.max() <= 1 and torch.equal(out[0], out[1])
    img = out[0, 0].numpy()
    f_row, t_col = np.unravel_index(np.argmax(img), img.shape)
    k20 = CFG["cqt_bins_per_octave"] * np.log2(20.0 / fe.freqs[0].item())          # CQT bin of 20 Hz
    row20 = (k20 + 0.5) * CFG["freq_bins"] / len(fe.freqs) - 0.5                 # after bilinear resize
    assert abs(f_row - row20) <= 8, (f_row, row20)
    assert abs(t_col - 300) <= 3, t_col


@check("frontend: SpecAugment only in train mode and only for aug > 0 windows; float32 under autocast")
def _():
    fe = common.CQTFrontend(CFG)
    x = torch.randn(3, 1, fe.block, generator=torch.Generator().manual_seed(1)) * 0.01
    fe.eval()
    clean = fe(x)
    fe.train()
    torch.manual_seed(0)
    zero_frac = []
    for _ in range(10):
        out = fe(x, torch.tensor([0, 2, 0]))
        assert torch.equal(out[0], clean[0]) and torch.equal(out[2], clean[2])
        zero_frac.append((out[1] == 0).float().mean().item())
    assert max(zero_frac) > (clean[1] == 0).float().mean().item(), "no stripes ever drawn"
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        assert fe.eval()(x).dtype == torch.float32


def _write_noise_wav(path, seconds, sr=SR, seed=0, click_at=None):
    rng = np.random.default_rng(seed)
    y = 0.01 * rng.standard_normal(int(seconds * sr))
    if click_at is not None:
        y[int(click_at * sr)] = 1.0
    sf.write(path, y.astype("float32"), sr)


@check("dataset band=cqt: fixed-length wave, window always at [pad, pad+60) s (zeros where the file ends); aug flag")
def _():
    with tempfile.TemporaryDirectory() as root:
        stem = "hermine_SN8059_2024_03_17_00_00_05"
        p = os.path.join(root, stem + ".wav")
        _write_noise_wav(p, 90, click_at=40.0)          # windows 0-60 and 30-90 s; the 2nd's right pad (90-92 s) is past the end
        idx = {stem: p}
        empty = pd.DataFrame(columns=["file", "start_datetime", "end_datetime", "raw_label", "source", "label",
                                      "stem", "t_1_s", "t_2_s"])
        ds = common.PamSequenceDataset([stem], idx, empty, band="cqt", augment=False, return_mask=True)
        assert list(ds.df_data["start_sec"]) == [0.0, 30.0]
        pad = CFG["cqt_pad_s"]
        n_total = int((CFG["snippet_length_s"] + 2 * pad) * SR)
        for i, start in enumerate([0.0, 30.0]):
            x, labels, mask = ds[i]
            w = x["wave"][0]
            assert w.shape == (n_total,) and int(x["aug"]) == 0
            assert labels.shape == mask.shape == (len(common.LABELS), 121)
            assert int(torch.argmax(w.abs())) == int(round((pad + 40.0 - start) * SR)), (start, int(torch.argmax(w.abs())))
        w0 = ds[0][0]["wave"][0]
        assert w0[: int(pad * SR)].abs().max() == 0, "left pad before the file start must be zeros"
        w1 = ds[1][0]["wave"][0]
        assert w1[-int(pad * SR):].abs().max() == 0, "right pad past the file end must be zeros"
        dsa = common.PamSequenceDataset([stem], idx, empty, band="cqt", augment=True, noise_pool_size=1)
        assert int(dsa[len(dsa) - 1][0]["aug"]) == 2


@check("model: single CQT tower; dict batch -> (B, C, 121) logits; frontend buffers not in the state_dict")
def _():
    small = {**CFG, "hidden_size": 32, "gru_layers": 1, "pooled_freq_rows": 8}
    m = common.build_model(small, len(common.LABELS), use_attention=True).eval()
    n = m.frontend.block
    x = {"wave": torch.randn(2, 1, n) * 0.01, "aug": torch.zeros(2, dtype=torch.int64)}
    with torch.no_grad():
        y = m(x)
    assert y.shape == (2, len(common.LABELS), 121), y.shape
    assert not any(k.startswith("frontend.") for k in m.state_dict()), "CQT kernels leaked into the checkpoint"
    assert not any(k.startswith(("cnn_hf", "cnn_lf")) for k in m.state_dict()) and hasattr(m, "cnn")
    assert m.cnn[0].in_channels == 1


# ---------------------------------------------------------------- v19

@check("v19 config: downsweep masked for bluefin_kerguelen2015; Humpback masked for all BlueFin; ABZ kept")
def _():
    L = common.LABELS
    k15 = common.CLASS_CFG.scope("bluefin_kerguelen2015", L)
    k14 = common.CLASS_CFG.scope("bluefin_kerguelen2014", L)
    ds = L.index("Fin whale downsweep")
    assert not k15[ds] and k14[ds]
    assert all(k15[i] == k14[i] for i, lb in enumerate(L) if lb not in ("Fin whale downsweep", "Minke whale"))
    assert "Blue whale ABZ call" in L
    hb = L.index("Humpback")
    for src in ("bluefin_ElephantIsland2013Aural", "bluefin_ElephantIsland2014", "bluefin_kerguelen2005",
                "bluefin_kerguelen2014", "bluefin_kerguelen2015"):
        assert not common.CLASS_CFG.scope(src, L)[hb], f"Humpback must be masked for {src}"
    assert common.CLASS_CFG.scope("hermine_SN8059_aplose_subset_2h", L)[hb] and common.CLASS_CFG.scope("2024_soi_train", L)[hb]


@check("v19 fix script: removes only kerguelen2015 downsweep rows, keeps a backup, idempotent")
def _():
    import subprocess
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.csv")
        pd.DataFrame({"filename": ["2015_03_12_11_00_00_kerguelen2015_x", "2015_03_12_11_00_00_kerguelen2015_x",
                                   "2014_03_25_01_00_00_kerguelen2014_y"],
                      "annotation": ["Fin whale downsweep", "Fin whale 20 Hz", "Fin whale downsweep"]}).to_csv(p, index=False)
        for _ in range(2):
            subprocess.run([sys.executable, os.path.join(HERE, "fix_kerguelen2015_downsweep.py"), p],
                           check=True, capture_output=True)
        out = pd.read_csv(p)
        assert len(out) == 2 and not ((out.annotation == "Fin whale downsweep") & out.filename.str.contains("2015")).any()
        assert len(pd.read_csv(os.path.join(d, "t.pre_v19_fix.csv"))) == 3


def _loss_inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(3, 4, 7, generator=g) * 3
    targets = (torch.rand(3, 4, 7, generator=g) > 0.7).float()
    mask = torch.ones(3, 4, 7)
    mask[1, 2] = 0
    return logits, targets, mask


@check("v19 robust loss: none / no noisy windows == v18 loss; bootstrap beta=1 == v18; masked elements get no gradient")
def _():
    w = torch.tensor([1.0, 2.0, 0.5, 1.5])
    logits, targets, mask = _loss_inputs()
    plain = common.WeightedCombinedLoss(class_weights=w)(logits, targets, mask)
    for kind in common.ROBUST_LOSSES:
        crit = common.WeightedCombinedLoss(class_weights=w, robust=kind)
        assert torch.allclose(crit(logits, targets, mask, noisy=torch.zeros(3)), plain)
        assert torch.allclose(crit(logits, targets, mask), plain)
    # the element-wise re-implementation used by the robust branch == the v18 loss
    boot1 = common.WeightedCombinedLoss(class_weights=w, robust="bootstrap", robust_beta=1.0)
    assert torch.allclose(boot1(logits, targets, mask, noisy=torch.ones(3)), plain, atol=1e-6)
    for kind in ("bootstrap", "gce"):
        x = logits.clone().requires_grad_(True)
        common.WeightedCombinedLoss(class_weights=w, robust=kind)(x, targets, mask, noisy=torch.ones(3)).backward()
        assert torch.all(x.grad[1, 2] == 0) and x.grad.abs().sum() > 0


@check("v19 robust loss: a confidently contradicted label costs less (bootstrap) / is bounded (GCE); clean windows untouched")
def _():
    # one element: label 1, model says p ~ 0 (logit -8)
    logit = torch.full((1, 1, 1), -8.0)
    y = torch.ones(1, 1, 1)
    plain = common.WeightedCombinedLoss()(logit, y)
    boot = common.WeightedCombinedLoss(robust="bootstrap", robust_beta=0.8)(logit, y, noisy=torch.ones(1))
    gce = common.WeightedCombinedLoss(robust="gce", robust_q=0.7)(logit, y, noisy=torch.ones(1))
    assert boot < plain and gce <= 1 / 0.7 + 1e-6 and plain > 5, (plain, boot, gce)
    # only the noisy window of a batch changes
    logits, targets, mask = _loss_inputs(1)
    crit = common.WeightedCombinedLoss(robust="gce")
    el_plain = common.WeightedCombinedLoss()._elementwise(logits, targets)
    noisy = torch.tensor([0.0, 1.0, 0.0])
    full = crit(logits, targets, None, noisy=noisy)
    p = torch.sigmoid(logits)
    gce_el = (1 - torch.where(targets > 0.5, p, 1 - p).clamp_min(1e-7) ** 0.7) / 0.7
    expect = torch.cat([el_plain[:1], gce_el[1:2], el_plain[2:]]).mean()
    assert torch.allclose(full, expect, atol=1e-6)


@check("v19 dataset: x['noisy'] = 1 only for windows of --robust-sources files")
def _():
    with tempfile.TemporaryDirectory() as root:
        stems = {"hermine_SN8059_2024_03_17_00_00_05": None, "2005_02_20_15_45_00_kerguelen2005_x_s01": None}
        idx = {}
        for s in stems:
            p = os.path.join(root, s + ".wav")
            _write_noise_wav(p, 61)
            idx[s] = p
        empty = pd.DataFrame(columns=["file", "start_datetime", "end_datetime", "raw_label", "source", "label",
                                      "stem", "t_1_s", "t_2_s"])
        ds = common.PamSequenceDataset(list(idx), idx, empty, band="cqt", noisy_sources=["bluefin_*"])
        flags = {ds.df_data.loc[i, "stem"]: float(ds[i][0]["noisy"]) for i in range(len(ds))}
        assert flags == {"hermine_SN8059_2024_03_17_00_00_05": 0.0, "2005_02_20_15_45_00_kerguelen2005_x_s01": 1.0}
        assert float(common.PamSequenceDataset(list(idx), idx, empty, band="cqt")[0][0]["noisy"]) == 0.0


@check("v19 masked eval: continuous inference carries the masks; thresholds/AP/events ignore masked bins")
def _():
    import predict_eval
    L = common.LABELS
    with tempfile.TemporaryDirectory() as root:
        stem = "2005_02_20_15_45_00_kerguelen2015_x_s01"   # bluefin_kerguelen2015: downsweep out of scope
        p = os.path.join(root, stem + ".wav")
        _write_noise_wav(p, 61)
        idx = {stem: p}
        empty = pd.DataFrame(columns=["file", "start_datetime", "end_datetime", "raw_label", "source", "label",
                                      "stem", "t_1_s", "t_2_s"])
        small = {**CFG, "hidden_size": 16, "gru_layers": 1, "pooled_freq_rows": 4}
        model = common.build_model(small, len(L), use_attention=False).eval()
        res = common.run_inference_continuous_singleprocess(
            model, common.PamSequenceDataset([stem], idx, empty, band="cqt", return_mask=True), torch.device("cpu"))
        m = res[stem]["mask"][:, res[stem]["covered"]]
        assert m[L.index("Fin whale downsweep")].max() == 0           # out of scope -> not scored
        assert m[L.index("Fin whale 20 Hz")].min() == 1                # in scope -> scored
        res2 = common.run_inference_continuous_singleprocess(
            model, common.PamSequenceDataset([stem], idx, empty, band="cqt"), torch.device("cpu"))
        assert res2[stem]["mask"][:, res2[stem]["covered"]].min() == 1   # no masks requested -> all scored

    # fit_thresholds: a masked block of confident "false positives" must not change AP
    rng = np.random.default_rng(0)
    y = (rng.random((400, 1)) > 0.8).astype(float)
    s = y + rng.normal(0, 0.6, y.shape)
    junk_y, junk_s = np.zeros((200, 1)), np.full((200, 1), 5.0)
    Y, S = np.vstack([y, junk_y]), np.vstack([s, junk_s])
    M = np.vstack([np.ones_like(y), np.zeros_like(junk_y)])
    _, _, ap_masked, _ = predict_eval.fit_thresholds(Y, S, ["c"], mask_2d=M)
    _, _, ap_clean, _ = predict_eval.fit_thresholds(y, s, ["c"])
    _, _, ap_unmasked, _ = predict_eval.fit_thresholds(Y, S, ["c"])
    assert abs(ap_masked[0] - ap_clean[0]) < 1e-9 and ap_unmasked[0] < ap_clean[0] - 0.1
    _, f1_none, ap_none, _ = predict_eval.fit_thresholds(np.zeros((10, 1)), np.arange(10.0)[:, None], ["c"])
    assert np.isnan(ap_none[0]) and np.isnan(f1_none[0])

    # event metrics: a predicted event inside masked bins is not a false positive
    cont = {"f": {"scores": np.array([[5.0] * 6 + [-5.0] * 6]), "true": np.zeros((1, 12)),
                  "covered": np.ones(12, bool), "mask": np.array([[0.0] * 6 + [1.0] * 6]), "duration": 6.0}}
    assert common.event_level_metrics(cont, 0, 0.0, 0.5)["fp"] == 1
    assert common.event_level_metrics(cont, 0, 0.0, 0.5, use_mask=True)["fp"] == 0


if "--smoke" in sys.argv:
    def _iso(stem, sec):
        m = re.search(r"\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}", stem)
        t0 = dt.datetime.strptime(m[0], "%Y_%m_%d_%H_%M_%S") + dt.timedelta(seconds=sec)
        return t0.strftime("%Y-%m-%dT%H:%M:%S.000+00:00")

    @check("SMOKE: 1 epoch of band=cqt (v19: robust bootstrap loss on hermine*) through run_training, then rebuild + inference")
    def _():
        import somad_args as train_lf
        with tempfile.TemporaryDirectory() as root:
            data = os.path.join(root, "data")
            herm = os.path.join(data, "hermine_SN8059_aplose_subset_2h")
            soit = os.path.join(data, "2024_soi_train")
            os.makedirs(herm); os.makedirs(soit)
            hs = [f"hermine_SN8059_2024_03_17_00_0{i}_05" for i in range(5)]
            for i, s in enumerate(hs):
                _write_noise_wav(os.path.join(herm, s + ".wav"), 90, seed=i)
            rows = [dict(dataset="hermine_SN8059_aplose_subset_2h", filename=s, start_datetime=_iso(s, 10),
                         end_datetime=_iso(s, 14), annotation="Humpback song", type="BOX") for s in hs]
            pd.DataFrame(rows).to_csv(os.path.join(data, "Gerlache_Strait_SN8059_-_2024_-_subset_1h_results (3).csv"),
                                      index=False)
            pd.DataFrame(columns=["audiofilename", "t1", "t2", "label"]).to_csv(
                os.path.join(soit, "annotation_aural_subset_2024_correct.csv"), index=False)
            out = os.path.join(root, "out")
            args = train_lf.resolve_defaults(train_lf.build_argparser().parse_args([
                "--data-root", data, "--output-dir", out, "--run-name", "bigru_cqt", "--epochs", "1",
                "--batch-size", "2", "--num-workers", "0", "--patience", "1", "--noise-pool-size", "2",
                "--min-val-positives", "1", "--robust-loss", "bootstrap", "--robust-sources", "hermine*"]))
            common.run_training("cqt", args)
            ck = torch.load(os.path.join(out, "latest_checkpoint.pth"), weights_only=False)
            assert ck["band"] == "cqt" and ck["fusion_mode"] == "cqt"
            assert ck["robust_loss"]["type"] == "bootstrap" and ck["robust_loss"]["sources"] == ["hermine*"]
            assert ck["band_config"]["cqt_gamma_hz"] == 3.0 and ck["band_config"]["freq_bins"] == 500
            assert not any(k.startswith("frontend.") for k in ck["model_state_dict"])
            model = common.load_model_from_checkpoint(ck, torch.device("cpu"))
            val = open(os.path.join(out, "val_files.txt")).read().split()
            idx = common.build_audio_index([herm])
            empty = pd.DataFrame(columns=["file", "start_datetime", "end_datetime", "raw_label", "source", "label",
                                          "stem", "t_1_s", "t_2_s"])
            ds = common.PamSequenceDataset(val, idx, empty, band="cqt", augment=False, full_coverage=True)
            res = common.run_inference_continuous_singleprocess(model, ds, torch.device("cpu"), batch_size=2)
            for stem in val:
                assert res[stem]["covered"].all() and np.isfinite(res[stem]["scores"]).all()


fails = [r for r in RESULTS if not r[1]]
print(f"\n{len(RESULTS) - len(fails)}/{len(RESULTS)} passed")
sys.exit(1 if fails else 0)
