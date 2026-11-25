# -*- coding: utf-8 -*-
# AI-assisted code: developed with Claude Code (Anthropic); reviewed and tested by the SOMAD authors.
"""SOMAD v17: shared data pipeline, model and loss code for the CNN-BiGRU
sequence tagger that scores all classes (class_config.json) at once from
60 s windows of 32 kHz audio.

v17 input: ONE variable-Q CQT spectrogram (11 octaves x 24 bins/octave,
7.8 Hz - 16 kHz, gamma 3 Hz), resized to 500 freq x 600 time bins, through
ONE CNN tower. It replaces v13's two sub-band spectrograms (mel 32 kHz + linear
500 Hz) and two CNN towers, and v13's separate lf/hf/joint bands, fusion modes
and SimSiam tower pretraining (use the v13 code for those). The spectrogram is
computed inside the model (CQTFrontend) on the GPU; the DataLoader only
decodes audio and applies waveform augmentation. See CHANGES_v17.md.

Classes keep v13's two blocks (class_config.json "band": lf / hf) only as an
ordering convention: LF classes first, then HF classes, all scored jointly.
Perch's "Background" class is deliberately not reproduced: "no class active"
is the implicit negative of these multi-label sequence taggers.
"""

import csv
import datetime as dt
import glob
import os
import re
import time
from collections import OrderedDict

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, sosfiltfilt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# --------------------------------------------------------------------------
# torchaudio.load/.info compat shim -- dropped from torchaudio's top-level
# API in the 2.9.1 release this project moved to for RTX 5090 (Blackwell,
# sm_120) support (torchaudio.transforms.* used elsewhere in this file is
# unaffected, only the I/O functions). soundfile is already a hard
# dependency (via librosa), so it replaces them directly rather than
# chasing torchaudio's own I/O backend churn.
# --------------------------------------------------------------------------
def _load_audio(path: str, frame_offset: int = 0, num_frames: int = -1):
    data, sr = sf.read(path, start=frame_offset, frames=num_frames, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T.copy())
    return waveform, sr


def _audio_info(path: str):
    info = sf.info(path)
    return info.samplerate, info.frames


def _save_audio(path: str, waveform: torch.Tensor, sr: int):
    """torchaudio.save-compatible replacement (same 2.9.1 API removal as
    _load_audio/_audio_info above)."""
    data = waveform.numpy()
    data = data[0] if data.shape[0] == 1 else data.T
    sf.write(path, data, sr)


def _to_device(x, device):
    """Tensor.to(device), also for the {"wave", "aug"} dict batches
    PamSequenceDataset returns (the default collate_fn turns a batch of dicts
    into one dict of batched tensors)."""
    if isinstance(x, dict):
        return {k: v.to(device, non_blocking=True) for k, v in x.items()}
    return x.to(device, non_blocking=True)


# --------------------------------------------------------------------------
# v8 debug instrumentation (option 2 of the training-speed diagnosis, see
# README's "v8 changes") -- off by default (near-zero overhead when off:
# one bool check + no-op'd time.perf_counter() calls skipped entirely), on
# via PAM_DEBUG_TIMING=1 in the environment. Prints a per-stage timing +
# cache hit-rate summary from PamSequenceDataset.__getitem__ every
# DEBUG_TIMING_PRINT_EVERY calls (per worker -- each DataLoader worker
# process has its own counters, so with num-workers>1 you'll see this many
# times over, staggered, not one single global count) so the actual
# bottleneck (decode? resample+dither? augmentation? spectrogram compute?
# cache miss rate?) can be read directly instead of guessed at.
# --------------------------------------------------------------------------

DEBUG_TIMING = os.environ.get("PAM_DEBUG_TIMING", "0") == "1"
DEBUG_TIMING_PRINT_EVERY = 100


class _TimingStats:
    """Accumulates wall-clock seconds per named stage plus cache hit/miss
    counts, and prints a summary every `print_every` calls to record()."""

    def __init__(self, print_every: int = DEBUG_TIMING_PRINT_EVERY):
        self.print_every = print_every
        self.totals = {}
        self.n_calls = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def add(self, stage: str, seconds: float):
        self.totals[stage] = self.totals.get(stage, 0.0) + seconds

    def record_cache(self, hit: bool):
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

    def maybe_print(self, tag: str):
        self.n_calls += 1
        if self.n_calls % self.print_every != 0:
            return
        n = self.cache_hits + self.cache_misses
        hit_pct = 100.0 * self.cache_hits / n if n else float("nan")
        stage_str = ", ".join(f"{k}={v / self.n_calls * 1000:.1f}ms/call" for k, v in sorted(self.totals.items()))
        print(f"[DEBUG_TIMING][{tag}] after {self.n_calls} calls: {stage_str} -- "
              f"clean-spec cache hit rate {hit_pct:.1f}% ({self.cache_hits}/{n})")


# --------------------------------------------------------------------------
# Canonical label sets (must match perch_training/label_map*.json naming)
# --------------------------------------------------------------------------

LF_LABELS = [
    "Fin whale 20 Hz",
    "Fin whale downsweep",
    "Blue whale AB call",      # v5: A call + B call combined -- see APLOSE_LABEL_MAP's comment
    "Blue whale D call",
    "Blue whale Z call",       # v5: new, bluefin_library-only
    "Minke whale",             # v5: new, bluefin_library-only (bioduck + downsweep/other minke calls combined)
]

HF_LABELS = [
    "Humpback",
    "Leopard seal",
    "Weddell seal",
    "Crabeater seal",
    "Orca",
]

# All classes, LF block first -- the model's output order.
JOINT_LABELS = LF_LABELS + HF_LABELS

# One band, "cqt", scoring every class (same list object as JOINT_LABELS, so
# configure_classes' in-place update reaches it too).
BAND_LABELS = {"cqt": JOINT_LABELS}

# --------------------------------------------------------------------------
# Raw annotation label -> canonical label maps, one per source dataset.
# Superset of perch_training/label_map.json and label_map_soi2024.json, plus
# the blue whale call mappings those perch label maps don't have (perch's
# label maps drop bw_a/bw_dcall/bw_zcall and Blue whale A/D entirely).
# Labels not present here (e.g. Ship, vessel_noise, minke, odontocete,
# Unknown baleen whale, bw_tonal, ...) are simply dropped -- they don't
# contribute a positive label to any class, same as leaving a stretch of
# audio unannotated.
# --------------------------------------------------------------------------

APLOSE_LABEL_MAP = {
    "Humpback squeaking": "Humpback",
    "Humbpack groaning": "Humpback",
    "Humpback song": "Humpback",
    "Humpback LF tonal": "Humpback",
    "Humpback impulse": "Humpback",
    "Humpback bursts": "Humpback",
    "Fin whale 20 Hz": "Fin whale 20 Hz",
    "Fin whale downsweep": "Fin whale downsweep",
    "Leopard seal": "Leopard seal",
    "Weddell seal": "Weddell seal",
    "Crabeater seal": "Crabeater seal",
    "Orca": "Orca",
    # v5: A call and B call are merged into one "Blue whale AB call" class,
    # right here -- this is the ONE place the merge happens, for every raw
    # source's own spelling of A/B call, not baked in at any source's own
    # conversion step. B call alone is far too rare to train separately (1
    # occurrence total in the original Gerlache/hermine data) and both are
    # part of the same stereotyped Antarctic blue whale call sequence
    # anyway -- see README's "v5 changes".
    "Blue whale A": "Blue whale AB call",        # original Gerlache/hermine export's own raw string
    "Blue whale B": "Blue whale AB call",        # ditto (the single occurrence noted above)
    "Blue whale D": "Blue whale D call",         # original Gerlache/hermine export's own raw string
    # v5: convert_bluefin_library.py's output (a second, separate
    # APLOSE-format CSV -- see that script's docstring and README's "v5
    # changes") writes its OWN raw annotation strings, distinct from the
    # original Gerlache spellings above -- "Fin whale 20 Hz"/"Fin whale
    # downsweep" above already happen to match (same canonical string
    # either source would use), everything below does not. It deliberately
    # writes "Blue whale A call"/"Blue whale B call" as two SEPARATE raw
    # strings (matching each site's own distinct Raven selection files),
    # not pre-merged -- both map to the same "Blue whale AB call" target
    # here, same as the original source's "Blue whale A"/"Blue whale B"
    # above, so there's exactly one merge mechanism for every source
    # instead of one source special-casing it at conversion time.
    "Blue whale A call": "Blue whale AB call",
    "Blue whale B call": "Blue whale AB call",
    "Blue whale D call": "Blue whale D call",
    "Blue whale Z call": "Blue whale Z call",
    "Minke whale": "Minke whale",
}

SOI_LABEL_MAP = {
    "fw_20hz": "Fin whale 20 Hz",
    "fw_downsweep": "Fin whale downsweep",
    "humpback": "Humpback",
    "humpbacks": "Humpback",
    "humpack?": "Humpback",
    "leopard_seal": "Leopard seal",
    "weddell_seal": "Weddell seal",
    "crabeater_seal": "Crabeater seal",
    "orca": "Orca",
    "orca?": "Orca",
    "orcas?": "Orca",
    "bw_a": "Blue whale AB call",  # v5: merged label -- no separate "bw_b" key exists in this source
    "bw_dcall": "Blue whale D call",
    "bw_zcall": "Blue whale Z call",
}

# --------------------------------------------------------------------------
# v13: classes, label maps and per-source scope now come from
# class_config.json (see class_config.py). The literal lists/maps above are
# only the v12 defaults the config is mutated INTO, in place -- every
# existing reference (BAND_LABELS, the dataset, scripts that import
# common.JOINT_LABELS...) keeps pointing at the same list/dict objects.
# Adding a class later = edit class_config.json, no code change here.
# --------------------------------------------------------------------------
import class_config as _class_config

UNCERTAIN_LABEL_MAP = {}   # raw label -> class whose bins get MASKED (not negative, not positive)
CLASS_CFG = None


def configure_classes(path_or_dict=None):
    """(Re)load the class/source config and update LF/HF/JOINT_LABELS and the
    label maps in place. Called once at import with the shipped
    class_config.json (or $SOMAD_CLASS_CONFIG); call again with a path/dict
    to switch (somad_args.resolve_defaults does, for --class-config)."""
    global CLASS_CFG
    cfg = _class_config.load_class_config(path_or_dict)
    LF_LABELS[:] = cfg.lf_labels
    HF_LABELS[:] = cfg.hf_labels
    JOINT_LABELS[:] = LF_LABELS + HF_LABELS
    APLOSE_LABEL_MAP.clear()
    APLOSE_LABEL_MAP.update(cfg.aplose_map)
    SOI_LABEL_MAP.clear()
    SOI_LABEL_MAP.update(cfg.soi_map)
    UNCERTAIN_LABEL_MAP.clear()
    UNCERTAIN_LABEL_MAP.update(cfg.uncertain_map)
    CLASS_CFG = cfg
    return cfg


configure_classes()

# --------------------------------------------------------------------------
# Audio / spectrogram / windowing configuration (one band: "cqt").
# --------------------------------------------------------------------------

# v17: ONE variable-Q CQT spectrogram of the full 32 kHz band as the model
# input, through a single CNN tower, replacing v13's two sub-band
# spectrograms (mel HF + linear LF) and two towers (and v13's lf/hf/joint
# bands). The spectrogram is NOT computed in the DataLoader:
# PamSequenceDataset returns the raw (padded) window waveform, and
# ImprovedCNNBiGRU's CQTFrontend
# computes CQT -> power-pool to time_bins -> log10 -> resize to freq_bins ->
# per-window min-max -> SpecAugment on the model's device (GPU). 11 octaves x
# 24 bins/octave at 32 kHz = 264 log-spaced bins from 7.8 Hz to 16 kHz;
# gamma = 3 Hz widens every band by 3 Hz (variable-Q, as librosa.vqt), which
# only matters for the low bins: ~0.24 s window at 20 Hz instead of ~0.9 s
# for a plain CQT, so fin whale 20 Hz pulses keep their downsweep shape.
BAND_CONFIG = {"cqt": dict(
    sample_rate=32000,
    cqt_octaves=11,
    cqt_bins_per_octave=24,
    cqt_gamma_hz=3.0,
    # seconds of real audio (or zeros at a file edge) on each side of the 60 s
    # window inside the one CQT block: the transform is circular, so wrap-
    # around and the low bins' long windows only touch this padding, which
    # is cropped off before pooling. Must equal PamSequenceDataset.WINDOW_LOAD_PAD_S.
    cqt_pad_s=2.0,
    # windows per CQT call inside CQTFrontend (GPU memory: ~0.5 GB per window
    # at these settings, freed after each chunk)
    cqt_chunk=4,
    freq_bins=500,
    time_bins=600,
    snippet_length_s=60,
    snippet_overlap=2,
    label_interval_s=0.5,
    hidden_size=512,
    gru_layers=3,
    pooled_freq_rows=48,
    spec_augment=True,
    max_time_mask_frac=0.10,
    max_freq_mask_frac=0.10,
)}


def timesteps_for_band(band: str) -> int:
    cfg = BAND_CONFIG[band]
    return int(cfg["snippet_length_s"] / cfg["label_interval_s"]) + 1


# --------------------------------------------------------------------------
# Filename -> recording start time, shared by both audio sources: both
# "hermine_SN8059_2024_03_17_00_00_05.wav" and
# "2024_02_12_06_29_42_1_000030.wav" embed a %Y_%m_%d_%H_%M_%S timestamp.
# --------------------------------------------------------------------------

_TIME_PATTERN = re.compile(r"\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}")

# Directives gettime()'s optional `fmt` override supports, and the regex
# fragment used to locate each one's digits in a filename before handing
# the matched substring to strptime. Deliberately a small, explicit set
# (not every strptime directive) -- these cover the numeric date/time
# fields real recorder filenames actually use; unsupported directives
# raise clearly rather than silently building a wrong/loose regex.
_FMT_DIRECTIVE_TO_REGEX = {
    "%Y": r"\d{4}", "%y": r"\d{2}", "%m": r"\d{2}", "%d": r"\d{2}",
    "%H": r"\d{2}", "%M": r"\d{2}", "%S": r"\d{2}",
}


def _format_to_regex(fmt: str) -> re.Pattern:
    """Build a regex that locates a `fmt`-shaped timestamp substring inside
    an arbitrary filename, so it can be handed to strptime -- strptime
    itself can only parse a substring it's already been given, it can't
    search for one. Literal characters in `fmt` (separators like `_`/`-`)
    are matched exactly; see _FMT_DIRECTIVE_TO_REGEX for supported
    directives."""
    pattern = ""
    i = 0
    while i < len(fmt):
        directive = fmt[i:i + 2]
        if fmt[i] == "%" and directive in _FMT_DIRECTIVE_TO_REGEX:
            pattern += _FMT_DIRECTIVE_TO_REGEX[directive]
            i += 2
        elif fmt[i] == "%":
            raise ValueError(f"Unsupported strptime directive {directive!r} in {fmt!r} -- "
                              f"gettime()'s fmt override only supports {sorted(_FMT_DIRECTIVE_TO_REGEX)}")
        else:
            pattern += re.escape(fmt[i])
            i += 1
    return re.compile(pattern)


def gettime(filename: str, fmt: str = None) -> dt.datetime:
    """Extract a recording-start timestamp from `filename`. Default: the
    project's own %Y_%m_%d_%H_%M_%S convention (see module comment above).
    Pass `fmt` (a strptime format string, e.g. "%Y%m%d_%H%M%S") to parse a
    different filename convention instead -- e.g. run_inference.py's
    --filename-time-format, for deploying over audio that isn't named the
    way this project's own training data is. Test a candidate `fmt`
    against your actual files first with check_wav_timestamps.py --format,
    which shares this same parsing logic."""
    if fmt is not None:
        pattern = _format_to_regex(fmt)
        m = pattern.search(os.path.basename(filename))
        if not m:
            raise ValueError(f"Could not find a {fmt!r}-shaped timestamp in {filename!r}")
        return dt.datetime.strptime(m[0], fmt)
    m = _TIME_PATTERN.search(os.path.basename(filename))
    if not m:
        raise ValueError(f"Could not find a %Y_%m_%d_%H_%M_%S timestamp in {filename!r}")
    return dt.datetime.strptime(m[0], "%Y_%m_%d_%H_%M_%S")


def format_aplose_dt(dt_val: pd.Timestamp) -> str:
    """Match APLOSE's export datetime format, e.g. 2024-03-17T00:00:05.000+00:00
    -- same convention as perch_training/perch_common.py's format_aplose_dt,
    duplicated here rather than imported so this module doesn't depend on
    the (separate) Perch pipeline."""
    return dt_val.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt_val.microsecond // 1000:03d}+00:00"


# --------------------------------------------------------------------------
# Annotation loading / merging
# --------------------------------------------------------------------------

def load_aplose_annotations(csv_path: str) -> pd.DataFrame:
    """Load the Gerlache/hermine APLOSE-export CSV into the common schema.

    Drops APLOSE "WEAK" annotations (`type` column; APLOSE's own presence-
    only category, no precise time box -- confirmed on the real export:
    100% of these have (end_datetime - start_datetime) spanning the whole
    ~60s reviewed clip, vs. APLOSE "BOX" annotations, which are all a few
    seconds). PamSequenceDataset rasterizes an annotation's span directly
    into per-bin positive labels, so a WEAK row would otherwise teach the
    model "predict positive for the entire window" -- confirmed as the
    root cause of a small number (~1-2%) of real deployment files showing
    a class detected for 100% of the file with no on/off structure, even
    though a real call genuinely is present somewhere in the recording
    (see README's "v3 changes" #14/#15). Dropping these is a
    simplification, not a perfect fix: it treats a WEAK-only window the
    same as an unannotated one (no positive signal contributed at all,
    same as the rest of never-reviewed audio -- see "No explicit
    'Background' class" above), rather than the more correct but bigger
    change of masking that window/class out of the loss entirely
    (excluding it from both positive AND negative gradient contributions,
    since presence is confirmed even if timing isn't). Good enough given
    how few of these there are relative to precisely-boxed annotations.
    """
    df = pd.read_csv(csv_path)
    out = pd.DataFrame({
        "file": df["filename"].astype(str),
        "start_datetime": pd.to_datetime(df["start_datetime"], utc=True),
        "end_datetime": pd.to_datetime(df["end_datetime"], utc=True),
        "raw_label": df["annotation"],
        "source": "aplose",
        "dataset": df["dataset"].astype(str) if "dataset" in df.columns else "aplose",
        "weak": df.get("type", pd.Series("BOX", index=df.index)).eq("WEAK"),
    })
    out["label"] = out["raw_label"].map(APLOSE_LABEL_MAP)
    # v13: "uncertain" raw labels (class_config.json -> uncertain_labels) keep
    # a row with label=NaN and mask_label=<class>: never a positive, but the
    # dataset masks those bins out of that class's loss instead of treating
    # them as confirmed background.
    out["mask_label"] = out["raw_label"].map(UNCERTAIN_LABEL_MAP)
    out = out[out["label"].notna() | out["mask_label"].notna()].reset_index(drop=True)

    n_weak = int(out["weak"].sum())
    if n_weak:
        print(f"Dropping {n_weak} APLOSE 'WEAK' (presence-only, no precise timing) annotation rows "
              f"-- would otherwise rasterize as 'entire window positive':")
        print(out.loc[out["weak"], "label"].value_counts().to_string())
    return out[~out["weak"]].drop(columns="weak").reset_index(drop=True)


def load_soi_annotations(csv_path: str) -> pd.DataFrame:
    """Load the SOI/AURAL annotation CSV (t1/t2/label/audiofilename) into the
    common schema. v13: also accepts an APLOSE-format export of the same
    annotations (filename/start_datetime/end_datetime/annotation columns),
    mapped with the SOI label map."""
    df = pd.read_csv(csv_path)
    if "audiofilename" not in df.columns and {"filename", "annotation", "start_datetime"} <= set(df.columns):
        out = pd.DataFrame({
            "file": df["filename"].astype(str).apply(os.path.basename),
            "start_datetime": pd.to_datetime(df["start_datetime"], utc=True),
            "end_datetime": pd.to_datetime(df["end_datetime"], utc=True),
            "raw_label": df["annotation"],
            "source": "soi",
            "dataset": df["dataset"].astype(str) if "dataset" in df.columns else "soi",
        })
        weak = df.get("type", pd.Series("BOX", index=df.index)).eq("WEAK").to_numpy()
        out = out[~weak]
    else:
        out = pd.DataFrame({
            "file": df["audiofilename"].apply(os.path.basename),
            "start_datetime": pd.to_datetime(df["t1"], utc=True),
            "end_datetime": pd.to_datetime(df["t2"], utc=True),
            "raw_label": df["label"],
            "source": "soi",
            "dataset": "soi",
        })
    out["label"] = out["raw_label"].map(SOI_LABEL_MAP)
    out["mask_label"] = out["raw_label"].map(UNCERTAIN_LABEL_MAP)
    return out[out["label"].notna() | out["mask_label"].notna()].reset_index(drop=True)


def build_audio_index(audio_dirs) -> dict:
    """Map file stem (no extension) -> full path, across all given directories."""
    index = {}
    for d in audio_dirs:
        for path in glob.glob(os.path.join(d, "*.wav")):
            index[os.path.splitext(os.path.basename(path))[0]] = path
    return index


def resolve_audio_dirs(args, band: str) -> list:
    """Audio folders to index: hermine/soi, the BlueFin library conversion
    (args.bluefin_dir, 500 Hz audio -- upsampled to 32 kHz with shaped dither
    on load, or read from --bluefin-hf-dir's precomputed 32 kHz copy) and any
    --extra-audio-dir (v13: e.g. DCLDE, SRW). `band` is kept for the call
    signature; v17 has only "cqt", which needs all of them."""
    dirs = [args.hermine_dir, args.soi_train_dir, args.soi_val_dir, getattr(args, "bluefin_dir", None)]
    dirs.extend(getattr(args, "extra_audio_dir", None) or [])   # v13: e.g. the DCLDE folder
    return [d for d in dirs if d]


def filter_bluefin_sites(audio_index: dict, bluefin_dir: str, sites) -> dict:
    """v7: drop any bluefin_dir-sourced stem whose filename doesn't contain
    one of `sites` (site-year names, e.g. "ElephantIsland2013Aural" --
    matching convert_bluefin_library.py's own SITES keys exactly, since
    that script's output filenames always embed the site name) -- lets
    training restrict bluefin_library to a subset of its 11 site-years
    (--bluefin-sites) without touching hermine/soi stems.
    Identified by directory (bluefin_dir is one flat folder, see
    convert_bluefin_library.py's docstring), not by name pattern, so this
    can't accidentally misclassify a hermine/soi stem. No-op if `sites` is
    falsy (every bluefin site stays -- the default) or bluefin_dir wasn't
    resolved."""
    if not sites or not bluefin_dir:
        return audio_index
    bluefin_dir_norm = os.path.normpath(bluefin_dir)
    out = {}
    n_dropped = 0
    for stem, path in audio_index.items():
        if os.path.normpath(os.path.dirname(path)) == bluefin_dir_norm and not any(s in stem for s in sites):
            n_dropped += 1
            continue
        out[stem] = path
    print(f"--bluefin-sites {sites}: dropped {n_dropped} bluefin file(s) from other site-years, "
          f"{len(out)} file(s) remain")
    return out


def build_dataset_table(aplose_csv, soi_csv: str, audio_dirs, bluefin_dir: str = None,
                         bluefin_sites=None) -> tuple[pd.DataFrame, dict]:
    """Merge both annotation sources, restrict to files actually present on
    disk, and attach each file's start time / relative annotation offsets
    (t_1_s / t_2_s) so windows can be cut per-file at train time.

    `aplose_csv` accepts either a single path (v1-v4 behavior) or a list of
    paths (v5+: e.g. the original Gerlache/hermine export plus
    convert_bluefin_library.py's bluefin_library_annotations.csv) -- every
    APLOSE-format source gets loaded and concatenated the same way multiple
    real APLOSE exports would be. All must share the real APLOSE export
    schema load_aplose_annotations expects.

    `bluefin_dir`/`bluefin_sites` (v7, both optional, default None = no
    filtering, backward compatible with every existing caller) restrict
    bluefin_library to a subset of its site-years via filter_bluefin_sites
    -- see --bluefin-sites in somad_args.build_argparser.

    Returns (df_anno, audio_index) where df_anno has columns
    [file, label, source, t_1_s, t_2_s] and audio_index maps
    file stem -> wav path for every audio file found (annotated or not --
    unannotated files/regions are valid, they're implicit negatives).
    """
    audio_index = build_audio_index(audio_dirs)
    audio_index = filter_bluefin_sites(audio_index, bluefin_dir, bluefin_sites)

    aplose_csvs = [aplose_csv] if isinstance(aplose_csv, str) else list(aplose_csv or [])
    frames = [load_aplose_annotations(path) for path in aplose_csvs]
    if soi_csv:
        frames.append(load_soi_annotations(soi_csv))
    df_anno = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["file", "start_datetime", "end_datetime", "raw_label", "source", "label"]
    )

    df_anno["stem"] = df_anno["file"].apply(lambda f: os.path.splitext(f)[0])
    df_anno = df_anno[df_anno["stem"].isin(audio_index)].reset_index(drop=True)

    starttimes = df_anno["stem"].apply(gettime)
    starttimes = pd.to_datetime(starttimes, utc=True)
    df_anno["t_1_s"] = (df_anno["start_datetime"] - starttimes).dt.total_seconds()
    df_anno["t_2_s"] = (df_anno["end_datetime"] - starttimes).dt.total_seconds()

    # A handful of rows in the SOI annotation CSV have a start/end datetime
    # wildly inconsistent with their own audiofilename's embedded timestamp
    # (bad metadata upstream, not something this pipeline can repair). Files
    # here are at most a few minutes long, so anything wildly outside that
    # can't be real and would otherwise corrupt duration-based class weights.
    n_before = len(df_anno)
    plausible = (df_anno["t_1_s"] > -60) & (df_anno["t_2_s"] < 3600)
    dropped = df_anno.loc[~plausible]
    if len(dropped):
        print(f"Dropping {len(dropped)} annotation rows with implausible offsets "
              f"(bad source metadata) from: {sorted(dropped['file'].unique())}")
    df_anno = df_anno[plausible].reset_index(drop=True)
    assert len(df_anno) == n_before - len(dropped)

    df_anno = drop_out_of_scope_annotations(df_anno, audio_index)
    return df_anno, audio_index


def add_pooled_class_rows(df_anno: pd.DataFrame) -> pd.DataFrame:
    """v18: for every class with "pooled_from" in class_config.json, append a copy of each member
    class's annotation rows labelled with the pooled class. The member classes keep their rows."""
    pooled = getattr(CLASS_CFG, "pooled", None)
    if not pooled or df_anno is None or df_anno.empty:
        return df_anno
    extra = []
    for name, members in pooled.items():
        if (df_anno["label"] == name).any():
            continue
        d = df_anno[df_anno["label"].isin(members)].copy()
        d["label"] = name
        extra.append(d)
        print(f"pooled class {name!r}: +{len(d)} rows from {members}")
    return pd.concat([df_anno] + extra, ignore_index=True) if extra else df_anno


def drop_out_of_scope_annotations(df_anno: pd.DataFrame, audio_index: dict) -> pd.DataFrame:
    """v13: remove annotation rows whose class is outside the scope of the
    file's source (class_config.json -> annotated_classes / annotated_in).
    Masking alone zeroes their loss, but they would still count as positives
    in the rare-class sampler, class weights and bin tables; dropping them
    makes all of those consistent (e.g. Kerguelen's D-call/downsweep rows)."""
    if df_anno is None or df_anno.empty or "label" not in df_anno.columns:
        return df_anno
    df_anno = add_pooled_class_rows(df_anno)
    labels = list(JOINT_LABELS)
    li = {lb: i for i, lb in enumerate(labels)}
    scope = stem_scope_matrix(df_anno["stem"].unique(), audio_index, labels)
    keep = [
        (lb not in li) or bool(scope[st][li[lb]]) if isinstance(lb, str) else True
        for st, lb in zip(df_anno["stem"].tolist(), df_anno["label"].tolist())
    ]
    keep = np.array(keep, dtype=bool)
    n_drop = int((~keep).sum())
    if n_drop:
        d = df_anno.loc[~keep]
        print(f"Dropping {n_drop} annotation rows of classes outside their source's scope: "
              f"{d.groupby(['label']).size().to_dict()}")
    return df_anno[keep].reset_index(drop=True)


def stem_sources(audio_index: dict) -> dict:
    """v13: stem -> source name (class_config.json stem rules, else the
    audio folder's name)."""
    return {s: CLASS_CFG.source_of(s, p) for s, p in audio_index.items()}


def stem_scope_matrix(stems, audio_index: dict, label_list) -> dict:
    """v13: stem -> bool array (len(label_list)): was this class actually
    annotated for this file's source? False = masked out of loss/metrics."""
    cache, out = {}, {}
    for s in stems:
        src = CLASS_CFG.source_of(s, audio_index.get(s))
        if src not in cache:
            cache[src] = np.array(CLASS_CFG.scope(src, label_list), dtype=bool)
        out[s] = cache[src]
    return out


def split_files(audio_index: dict, val_fraction: float = 0.30, seed: int = 0,
                df_anno: pd.DataFrame = None, verbose: bool = True):
    """v13 deterministic file-level train/val split (no clip ever leaks
    across the split).

    Roles come from class_config.json -> sources[*].split:
      train_only  -- never in validation (e.g. the DCLDE Pacific orca data)
      val_only    -- never in training
      random      -- eligible for the random val_fraction draw
    plus, for random sources, `val_only_classes` (Southern Ocean Orca): any
    FILE containing an annotation of such a class is forced into
    validation, so those annotations can never reach training.

    val_fraction is a fraction of the random-eligible files (random-source
    files incl. the forced ones): forced files count toward it, the rest of
    the quota is drawn stratified by source. With df_anno=None no file can be
    forced (callers that need the real split must pass df_anno)."""
    from sklearn.model_selection import train_test_split

    stems = sorted(audio_index.keys())
    src = stem_sources(audio_index)
    role = {s: CLASS_CFG.split_role(src[s]) for s in stems}

    forced = set()
    if df_anno is not None and len(df_anno):
        d = df_anno[df_anno["stem"].isin(src.keys()) & df_anno["label"].notna()]
        for source in sorted({src[s] for s in d["stem"].unique()}):
            vo = CLASS_CFG.val_only_classes(source)
            if not vo:
                continue
            ss = d[d["stem"].map(src) == source]
            forced |= set(ss.loc[ss["label"].isin(vo), "stem"])
    conflict = [s for s in forced if role[s] != "random"]
    if conflict and verbose:
        print(f"WARNING: {len(conflict)} file(s) hold val-only-class annotations but their source's split "
              f"role is not 'random' -- split role wins: {conflict[:5]}")
    forced = {s for s in forced if role[s] == "random"}

    ignored = {s for s in stems if role[s] == "ignore"}      # in neither split (see class_config.json)
    train = {s for s in stems if role[s] == "train_only"}
    val = {s for s in stems if role[s] == "val_only"} | forced
    pool = [s for s in stems if role[s] == "random" and s not in forced]

    n_eligible = len(pool) + len(forced)
    target = int(round(val_fraction * n_eligible))
    need = max(0, min(len(pool), target - len(forced)))
    extra = set()
    if need and need < len(pool):
        strat = [src[s] for s in pool]
        counts = {k: strat.count(k) for k in set(strat)}
        strat = [k if counts[k] >= 2 else "_rare" for k in strat]
        try:
            _, extra_l = train_test_split(pool, test_size=need, random_state=seed, stratify=strat)
        except ValueError:
            _, extra_l = train_test_split(pool, test_size=need, random_state=seed)
        extra = set(extra_l)
    elif need >= len(pool):
        extra = set(pool)
    val |= extra
    train |= set(pool) - extra

    if verbose:
        rows = {}
        for s in stems:
            r = rows.setdefault(src[s], {"train": 0, "val": 0, "forced_val": 0, "ignored": 0})
            if s in ignored:
                r["ignored"] += 1
            elif s in val:
                r["val"] += 1
                r["forced_val"] += int(s in forced)
            else:
                r["train"] += 1
        print("split by source (files): " + "; ".join(
            (f"{k}: IGNORED {v['ignored']}" if v["ignored"] and not (v["train"] or v["val"]) else
             f"{k}: train {v['train']} / val {v['val']}" + (f" (forced {v['forced_val']})" if v["forced_val"] else ""))
            for k, v in sorted(rows.items())))
        print(f"split: {len(train)} train / {len(val)} val files, {len(ignored)} ignored "
              f"({100 * len(val) / max(1, len(stems) - len(ignored)):.1f}% of used files; "
              f"{100 * len(val) / max(1, n_eligible):.1f}% of the random-eligible pool; "
              f"{len(forced)} forced-val for val-only classes)")
    return sorted(train), sorted(val)


def class_bin_table(df_anno: pd.DataFrame, train_stems, val_stems, audio_index: dict, label_list,
                    interval: float = 0.5) -> pd.DataFrame:
    """v13: unique positive bins (the labels the model is actually trained /
    scored on: 0.5 s bins touched by >=1 annotation of the class, overlapping
    annotations counted once) per class x source x split. Windows overlap and
    are duplicated for augmentation, so this is the honest "how much label
    signal is there" number, not a window count."""
    train_set, val_set = set(train_stems), set(val_stems)
    d = df_anno[df_anno["label"].isin(list(label_list))]
    src_cache = {}
    rows = []
    for (stem, lb), g in d.groupby(["stem", "label"]):
        a = np.floor(g["t_1_s"].to_numpy(dtype=float) / interval).astype(int)
        b = np.maximum(np.ceil(g["t_2_s"].to_numpy(dtype=float) / interval).astype(int), a + 1)
        order = np.argsort(a, kind="stable")
        a, b = a[order], b[order]
        total, cs, ce = 0, a[0], b[0]
        for x, y in zip(a[1:], b[1:]):
            if x <= ce:
                ce = max(ce, y)
            else:
                total += ce - cs
                cs, ce = x, y
        total += ce - cs
        if stem not in src_cache:
            src_cache[stem] = CLASS_CFG.source_of(stem, audio_index.get(stem))
        split = "train" if stem in train_set else ("val" if stem in val_set else "unused")
        rows.append((lb, src_cache[stem], split, stem, len(g), int(total)))
    t = pd.DataFrame(rows, columns=["class", "source", "split", "stem", "annotations", "bins"])
    if t.empty:
        return pd.DataFrame(columns=["class", "source", "split", "files", "annotations", "positive_bins"])
    return (t.groupby(["class", "source", "split"])
             .agg(files=("stem", "nunique"), annotations=("annotations", "sum"), positive_bins=("bins", "sum"))
             .reset_index())


def _print_split_class_bins(df_anno, train_stems, val_stems, audio_index, label_list):
    t = class_bin_table(df_anno, train_stems, val_stems, audio_index, label_list)
    if t.empty:
        print("(no annotated bins found)")
        return
    piv = t.pivot_table(index="class", columns="split", values="positive_bins", aggfunc="sum", fill_value=0)
    for col in ("train", "val"):
        if col not in piv.columns:
            piv[col] = 0
    print("positive 0.5 s bins per class (unique):")
    for lb in label_list:
        tr, va = (int(piv.loc[lb, "train"]), int(piv.loc[lb, "val"])) if lb in piv.index else (0, 0)
        flag = "  <-- no validation positives (excluded from the selection score)" if va < MIN_VAL_POSITIVE_BINS else ""
        print(f"  {lb:22s} train {tr:9d}   val {va:7d}{flag}")


def get_split(args, audio_index: dict, df_anno: pd.DataFrame, verbose: bool = True):
    """v13: the split every script should use -- a precomputed one from
    --split-dir (train_files.txt/val_files.txt, see make_split.py) when given,
    else computed with split_files. Keeps training, evaluation and the plots
    on one identical split."""
    split_dir = getattr(args, "split_dir", None)
    if split_dir:
        def _read(n):
            with open(os.path.join(split_dir, n)) as f:
                return [l.strip() for l in f if l.strip()]
        tr = [s for s in _read("train_files.txt") if s in audio_index]
        va = [s for s in _read("val_files.txt") if s in audio_index]
        if verbose:
            print(f"split read from {split_dir}: {len(tr)} train / {len(va)} val files "
                  f"(of {len(audio_index)} audio files found)")
        return sorted(tr), sorted(va)
    return split_files(audio_index, val_fraction=getattr(args, "val_fraction", 0.30),
                       seed=getattr(args, "seed", 0), df_anno=df_anno, verbose=verbose)


MIN_CLASS_COUNT_WARNING = 20


def low_sample_classes(df_anno_train: pd.DataFrame, label_list, min_count: int = MIN_CLASS_COUNT_WARNING):
    """Classes with fewer than `min_count` annotated instances in the
    training split -- flagged because no amount of architecture/training
    tuning fixes a class with a handful of examples (e.g. Blue whale Z call,
    typically ~12 instances total); the fix is more annotations."""
    counts = df_anno_train["label"].value_counts()
    return [lb for lb in label_list if counts.get(lb, 0) < min_count]


def compute_window_class_presence(df_data: pd.DataFrame, df_anno: pd.DataFrame, label_list) -> np.ndarray:
    """For every window in df_data, whether each class has at least one
    annotation overlapping it -- pure metadata/interval-overlap computation
    (no audio decoding), used to build a rare-class-aware sampling weight.
    Returns a bool array of shape (len(df_data), len(label_list)).
    """
    presence = np.zeros((len(df_data), len(label_list)), dtype=bool)
    df_data = df_data.reset_index(drop=True)

    for stem, idx in df_data.groupby("stem").groups.items():
        idx = np.asarray(idx)
        w_start = df_data.loc[idx, "start_sec"].to_numpy()[:, None]
        w_stop = df_data.loc[idx, "stop_sec"].to_numpy()[:, None]

        df_f = df_anno[df_anno["stem"] == stem]
        if len(df_f) == 0:
            continue

        for c, lb in enumerate(label_list):
            rows = df_f[df_f["label"] == lb]
            if len(rows) == 0:
                continue
            a_start = rows["t_1_s"].to_numpy()[None, :]
            a_stop = rows["t_2_s"].to_numpy()[None, :]
            overlap = (w_start < a_stop) & (w_stop > a_start)  # (n_windows_in_stem, n_annotations)
            presence[idx, c] |= overlap.any(axis=1)

    return presence


# v10.2: 100 real background windows per split for augment_waveform's noise
# mixing (a small pool gives thin coverage of real background diversity).
# Default only -- see PamSequenceDataset's noise_pool_size / --noise-pool-size.
NOISE_POOL_SIZE = 100

# Shared shaped-dither constants for upsampling PAST a file's own native
# Nyquist -- see make_shaped_dither's docstring. RAMP_START_FRAC/REF_BAND_FRAC
# match convert_bluefin_library.py's own (approved via preview_bluefin_
# library.py's checkpoint round for THAT script's 500 Hz LF target) exactly,
# so calling make_shaped_dither with rolloff_db_per_oct=DITHER_ROLLOFF_DB_PER_OCT_LF
# reproduces that script's behavior byte-for-byte. v7's new joint model needs
# a SEPARATE, much steeper rolloff for its on-the-fly HF-target (32 kHz)
# upsampling of bluefin-sourced files (native rate <= 2048 Hz, so this can be
# a ~6-7 octave upsample instead of LF's ~1 octave one) -- reusing the LF
# constant there would leave an implausibly present noise floor most of the
# way to 16 kHz; see _spectrogram_for_subband's own comment for where this
# gets used.
RAMP_START_FRAC = 0.85
REF_BAND_FRAC = (0.6, 0.98)
DITHER_ROLLOFF_DB_PER_OCT_LF = 4.0    # matches convert_bluefin_library.py's LF-target dither exactly
DITHER_ROLLOFF_DB_PER_OCT_HF = 12.0   # v7: steeper, for the much larger LF-file -> HF-target upsample
NOISE_BOOST = 1.6


def _band_rms(y: np.ndarray, sr: int, f_lo: float, f_hi: float) -> float:
    f_lo = max(f_lo, 1.0)
    f_hi = min(f_hi, sr / 2.0 - 1.0)
    if f_hi <= f_lo:
        return float(np.sqrt(np.mean(y ** 2)))
    sos = butter(4, [f_lo, f_hi], btype="band", fs=sr, output="sos")
    filtered = sosfiltfilt(sos, y)
    return float(np.sqrt(np.mean(filtered ** 2)))


def make_shaped_dither(y_resampled: np.ndarray, sr_out: int, native_sr: int,
                        rng: np.random.Generator, rolloff_db_per_oct: float = DITHER_ROLLOFF_DB_PER_OCT_LF
                        ) -> np.ndarray:
    """Noise to fill in above a file's own native Nyquist limit, shaped to
    blend in (raised-cosine ramp-in at the native cutoff, then a per-octave
    rolloff above it, level-matched to the real signal's own noise floor
    just below cutoff) rather than read as a flat, independent noise floor
    or a suspicious hard silent edge -- ported from
    convert_bluefin_library.py's function of the same name (see its
    docstring, "Why the dither, v2", for the full reasoning), generalized
    here with an explicit `rolloff_db_per_oct` so a much larger upsample
    ratio (v7's on-the-fly resample to HF's 32 kHz, vs. that script's fixed
    500 Hz LF target) can use a steeper rolloff instead of leaving an
    implausible amount of energy present many octaves above where the
    recording could ever have had any. Returns an array the same length as
    y_resampled to add to it."""
    cutoff = native_sr / 2.0
    ref_rms = _band_rms(y_resampled, sr_out, REF_BAND_FRAC[0] * cutoff, REF_BAND_FRAC[1] * cutoff)

    n = len(y_resampled)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr_out)
    env = np.zeros_like(freqs)

    ramp_lo = RAMP_START_FRAC * cutoff
    in_ramp = (freqs >= ramp_lo) & (freqs <= cutoff)
    env[in_ramp] = 0.5 * (1 - np.cos(np.pi * (freqs[in_ramp] - ramp_lo) / (cutoff - ramp_lo)))

    above = freqs > cutoff
    octaves_above = np.log2(np.maximum(freqs[above], cutoff) / cutoff)
    env[above] = 10 ** (-rolloff_db_per_oct * octaves_above / 20.0)

    white = rng.normal(0.0, 1.0, size=n)
    shaped = np.fft.irfft(np.fft.rfft(white) * env, n=n)
    shaped_rms = np.sqrt(np.mean(shaped ** 2)) + 1e-12
    shaped = shaped * (ref_rms * NOISE_BOOST / shaped_rms)
    return shaped.astype("float32")


def _load_window_waveform(path: str, sr: int, start_sec: float, stop_sec: float) -> np.ndarray:
    """Load one window of a file as a 1D numpy array at `sr`. Used only to
    build the background noise pool (a handful of calls at dataset init),
    not on the hot __getitem__ path."""
    waveform, sr_org = _load_audio(path)
    if sr != sr_org:
        waveform = torchaudio.transforms.Resample(orig_freq=sr_org, new_freq=sr)(waveform)
    start_sample = int(start_sec * sr)
    stop_sample = int(stop_sec * sr)
    return waveform[:, start_sample:stop_sample].numpy().squeeze(0)


def augment_waveform(audio: np.ndarray, rng: np.random.Generator, noise_pool: list = None,
                     cfg: dict = None) -> np.ndarray:
    """Gain jitter + additive noise (mixed in from `noise_pool` -- real
    background windows from this training split -- when available, else
    plain Gaussian noise). Mirrors perch_training's
    perch_common.augment_clip, with one deliberate difference: no time
    shift. This pipeline computes a spectrogram over the *whole* recording
    and only slices out the annotated window afterward (see
    PamSequenceDataset.__getitem__); labels are rasterized against that same
    unshifted time axis. Shifting the waveform in time would desync it from
    its own labels, so this only uses time-invariant transforms (gain,
    additive noise), which are safe to apply to the whole waveform before
    slicing.

    v3: toned down from v2. The v2 LF run showed Blue whale D call -- a
    short, comparatively weak downsweep chirp -- regress on both precision
    and recall relative to v1 (no waveform augmentation), while longer/
    stronger calls held steady or improved; the leading suspect is this
    noise mixing burying short, weak calls a bit too often at a bit too
    high an intensity. Lowered the SNR range (noise now always at least
    6dB down instead of allowing near 0dB / equal-power mixing), lowered
    the probability of mixing in noise at all, and quietened the Gaussian
    fallback to match.
    """
    # `cfg` (optional) overrides the three knobs below; training uses the v3
    # conservative defaults (gain +-6 dB, noise 50% of the time at 6-20 dB SNR).
    cfg = cfg or {}
    g_lo, g_hi = cfg.get("gain_db", (-6.0, 6.0))
    p_noise = cfg.get("noise_prob", 0.5)
    snr_lo, snr_hi = cfg.get("snr_db", (6.0, 20.0))

    out = audio.copy()

    gain_db = rng.uniform(g_lo, g_hi)
    out = out * (10.0 ** (gain_db / 20.0))

    if noise_pool and rng.random() < p_noise:
        noise = noise_pool[rng.integers(len(noise_pool))]
        if len(noise) != len(out):
            reps = int(np.ceil(len(out) / len(noise)))
            noise = np.tile(noise, reps)[:len(out)]
        snr_db = rng.uniform(snr_lo, snr_hi)
        sig_power = np.mean(out ** 2) + 1e-12
        noise_power = np.mean(noise ** 2) + 1e-12
        target_noise_power = sig_power / (10.0 ** (snr_db / 10.0))
        noise = noise * np.sqrt(target_noise_power / noise_power)
    else:
        noise_amp = rng.uniform(0.0005, 0.005)
        noise = rng.normal(0.0, noise_amp, size=out.shape)

    out = out + noise
    return out.astype("float32")


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

def _duration_with_retries(path, tries=5):
    """sf.info(path).duration, retried with backoff: network drives (e.g. Z: on the analysis laptop) sporadically
    report existing files as missing / unreadable under load (v19)."""
    for k in range(tries):
        try:
            return sf.info(path).duration
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


class PamSequenceDataset(Dataset):
    """Cuts each recording into overlapping ``snippet_length_s`` windows and
    returns, per window, the window's raw audio (for CQTFrontend, which
    computes the spectrogram on the model's device) plus a (num_classes,
    timesteps) multi-hot label grid at ``label_interval_s`` resolution (and,
    with return_mask=True, the matching loss/metric mask). One dataset
    instance covers one split (a list of file stems).

    __getitem__ returns ({"wave": (1, n_total) float32, "aug": int64 scalar},
    labels[, mask]). "wave" is (snippet_length_s + 2 * WINDOW_LOAD_PAD_S) * sr
    samples with the window itself always at [pad * sr, (pad + snippet) * sr);
    "aug" is the number of SpecAugment stripes CQTFrontend draws for this
    window in training (0 for the clean copy of every window, 2 for the
    augmented copy -- see __init__).
    """

    # v8: per-worker (each DataLoader worker process gets its own copy of
    # this dataset, so this cache is NOT shared across workers) bounded FIFO
    # cache of decoded window waveforms -- each window appears twice in a
    # training epoch (clean + augmented copy), so the second access skips the
    # decode. Bounded so a large dataset can't blow up worker memory.
    _CACHE_MAX_ENTRIES = 64

    # Seconds of extra audio loaded on each side of a window's own
    # [start_sec, stop_sec): CQTFrontend computes ONE circular CQT over the
    # padded window, so wrap-around and the low bins' long windows only touch
    # this padding, which it crops off. Must equal BAND_CONFIG["cqt"]["cqt_pad_s"].
    WINDOW_LOAD_PAD_S = 2.0

    def __init__(self, file_stems, audio_index: dict, df_anno: pd.DataFrame,
                 band: str = "cqt", augment: bool = False, full_coverage: bool = False,
                 bluefin_hf_dir: str = None, noise_pool_size: int = NOISE_POOL_SIZE,
                 return_mask: bool = False, noisy_sources=None):
        assert band in BAND_CONFIG, f"unknown band {band!r} (v17 has only {list(BAND_CONFIG)})"
        self.band = band
        self.cfg = BAND_CONFIG[band]
        assert self.cfg["cqt_pad_s"] == self.WINDOW_LOAD_PAD_S, \
            "BAND_CONFIG['cqt']['cqt_pad_s'] must equal PamSequenceDataset.WINDOW_LOAD_PAD_S"
        self.model_labels = list(BAND_LABELS[band])
        self.num_classes = len(self.model_labels)
        # v13: when True, __getitem__ returns (x, labels, mask) -- mask is
        # (num_classes, timesteps), 1 = this class/bin contributes to loss and
        # metrics, 0 = masked (class not annotated for this file's source, or
        # an "uncertain" annotation covers the bin). Default False keeps the
        # (x, labels) contract inference/plot scripts rely on.
        self.return_mask = return_mask
        self.stem_scope = stem_scope_matrix(file_stems, audio_index, self.model_labels)
        # v19: windows of these sources (class_config source names, glob patterns) are flagged
        # x["noisy"] = 1 -> the noise-robust loss branch (WeightedCombinedLoss robust=...).
        self.noisy_stems = set()
        if noisy_sources:
            import fnmatch
            for st in file_stems:
                if any(fnmatch.fnmatch(CLASS_CFG.source_of(st, audio_index.get(st)), pat) for pat in noisy_sources):
                    self.noisy_stems.add(st)
            print(f"[{band}] noisy-label sources {list(noisy_sources)}: {len(self.noisy_stems)}/{len(file_stems)} files")
        self.augment = augment
        self.noise_pool_size = noise_pool_size

        # v8 (option C): a bluefin-sourced stem with a same-named file here
        # (precompute_bluefin_hf_cache.py: bluefin files pre-resampled+dithered
        # to 32 kHz once, offline) loads its audio from here instead of
        # live-resampling its 500 Hz file on every access. Not needed with
        # --training-data, where every file is already 32 kHz.
        self.bluefin_hf_index = build_audio_index([bluefin_hf_dir]) if bluefin_hf_dir else {}
        if bluefin_hf_dir:
            print(f"[{band}] bluefin_hf_dir={bluefin_hf_dir!r}: {len(self.bluefin_hf_index)} cached 32 kHz "
                  f"file(s) -- matching bluefin stems load from here instead of live-resampling")

        # v8 debug instrumentation, see DEBUG_TIMING's own module-level
        # comment -- None (and every timing check below a no-op) unless
        # PAM_DEBUG_TIMING=1 is set.
        self._timing = _TimingStats() if DEBUG_TIMING else None

        # (resolved path, tag, start_sec, stop_sec) -> (decoded/resampled/
        # dithered window waveform, pad_start_s, pad_stop_s)
        self._waveform_cache = OrderedDict()
        # v9.1: filenames we've already printed a truncated-file warning for
        # (see _load_padded_waveform) -- one print per file, not per window.
        self._truncated_file_warned = set()

        self.df_anno = df_anno[df_anno["stem"].isin(file_stems)].reset_index(drop=True)

        snippet_length = self.cfg["snippet_length_s"]
        overlap = self.cfg["snippet_overlap"]
        hop = snippet_length / overlap if overlap > 1 else snippet_length

        df_data = pd.DataFrame([])
        i = 0
        skipped_files = []
        for stem in file_stems:
            path = audio_index[stem]
            try:
                dur = _duration_with_retries(path)
            except Exception:
                try:
                    dur = librosa.get_duration(path=path)
                except Exception as exc:
                    # A file neither soundfile nor librosa/audioread can read the
                    # duration of (missing codec backend, corrupt header): skip it
                    # instead of letting one bad file crash the whole dataset build.
                    skipped_files.append(stem)
                    print(f"WARNING: skipping {path!r} -- duration unreadable by both soundfile and "
                          f"librosa/audioread ({exc!r}); file may be corrupt or in an unsupported format")
                    continue
            n_steps = max(1, int(np.floor((dur - snippet_length) / hop)) + 1) if dur > snippet_length else 1
            starts = np.arange(n_steps) * hop
            # The regular grid always starts at 0 but its last window can stop
            # short of the file's end. Deployment scoring (run_inference.py,
            # full_coverage=True) adds one extra window ending exactly at the
            # file's end; training/validation do not (it would add an extra,
            # differently-weighted window to the loss/threshold calibration).
            if full_coverage and dur > snippet_length:
                last_stop = starts[-1] + snippet_length
                if last_stop < dur - 1e-6:
                    starts = np.append(starts, dur - snippet_length)
            for start_sec in starts:
                df_data.loc[i, "filename"] = path
                df_data.loc[i, "stem"] = stem
                df_data.loc[i, "start_sec"] = start_sec
                df_data.loc[i, "stop_sec"] = start_sec + snippet_length
                i += 1
        if skipped_files:
            print(f"WARNING: {len(skipped_files)}/{len(file_stems)} file(s) skipped entirely "
                  f"(unreadable duration): {skipped_files[:20]}"
                  f"{' ... (truncated)' if len(skipped_files) > 20 else ''}")
        df_data["n_augment_stripes"] = 0

        # Training: every window twice -- once clean, once with waveform
        # augmentation (here) and 2 SpecAugment stripes (CQTFrontend).
        if augment:
            df_data_2 = df_data.copy()
            df_data_2["n_augment_stripes"] = 2
            df_data = pd.concat([df_data, df_data_2]).reset_index(drop=True)

        self.df_data = df_data
        self.num_samples = len(df_data)
        self.spec_dim = [self.cfg["freq_bins"], self.cfg["time_bins"]]
        self.label_interval = self.cfg["label_interval_s"]
        self.timesteps = int(snippet_length / self.label_interval) + 1

        # Per-window (num_samples, num_classes) bool: does each window overlap
        # an annotation of that class -- used by run_training's rare-class
        # aware WeightedRandomSampler.
        self.class_presence = compute_window_class_presence(self.df_data, self.df_anno, self.model_labels)

        # v2: real background windows for augment_waveform's noise mixing,
        # sampled once here with a fixed seed (same pool run-to-run per split).
        self.noise_pool = []
        if augment:
            bg_idx = np.where(~self.class_presence.any(axis=1))[0]
            pool_rng = np.random.default_rng(0)
            pick = pool_rng.choice(bg_idx, size=min(self.noise_pool_size, len(bg_idx)), replace=False) \
                if len(bg_idx) else []
            for i in pick:
                r = self.df_data.loc[i]
                try:
                    self.noise_pool.append(
                        _load_window_waveform(r["filename"], self.cfg["sample_rate"], r["start_sec"], r["stop_sec"])
                    )
                except Exception as e:
                    print(f"WARNING: couldn't load noise-pool window from {r['filename']}: {e}")
            print(f"[{band}] noise pool: {len(self.noise_pool)} background windows")

        print(f"[{band}] {self.num_samples} windows from {len(file_stems)} files")

    def __len__(self):
        return self.num_samples

    def _cache_put(self, cache: OrderedDict, key, value):
        """FIFO-bounded insert (oldest-inserted first out)."""
        cache[key] = value
        if len(cache) > self._CACHE_MAX_ENTRIES:
            cache.popitem(last=False)

    def _load_padded_waveform(self, filename, sr, start_sec, stop_sec, cache_key=None, min_len=1):
        """v17: the decode -> (resample + shaped dither) -> silence-pad part of
        _spectrogram_for_subband, split out unchanged so band="cqt" can reuse it
        without computing a spectrogram here. Returns (waveform (1, n) at `sr`,
        pad_start_s, pad_stop_s): the slice [start_sec - WINDOW_LOAD_PAD_S,
        stop_sec + WINDOW_LOAD_PAD_S] clamped to the file, cached per window in
        self._waveform_cache under `cache_key` (already including start/stop)."""
        timing = self._timing
        cached = self._waveform_cache.get(cache_key) if cache_key is not None else None
        if cached is not None:
            waveform, pad_start_s, pad_stop_s = cached
        else:
            t0 = time.perf_counter() if timing is not None else None
            # _audio_info reads only the header (sample_rate/
            # num_frames), not the audio itself -- needed to convert
            # the padded window's seconds into native-rate sample
            # offsets before loading, and to clamp the padding at the
            # file's own true end.
            sr_org, total_frames = _audio_info(filename)
            file_duration_s = total_frames / sr_org
            pad_start_s = max(0.0, start_sec - self.WINDOW_LOAD_PAD_S)
            pad_stop_s = min(file_duration_s, stop_sec + self.WINDOW_LOAD_PAD_S)
            frame_offset = int(round(pad_start_s * sr_org))
            num_frames = max(1, int(round((pad_stop_s - pad_start_s) * sr_org)))
            waveform, _ = _load_audio(filename, frame_offset=frame_offset, num_frames=num_frames)
            if waveform.shape[-1] < num_frames:
                # Real field-recorder files occasionally have a WAV
                # header whose frame count overstates what's actually
                # on disk (e.g. the mooring lost power or filled its
                # disk mid-write) -- sf.info()'s header-reported
                # duration then overstates the true, readable
                # duration, so a window/pad computed from it (both
                # here and in __init__'s window grid, which uses the
                # same sf.info().duration) can land at or past the
                # real end of the data chunk, where sf.read() returns
                # fewer frames than requested -- anywhere from a
                # FULLY empty read (frame_offset already past the
                # real end of data) down to just a handful of
                # leftover samples (frame_offset lands just before
                # it). Either way, padding the shortfall with
                # trailing silence keeps a many-thousand-file
                # unattended run going instead of crashing the whole
                # job over one truncated file's tail.
                if filename not in self._truncated_file_warned:
                    self._truncated_file_warned.add(filename)
                    print(f"WARNING: {filename} returned {waveform.shape[-1]}/{num_frames} requested "
                          f"audio frames at sample offset {frame_offset} (file may be truncated/"
                          f"corrupted -- header duration overstates what's actually on disk); "
                          f"padding the shortfall with silence instead of aborting the run")
                pad_amount = num_frames - waveform.shape[-1]
                waveform = torch.cat([waveform, torch.zeros(1, pad_amount, dtype=waveform.dtype)], dim=-1)
            if timing is not None:
                timing.add("decode", time.perf_counter() - t0)
            if sr != sr_org:
                t0 = time.perf_counter() if timing is not None else None
                waveform = torchaudio.transforms.Resample(orig_freq=sr_org, new_freq=sr)(waveform)
                if sr > sr_org:
                    # Upsampling past this file's own native Nyquist --
                    # fill the newly-opened band with shaped dither (see
                    # make_shaped_dither's docstring) instead of a
                    # suspiciously clean, hard-silent edge the model
                    # could learn to associate with these files
                    # specifically rather than the real acoustic
                    # content. This is the case a bluefin-derived file
                    # (already resampled+dithered to 500 Hz by
                    # convert_bluefin_library.py) hits when loaded for
                    # the joint model's HF sub-band (target 32 kHz) -- a
                    # ~6-7 octave upsample, so the steeper HF rolloff
                    # constant is used once the ratio is clearly in that
                    # regime (>=4x); a small-ratio upsample (the kind
                    # convert_bluefin_library.py's own LF-target
                    # conversion already handles offline) gets that
                    # script's own, gentler, already-approved constant
                    # instead. Skipped entirely when the caller already
                    # resolved `filename` to a precomputed 32 kHz cache
                    # (see PamSequenceDataset's bluefin_hf_index /
                    # precompute_bluefin_hf_cache.py, option C) -- there,
                    # sr_org == sr already, so this whole branch is a
                    # no-op and the live cost it describes never happens.
                    # NOTE (v8.1): the dither draw is independent per
                    # window now rather than shared across a whole
                    # file's overlapping windows -- harmless, since this
                    # noise is synthetic filler standing in for "no real
                    # information" either way, not real signal that
                    # needs to stay consistent across nearby windows.
                    rolloff = DITHER_ROLLOFF_DB_PER_OCT_HF if sr >= 4 * sr_org else DITHER_ROLLOFF_DB_PER_OCT_LF
                    dither = make_shaped_dither(waveform.numpy().squeeze(0), sr, sr_org,
                                                 np.random.default_rng(), rolloff)
                    waveform = waveform + torch.from_numpy(dither).unsqueeze(0)
                if timing is not None:
                    timing.add("resample_dither", time.perf_counter() - t0)

            # Final belt-and-suspenders floor, independent of the
            # truncated-file padding above: torch.stft's default
            # center=True reflect-pads by fft_size//2 on each side,
            # which itself raises if the input isn't longer than that
            # pad -- a real risk any time a window's own true span
            # (not just a truncated file's missing tail) is shorter
            # than the sub-band's own FFT window, e.g. a genuinely
            # very short recording file. Padding with silence up to
            # fft_size here (rather than letting _compute_spectrogram
            # crash on it) trades one nearly-empty window's score for
            # the run's ability to keep going.
            if waveform.shape[-1] < min_len:
                waveform = torch.cat(
                    [waveform, torch.zeros(1, min_len - waveform.shape[-1], dtype=waveform.dtype)], dim=-1)

            if cache_key is not None:
                self._cache_put(self._waveform_cache, cache_key, (waveform, pad_start_s, pad_stop_s))
        return waveform, pad_start_s, pad_stop_s

    def _cqt_window_waveform(self, filename, start_sec, stop_sec, n_augment, cache_key=None):
        """v17, band="cqt": the window's audio as a FIXED-length (1, (snippet_length_s
        + 2 * cqt_pad_s) * sr) float32 tensor with the window itself always at samples
        [pad * sr, (pad + snippet_length_s) * sr) -- CQTFrontend crops it by fixed
        indices on the GPU. Padding that falls outside the file (a file's first/last
        window, or a file shorter than the window) is zeros. Only the waveform-domain
        augmentation (gain + noise mixing, augment_waveform) runs here, on the CPU
        worker, because it needs this split's noise pool; the spectrogram, resizing,
        normalisation and SpecAugment all run in CQTFrontend."""
        cfg = self.cfg
        sr = cfg["sample_rate"]
        pad = self.WINDOW_LOAD_PAD_S
        timing = self._timing
        if cache_key is not None:
            cache_key = (*cache_key, round(start_sec, 3), round(stop_sec, 3))
        waveform, pad_start_s, _ = self._load_padded_waveform(filename, sr, start_sec, stop_sec, cache_key=cache_key)
        if n_augment > 0:
            t0 = time.perf_counter() if timing is not None else None
            augmented = augment_waveform(waveform.numpy().squeeze(0), np.random.default_rng(),
                                         self.noise_pool)
            waveform = torch.from_numpy(augmented).unsqueeze(0)
            if timing is not None:
                timing.add("augment", time.perf_counter() - t0)
        n_total = int(round((cfg["snippet_length_s"] + 2 * pad) * sr))
        offset = int(round((pad_start_s - (start_sec - pad)) * sr))   # > 0 only when the left pad was clamped at 0 s
        out = torch.zeros(1, n_total, dtype=torch.float32)
        n = max(0, min(waveform.shape[-1], n_total - offset))
        out[:, offset:offset + n] = waveform[:, :n]
        return out

    def __getitem__(self, idx):
        item_t0 = time.perf_counter() if self._timing is not None else None
        row = self.df_data.loc[idx]
        filename = row["filename"]
        stem = row["stem"]
        n_augment = int(row["n_augment_stripes"])

        # bluefin stems: the precomputed 32 kHz copy if there is one (see __init__)
        wave_path = self.bluefin_hf_index.get(stem, filename)
        x = {"wave": self._cqt_window_waveform(wave_path, row["start_sec"], row["stop_sec"], n_augment,
                                               cache_key=(wave_path, "cqt")),
             "aug": torch.tensor(n_augment, dtype=torch.int64),
             "noisy": torch.tensor(1.0 if stem in self.noisy_stems else 0.0)}

        label_interval = self.label_interval
        t = np.arange(row["start_sec"], row["stop_sec"] + label_interval, label_interval)

        stem = row["stem"]
        df_f = self.df_anno[self.df_anno["stem"] == stem]
        inside = (df_f["t_1_s"] >= row["start_sec"]) & (df_f["t_2_s"] <= row["stop_sec"])
        start_inside = (df_f["t_1_s"] >= row["start_sec"]) & (df_f["t_1_s"] <= row["stop_sec"]) & (df_f["t_2_s"] >= row["stop_sec"])
        end_inside = (df_f["t_1_s"] <= row["start_sec"]) & (df_f["t_2_s"] <= row["stop_sec"]) & (df_f["t_2_s"] >= row["start_sec"])
        df_anno_file = df_f[inside | start_inside | end_inside].reset_index(drop=True)

        score = np.zeros([len(t), self.num_classes], dtype=bool)
        for ix_class, lb in enumerate(self.model_labels):
            rows = df_anno_file[df_anno_file["label"] == lb]
            for _, r in rows.iterrows():
                b1 = np.digitize(r["t_1_s"], t) - 1
                b2 = np.digitize(r["t_2_s"], t)
                b1 = max(b1, 0)
                if b1 == b2:
                    score[b1, ix_class] = 1
                else:
                    score[b1:b2, ix_class] = 1

        labels = torch.from_numpy(score.T.astype("float32"))

        mask = None
        if self.return_mask:
            keep = self.stem_scope.get(stem)
            mask_np = np.ones((self.num_classes, len(t)), dtype="float32")
            if keep is not None:
                mask_np[~keep, :] = 0.0
            if "mask_label" in df_anno_file.columns:
                unc = df_anno_file[df_anno_file["mask_label"].notna()]
                for _, r in unc.iterrows():
                    if r["mask_label"] not in self.model_labels:
                        continue
                    ix_class = self.model_labels.index(r["mask_label"])
                    b1 = max(np.digitize(r["t_1_s"], t) - 1, 0)
                    b2 = np.digitize(r["t_2_s"], t)
                    mask_np[ix_class, b1:max(b2, b1 + 1)] = 0.0
            mask = torch.from_numpy(mask_np)

        if self._timing is not None:
            self._timing.add("__getitem__ total", time.perf_counter() - item_t0)
            self._timing.maybe_print(self.band)

        if self.return_mask:
            return x, labels, mask
        return x, labels


def compute_class_weights(df_anno_train: pd.DataFrame, file_stems, audio_index: dict,
                           label_list, snippet_length_s: float) -> torch.Tensor:
    """Inverse-frequency class weights from annotated seconds per class,
    normalized to sum to num_classes (same formula the original scripts used,
    just computed from annotation durations instead of a slow full dataset
    pass).

    v13: scope-aware. A class's "total duration" is only the audio of files
    whose source was actually annotated for that class (a class masked for
    a file contributes neither positives nor negatives there), so adding a
    big single-class dataset doesn't dilute every other class's weight."""
    scope = stem_scope_matrix(file_stems, audio_index, label_list)
    dur_per_class = np.zeros(len(label_list))
    for stem in file_stems:
        try:
            d = sf.info(audio_index[stem]).duration
        except Exception:
            d = librosa.get_duration(path=audio_index[stem])
        dur_per_class += d * scope[stem]

    class_seconds = np.zeros(len(label_list))
    for i, lb in enumerate(label_list):
        rows = df_anno_train[df_anno_train["label"] == lb]
        class_seconds[i] = (rows["t_2_s"] - rows["t_1_s"]).clip(lower=0).sum()

    weights = np.maximum(dur_per_class, 1.0) / (len(label_list) * (class_seconds + 1))
    weights = weights * len(label_list) / weights.sum()
    return torch.tensor(weights, dtype=torch.float32)


# --------------------------------------------------------------------------
# v17: CQT frontend (runs inside the model, i.e. on the GPU)
# --------------------------------------------------------------------------

def cqt_kernels(num_octaves: int, bins_per_octave: int, sample_rate: int, block_length: int,
                gamma_hz: float = 0.0):
    """Frequency-domain windows of a nonstationary-Gabor CQT, built exactly as
    cqt-pytorch's CQT.__init__ builds them (archinetai/cqt-pytorch, MIT licence:
    get_center_frequencies / get_bandwidths / get_windows_range_idx / get_windows),
    plus one addition: `gamma_hz` is added to every band's width (variable-Q
    transform, Schoerkhuber et al. 2014; librosa.vqt's `gamma`). Only the
    positive-frequency bins are built, which is all CQT.encode uses.
    Returns (freqs (K,), windows_range_idx (K, L) long, windows (K, L) float32),
    K = num_octaves * bins_per_octave, L = the longest window."""
    nyq = sample_rate / 2
    K = num_octaves * bins_per_octave
    freqs = nyq / (2 ** num_octaves) * (2 ** (torch.arange(K) / bins_per_octave))
    freqs_all = torch.cat([freqs, torch.tensor([nyq]), sample_rate - torch.flip(freqs, dims=[0])])
    q = 1.0 / (2 ** (1.0 / bins_per_octave) - 2 ** (-1.0 / bins_per_octave))
    bandwidths = freqs_all[1:K + 1] / q + gamma_hz     # cqt-pytorch's indexing: bin k uses centre k+1
    lengths = torch.round(bandwidths * block_length / sample_rate)
    max_len = int(lengths.max())
    positions = torch.round(freqs * block_length / sample_rate)
    idx = torch.stack([torch.arange(start=p - max_len // 2, end=p - max_len // 2 + max_len)
                       for p in positions]).long() % block_length
    windows = []
    for n in lengths:
        pad_left = int(np.floor(max_len / 2 - n / 2))
        pad_right = int(max_len - n - pad_left)
        windows.append(F.pad(torch.hann_window(int(n)), pad=(pad_left, pad_right)))
    return freqs, idx, torch.stack(windows)


class CQTFrontend(nn.Module):
    """v17: (B, 1, N) raw window audio -> (B, 1, freq_bins, time_bins) model input,
    entirely in torch on the model's device: one variable-Q CQT over the whole
    padded window (one FFT block, so no block-edge artefacts) -> |c|^2 -> crop
    the padding -> average power into time_bins -> log10 -> bilinear resize of
    the K log-spaced bins to freq_bins -> per-window min-max to [0, 1] ->
    SpecAugment (training only, windows with aug > 0, same stripe rules as
    _apply_specaugment). No learnable parameters, run without autograd and
    outside autocast (FFTs stay float32). The CQT kernels are non-persistent
    buffers, rebuilt from the config, so they never enter a checkpoint."""

    def __init__(self, cfg: dict):
        super().__init__()
        sr = cfg["sample_rate"]
        self.pad_s = cfg["cqt_pad_s"]
        self.win_s = cfg["snippet_length_s"]
        total_s = self.win_s + 2 * self.pad_s
        self.block = int(round(total_s * sr))
        self.freq_bins, self.time_bins = cfg["freq_bins"], cfg["time_bins"]
        self.chunk = int(cfg.get("cqt_chunk", 4))
        self.spec_augment = cfg.get("spec_augment", True)
        self.max_time_mask_frac = cfg.get("max_time_mask_frac", 0.10)
        self.max_freq_mask_frac = cfg.get("max_freq_mask_frac", 0.10)
        freqs, idx, windows = cqt_kernels(cfg["cqt_octaves"], cfg["cqt_bins_per_octave"], sr, self.block,
                                          cfg.get("cqt_gamma_hz", 0.0))
        self.register_buffer("freqs", freqs, persistent=False)
        self.register_buffer("windows_range_idx", idx, persistent=False)
        self.register_buffer("windows", windows, persistent=False)
        # CQT output sample i of the block is at time i * total_s / L
        L = idx.shape[1]
        self.t0 = int(round(self.pad_s / total_s * L))
        self.t1 = int(round((self.pad_s + self.win_s) / total_s * L))

    def cqt_power(self, wave: torch.Tensor) -> torch.Tensor:
        """(B, N) float32 -> (B, K, time_bins) CQT power of the window region."""
        out = []
        for w in wave.split(self.chunk):
            crops = torch.fft.fft(w)[:, self.windows_range_idx] * self.windows   # (b, K, L)
            c = torch.fft.ifft(crops)
            p = (c.real ** 2 + c.imag ** 2)[..., self.t0:self.t1]
            out.append(F.adaptive_avg_pool1d(p, self.time_bins))
        return torch.cat(out)

    def _spec_augment(self, x: torch.Tensor, aug: torch.Tensor) -> torch.Tensor:
        drop_t = int(self.time_bins * self.max_time_mask_frac)
        drop_f = int(self.freq_bins * self.max_freq_mask_frac)
        for b, n in enumerate(aug.tolist()):
            for _ in range(int(n)):
                if torch.randint(0, 2, (1,)).item() == 0:
                    d = torch.randint(0, max(drop_t, 1), (1,)).item()
                    s = torch.randint(0, self.time_bins - d, (1,)).item()
                    x[b, :, :, s:s + d] = 0
                else:
                    d = torch.randint(0, max(drop_f, 1), (1,)).item()
                    s = torch.randint(0, self.freq_bins - d, (1,)).item()
                    x[b, :, s:s + d, :] = 0
        return x

    @torch.no_grad()
    def forward(self, wave: torch.Tensor, aug: torch.Tensor = None) -> torch.Tensor:
        with torch.autocast(device_type=wave.device.type, enabled=False):
            wave = wave.float().reshape(wave.shape[0], -1)
            if wave.shape[1] != self.block:
                raise ValueError(f"CQTFrontend expects {self.block} samples per window, got {wave.shape[1]}")
            x = torch.log10(self.cqt_power(wave) + 1e-12).unsqueeze(1)          # (B, 1, K, T)
            x = F.interpolate(x, size=(self.freq_bins, self.time_bins), mode="bilinear", align_corners=False)
            lo = x.amin(dim=(1, 2, 3), keepdim=True)
            hi = x.amax(dim=(1, 2, 3), keepdim=True)
            x = (x - lo) / (hi - lo + 1e-12)
            if self.training and self.spec_augment and aug is not None:
                x = self._spec_augment(x, aug)
        return x


def cqt_spectrogram_db(waveform: torch.Tensor, cfg: dict = None, device=None):
    """Display spectrogram of arbitrary-length audio (already at cfg's sample
    rate), computed the way the model sees it: consecutive snippet_length_s
    windows, each with cqt_pad_s of neighbouring audio (zeros at the file
    edges) on both sides, through CQTFrontend.cqt_power -- but without the
    resize to freq_bins and the min-max. Returns (spec_db (1, K, T) tensor in
    dB, freqs (K,) numpy array in Hz); T = time_bins columns per window,
    trimmed to the audio's duration. Used by the plot scripts."""
    cfg = cfg or BAND_CONFIG["cqt"]
    fe = CQTFrontend(cfg).to(device or "cpu")
    sr = cfg["sample_rate"]
    win = int(round(cfg["snippet_length_s"] * sr))
    pad = int(round(cfg["cqt_pad_s"] * sr))
    x = waveform.reshape(-1).float()
    n_win = max(1, int(np.ceil(len(x) / win)))
    padded = F.pad(x, (pad, pad + n_win * win - len(x)))
    windows = torch.stack([padded[i * win: i * win + win + 2 * pad] for i in range(n_win)])
    with torch.no_grad():
        p = fe.cqt_power(windows.to(fe.windows.device)).cpu()        # (n_win, K, time_bins)
    p = p.permute(1, 0, 2).reshape(p.shape[1], -1)
    n_cols = max(1, int(round(len(x) / win * cfg["time_bins"])))
    return (10.0 * torch.log10(p[:, :n_cols] + 1e-12)).unsqueeze(0), fe.freqs.cpu().numpy()


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

class ImprovedCNNBiGRU(nn.Module):
    """v17: CQTFrontend -> ONE CNN stem -> BiGRU (+ optional attention) ->
    per-timestep classifier. Takes the dataset's {"wave", "aug"} batch and
    returns (B, num_classes, timesteps) logits. `frontend_cfg` is the band
    config the model is trained/loaded with (a checkpoint's "band_config";
    BAND_CONFIG["cqt"] when not given)."""

    @staticmethod
    def _build_cnn_stem(in_channels, pooled_freq_rows, timesteps):
        return nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(7, 7), padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(0.1),

            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(0.1),

            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Dropout2d(0.2),

            nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # pools whatever (freq, time) size the conv stack produces to
            # (pooled_freq_rows, timesteps) -- the model is input-size agnostic
            nn.AdaptiveAvgPool2d((pooled_freq_rows, timesteps)),
        )

    def __init__(self, num_classes=11, timesteps=121, hidden_size=512, gru_layers=3,
                 use_attention=True, pooled_freq_rows=48, frontend_cfg=None):
        super().__init__()
        self.use_attention = use_attention
        self.pooled_freq_rows = pooled_freq_rows
        self.frontend = CQTFrontend(frontend_cfg or BAND_CONFIG["cqt"])
        self.cnn = self._build_cnn_stem(1, pooled_freq_rows, timesteps)

        self.gru = nn.GRU(
            input_size=512 * pooled_freq_rows,
            hidden_size=hidden_size,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if gru_layers > 1 else 0.0,
        )
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size * 2, num_heads=8, dropout=0.2, batch_first=True,
            )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        x = self.frontend(x["wave"], x.get("aug"))                  # (B, 1, freq_bins, time_bins)
        feat = self.cnn(x).permute(0, 3, 1, 2)                       # (B, timesteps, 512, pooled_freq_rows)
        x = feat.reshape(feat.size(0), feat.size(1), -1)
        gru_out, _ = self.gru(x)
        if self.use_attention:
            attn_out, _ = self.attention(gru_out, gru_out, gru_out)
            x = gru_out + attn_out
        else:
            x = gru_out
        return self.classifier(x).permute(0, 2, 1)


def build_model(band_config: dict, num_classes: int, use_attention: bool) -> "ImprovedCNNBiGRU":
    """The one place a model is constructed from a band config (run_training
    with BAND_CONFIG["cqt"]; eval/inference/plot scripts with a checkpoint's
    saved "band_config")."""
    return ImprovedCNNBiGRU(
        num_classes=num_classes,
        timesteps=int(band_config["snippet_length_s"] / band_config["label_interval_s"]) + 1,
        hidden_size=band_config["hidden_size"],
        gru_layers=band_config.get("gru_layers", 3),
        use_attention=use_attention,
        pooled_freq_rows=band_config.get("pooled_freq_rows", 48),
        frontend_cfg=band_config,
    )


def load_model_from_checkpoint(checkpoint: dict, device) -> "ImprovedCNNBiGRU":
    """Rebuild + load a v17 checkpoint (architecture and label list as saved)."""
    if checkpoint.get("fusion_mode", "cqt") != "cqt" or "cqt_octaves" not in checkpoint.get("band_config", {}):
        raise SystemExit("ERROR: not a v17 (CQT single-tower) checkpoint -- use the v13 code for "
                         "joint/twotower checkpoints.")
    model = build_model(checkpoint["band_config"], len(checkpoint["model_labels"]),
                        checkpoint.get("use_attention", False)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval()


# --------------------------------------------------------------------------
# Loss + metrics
# --------------------------------------------------------------------------

class WeightedFocalLoss(nn.Module):
    def __init__(self, class_weights=None, alpha=1, gamma=2, reduction="mean"):
        super().__init__()
        self.class_weights = class_weights
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets, mask=None):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.class_weights is not None:
            weights = self.class_weights.view(1, -1, 1).to(inputs.device)
            focal_loss = focal_loss * weights

        # v13: mask (same shape as targets, 1 = counts, 0 = ignored) -- the
        # masked mean divides by the number of COUNTED elements, so with an
        # all-ones mask this is exactly the old .mean().
        if mask is not None:
            mask = mask.to(inputs.device, dtype=focal_loss.dtype)
            focal_loss = focal_loss * mask
            if self.reduction == "mean":
                return focal_loss.sum() / mask.sum().clamp_min(1.0)

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


ROBUST_LOSSES = ("none", "bootstrap", "gce")


class WeightedCombinedLoss(nn.Module):
    """alpha * focal + beta * BCE, both class-weighted and masked (masked mean over counted
    elements). v19: an optional noise-robust variant for windows flagged as noisy (forward's
    `noisy`, (B,) 0/1 -- see PamSequenceDataset noisy_sources):
      robust="bootstrap"  soft bootstrapping (Reed et al. 2015): the target of a noisy window becomes
                          robust_beta * label + (1 - robust_beta) * sigmoid(logit).detach(), so the
                          model's own confident opinion can partly overrule a wrong label;
      robust="gce"        generalized cross-entropy (Zhang & Sabuncu 2018) per element,
                          (1 - p_t^q) / q with p_t the probability of the given label: bounded, so a
                          confidently contradicted (probably wrong) label can't dominate the gradient.
    Clean windows (noisy 0) and robust="none" give exactly the v18 loss."""

    def __init__(self, class_weights=None, alpha=0.7, beta=0.3, focal_gamma=2,
                 robust="none", robust_beta=0.8, robust_q=0.7):
        super().__init__()
        assert robust in ROBUST_LOSSES, f"robust must be one of {ROBUST_LOSSES}, got {robust!r}"
        self.alpha = alpha
        self.beta = beta
        self.focal_loss = WeightedFocalLoss(class_weights=class_weights, gamma=focal_gamma)
        self.class_weights = class_weights
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none" if class_weights is not None else "mean")
        self.robust = robust
        self.robust_beta = robust_beta
        self.robust_q = robust_q

    def _elementwise(self, inputs, targets):
        """alpha * focal + beta * BCE per element, class-weighted (no reduction)."""
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce)
        el = self.alpha * self.focal_loss.alpha * (1 - pt) ** self.focal_loss.gamma * bce + self.beta * bce
        if self.class_weights is not None:
            el = el * self.class_weights.view(1, -1, 1).to(inputs.device)
        return el

    def _robust_forward(self, inputs, targets, mask, noisy):
        el = self._elementwise(inputs, targets)
        noisy_el = noisy.to(inputs.device).view(-1, 1, 1).bool().expand_as(el)
        if self.robust == "bootstrap":
            soft = self.robust_beta * targets + (1 - self.robust_beta) * torch.sigmoid(inputs).detach()
            el = torch.where(noisy_el, self._elementwise(inputs, soft), el)
        else:  # gce
            prob = torch.sigmoid(inputs.float())
            p_t = torch.where(targets > 0.5, prob, 1 - prob).clamp_min(1e-7)
            gce = (1 - p_t ** self.robust_q) / self.robust_q
            if self.class_weights is not None:
                gce = gce * self.class_weights.view(1, -1, 1).to(inputs.device)
            el = torch.where(noisy_el, gce.to(el.dtype), el)
        if mask is None:
            return el.mean()
        m = mask.to(inputs.device, dtype=el.dtype)
        return (el * m).sum() / m.sum().clamp_min(1.0)

    def forward(self, inputs, targets, mask=None, noisy=None):
        if self.robust != "none" and noisy is not None and bool((noisy > 0).any()):
            return self._robust_forward(inputs, targets, mask, noisy)
        focal = self.focal_loss(inputs, targets, mask)
        if self.class_weights is not None or mask is not None:
            bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
            if self.class_weights is not None:
                bce = bce * self.class_weights.view(1, -1, 1).to(inputs.device)
            if mask is not None:
                m = mask.to(inputs.device, dtype=bce.dtype)
                bce = (bce * m).sum() / m.sum().clamp_min(1.0)
            else:
                bce = bce.mean()
        else:
            bce = self.bce_loss(inputs, targets)
        return self.alpha * focal + self.beta * bce


def compute_class_wise_f1(preds, targets, threshold=0.5, num_classes=5, masks=None):
    """Frame-level F1 per class at a fixed threshold. v13: `masks` (same shape
    as targets) excludes masked frames from TP/FP/FN, and the macro average is
    over classes that have at least one positive among the counted frames
    (a class with no positives in this split can't score, and must not drag
    the mean down as a hard 0)."""
    preds_bin = (torch.sigmoid(preds) > threshold).float()
    f1_scores, supported = [], []
    for class_idx in range(num_classes):
        pred_class = preds_bin[:, class_idx, :].flatten()
        target_class = targets[:, class_idx, :].flatten()
        if masks is not None:
            keep = masks[:, class_idx, :].flatten() > 0.5
            pred_class, target_class = pred_class[keep], target_class[keep]

        tp = (pred_class * target_class).sum().item()
        fp = (pred_class * (1 - target_class)).sum().item()
        fn = ((1 - pred_class) * target_class).sum().item()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_scores.append(f1)
        supported.append(float(target_class.sum().item()) > 0)

    if masks is None:
        return float(np.mean(f1_scores)), f1_scores
    sup = [f for f, s in zip(f1_scores, supported) if s]
    return (float(np.mean(sup)) if sup else 0.0), f1_scores


MIN_VAL_POSITIVE_BINS = 20   # a class needs this many positive 0.5 s bins in val to enter the macro score


def masked_class_metrics(logits, targets, masks=None, threshold=0.5, min_pos=None):
    """v13 validation metrics, computed ONLY on frames where the class is in
    scope (masks == 1):

      ap      -- per-class average precision (threshold-free, robust to the
                 extreme class imbalance); NaN for a class with fewer than
                 `min_pos` positive bins (too few to score reliably).
      f1      -- per-class F1 at `threshold` (kept for continuity with v12).
      macro_ap / macro_f1 -- means over the supported classes only.

    macro_ap is the model-selection / LR-scheduler / early-stopping score in
    run_training (v12 used the unweighted mean F1@0.5 over all classes)."""
    from sklearn.metrics import average_precision_score
    min_pos = MIN_VAL_POSITIVE_BINS if min_pos is None else min_pos
    probs = torch.sigmoid(logits)
    C = logits.shape[1]
    ap, f1, n_pos, n_valid = [], [], [], []
    for c in range(C):
        y = targets[:, c, :].reshape(-1)
        p = probs[:, c, :].reshape(-1)
        if masks is not None:
            keep = masks[:, c, :].reshape(-1) > 0.5
            y, p = y[keep], p[keep]
        npos = int((y > 0.5).sum().item())
        n_pos.append(npos)
        n_valid.append(int(y.numel()))
        pred = p > threshold
        pos = y > 0.5
        tp = float((pred & pos).sum().item())
        fp = float((pred & ~pos).sum().item())
        fn = float((~pred & pos).sum().item())
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1.append(2 * prec * rec / (prec + rec + 1e-8))
        if npos >= min_pos and npos < y.numel():
            ap.append(float(average_precision_score(pos.numpy().astype(np.uint8), p.numpy())))
        else:
            ap.append(float("nan"))
    ap_arr = np.array(ap, dtype=float)
    sup_f1 = [f for f, n in zip(f1, n_pos) if n > 0]
    return {
        "ap": ap, "f1": f1, "n_pos": n_pos, "n_valid": n_valid,
        "macro_ap": float(np.nanmean(ap_arr)) if np.isfinite(ap_arr).any() else 0.0,
        "macro_f1": float(np.mean(sup_f1)) if sup_f1 else 0.0,
    }


def smooth_binary_predictions(binary_seq: np.ndarray, min_duration_bins: int = 1,
                               max_gap_bins: int = 0) -> np.ndarray:
    """Clean up a per-class binary frame sequence for deployment/inference
    output: bridges gaps of up to `max_gap_bins` between positive runs
    (closes single-frame flicker holes inside a call), then drops any
    positive run shorter than `min_duration_bins` (kills single-frame
    flicker false positives). Operates on a 1D array of 0/1 per class.

    Deliberately NOT used in predict_eval.py's PR-curve / frame-F1
    computation -- that metric is being kept as plain per-frame, per-0.5s-bin
    F1 for now. This is for cleaning up predictions at actual deployment
    time (e.g. a future apply-to-new-recordings script), where "one isolated
    frame flips on for 0.5s in the middle of nothing" is obviously noise but
    still counts against frame-level F1 either way.
    """
    seq = binary_seq.astype(bool).copy()

    if max_gap_bins > 0:
        i = 0
        n = len(seq)
        while i < n:
            if seq[i]:
                i += 1
                continue
            j = i
            while j < n and not seq[j]:
                j += 1
            gap = j - i
            if gap <= max_gap_bins and i > 0 and j < n and seq[i - 1] and seq[j]:
                seq[i:j] = True
            i = j

    if min_duration_bins > 1:
        i = 0
        n = len(seq)
        while i < n:
            if not seq[i]:
                i += 1
                continue
            j = i
            while j < n and seq[j]:
                j += 1
            if (j - i) < min_duration_bins:
                seq[i:j] = False
            i = j

    return seq.astype(binary_seq.dtype)


# --------------------------------------------------------------------------
# Event-level scoring (v3)
#
# Frame-level F1 (compute_class_wise_f1 above) scores every 0.5s bin
# independently. That's sensitive to boundary noise a call detector user
# doesn't actually care about: a prediction that's shifted by one bin from
# the annotated boundary is functionally a correct detection but gets
# punished at the frame level. Event-level scoring instead asks "did we
# detect this call at all" -- run-length-encode both the thresholded
# predictions and the ground truth into (start_s, stop_s) events, then
# match predicted events to true events by overlap.
#
# This requires a *continuous* per-file timeline, not the raw per-window
# predictions used for frame metrics: PamSequenceDataset's windows overlap
# 50%, so concatenating them end-to-end would double-count time and
# fragment events at window boundaries. run_inference_continuous below
# stitches overlapping windows back into one timeline per file (max score
# at each bin across whichever windows cover it) before this is applied.
# --------------------------------------------------------------------------

def frames_to_events(binary_seq: np.ndarray, label_interval: float):
    """Run-length encode a 0/1 sequence into a list of (start_s, stop_s)."""
    events = []
    i = 0
    n = len(binary_seq)
    while i < n:
        if not binary_seq[i]:
            i += 1
            continue
        j = i
        while j < n and binary_seq[j]:
            j += 1
        events.append((i * label_interval, j * label_interval))
        i = j
    return events


def match_events(pred_events: list, true_events: list):
    """Greedy any-overlap matching, one-to-one: each true event can match
    at most one predicted event. Returns (tp, fp, fn) counts."""
    matched_true = set()
    tp = 0
    for ps, pe in pred_events:
        hit = None
        for ti, (ts, te) in enumerate(true_events):
            if ti in matched_true:
                continue
            if ps < te and pe > ts:
                hit = ti
                break
        if hit is not None:
            matched_true.add(hit)
            tp += 1
    fp = len(pred_events) - tp
    fn = len(true_events) - len(matched_true)
    return tp, fp, fn


def event_level_metrics(continuous: dict, class_idx: int, threshold: float, label_interval: float,
                         min_duration_bins: int = 2, max_gap_bins: int = 1, use_mask: bool = False):
    """Event-level precision/recall/F1 for one class, pooled across every
    file in `continuous` (as returned by run_inference_continuous).
    `threshold` is compared against raw model output (logits, same scale
    predict_eval.py's frame-level threshold search uses -- NOT a
    0-1 probability). Predicted events are lightly cleaned up with
    smooth_binary_predictions first (bridge 1-bin gaps, drop <1s blips)
    since that's a closer match to how a detector would actually be used
    than scoring every raw bin flip as its own event.

    v19 use_mask: files where the class is outside the annotation scope are
    skipped, and masked bins (e.g. under an "uncertain" annotation) are
    removed from both predictions and truth before events are formed.
    """
    total_tp = total_fp = total_fn = 0
    for stem, d in continuous.items():
        scores = d["scores"][class_idx]
        true = d["true"][class_idx]
        if use_mask and "mask" in d:
            keep = d["mask"][class_idx] > 0
            if not keep.any():
                continue
            scores = np.where(keep, scores, -1e4)
            true = np.where(keep, true, 0)
        pred_bin = (scores > threshold).astype(int)
        pred_bin = smooth_binary_predictions(pred_bin, min_duration_bins=min_duration_bins,
                                              max_gap_bins=max_gap_bins)
        pred_events = frames_to_events(pred_bin, label_interval)
        true_events = frames_to_events(true.astype(int), label_interval)
        tp, fp, fn = match_events(pred_events, true_events)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return dict(precision=precision, recall=recall, f1=f1, tp=total_tp, fp=total_fp, fn=total_fn)


def _worker_init_single_thread(worker_id):
    """DataLoader worker_init_fn: caps each worker PROCESS to a single
    PyTorch CPU thread for its own ops (the resample/MelSpectrogram/Resize
    calls inside PamSequenceDataset.__getitem__ all run through torch,
    which multi-threads by default). Without this, --num-workers worker
    processes each ALSO spawn their own multi-threaded torch op execution
    -- on a CPU-only run (no GPU to hide it behind), that's often FAR more
    total threads than the machine's physical cores, and the resulting
    contention/context-switching can make a "parallel" run slower than a
    single-worker one, not faster. Must be a module-level (picklable)
    function, not a local closure, since each worker is a separate process
    spawned via pickling the DataLoader's own arguments. Irrelevant (never
    called) when num_workers=0, since then there ARE no separate worker
    processes -- everything runs in the main process's own thread pool."""
    torch.set_num_threads(1)


@torch.no_grad()
def run_inference_continuous(model, dataset, device, batch_size: int = 16, num_workers: int = 4):
    """Run `model` over every window in `dataset` (expects shuffle=False
    iteration order -- pass a fresh DataLoader here, don't reuse a shuffled
    one) and stitch the per-window outputs into one continuous per-file
    timeline per class, so events aren't fragmented/double-counted across
    PamSequenceDataset's overlapping windows. Where windows overlap, takes
    the max raw model output (logit) at each bin -- i.e. any window that
    saw a call "wins" at that bin, same spirit as the OR-like max used for
    ground truth.

    Returns dict[stem] -> {"duration": float, "scores": (C, n_bins) raw
    logits, "true": (C, n_bins) 0/1 ground truth, "covered": (n_bins,) bool
    -- whether any window actually wrote to that bin, "mask": (C, n_bins) 1 =
    scored (v19; from the dataset's masks when built with return_mask=True,
    else all ones)}. Bin i is time
    i * dataset.label_interval seconds into the file.

    Bins outside every window's coverage (can happen at a file's tail if
    its duration isn't an exact multiple of the window hop) are left at
    the -1e4 sentinel in "scores" -- NOT a real model output. Callers that
    do a fixed >threshold comparison (event/per-file metrics) are
    unaffected since -1e4 always reads as negative, but anything that
    derives a *range* from these scores (e.g. predict_eval.py's PR-curve
    threshold search, which takes min/max as interpolation bounds) must
    mask by "covered" first or a single leftover sentinel bin silently
    collapses that range from a few logits wide to ~10000 wide.
    """
    model.eval()
    label_interval = dataset.label_interval
    num_classes = dataset.num_classes

    file_durations = {}
    for stem in dataset.df_data["stem"].unique():
        path = dataset.df_data.loc[dataset.df_data["stem"] == stem, "filename"].iloc[0]
        try:
            file_durations[stem] = sf.info(path).duration
        except Exception:
            file_durations[stem] = librosa.get_duration(path=path)

    results = {}
    for stem, dur in file_durations.items():
        n_bins = int(dur / label_interval) + 1
        results[stem] = {
            "duration": dur,
            "scores": np.full((num_classes, n_bins), -1e4, dtype=np.float32),
            "true": np.zeros((num_classes, n_bins), dtype=np.float32),
            "covered": np.zeros(n_bins, dtype=bool),
            # v19: 1 = this class/bin is scored (in the file's annotation scope, not under an
            # "uncertain" annotation); all ones unless the dataset was built with return_mask=True
            "mask": np.zeros((num_classes, n_bins), dtype=np.float32),
        }

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                         worker_init_fn=_worker_init_single_thread if num_workers > 0 else None)
    cursor = 0
    for batch in tqdm(loader, desc="Inference (continuous)"):
        x, y = batch[0], batch[1]
        m = batch[2].numpy() if len(batch) == 3 else np.ones(y.shape, dtype=np.float32)
        x = _to_device(x, device)
        scores = model(x).cpu().numpy()  # (B, C, T) raw logits
        y = y.numpy()
        for k in range(y.shape[0]):
            row = dataset.df_data.iloc[cursor]
            stem = row["stem"]
            start_bin = int(round(row["start_sec"] / label_interval))
            T = scores.shape[2]
            n_bins = results[stem]["scores"].shape[1]
            end_bin = min(start_bin + T, n_bins)
            length = end_bin - start_bin
            if length > 0:
                results[stem]["scores"][:, start_bin:end_bin] = np.maximum(
                    results[stem]["scores"][:, start_bin:end_bin], scores[k, :, :length]
                )
                results[stem]["true"][:, start_bin:end_bin] = np.maximum(
                    results[stem]["true"][:, start_bin:end_bin], y[k, :, :length]
                )
                results[stem]["mask"][:, start_bin:end_bin] = np.maximum(
                    results[stem]["mask"][:, start_bin:end_bin], m[k, :, :length]
                )
                results[stem]["covered"][start_bin:end_bin] = True
            cursor += 1

    return results


@torch.no_grad()
def run_inference_continuous_singleprocess(model, dataset, device, batch_size: int = 16):
    """Same contract/output as run_inference_continuous (dict[stem] ->
    {"duration", "scores", "true", "covered"}), but with NO DataLoader and
    NO worker processes at all -- windows are fetched and batched directly
    in this one process, relying purely on PyTorch's own intra-op
    threading (torch.set_num_threads) instead of DataLoader's multi-
    PROCESS parallelism.

    Meant as a simpler, easier-to-reason-about alternative for a CPU-only
    deployment run, where the multiprocessing machinery itself has been a
    real source of operational friction: per-worker thread-oversubscription
    tuning (_worker_init_single_thread), --num-workers/--batch-size needing
    explicit plumbing through run_inference.sh's arg parser, and a single
    worker process crashing taking the whole DataLoader iterator -- and the
    whole many-thousand-file run -- down with it. Single-process execution
    also makes per-window fault tolerance trivial (see the try/except
    below): a worker-process exception can't be caught from the main
    process the way a plain Python exception in this same process can.

    Trade-off: this does NOT overlap audio-decode/spectrogram CPU work
    (this process, while idle waiting on the model) with the model's own
    forward pass the way a >0-worker DataLoader would (decoding window N+1
    while window N is being scored). On a CPU-only box where both the
    decode and the model math ultimately compete for the same physical
    cores anyway, that overlap is often not worth much -- but if decode is
    genuinely the bottleneck (e.g. audio on a slow network filesystem,
    model math cheap relative to it), run_inference_continuous with
    --num-workers > 0 can still meaningfully help. Reach for this one
    first for its simplicity; it's not a strict downgrade.
    """
    model.eval()
    label_interval = dataset.label_interval
    num_classes = dataset.num_classes

    file_durations = {}
    for stem in dataset.df_data["stem"].unique():
        path = dataset.df_data.loc[dataset.df_data["stem"] == stem, "filename"].iloc[0]
        try:
            file_durations[stem] = sf.info(path).duration
        except Exception:
            file_durations[stem] = librosa.get_duration(path=path)

    results = {}
    for stem, dur in file_durations.items():
        n_bins = int(dur / label_interval) + 1
        results[stem] = {
            "duration": dur,
            "scores": np.full((num_classes, n_bins), -1e4, dtype=np.float32),
            "true": np.zeros((num_classes, n_bins), dtype=np.float32),
            "covered": np.zeros(n_bins, dtype=bool),
            # v19: 1 = this class/bin is scored (in the file's annotation scope, not under an
            # "uncertain" annotation); all ones unless the dataset was built with return_mask=True
            "mask": np.zeros((num_classes, n_bins), dtype=np.float32),
        }

    n = len(dataset)
    with tqdm(total=n, desc="Inference (single-process)") as pbar:
        cursor = 0
        while cursor < n:
            end = min(cursor + batch_size, n)
            specs, labels, masks, rows = [], [], [], []
            for i in range(cursor, end):
                try:
                    item = dataset[i]
                    spec, y = item[0], item[1]
                    masks.append(item[2] if len(item) == 3 else torch.ones_like(y))
                except Exception as exc:
                    # A window that slips past common.py's own truncated-
                    # file handling and still fails outright (e.g. a
                    # genuinely unreadable file) -- skipped rather than
                    # aborting the whole run over one bad window, same
                    # spirit as the silence-padding fallback inside
                    # _spectrogram_for_subband, just for whatever it
                    # doesn't already catch.
                    print(f"WARNING: skipping window {i} (stem={dataset.df_data.iloc[i]['stem']!r}) "
                          f"-- {exc!r}")
                    continue
                specs.append(spec)
                labels.append(y)
                rows.append(dataset.df_data.iloc[i])
            pbar.update(end - cursor)
            cursor = end
            if not specs:
                continue

            if isinstance(specs[0], dict):
                x = {k: torch.stack([s[k] for s in specs]) for k in specs[0]}
            else:
                x = torch.stack(specs)
            y_batch = torch.stack(labels)
            m_np = torch.stack(masks).numpy()

            x = _to_device(x, device)
            scores = model(x).cpu().numpy()  # (B, C, T) raw logits
            y_np = y_batch.numpy()

            for k, row in enumerate(rows):
                stem = row["stem"]
                start_bin = int(round(row["start_sec"] / label_interval))
                T = scores.shape[2]
                n_bins_stem = results[stem]["scores"].shape[1]
                end_bin = min(start_bin + T, n_bins_stem)
                length = end_bin - start_bin
                if length > 0:
                    results[stem]["scores"][:, start_bin:end_bin] = np.maximum(
                        results[stem]["scores"][:, start_bin:end_bin], scores[k, :, :length]
                    )
                    results[stem]["true"][:, start_bin:end_bin] = np.maximum(
                        results[stem]["true"][:, start_bin:end_bin], y_np[k, :, :length]
                    )
                    results[stem]["mask"][:, start_bin:end_bin] = np.maximum(
                        results[stem]["mask"][:, start_bin:end_bin], m_np[k, :, :length]
                    )
                    results[stem]["covered"][start_bin:end_bin] = True

    return results


# --------------------------------------------------------------------------
# Training / validation loops
# --------------------------------------------------------------------------

def _unpack_batch(batch, device):
    """v13: loaders built with return_mask=True yield (x, y, mask); older
    (x, y) loaders still work (mask=None)."""
    if len(batch) == 3:
        x, y, m = batch
        return _to_device(x, device), y.to(device, non_blocking=True), m.to(device, non_blocking=True)
    x, y = batch
    return _to_device(x, device), y.to(device, non_blocking=True), None


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, num_classes,
                     scaler=None):
    """Returns a dict: loss, f1 (masked macro F1@0.5, supported classes only),
    class_f1. (AP is only computed in validate -- on the training stream it
    would be both slow and distorted by the rare-class sampler.)"""
    model.train()
    total_loss = 0
    all_preds, all_targets, all_masks = [], [], []

    for batch in tqdm(loader, desc=f"Epoch {epoch} train"):
        x, y, m = _unpack_batch(batch, device)

        optimizer.zero_grad()
        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(x)
                loss = criterion(outputs, y, m, noisy=x.get("noisy") if isinstance(x, dict) else None)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(x)
            loss = criterion(outputs, y, m, noisy=x.get("noisy") if isinstance(x, dict) else None)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        all_preds.append(outputs.detach().float().cpu())
        all_targets.append(y.detach().cpu())
        if m is not None:
            all_masks.append(m.detach().cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_masks = torch.cat(all_masks, dim=0) if all_masks else None
    avg_f1, class_f1s = compute_class_wise_f1(all_preds, all_targets, num_classes=num_classes, masks=all_masks)
    return {"loss": total_loss / len(loader), "f1": avg_f1, "class_f1": class_f1s}


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes, min_pos=None):
    """Returns the masked_class_metrics dict plus 'loss'."""
    model.eval()
    total_loss = 0
    all_preds, all_targets, all_masks = [], [], []

    for batch in tqdm(loader, desc="Validation"):
        x, y, m = _unpack_batch(batch, device)
        outputs = model(x)
        loss = criterion(outputs, y, m)
        total_loss += loss.item()
        all_preds.append(outputs.float().cpu())
        all_targets.append(y.cpu())
        if m is not None:
            all_masks.append(m.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_masks = torch.cat(all_masks, dim=0) if all_masks else None
    res = masked_class_metrics(all_preds, all_targets, all_masks, min_pos=min_pos)
    res["loss"] = total_loss / len(loader)
    return res


def run_training(band: str, args):
    """Training entrypoint (train_cqt.py). `args` is an argparse.Namespace from
    somad_args.build_argparser/resolve_defaults; `band` is "cqt" (the only one)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Band: {band} | device: {device}")
    if getattr(args, "class_config", None):
        configure_classes(args.class_config)
    print(f"Classes ({len(BAND_LABELS[band])}): {BAND_LABELS[band]}")

    audio_dirs = resolve_audio_dirs(args, band)
    df_anno, audio_index = build_dataset_table(
        args.aplose_csv, args.soi_csv, audio_dirs,
        bluefin_dir=getattr(args, "bluefin_dir", None),
        bluefin_sites=getattr(args, "bluefin_sites", None),
    )
    print(f"Found {len(audio_index)} audio files, {len(df_anno)} usable annotations")
    print(df_anno["label"].value_counts())
    n_unc = int(df_anno["mask_label"].notna().sum()) if "mask_label" in df_anno.columns else 0
    if n_unc:
        print(f"{n_unc} 'uncertain' annotation rows -> their bins are masked for the class, not negatives")

    train_stems, val_stems = get_split(args, audio_index, df_anno)
    print(f"Train files: {len(train_stems)} | Val files: {len(val_stems)}")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "train_files.txt"), "w") as f:
        f.write("\n".join(train_stems))
    with open(os.path.join(args.output_dir, "val_files.txt"), "w") as f:
        f.write("\n".join(val_stems))

    # v13: what each class actually has to learn from / be scored on, per
    # split (unique positive 0.5 s bins, scope-aware) -- also in the run log.
    try:
        _print_split_class_bins(df_anno, train_stems, val_stems, audio_index, BAND_LABELS[band])
    except Exception as exc:  # never let a reporting helper kill a long job
        print(f"(class/bin summary skipped: {exc!r})")

    bluefin_hf_dir = getattr(args, "bluefin_hf_dir", None)
    noise_pool_size = getattr(args, "noise_pool_size", NOISE_POOL_SIZE)
    robust = getattr(args, "robust_loss", "none") or "none"
    noisy_sources = getattr(args, "robust_sources", None) if robust != "none" else None
    train_set = PamSequenceDataset(train_stems, audio_index, df_anno, band=band, augment=True,
                                    bluefin_hf_dir=bluefin_hf_dir,
                                    noise_pool_size=noise_pool_size, return_mask=True,
                                    noisy_sources=noisy_sources)
    val_set = PamSequenceDataset(val_stems, audio_index, df_anno, band=band, augment=False,
                                  bluefin_hf_dir=bluefin_hf_dir,
                                  noise_pool_size=noise_pool_size, return_mask=True)

    label_list = BAND_LABELS[band]
    df_anno_train = df_anno[df_anno["stem"].isin(train_stems)]
    class_weights = compute_class_weights(
        df_anno_train, train_stems, audio_index,
        label_list, BAND_CONFIG[band]["snippet_length_s"],
    )
    print("Class labels:", label_list)
    print("Class weights:", class_weights)

    rare_classes = low_sample_classes(df_anno_train, label_list)
    if rare_classes:
        counts = df_anno_train["label"].value_counts()
        print(f"WARNING: low training sample count (< {MIN_CLASS_COUNT_WARNING} instances) for "
              f"{ {lb: int(counts.get(lb, 0)) for lb in rare_classes} } -- treat these classes' "
              f"metrics/thresholds as unreliable regardless of architecture/training choices; "
              f"the fix is more annotations, not more tuning.")

    # Rare-class-aware sampling: loss-level class weighting alone doesn't
    # help if the model rarely even sees a rare-class window in the first
    # place. Boost the sampling probability of windows that contain a rare
    # class; background-only windows keep weight 1.0 so they're still seen,
    # just not as the overwhelming majority every epoch.
    sample_weights = 1.0 + train_set.class_presence.astype(np.float32) @ class_weights.numpy()
    # v13: per-source draw multiplier (class_config.json -> sample_weight),
    # e.g. keep the 17k-box DCLDE-CRP set from dominating every batch.
    src_w = {s: CLASS_CFG.sample_weight(CLASS_CFG.source_of(s, audio_index.get(s))) for s in train_stems}
    sample_weights = sample_weights * train_set.df_data["stem"].map(src_w).fillna(1.0).to_numpy(dtype=np.float32)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_set), replacement=True
    )

    # Same worker-thread-oversubscription fix as run_inference_continuous's
    # own DataLoader -- see _worker_init_single_thread's docstring. v17: the
    # workers only decode audio (+ resample/dither if needed) and apply the
    # waveform augmentation; the spectrogram is computed by the model.
    _worker_init = _worker_init_single_thread if args.num_workers > 0 else None
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler,
                               num_workers=args.num_workers, pin_memory=True, drop_last=True,
                               worker_init_fn=_worker_init)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True,
                             worker_init_fn=_worker_init)

    # v17: CQTFrontend (spectrogram on the GPU) -> one CNN tower -> BiGRU.
    model = build_model(BAND_CONFIG[band], len(label_list), args.use_attention).to(device)
    print(f"Model: CQT {BAND_CONFIG[band]['cqt_octaves']} oct x {BAND_CONFIG[band]['cqt_bins_per_octave']}/oct, "
          f"gamma {BAND_CONFIG[band]['cqt_gamma_hz']} Hz -> {BAND_CONFIG[band]['freq_bins']} x "
          f"{BAND_CONFIG[band]['time_bins']} input, single CNN tower, "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters")

    criterion = WeightedCombinedLoss(class_weights=class_weights, robust=robust,
                                     robust_beta=getattr(args, "robust_beta", 0.8),
                                     robust_q=getattr(args, "robust_q", 0.7))
    if robust != "none":
        print(f"v19 robust loss: {robust} (beta={criterion.robust_beta}, q={criterion.robust_q}) on sources "
              f"{noisy_sources}; validation loss and the selection metric are unchanged (plain loss, masked AP)")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                   betas=(0.9, 0.999))
    # v2: CosineAnnealingWarmRestarts was firing hard restarts (epochs 11,
    # 31, ...) that visibly knocked train/val F1 down each time in the v1 LF
    # run, without ever helping val F1 escape its plateau. ReduceLROnPlateau
    # only cuts the LR when val F1 actually stalls, so it can't self-inflict
    # that kind of regression.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
    )
    scaler = torch.amp.GradScaler("cuda") if (args.amp and device.type == "cuda") else None

    best_val_f1 = 0.0
    patience_counter = 0
    epoch_start = 1

    # v8: --resume -- restores model/optimizer/scheduler/AMP-scaler state,
    # epoch number, best-val-F1-so-far, and the early-stop patience counter,
    # so training picks up from exactly where it left off instead of
    # restarting at epoch 1 with a fresh optimizer (which would re-warm the
    # LR schedule and re-zero patience, silently discarding real progress).
    # Refuses to load a checkpoint trained for a different band or label
    # list outright (model.load_state_dict would fail on a shape mismatch
    # anyway for a freq_bins/hidden_size change, but band/label_list can
    # mismatch without necessarily changing tensor shapes -- e.g. two
    # different bands that happen to share hidden_size/gru_layers -- so
    # this check catches that case too, not just the shape-mismatch one
    # PyTorch would already reject).
    resume_path = getattr(args, "resume", None)
    if resume_path:
        print(f"Resuming from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        if checkpoint.get("band") != band:
            raise SystemExit(f"ERROR: {resume_path} was trained as band={checkpoint.get('band')!r}, "
                              f"cannot resume as band={band!r} -- these are different models.")
        if checkpoint.get("model_labels") != label_list:
            raise SystemExit(f"ERROR: {resume_path}'s label list {checkpoint.get('model_labels')} doesn't "
                              f"match this run's {label_list} -- not resumable (a different class set is a "
                              f"different model, not a continuation of this one).")
        # v17: the CQT/input settings are part of the architecture
        if checkpoint.get("band_config") != BAND_CONFIG[band]:
            raise SystemExit(f"ERROR: {resume_path} was trained with band_config {checkpoint.get('band_config')}, "
                              f"this run uses {BAND_CONFIG[band]} -- not resumable.")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        best_val_f1 = checkpoint.get("best_f1", 0.0)
        patience_counter = checkpoint.get("patience_counter", 0)
        epoch_start = checkpoint.get("epoch", 0) + 1
        print(f"Resumed at epoch {epoch_start}, best_val_f1={best_val_f1:.4f}, "
              f"patience_counter={patience_counter}/{args.patience}")

    # v12: --resume-add-classes -- a DIFFERENT operation from --resume
    # above, not a variant of it: this warm-starts a NEW training phase
    # (fresh optimizer/scheduler/epoch-count/best-val-f1/patience, since
    # adding classes genuinely changes the loss landscape -- the old run's
    # best_val_f1 isn't comparable to this one's) from an OLD checkpoint
    # trained on a SUBSET of this run's current label_list. Mutually
    # exclusive with --resume (which requires an IDENTICAL class set and
    # continues the exact same training run, unchanged above).
    #
    # Mechanism: every parameter except the classifier's final Linear layer
    # has a shape that depends only on hidden_size/gru_layers/
    # pooled_freq_rows/use_attention -- NOT on num_classes -- so the whole
    # CNN towers + BiGRU + attention load unchanged regardless of how many
    # classes were added. Only the classifier's last Linear layer's output
    # dimension actually depends on num_classes; its rows are copied over
    # by NAME (not position), since a new class inserted in the MIDDLE of
    # JOINT_LABELS (e.g. added to LF_LABELS) shifts every later class's
    # index -- position-based copying would silently scramble which row
    # means which class. New classes' rows are left at the classifier's
    # own fresh random init (nothing to warm-start them from).
    #
    # Optimizer/scheduler state is NOT restored (unlike --resume) -- AdamW's
    # per-parameter momentum/variance buffers are tied to the OLD
    # classifier tensor's exact shape/identity, which no longer exists once
    # that layer's output dimension changes; starting the new, larger
    # classifier's optimizer state fresh is the standard, expected approach
    # for incremental-class fine-tuning, not a shortcut taken here.
    resume_add_classes_path = getattr(args, "resume_add_classes", None)
    if resume_add_classes_path:
        if resume_path:
            raise SystemExit("ERROR: --resume and --resume-add-classes are mutually exclusive -- "
                              "--resume continues an identical-architecture/identical-class-set run; "
                              "--resume-add-classes starts a NEW training phase warm-started from an "
                              "old, smaller class set.")
        old_checkpoint = torch.load(resume_add_classes_path, map_location=device, weights_only=False)
        if old_checkpoint.get("band") != band:
            raise SystemExit(f"ERROR: {resume_add_classes_path} was trained as band="
                              f"{old_checkpoint.get('band')!r}, this run is band={band!r} -- these are "
                              f"different models.")
        if old_checkpoint.get("band_config") != BAND_CONFIG[band]:
            raise SystemExit(f"ERROR: {resume_add_classes_path} was trained with band_config "
                              f"{old_checkpoint.get('band_config')}, this run uses {BAND_CONFIG[band]} -- a "
                              f"different input/architecture, not just a class-set change.")
        old_label_list = old_checkpoint.get("model_labels", [])
        missing = [lb for lb in old_label_list if lb not in label_list]
        if missing:
            raise SystemExit(f"ERROR: {resume_add_classes_path}'s classes {missing} are not present in "
                              f"this run's current label list ({label_list}) -- --resume-add-classes "
                              f"only supports ADDING classes, not removing or renaming existing ones.")
        new_classes = [lb for lb in label_list if lb not in old_label_list]
        if not new_classes:
            print(f"WARNING: {resume_add_classes_path}'s classes already match this run's label list "
                  f"exactly -- nothing to add. Consider --resume instead if you meant to continue this "
                  f"exact run.")
        print(f"Expanding from {len(old_label_list)} to {len(label_list)} classes -- new: {new_classes}")

        old_state = old_checkpoint["model_state_dict"]
        new_state = model.state_dict()
        # Shape-mismatched keys, derived rather than hardcoded (e.g.
        # "classifier.4.*") so this keeps working if the classifier head's
        # own internal structure ever changes -- anything whose shape
        # differs between the old and new state dicts is, by construction,
        # exactly the num_classes-dependent final layer and nothing else
        # (every other parameter's shape is independent of num_classes).
        mismatched_keys = {k for k in old_state if k in new_state and old_state[k].shape != new_state[k].shape}
        compatible_state = {k: v for k, v in old_state.items() if k not in mismatched_keys}
        missing_keys, unexpected_keys = model.load_state_dict(compatible_state, strict=False)
        truly_missing = set(missing_keys) - mismatched_keys
        if truly_missing or unexpected_keys:
            raise SystemExit(f"ERROR: {resume_add_classes_path} isn't architecturally compatible beyond "
                              f"the classifier's final layer -- missing {truly_missing}, unexpected "
                              f"{unexpected_keys}. --resume-add-classes only handles a changed class "
                              f"count, not other architecture differences (hidden_size, gru_layers, "
                              f"pooled_freq_rows, use_attention, ...) -- those need --resume (identical "
                              f"architecture) or training from scratch.")
        if not mismatched_keys:
            print("WARNING: no shape-mismatched classifier keys found -- old and new models may have "
                  "the exact same num_classes already (see the 'nothing to add' warning above).")
        else:
            final_weight_key = next(k for k in mismatched_keys if k.endswith(".weight"))
            final_bias_key = next(k for k in mismatched_keys if k.endswith(".bias"))
            with torch.no_grad():
                new_weight = new_state[final_weight_key]
                new_bias = new_state[final_bias_key]
                old_weight = old_state[final_weight_key]
                old_bias = old_state[final_bias_key]
                for old_idx, lb in enumerate(old_label_list):
                    new_idx = label_list.index(lb)
                    new_weight[new_idx].copy_(old_weight[old_idx])
                    new_bias[new_idx].copy_(old_bias[old_idx])
            model.load_state_dict(new_state)
            print(f"Warm-started {len(old_label_list)}/{len(label_list)} classifier rows from "
                  f"{resume_add_classes_path}; {len(new_classes)} new class row(s) left at random init. "
                  f"Starting fresh optimizer/scheduler/epoch count (epoch 1) -- this is a new training "
                  f"phase, not a continuation.")

    if os.environ.get("FREEZE_TRUNK") == "1":
        trunk = [model.cnn, model.gru] + ([model.attention] if getattr(model, "attention", None) is not None else [])
        for m in trunk:
            for prm in m.parameters():
                prm.requires_grad = False
        _orig_train = model.train
        def _train_keep_trunk_eval(mode=True):
            _orig_train(mode)
            for m in trunk:
                m.eval()
            return model
        model.train = _train_keep_trunk_eval
        n_tr = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
        print(f"FREEZE_TRUNK: CNN + BiGRU + attention frozen (eval mode, BatchNorm stats fixed); "
              f"training {n_tr / 1e6:.2f} M classifier parameters only")

    # v13: selection metric = masked macro average precision over the classes
    # with >= --min-val-positives positive bins in the validation split (see
    # masked_class_metrics). F1@0.5 stays in the log for continuity with v12.
    min_val_pos = getattr(args, "min_val_positives", MIN_VAL_POSITIVE_BINS)
    log_file = os.path.join(args.output_dir, f"{args.run_name}.csv")
    if not (resume_path and os.path.exists(log_file)):
        with open(log_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_f1", "val_loss", "val_f1", "val_macro_ap", "lr"]
                            + [f"val_ap__{lb}" for lb in label_list] + [f"val_f1__{lb}" for lb in label_list])

    for epoch in range(epoch_start, args.epochs + 1):
        tr = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            num_classes=len(label_list), scaler=scaler,
        )
        va = validate(model, val_loader, criterion, device, num_classes=len(label_list), min_pos=min_val_pos)
        train_loss, train_f1 = tr["loss"], tr["f1"]
        val_loss, val_f1, val_ap = va["loss"], va["macro_f1"], va["macro_ap"]
        val_class_f1s, val_class_ap = va["f1"], va["ap"]
        # fall back to F1 only if NO class had enough validation positives for AP
        val_score = val_ap if np.isfinite(np.array(val_class_ap)).any() else val_f1
        scheduler.step(val_score)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\nEpoch {epoch} | train_loss={train_loss:.4f} train_f1={train_f1:.4f} "
              f"| val_loss={val_loss:.4f} val_macro_AP={val_ap:.4f} (select) val_f1@0.5={val_f1:.4f} "
              f"| lr={current_lr:.2e}")
        print("Val class AP:", {lb: (None if np.isnan(a) else round(a, 3)) for lb, a in zip(label_list, val_class_ap)})
        print("Val class F1@0.5:", dict(zip(label_list, [round(f, 3) for f in val_class_f1s])))
        print("Val positive bins:", dict(zip(label_list, va["n_pos"])))

        with open(log_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_f1, val_loss, val_f1, val_ap, current_lr]
                            + [("" if np.isnan(a) else a) for a in val_class_ap] + list(val_class_f1s))

        if val_score > best_val_f1:
            best_val_f1 = val_score
            patience_counter = 0
        else:
            patience_counter += 1

        # v8: shared by both checkpoints below -- keeping best_model.pth
        # (only overwritten on a new best val F1, used for eval/deployment)
        # and latest_checkpoint.pth (overwritten every epoch regardless,
        # used only by --resume) as separate files matters: resuming from
        # best_model.pth would silently throw away every non-improving
        # epoch since that best, restarting the patience clock from a
        # stale point instead of actually continuing where training left
        # off.
        checkpoint_dict = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            # v13: "best_f1" kept as the key name every v12 script/resume path
            # reads, but it now holds the SELECTION score (masked macro AP);
            # selection_metric says so explicitly. val_f1_at_epoch is the old
            # F1@0.5 of this epoch.
            "best_f1": best_val_f1,
            "selection_metric": "masked_macro_average_precision",
            "val_f1_at_epoch": val_f1,
            "class_ap": val_class_ap,
            "min_val_positive_bins": min_val_pos,
            "class_config": CLASS_CFG.raw,
            "patience_counter": patience_counter,
            "class_f1s": val_class_f1s,
            "band": band,
            "model_labels": list(label_list),
            "band_config": BAND_CONFIG[band],
            "use_attention": args.use_attention,
            "rare_classes": rare_classes,
            "fusion_mode": "cqt",   # v17: kept so v13-era scripts can tell this checkpoint apart
            "resume_add_classes_from": getattr(args, "resume_add_classes", None),
            "robust_loss": dict(type=robust, sources=noisy_sources, beta=criterion.robust_beta, q=criterion.robust_q),
        }
        torch.save(checkpoint_dict, os.path.join(args.output_dir, "latest_checkpoint.pth"))

        if patience_counter == 0:  # i.e. this epoch WAS the new best (see above)
            torch.save(checkpoint_dict, os.path.join(args.output_dir, "best_model.pth"))
            print(f"Saved new best model, val_macro_AP={best_val_f1:.4f}")

        if patience_counter >= args.patience:
            print(f"Early stopping after {args.patience} epochs without improvement")
            break

    print(f"Training done. Best val macro AP: {best_val_f1:.4f}")
