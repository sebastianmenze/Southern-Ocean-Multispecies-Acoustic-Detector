# -*- coding: utf-8 -*-
"""Shared data pipeline, model and loss code for the BiGRU sequence-tagger
classifiers (LF = low-frequency baleen whale calls, HF = higher-frequency
species), trained on the combined APLOSE (Gerlache/hermine) + SOI/AURAL 2024
annotation sets -- the same two annotation sources the Perch classifier in
``aplose_training/perch_training*`` is built from (see ``train_perchv2.txt``).

Label set is aligned with the Perch model's canonical species names
(see ``perch_training/label_map*.json``), split into two bands:

  LF band (audio resampled to a few hundred Hz; Perch's own frontend can't
  resolve energy below ~70-150 Hz -- see perch_training_wideband/perch_common.py):
    Fin whale 20 Hz, Fin whale downsweep,
    Blue whale AB call, Blue whale D call, Blue whale Z call,
    Minke whale
    (v5: Blue whale A call and B call are merged into one "AB call" label --
    B call alone was a single occurrence in the original Gerlache/hermine
    data, nowhere near enough to train separately, and both are part of the
    same stereotyped call sequence anyway. Blue whale Z call and Minke
    whale are new, sourced entirely from convert_bluefin_library.py's
    AcousticTrends_BlueFinLibrary conversion, not the original Gerlache/
    hermine APLOSE export -- see README's "v5 changes")

  HF band (audio kept near its native 32 kHz rate):
    Humpback, Leopard seal, Weddell seal, Crabeater seal, Orca

Perch's "Background" class is a hand-picked ambient-noise clip category and
is deliberately NOT reproduced as an explicit output here: these models are
multi-label sequence taggers over continuous recordings, and the vast
majority of each recording is unannotated audio anyway, so "no species
active" already serves as the implicit background/negative case (this is
also how the original claude_18_hf / claude_19_lf models in
``nn_bigru_classifier`` worked).
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
import torchvision.transforms as transforms
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


# --------------------------------------------------------------------------
# v9: fusion-mode helpers for band="joint" -- see train_lf.py's --fusion-mode
# comment for the full reasoning behind each mode.
# --------------------------------------------------------------------------
def _minmax_norm(spec: torch.Tensor) -> torch.Tensor:
    return (spec - spec.min()) / (spec.max() - spec.min() + 1e-12)


def _resize_freq(spec: torch.Tensor, target_freq_bins: int) -> torch.Tensor:
    """Resize a (channels, freq_bins, time_bins) spec to a different
    freq_bins, leaving time_bins untouched. No-op if already that size."""
    if spec.shape[1] == target_freq_bins:
        return spec
    return transforms.Resize((target_freq_bins, spec.shape[2]))(spec)


def _to_device(x, device):
    """Like Tensor.to(device), but also handles the dict-of-tensors batch
    shape --fusion-mode=twotower produces (PamSequenceDataset.__getitem__
    returns {"hf": ..., "lf": ...} instead of one tensor for that mode, and
    the default DataLoader collate_fn turns a batch of those dicts into one
    dict of batched tensors)."""
    if isinstance(x, dict):
        return {k: v.to(device, non_blocking=True) for k, v in x.items()}
    return x.to(device, non_blocking=True)


def _apply_specaugment(spec: torch.Tensor, freq_bins: int, time_bins: int, cfg: dict, n_augment: int) -> torch.Tensor:
    """SpecAugment (stripe dropout) on one (channels, freq_bins, time_bins)
    tensor -- extracted so --fusion-mode=channels/twotower can apply it per
    sub-band (each with its own freq_bins) instead of only against the
    single combined spec_dim the original inline version assumed."""
    if n_augment <= 0 or not cfg.get("spec_augment", True):
        return spec
    stripes_ver = stripes_hor = 0
    for _ in range(n_augment):
        if torch.randint(low=0, high=2, size=(1,))[0] == 0:
            stripes_ver += 1
        else:
            stripes_hor += 1

    drop_ver = int(time_bins * cfg["max_time_mask_frac"])
    drop_hor = int(freq_bins * cfg["max_freq_mask_frac"])

    for _ in range(stripes_ver):
        distance = torch.randint(low=0, high=max(drop_ver, 1), size=(1,))[0]
        bgn = torch.randint(low=0, high=time_bins - distance, size=(1,))[0]
        spec[:, :, bgn:bgn + distance] = 0
    for _ in range(stripes_hor):
        distance = torch.randint(low=0, high=max(drop_hor, 1), size=(1,))[0]
        bgn = torch.randint(low=0, high=freq_bins - distance, size=(1,))[0]
        spec[:, bgn:bgn + distance, :] = 0
    return spec


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

# v7: for the new joint LF+HF model (band="joint", see BAND_CONFIG["joint"]
# below and train_joint.py) -- the union of LF_LABELS/HF_LABELS, ONE shared
# output per class instead of v6's approach (train Humpback/Leopard/
# Weddell/Crabeater independently on both bands, two separate decisions per
# class). A joint spectrogram lets the model see both bands' evidence for
# the same class at once, which is strictly more informative than
# reconciling two independent per-band predictions after the fact -- see
# README's "v7 changes" for the full reasoning behind building this at all.
# LF_LABELS/HF_LABELS above are reverted to their original v5 six/five (v6's
# cross-band duplication was v6-specific, superseded here) so train_lf.py/
# train_hf.py still work unchanged from this same v7 checkout if you want
# the old separate-band models for comparison.
JOINT_LABELS = LF_LABELS + HF_LABELS  # 6 + 5 = 11, no overlap now that LF_LABELS is back to v5's six

BAND_LABELS = {"lf": LF_LABELS, "hf": HF_LABELS, "joint": JOINT_LABELS}

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
# Per-band audio / spectrogram / windowing configuration.
# LF settings match the old claude_19_lf model; HF settings match the old
# claude_18_hf model (see nn_bigru_classifier/train_pam_naut_19_lf.py and
# apply_to_all_data_cnn_18hf_nautilus.py).
# --------------------------------------------------------------------------

BAND_CONFIG = {
    "lf": dict(
        sample_rate=500,
        # v7: fft_size 2**9->2**8 (512->256 samples, ~1.02s->~0.51s window).
        # fft_overlap briefly tried at 0.9, reverted back to 0.1 -- note
        # what this field actually controls (hop_length = fft_size *
        # fft_overlap, see _compute_spectrogram): despite the name, a
        # BIGGER value means a BIGGER hop, i.e. LESS real overlap between
        # consecutive STFT frames. At 0.1 (here), hop=25.6 samples -> ~90%
        # real overlap between windows.
        fft_size=2 ** 8,
        fft_overlap=0.1,
        n_mels=None,          # linear spectrogram
        f_min=None,
        f_max=None,
        freq_bins=200,
        time_bins=600,
        snippet_length_s=60,
        snippet_overlap=2,    # 2 = 50% hop between snippet starts
        label_interval_s=0.5,
        # v2: LF and HF used to share identical capacity (hidden_size=512,
        # 3 GRU layers) despite LF's input being far lower-information --
        # 500 Hz sample rate, 200 freq bins, effectively content under
        # ~250 Hz. The v1 run massively overfit on LF (train F1 -> 0.89,
        # val F1 stuck ~0.47 from epoch ~25 on): cut GRU capacity hard here.
        hidden_size=192,
        gru_layers=2,
        # v3: SpecAugment (zeroing out random time/freq stripes) dropped
        # for LF entirely. Real recordings never contain literal zeroed-out
        # blocks -- it's an out-of-distribution artifact, and LF calls are
        # narrow/short enough (a fin whale 20 Hz pulse is ~1s) that a mask
        # can erase one outright rather than just partially occluding it.
        # Waveform-level augmentation (augment_waveform: gain + noise, both
        # of which real recordings *do* look like) stays on.
        spec_augment=False,
    ),
    "hf": dict(
        sample_rate=32000,
        fft_size=2 ** 13,
        fft_overlap=0.1,
        n_mels=700,
        f_min=10,
        f_max=16000,
        freq_bins=500,
        time_bins=600,
        snippet_length_s=60,
        snippet_overlap=2,
        label_interval_s=0.5,
        hidden_size=512,
        gru_layers=3,
        # Kept on for HF for now (only asked to drop it for LF). HF calls
        # (humpback song bouts, seal/orca calls) run longer, so a mask is
        # less likely to erase one outright than for LF's short pulses.
        spec_augment=True,
        max_time_mask_frac=0.10,   # 600*0.10 = 60 bins -> up to ~6s
        max_freq_mask_frac=0.10,   # 500*0.10 = 50 bins
    ),
}

# v7: the joint LF+HF model's config -- NOT a third independent band, a
# combination of the two existing ones above (added after BAND_CONFIG's own
# literal closes so "hf"/"lf" can be referenced directly instead of
# duplicating their fft/mel parameters a third time). Every window's input
# is built from the SAME source audio decoded/resampled TWICE -- once per
# sub_bands entry, through the exact same per-band spectrogram math
# BAND_CONFIG["hf"]/["lf"] already use for the standalone models -- then the
# two resulting (own freq_bins, time_bins) spectrograms are stacked along
# the frequency axis, HF on top of LF (matches how a real spectrogram reads
# top-to-bottom: HF's ~16 kHz ceiling above, LF's ~250 Hz ceiling below),
# giving one (hf freq_bins + lf freq_bins, time_bins) image -- computed
# below from BAND_CONFIG["hf"]/["lf"]'s own current values directly, not
# hardcoded, so it always stays correct if either changes.
BAND_CONFIG["joint"] = dict(
    sub_bands=dict(hf=BAND_CONFIG["hf"], lf=BAND_CONFIG["lf"]),
    freq_bins=BAND_CONFIG["hf"]["freq_bins"] + BAND_CONFIG["lf"]["freq_bins"],
    time_bins=600,
    snippet_length_s=60,
    snippet_overlap=2,
    label_interval_s=0.5,
    # >= HF's capacity (512/3), not LF's (192/2) -- the joint input carries
    # strictly more information than either band alone (both bands' full
    # content, for a class list that's now the union of both), so there's
    # no reason to under-provision it the way LF's small model was
    # deliberately sized for a low-information 200-bin input alone (see
    # "lf" config's own v2 comment above). Not pushed higher than this
    # without real evidence it helps -- see README's "v7 changes".
    hidden_size=512,
    gru_layers=3,
    # ImprovedCNNBiGRU's AdaptiveAvgPool2d((pooled_freq_rows, timesteps))
    # stage -- 32 in the existing lf/hf models regardless of their own
    # freq_bins (200 or 500), since that pooling already absorbs whatever
    # height the CNN's feature maps come in at. Bumped up here since a
    # joint (200+500=700-row) input is now carrying TWO bands' worth of
    # frequency detail into that same bottleneck; leaving it at 32 risked blurring
    # HF-half and LF-half detail together too aggressively. Not a required
    # architecture change (the model runs fine at 32 too), just a capacity
    # increase matched to the larger, more diverse input -- see
    # ImprovedCNNBiGRU's own pooled_freq_rows parameter.
    pooled_freq_rows=48,
    spec_augment=True,
    max_time_mask_frac=0.10,
    max_freq_mask_frac=0.10,
    # v9: only used by --fusion-mode=channels (see train_lf.py's own
    # comment for the full reasoning) -- the shared frequency resolution
    # BOTH sub-bands get resized to before being stacked as 2 input
    # channels instead of extra frequency rows. Set to HF's own native 500
    # (was 256) so HF is never downsampled/loses resolution; LF (native
    # 200) gets upsampled 2.5x to match instead -- LF's real content is
    # concentrated under ~250 Hz anyway (see BAND_CONFIG["lf"]'s own
    # comments), so upsampling it via Resize (bilinear interpolation, no
    # new information) is a smaller cost than throwing away real HF detail.
    channels_freq_bins=500,
)


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
        "weak": df.get("type", pd.Series("BOX", index=df.index)).eq("WEAK"),
    })
    out["label"] = out["raw_label"].map(APLOSE_LABEL_MAP)
    out = out.dropna(subset=["label"]).reset_index(drop=True)

    n_weak = int(out["weak"].sum())
    if n_weak:
        print(f"Dropping {n_weak} APLOSE 'WEAK' (presence-only, no precise timing) annotation rows "
              f"-- would otherwise rasterize as 'entire window positive':")
        print(out.loc[out["weak"], "label"].value_counts().to_string())
    return out[~out["weak"]].drop(columns="weak").reset_index(drop=True)


def load_soi_annotations(csv_path: str) -> pd.DataFrame:
    """Load the SOI/AURAL annotation CSV (t1/t2/label/audiofilename) into the
    common schema."""
    df = pd.read_csv(csv_path)
    out = pd.DataFrame({
        "file": df["audiofilename"].apply(os.path.basename),
        "start_datetime": pd.to_datetime(df["t1"], utc=True),
        "end_datetime": pd.to_datetime(df["t2"], utc=True),
        "raw_label": df["label"],
        "source": "soi",
    })
    out["label"] = out["raw_label"].map(SOI_LABEL_MAP)
    return out.dropna(subset=["label"]).reset_index(drop=True)


def build_audio_index(audio_dirs) -> dict:
    """Map file stem (no extension) -> full path, across all given directories."""
    index = {}
    for d in audio_dirs:
        for path in glob.glob(os.path.join(d, "*.wav")):
            index[os.path.splitext(os.path.basename(path))[0]] = path
    return index


def resolve_audio_dirs(args, band: str) -> list:
    """hermine/soi dirs apply to both bands; args.bluefin_dir (v5+: the
    AcousticTrends_BlueFinLibrary conversion, resampled to exactly 500 Hz
    -- the LF model's own working rate, see convert_bluefin_library.py) is
    LF-only. Its content has no real energy above a couple hundred Hz
    (bandlimited by the original recorders, some as low as 250 Hz natively
    -- see that script's docstring), so every window would be a
    guaranteed-background negative for every HF class (seals, orca,
    humpback all need much higher frequencies); including it in HF
    training/eval would only add wasted decode/spectrogram time with no
    benefit. `args.bluefin_dir` may not exist on older (v4 and earlier)
    argparsers -- getattr default keeps this call safe either way.

    v7: band=="joint" also includes bluefin_dir, same as "lf" -- a joint
    window still needs SOME audio for its HF half even over a bluefin-
    sourced file (see _spectrogram_for_subband's shaped-dither handling for
    why that's not just silence/a hard edge), and the joint model still
    wants bluefin's real LF-band content for its LF half regardless."""
    dirs = [args.hermine_dir, args.soi_train_dir, args.soi_val_dir]
    if band in ("lf", "joint"):
        dirs.append(getattr(args, "bluefin_dir", None))
    return [d for d in dirs if d]


def filter_bluefin_sites(audio_index: dict, bluefin_dir: str, sites) -> dict:
    """v7: drop any bluefin_dir-sourced stem whose filename doesn't contain
    one of `sites` (site-year names, e.g. "ElephantIsland2013Aural" --
    matching convert_bluefin_library.py's own SITES keys exactly, since
    that script's output filenames always embed the site name) -- lets
    training restrict bluefin_library to a subset of its 11 site-years
    (e.g. train_joint.py's own default) without touching hermine/soi stems.
    Identified by directory (bluefin_dir is one flat folder, see
    convert_bluefin_library.py's docstring), not by name pattern, so this
    can't accidentally misclassify a hermine/soi stem. No-op if `sites` is
    falsy (every bluefin site stays -- current default behavior for
    train_lf.py/train_hf.py) or bluefin_dir wasn't actually resolved (band
    doesn't use it at all)."""
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
    -- see train_joint.py's own default and --bluefin-sites in
    build_argparser.

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

    return df_anno, audio_index


def split_files(audio_index: dict, val_fraction: float = 0.15, seed: int = 0):
    """Deterministic file-level train/val split (no clip ever leaks across
    the split), stratified by source (aplose vs soi) so validation has
    representation from both recording sites."""
    from sklearn.model_selection import train_test_split

    stems = sorted(audio_index.keys())
    sources = ["aplose" if s.startswith("hermine") else "soi" for s in stems]

    train_stems, val_stems = train_test_split(
        stems, test_size=val_fraction, random_state=seed, stratify=sources
    )
    return sorted(train_stems), sorted(val_stems)


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


NOISE_POOL_SIZE = 24

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


def augment_waveform(audio: np.ndarray, rng: np.random.Generator, noise_pool: list = None) -> np.ndarray:
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
    out = audio.copy()

    gain_db = rng.uniform(-6.0, 6.0)
    out = out * (10.0 ** (gain_db / 20.0))

    if noise_pool and rng.random() < 0.5:
        noise = noise_pool[rng.integers(len(noise_pool))]
        if len(noise) != len(out):
            reps = int(np.ceil(len(out) / len(noise)))
            noise = np.tile(noise, reps)[:len(out)]
        snr_db = rng.uniform(6.0, 20.0)
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

class PamSequenceDataset(Dataset):
    """Cuts each recording into overlapping ``snippet_length_s`` windows,
    computes a spectrogram per window, and rasterizes the annotations
    falling in that window into a (num_classes, timesteps) multi-hot label
    grid at ``label_interval_s`` resolution. One dataset instance covers a
    single band ("lf" or "hf") and a single split (a list of file stems).
    """

    # v8: per-worker (each DataLoader worker process gets its own copy of
    # this dataset, so these caches are NOT shared across workers -- still
    # a large win, since a worker keeps landing on the same file's other
    # overlapping windows within one epoch's worth of its own share of
    # work) bounded caches -- see _spectrogram_for_subband's own docstring
    # for what's cached and why. Bounded (not unbounded) so a large dataset
    # can't blow up worker memory; a plain FIFO eviction (oldest-inserted
    # first) rather than true LRU -- simpler, and "the last N distinct
    # files/sub-bands touched" is a good enough proxy for temporal locality
    # here without tracking per-access recency.
    _CACHE_MAX_ENTRIES = 64

    def __init__(self, file_stems, audio_index: dict, df_anno: pd.DataFrame,
                 band: str, augment: bool = False, full_coverage: bool = False,
                 bluefin_hf_dir: str = None, fusion_mode: str = "stack"):
        assert band in BAND_CONFIG
        self.band = band
        self.cfg = BAND_CONFIG[band]
        self.model_labels = BAND_LABELS[band]
        self.num_classes = len(self.model_labels)
        self.augment = augment
        # v9: only consulted when band=="joint" -- see train_lf.py's
        # --fusion-mode comment.
        self.fusion_mode = fusion_mode

        # v8 (option C wiring): band=="joint" HF sub-band audio for a
        # bluefin-sourced stem gets loaded from here INSTEAD of the file's
        # normal (500 Hz) path, if a same-named file exists -- see
        # precompute_bluefin_hf_cache.py, which pre-resamples+dithers
        # bluefin_library_aplose up to HF's rate once, offline, so this
        # never has to happen live at train time. None/no match falls back
        # to the original path (live resample+dither in
        # _spectrogram_for_subband, same as before this existed).
        self.bluefin_hf_index = build_audio_index([bluefin_hf_dir]) if bluefin_hf_dir else {}
        if band == "joint":
            if bluefin_hf_dir:
                print(f"[joint] bluefin_hf_dir={bluefin_hf_dir!r}: {len(self.bluefin_hf_index)} cached "
                      f"HF-rate file(s) found -- bluefin stems matching one of these load their HF sub-band "
                      f"audio from here instead of live-resampling")
            else:
                print("[joint] --bluefin-hf-dir not set -- every bluefin HF sub-band access will be "
                      "live-resampled+dithered from its 500 Hz file (see precompute_bluefin_hf_cache.py to "
                      "avoid this)")

        # v8 debug instrumentation, see DEBUG_TIMING's own module-level
        # comment -- None (and every _spectrogram_for_subband timing check
        # below a no-op) unless PAM_DEBUG_TIMING=1 is set.
        self._timing = _TimingStats() if DEBUG_TIMING else None

        # v8 (option A), keyed per WINDOW since v8.1 (not per whole file --
        # see _spectrogram_for_subband's own docstring for the full
        # reasoning/history): (resolved path, sub-band tag, start_sec,
        # stop_sec) -> the decoded/resampled/dithered waveform or finished
        # spectrogram for that exact window. Mainly hits for the one case
        # v8.1's windowing guarantees will recur -- each window appears
        # TWICE in df_data (once clean, once to augment, see below), same
        # (path, start_sec, stop_sec) both times.
        self._waveform_cache = OrderedDict()
        self._clean_spec_cache = OrderedDict()
        # v9.1: filenames we've already printed a truncated-file warning
        # for (see _spectrogram_for_subband's 0-frame handling) -- one
        # print per file, not per window, so a badly truncated file
        # doesn't spam thousands of identical warnings across its tail
        # windows.
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
                dur = sf.info(path).duration
            except Exception:
                try:
                    dur = librosa.get_duration(path=path)
                except Exception as exc:
                    # A file neither soundfile nor librosa/audioread can
                    # even read the DURATION of -- genuinely unreadable
                    # (missing codec backend, badly corrupt header), not
                    # the milder truncated-tail case _raw_waveform_for_
                    # window/_spectrogram_for_subband already handle
                    # per-window. Skipped from the window grid entirely
                    # rather than letting the exception propagate and
                    # crash the WHOLE dataset's construction -- confirmed
                    # as a real risk, not hypothetical: one such file in a
                    # multi-thousand-file folder previously took down an
                    # entire inference job (run_all_sequential.sh's own
                    # per-job try/except caught THAT, but every file in
                    # the same job was lost with it -- this is the
                    # finer-grained fix, one bad file costing only itself).
                    skipped_files.append(stem)
                    print(f"WARNING: skipping {path!r} -- duration unreadable by both soundfile and "
                          f"librosa/audioread ({exc!r}); file may be corrupt or in an unsupported format")
                    continue
            n_steps = max(1, int(np.floor((dur - snippet_length) / hop)) + 1) if dur > snippet_length else 1
            starts = np.arange(n_steps) * hop
            # The regular hop-spaced grid above always starts its first
            # window at 0 (covering the file's true beginning), but its
            # LAST window can fall short of the file's true end whenever
            # (dur - snippet_length) isn't an exact multiple of hop -- the
            # remainder is simply dropped by floor(), leaving a real,
            # never-covered gap at the tail (confirmed on a real 300s file:
            # regular grid's last window stopped at 270s, leaving the final
            # ~30s permanently NaN in run_inference.py's output). Deployment
            # scoring (run_inference.py passes full_coverage=True) adds one
            # extra window anchored to end exactly at the file's true end
            # to close that gap -- it will typically overlap its
            # predecessor by more than the usual hop (a single, not
            # sliding-window, pass over that stretch specifically), which
            # is an accepted tradeoff for full coverage, not a bug. NOT
            # applied to training/validation (both leave full_coverage at
            # its False default) since that would add an extra, differently
            # -weighted window into what the loss/threshold calibration
            # sees -- out of scope for a deployment-coverage fix.
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

        if augment:
            df_data_2 = df_data.copy()
            df_data_2["n_augment_stripes"] = 2
            df_data = pd.concat([df_data, df_data_2]).reset_index(drop=True)

        self.df_data = df_data
        self.num_samples = len(df_data)
        self.spec_dim = [self.cfg["freq_bins"], self.cfg["time_bins"]]
        self.label_interval = self.cfg["label_interval_s"]
        self.timesteps = int(snippet_length / self.label_interval) + 1

        # Per-window (num_samples, num_classes) bool: does each window
        # overlap an annotation of that class. Pure metadata, cheap to
        # compute up front -- used by run_training to build a rare-class
        # aware WeightedRandomSampler (loss-level class weighting alone
        # doesn't help if the model rarely even sees a rare-class window).
        self.class_presence = compute_window_class_presence(self.df_data, self.df_anno, self.model_labels)

        # v2: real background windows for augment_waveform's noise mixing,
        # sampled once here (not per __getitem__ call -- that would mean
        # decoding a whole extra file on every augmented sample). Fixed seed
        # so the same pool is picked run-to-run for a given split/band.
        # v7: band=="joint" needs TWO pools (one per sub-band's own sample
        # rate -- self.cfg has no single top-level "sample_rate" for
        # "joint", see BAND_CONFIG["joint"]), a dict instead of the flat
        # list every other band uses. _spectrogram_for_subband picks the
        # matching one by sub-band name.
        self.noise_pool = {"hf": [], "lf": []} if band == "joint" else []
        if augment:
            bg_idx = np.where(~self.class_presence.any(axis=1))[0]
            pool_rng = np.random.default_rng(0)
            pick = pool_rng.choice(bg_idx, size=min(NOISE_POOL_SIZE, len(bg_idx)), replace=False) \
                if len(bg_idx) else []
            if band == "joint":
                for i in pick:
                    r = self.df_data.loc[i]
                    for sub in ("hf", "lf"):
                        try:
                            self.noise_pool[sub].append(_load_window_waveform(
                                r["filename"], self.cfg["sub_bands"][sub]["sample_rate"],
                                r["start_sec"], r["stop_sec"]))
                        except Exception as e:
                            print(f"WARNING: couldn't load {sub} noise-pool window from {r['filename']}: {e}")
                print(f"[joint] noise pool: {len(self.noise_pool['hf'])} hf + "
                      f"{len(self.noise_pool['lf'])} lf background windows")
            else:
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
        """FIFO-bounded insert -- see _CACHE_MAX_ENTRIES/__init__'s own
        comment for why FIFO (not true LRU) is good enough here."""
        cache[key] = value
        if len(cache) > self._CACHE_MAX_ENTRIES:
            cache.popitem(last=False)

    def _compute_spectrogram(self, waveform, cfg=None):
        # v7: optional explicit `cfg` -- band=="joint" calls this once per
        # sub-band (see _spectrogram_for_subband) with that sub-band's OWN
        # config, not self.cfg (which has no fft_size/n_mels/etc of its own
        # for "joint", only a `sub_bands` dict -- see BAND_CONFIG["joint"]).
        # Every existing lf/hf call site omits `cfg`, so this defaults to
        # self.cfg exactly as before -- no behavior change for those.
        cfg = cfg if cfg is not None else self.cfg
        sr = cfg["sample_rate"]
        fft_size = cfg["fft_size"]
        hop_length = int(fft_size * cfg["fft_overlap"])

        if cfg["n_mels"]:
            spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=sr, n_fft=fft_size, hop_length=hop_length,
                n_mels=cfg["n_mels"], f_min=cfg["f_min"], f_max=cfg["f_max"],
            )(waveform)
            # Drops mel filter rows that are STRUCTURALLY always zero (an
            # artifact of the filterbank/f_min/f_max/sample_rate
            # combination, not of any one file) -- but checking only frame
            # 0 (as this used to) is a fragile proxy for that: a file whose
            # very first frame happens to be genuinely silent (a near-empty
            # AURAL/SOI stub file, or just a quiet lead-in) reads every mel
            # bin as exactly 0 there, so ALL rows get dropped and the
            # (0-row) spectrogram crashes torchvision's Resize downstream
            # with "Input and output sizes should be greater than 0" --
            # confirmed as a real crash in deployment inference. Checking
            # for nonzero power in ANY frame across the whole file is far
            # more robust to one silent frame; the `valid.any()` fallback
            # additionally covers the (much rarer) case of a file silent in
            # literally every frame, keeping every bin unfiltered rather
            # than crashing -- a few always-zero rows surviving into a
            # normal file's spectrogram is a much smaller cost than losing
            # the whole inference run to one bad file.
            valid = (spec[0] > 0).any(dim=-1)
            if valid.any():
                spec = spec[:, valid, :]
        else:
            spec = torchaudio.transforms.Spectrogram(
                n_fft=fft_size, hop_length=hop_length,
            )(waveform)

        return torch.log10(spec + 1e-12)

    # v8.1: seconds of extra audio loaded/transformed on each side of a
    # window's own [start_sec, stop_sec) -- see _spectrogram_for_subband's
    # own docstring for why. Comfortably bigger than either sub-band's own
    # FFT window (LF: fft_size=256 @ 500 Hz = 0.512s; HF: fft_size=8192 @
    # 32 kHz = 0.256s), so the frames actually inside [start_sec, stop_sec)
    # always have real audio context on both sides, not a truncated/
    # zero-padded edge.
    WINDOW_LOAD_PAD_S = 2.0

    def _spectrogram_for_subband(self, filename, sub_cfg, freq_bins, time_bins,
                                  start_sec, stop_sec, n_augment, noise_pool, cache_key=None):
        """One band's windowed, resized (1, freq_bins, time_bins) spectrogram
        for one file/window -- the per-window process __getitem__ always did
        for its single band, extracted here so band=="joint" can call it
        TWICE (once per BAND_CONFIG["joint"]["sub_bands"] entry) instead of
        duplicating this logic; a plain "lf"/"hf" dataset still calls this
        exactly once, with sub_cfg=self.cfg/freq_bins=self.spec_dim[0]/
        noise_pool=self.noise_pool.

        v8.1: decodes and transforms ONLY this window's own slice (+/-
        WINDOW_LOAD_PAD_S of padding for FFT edge context), not the whole
        file. v8's first pass here (see git history / README's "v8 changes"
        for the full writeup) decoded+spectrogram'd the ENTIRE file on
        every access, relying on a per-file cache (see below) to avoid
        redoing that across a file's several overlapping windows -- but a
        real run showed the cache hitting 0% of the time under
        run_training's WeightedRandomSampler (draws close to i.i.d. across
        the whole dataset, so a bounded per-file cache rarely survives long
        enough to be revisited), meaning that version was paying full
        whole-file cost on nearly every single call regardless -- confirmed
        as the actual bottleneck via PAM_DEBUG_TIMING (spectrogram compute
        alone was ~87% of per-sample time on a real run, dominated by
        computing HF's 8192-point/700-mel spectrogram over close to an
        hour of audio to use 60 seconds of it). Loading only the needed
        slice fixes this directly, independent of cache behavior, by
        making the expensive part proportional to the WINDOW's duration
        instead of the FILE's.

        v8 (option A): `cache_key` (typically (resolved path, sub-band tag),
        None to disable caching entirely) still helps skip repeat work for
        the one case v8.1's windowing guarantees will recur: each window
        appears TWICE in df_data (once clean, once to augment -- see
        __init__), sharing the identical (path, start_sec, stop_sec) --
        cache_key includes start_sec/stop_sec now (not just path/sub-band)
        to key on that correctly.
          - self._clean_spec_cache: the finished, freq-resized WINDOW
            spectrogram for n_augment==0 rows -- reused verbatim if the
            identical window is requested again.
          - self._waveform_cache: the decoded/resampled/dithered WINDOW
            waveform (pre-augmentation) -- reused for n_augment>0 rows too,
            but never as a finished spectrogram (augment_waveform's gain
            jitter + noise mixing must be freshly randomized every single
            access; caching the augmented result would either freeze one
            random draw forever or require recomputing anyway).
        """
        sr = sub_cfg["sample_rate"]
        timing = self._timing  # None unless PAM_DEBUG_TIMING=1 -- see module-level DEBUG_TIMING comment

        if cache_key is not None:
            cache_key = (*cache_key, round(start_sec, 3), round(stop_sec, 3))

        clean_cached = self._clean_spec_cache.get(cache_key) if (cache_key is not None and n_augment == 0) else None
        if timing is not None and cache_key is not None and n_augment == 0:
            timing.record_cache(hit=clean_cached is not None)

        if clean_cached is not None:
            spec, pad_start_s, pad_stop_s = clean_cached
        else:
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
                min_len = sub_cfg["fft_size"]
                if waveform.shape[-1] < min_len:
                    waveform = torch.cat(
                        [waveform, torch.zeros(1, min_len - waveform.shape[-1], dtype=waveform.dtype)], dim=-1)

                if cache_key is not None:
                    self._cache_put(self._waveform_cache, cache_key, (waveform, pad_start_s, pad_stop_s))

            # v2: waveform-level augmentation (gain jitter + noise mixing),
            # same gate as the spectrogram-domain stripe augmentation below
            # -- only the "augmented half" of the training set gets either.
            # Applied to the whole loaded (padded) slice since both
            # transforms are time-invariant; see augment_waveform's
            # docstring for why no time-shift is used here. A fresh copy,
            # not applied in place -- `waveform` may be the shared cached
            # tensor this same window's clean row will reuse next.
            if n_augment > 0:
                t0 = time.perf_counter() if timing is not None else None
                rng = np.random.default_rng()
                augmented = augment_waveform(waveform.numpy().squeeze(0), rng, noise_pool)
                spec_waveform = torch.from_numpy(augmented).unsqueeze(0)
                if timing is not None:
                    timing.add("augment", time.perf_counter() - t0)
            else:
                spec_waveform = waveform

            t0 = time.perf_counter() if timing is not None else None
            spec = self._compute_spectrogram(spec_waveform, sub_cfg)

            resize_transform = transforms.Resize((freq_bins, spec.shape[2]))
            spec = resize_transform(spec)
            if timing is not None:
                timing.add("spectrogram+freq_resize", time.perf_counter() - t0)

            if n_augment == 0 and cache_key is not None:
                self._cache_put(self._clean_spec_cache, cache_key, (spec, pad_start_s, pad_stop_s))

        t0 = time.perf_counter() if timing is not None else None
        # Absolute (within-file) time axis: the loaded/transformed slice
        # spans [pad_start_s, pad_stop_s) (not [0, file_duration) -- see
        # above), so spec's own time bins must be anchored there, not at
        # the file's true start, for this comparison against start_sec/
        # stop_sec (also absolute-within-file) to land on the right bins.
        # pad_start_s/pad_stop_s are cached alongside spec/waveform
        # (not re-derived/approximated here) specifically because the
        # padding can be asymmetrically clamped near a file's true start or
        # end -- assuming the full +/-WINDOW_LOAD_PAD_S on both sides
        # unconditionally would misalign this axis for any window within
        # one pad-width of either edge.
        t = np.linspace(pad_start_s, pad_stop_s, spec.shape[2])
        ix = (t >= start_sec) & (t < stop_sec)
        spec = spec[:, :, ix]
        resize_transform = transforms.Resize((spec.shape[1], time_bins))
        spec = resize_transform(spec)
        if timing is not None:
            timing.add("time_slice+resize", time.perf_counter() - t0)
        return spec

    def __getitem__(self, idx):
        item_t0 = time.perf_counter() if self._timing is not None else None
        row = self.df_data.loc[idx]
        filename = row["filename"]
        stem = row["stem"]
        n_augment = int(row["n_augment_stripes"])
        cfg = self.cfg

        if self.band == "joint":
            sub = cfg["sub_bands"]
            # v8 (option C wiring): a bluefin-sourced stem with a matching
            # entry in bluefin_hf_index (see __init__/precompute_bluefin_hf_
            # cache.py) loads its HF sub-band audio from the precomputed
            # 32 kHz cache instead of the file's normal (500 Hz) path --
            # already at HF's target rate, so _spectrogram_for_subband's own
            # sr != sr_org check is a no-op and the live resample+dither
            # never happens for it. Falls back to `filename` (live resample)
            # for every non-bluefin stem, unchanged from before this existed.
            hf_path = self.bluefin_hf_index.get(stem, filename)
            hf_spec = self._spectrogram_for_subband(
                hf_path, sub["hf"], sub["hf"]["freq_bins"], self.spec_dim[1],
                row["start_sec"], row["stop_sec"], n_augment, self.noise_pool["hf"],
                cache_key=(hf_path, "hf"))
            lf_spec = self._spectrogram_for_subband(
                filename, sub["lf"], sub["lf"]["freq_bins"], self.spec_dim[1],
                row["start_sec"], row["stop_sec"], n_augment, self.noise_pool["lf"],
                cache_key=(filename, "lf"))

            # v9: how the two sub-bands get combined -- see train_lf.py's
            # --fusion-mode comment for the full reasoning behind each.
            if self.fusion_mode == "stack":
                # v7/v8 baseline, unchanged: concat on freq axis (HF on top
                # of LF, top-to-bottom like a real spectrogram), single
                # joint min-max normalize further down.
                spec = torch.cat([hf_spec, lf_spec], dim=1)
            elif self.fusion_mode == "stack_normfix":
                # option 1: normalize each sub-band independently BEFORE
                # stacking, so neither band's raw log-power range can
                # dominate/wash out the other's contrast under a single
                # joint min-max.
                spec = torch.cat([_minmax_norm(hf_spec), _minmax_norm(lf_spec)], dim=1)
            elif self.fusion_mode == "channels":
                # option 2: resize both sub-bands to the SAME freq
                # resolution and stack as separate input CHANNELS instead
                # of extra frequency rows, so the CNN's first conv layer
                # never convolves across the physically meaningless LF/HF
                # frequency seam "stack" creates.
                target_fb = cfg["channels_freq_bins"]
                hf_r = _minmax_norm(_resize_freq(hf_spec, target_fb))
                lf_r = _minmax_norm(_resize_freq(lf_spec, target_fb))
                spec = torch.cat([hf_r, lf_r], dim=0)
            elif self.fusion_mode == "twotower":
                # option 3: keep both sub-bands fully separate -- each gets
                # its own CNN tower in ImprovedCNNBiGRU, fused only after
                # each tower's own pooled features are computed.
                spec = {"hf": _minmax_norm(hf_spec), "lf": _minmax_norm(lf_spec)}
            else:
                raise ValueError(f"unknown fusion_mode {self.fusion_mode!r}")
        else:
            spec = self._spectrogram_for_subband(
                filename, cfg, self.spec_dim[0], self.spec_dim[1],
                row["start_sec"], row["stop_sec"], n_augment, self.noise_pool,
                cache_key=(filename, "single"))

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

        # v9: stack_normfix/channels/twotower already normalized (per
        # sub-band) above -- only "stack" (and every non-joint band) still
        # needs the original single joint min-max here.
        if isinstance(spec, dict):
            pass
        elif self.band != "joint" or self.fusion_mode == "stack":
            spec = _minmax_norm(spec)

        # v9: SpecAugment -- "stack"/"stack_normfix" (and non-joint bands)
        # produce one tensor shaped exactly like self.spec_dim, so the
        # original single-call form still applies unchanged; "channels"
        # has a different freq resolution (cfg["channels_freq_bins"], not
        # self.spec_dim[0]); "twotower" needs it applied per sub-band, each
        # at its own native freq_bins.
        if isinstance(spec, dict):
            for key, sub_spec in spec.items():
                spec[key] = _apply_specaugment(
                    sub_spec, sub[key]["freq_bins"], self.spec_dim[1], cfg, n_augment)
        elif self.band == "joint" and self.fusion_mode == "channels":
            spec = _apply_specaugment(spec, cfg["channels_freq_bins"], self.spec_dim[1], cfg, n_augment)
        else:
            spec = _apply_specaugment(spec, self.spec_dim[0], self.spec_dim[1], cfg, n_augment)

        if self._timing is not None:
            self._timing.add("__getitem__ total", time.perf_counter() - item_t0)
            self._timing.maybe_print(self.band)

        return spec, labels


def compute_class_weights(df_anno_train: pd.DataFrame, file_stems, audio_index: dict,
                           label_list, snippet_length_s: float) -> torch.Tensor:
    """Inverse-frequency class weights from annotated seconds per class,
    normalized to sum to num_classes (same formula the original scripts used,
    just computed from annotation durations instead of a slow full dataset
    pass)."""
    total_duration = 0.0
    for stem in file_stems:
        try:
            total_duration += sf.info(audio_index[stem]).duration
        except Exception:
            total_duration += librosa.get_duration(path=audio_index[stem])

    class_seconds = np.zeros(len(label_list))
    for i, lb in enumerate(label_list):
        rows = df_anno_train[df_anno_train["label"] == lb]
        class_seconds[i] = (rows["t_2_s"] - rows["t_1_s"]).clip(lower=0).sum()

    weights = total_duration / (len(label_list) * (class_seconds + 1))
    weights = weights * len(label_list) / weights.sum()
    return torch.tensor(weights, dtype=torch.float32)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

class ImprovedCNNBiGRU(nn.Module):
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
            # v7: pooled_freq_rows was a hardcoded 32 through v6, regardless
            # of fsteps (the constructor's own `fsteps` parameter above is,
            # in fact, entirely unused inside this class -- AdaptiveAvgPool2d
            # already made the model shape-agnostic to input frequency
            # height, so changing BAND_CONFIG's lf/hf/joint freq_bins values
            # never requires touching this class at all). Raised for the
            # joint model specifically (see BAND_CONFIG["joint"]'s own
            # pooled_freq_rows comment for current numbers/reasoning) --
            # left at the original 32 for standalone lf/hf.
            nn.AdaptiveAvgPool2d((pooled_freq_rows, timesteps)),
        )

    def __init__(self, in_channels=1, hidden_size=512, num_classes=5, timesteps=121, fsteps=200,
                 use_attention=False, gru_layers=3, pooled_freq_rows=32, fusion_mode="stack"):
        super().__init__()
        self.use_attention = use_attention
        self.pooled_freq_rows = pooled_freq_rows
        # v9: only "twotower" changes anything here -- see train_lf.py's
        # --fusion-mode comment. Every other mode (including plain lf/hf,
        # which never set this) behaves exactly as before: one shared CNN
        # stem, `in_channels` already tells it 1 (stack/stack_normfix) or 2
        # (channels).
        self.fusion_mode = fusion_mode

        if fusion_mode == "twotower":
            # option 3: separate CNN stem per sub-band -- each learns its
            # own early filters instead of a shared stem whose first conv
            # layer would otherwise convolve across the physically
            # meaningless LF/HF frequency seam "stack" creates. Fused only
            # after each tower's own pooled features are flattened, right
            # before the GRU (see forward()).
            self.cnn_hf = self._build_cnn_stem(1, pooled_freq_rows, timesteps)
            self.cnn_lf = self._build_cnn_stem(1, pooled_freq_rows, timesteps)
            gru_input_size = 512 * pooled_freq_rows * 2
        else:
            self.cnn = self._build_cnn_stem(in_channels, pooled_freq_rows, timesteps)
            gru_input_size = 512 * pooled_freq_rows

        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=hidden_size,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if gru_layers > 1 else 0.0,
        )

        # This constructor's own default is False (unused in practice --
        # every real caller passes use_attention explicitly: run_training
        # from args.use_attention, predict_eval.py/run_inference.py/
        # plot_errors.py from the checkpoint's saved value). The actual
        # train_lf.py/train_hf.py CLI default is True as of v4 (see
        # README's "v4 changes") -- --no-attention reverts to a plain
        # BiGRU. A 3-layer BiGRU is already a lot of temporal capacity for
        # the amount of training data here (a few hundred to a few
        # thousand positive windows per class), so the attention block's
        # extra parameters were an overfitting-risk concern in v3, but that
        # was never actually A/B'd cleanly -- every v3 attention run was
        # confounded by the threshold-calibration and WEAK-annotation bugs
        # "v3 changes" #11/#15 later found and fixed.
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

    def _cnn_to_seq(self, feat, batch_size):
        feat = feat.permute(0, 3, 1, 2)
        return feat.reshape(batch_size, feat.size(1), -1)

    def forward(self, x):
        if self.fusion_mode == "twotower":
            batch_size = x["hf"].size(0)
            hf_seq = self._cnn_to_seq(self.cnn_hf(x["hf"]), batch_size)
            lf_seq = self._cnn_to_seq(self.cnn_lf(x["lf"]), batch_size)
            x = torch.cat([hf_seq, lf_seq], dim=-1)
        else:
            batch_size = x.size(0)
            x = self._cnn_to_seq(self.cnn(x), batch_size)
        gru_out, _ = self.gru(x)
        if self.use_attention:
            attn_out, _ = self.attention(gru_out, gru_out, gru_out)
            x = gru_out + attn_out
        else:
            x = gru_out
        x = self.classifier(x)
        return x.permute(0, 2, 1)


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

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.class_weights is not None:
            weights = self.class_weights.view(1, -1, 1).to(inputs.device)
            focal_loss = focal_loss * weights

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class WeightedCombinedLoss(nn.Module):
    def __init__(self, class_weights=None, alpha=0.7, beta=0.3, focal_gamma=2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.focal_loss = WeightedFocalLoss(class_weights=class_weights, gamma=focal_gamma)
        self.class_weights = class_weights
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none" if class_weights is not None else "mean")

    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        if self.class_weights is not None:
            bce = self.bce_loss(inputs, targets)
            weights = self.class_weights.view(1, -1, 1).to(inputs.device)
            bce = (bce * weights).mean()
        else:
            bce = self.bce_loss(inputs, targets)
        return self.alpha * focal + self.beta * bce


def compute_class_wise_f1(preds, targets, threshold=0.5, num_classes=5):
    preds_bin = (torch.sigmoid(preds) > threshold).float()
    f1_scores = []
    for class_idx in range(num_classes):
        pred_class = preds_bin[:, class_idx, :].flatten()
        target_class = targets[:, class_idx, :].flatten()

        tp = (pred_class * target_class).sum().item()
        fp = (pred_class * (1 - target_class)).sum().item()
        fn = ((1 - pred_class) * target_class).sum().item()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_scores.append(f1)

    return float(np.mean(f1_scores)), f1_scores


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
                         min_duration_bins: int = 2, max_gap_bins: int = 1):
    """Event-level precision/recall/F1 for one class, pooled across every
    file in `continuous` (as returned by run_inference_continuous).
    `threshold` is compared against raw model output (logits, same scale
    predict_eval.py's frame-level threshold search uses -- NOT a
    0-1 probability). Predicted events are lightly cleaned up with
    smooth_binary_predictions first (bridge 1-bin gaps, drop <1s blips)
    since that's a closer match to how a detector would actually be used
    than scoring every raw bin flip as its own event.
    """
    total_tp = total_fp = total_fn = 0
    for stem, d in continuous.items():
        scores = d["scores"][class_idx]
        true = d["true"][class_idx]
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
    -- whether any window actually wrote to that bin}. Bin i is time
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
        }

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                         worker_init_fn=_worker_init_single_thread if num_workers > 0 else None)
    cursor = 0
    for x, y in tqdm(loader, desc="Inference (continuous)"):
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
        }

    n = len(dataset)
    with tqdm(total=n, desc="Inference (single-process)") as pbar:
        cursor = 0
        while cursor < n:
            end = min(cursor + batch_size, n)
            specs, labels, rows = [], [], []
            for i in range(cursor, end):
                try:
                    spec, y = dataset[i]
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
                    results[stem]["covered"][start_bin:end_bin] = True

    return results


# --------------------------------------------------------------------------
# Training / validation loops
# --------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, num_classes,
                     scaler=None):
    model.train()
    total_loss = 0
    all_preds, all_targets = [], []

    for x, y in tqdm(loader, desc=f"Epoch {epoch} train"):
        x, y = _to_device(x, device), y.to(device, non_blocking=True)

        optimizer.zero_grad()
        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(x)
                loss = criterion(outputs, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        all_preds.append(outputs.detach().cpu())
        all_targets.append(y.detach().cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    avg_f1, class_f1s = compute_class_wise_f1(all_preds, all_targets, num_classes=num_classes)
    return total_loss / len(loader), avg_f1, class_f1s


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []

    for x, y in tqdm(loader, desc="Validation"):
        x, y = _to_device(x, device), y.to(device, non_blocking=True)
        outputs = model(x)
        loss = criterion(outputs, y)
        total_loss += loss.item()
        all_preds.append(outputs.cpu())
        all_targets.append(y.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    avg_f1, class_f1s = compute_class_wise_f1(all_preds, all_targets, num_classes=num_classes)
    return total_loss / len(loader), avg_f1, class_f1s


def run_training(band: str, args):
    """Shared entrypoint used by train_lf.py / train_hf.py. `args` is an
    argparse.Namespace; see those scripts for the accepted fields."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Band: {band} | device: {device}")

    audio_dirs = resolve_audio_dirs(args, band)
    df_anno, audio_index = build_dataset_table(
        args.aplose_csv, args.soi_csv, audio_dirs,
        bluefin_dir=getattr(args, "bluefin_dir", None),
        bluefin_sites=getattr(args, "bluefin_sites", None),
    )
    print(f"Found {len(audio_index)} audio files, {len(df_anno)} usable annotations")
    print(df_anno["label"].value_counts())

    train_stems, val_stems = split_files(audio_index, val_fraction=args.val_fraction, seed=args.seed)
    print(f"Train files: {len(train_stems)} | Val files: {len(val_stems)}")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "train_files.txt"), "w") as f:
        f.write("\n".join(train_stems))
    with open(os.path.join(args.output_dir, "val_files.txt"), "w") as f:
        f.write("\n".join(val_stems))

    bluefin_hf_dir = getattr(args, "bluefin_hf_dir", None)
    # v9: only meaningful for band="joint" -- PamSequenceDataset defaults
    # to "stack" (the v7/v8 behavior) when not band="joint", same as if
    # this were never passed.
    fusion_mode = getattr(args, "fusion_mode", "stack")
    train_set = PamSequenceDataset(train_stems, audio_index, df_anno, band=band, augment=True,
                                    bluefin_hf_dir=bluefin_hf_dir, fusion_mode=fusion_mode)
    val_set = PamSequenceDataset(val_stems, audio_index, df_anno, band=band, augment=False,
                                  bluefin_hf_dir=bluefin_hf_dir, fusion_mode=fusion_mode)

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
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_set), replacement=True
    )

    # Same worker-thread-oversubscription fix as run_inference_continuous's
    # own DataLoader -- see _worker_init_single_thread's docstring. Matters
    # here regardless of training running on GPU: the audio decode/
    # spectrogram preprocessing this caps is CPU work either way.
    _worker_init = _worker_init_single_thread if args.num_workers > 0 else None
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler,
                               num_workers=args.num_workers, pin_memory=True, drop_last=True,
                               worker_init_fn=_worker_init)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True,
                             worker_init_fn=_worker_init)

    # v9: --fusion-mode=channels needs 2 input channels (both sub-bands
    # stacked as channels instead of extra frequency rows); twotower
    # ignores in_channels entirely (each of its 2 towers is hardcoded to 1,
    # see ImprovedCNNBiGRU._build_cnn_stem calls). Every other case
    # (including plain lf/hf) is the original single-channel spectrogram.
    in_channels = 2 if (band == "joint" and fusion_mode == "channels") else 1
    model = ImprovedCNNBiGRU(
        in_channels=in_channels,
        num_classes=len(label_list),
        timesteps=timesteps_for_band(band),
        fsteps=BAND_CONFIG[band]["freq_bins"],
        hidden_size=BAND_CONFIG[band]["hidden_size"],
        gru_layers=BAND_CONFIG[band]["gru_layers"],
        use_attention=args.use_attention,
        pooled_freq_rows=BAND_CONFIG[band].get("pooled_freq_rows", 32),
        fusion_mode=fusion_mode,
    ).to(device)

    criterion = WeightedCombinedLoss(class_weights=class_weights)
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
        # v9: a different fusion_mode is a different architecture (channels
        # has in_channels=2, twotower has two separate CNN stems instead of
        # one) -- model.load_state_dict below would already fail on most
        # such mismatches with a shape/key error, but check explicitly for
        # a clearer message (and to catch the rare shape-compatible case).
        if checkpoint.get("fusion_mode", "stack") != fusion_mode:
            raise SystemExit(f"ERROR: {resume_path} was trained with fusion_mode="
                              f"{checkpoint.get('fusion_mode', 'stack')!r}, cannot resume as "
                              f"fusion_mode={fusion_mode!r} -- these are different architectures.")
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

    log_file = os.path.join(args.output_dir, f"{args.run_name}.csv")
    if not (resume_path and os.path.exists(log_file)):
        with open(log_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_f1", "val_loss", "val_f1", "lr"])

    for epoch in range(epoch_start, args.epochs + 1):
        train_loss, train_f1, _ = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            num_classes=len(label_list), scaler=scaler,
        )
        val_loss, val_f1, val_class_f1s = validate(model, val_loader, criterion, device,
                                                     num_classes=len(label_list))
        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\nEpoch {epoch} | train_loss={train_loss:.4f} train_f1={train_f1:.4f} "
              f"| val_loss={val_loss:.4f} val_f1={val_f1:.4f} | lr={current_lr:.2e}")
        print("Val class F1s:", dict(zip(label_list, [round(f, 3) for f in val_class_f1s])))

        with open(log_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_f1, val_loss, val_f1, current_lr])

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
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
            "best_f1": best_val_f1,
            "patience_counter": patience_counter,
            "class_f1s": val_class_f1s,
            "band": band,
            "model_labels": label_list,
            "band_config": BAND_CONFIG[band],
            "use_attention": args.use_attention,
            "rare_classes": rare_classes,
            "fusion_mode": fusion_mode,
        }
        torch.save(checkpoint_dict, os.path.join(args.output_dir, "latest_checkpoint.pth"))

        if patience_counter == 0:  # i.e. this epoch WAS the new best (see above)
            torch.save(checkpoint_dict, os.path.join(args.output_dir, "best_model.pth"))
            print(f"Saved new best model, val_f1={best_val_f1:.4f}")

        if patience_counter >= args.patience:
            print(f"Early stopping after {args.patience} epochs without improvement")
            break

    print(f"Training done. Best val F1: {best_val_f1:.4f}")
