# -*- coding: utf-8 -*-
"""Shared CLI argument parser (`build_argparser`/`resolve_defaults`) for
SOMAD's training scripts, plus a standalone entry point that trains a
6-class, LF-only model (Fin whale 20 Hz, Fin whale downsweep, Blue whale
AB/D/Z call, Minke whale).

This module is imported by `train_joint.py`, which is what actually
trained the published SOMAD checkpoint (`--band joint --fusion-mode
twotower`, all 11 classes, both sub-bands). Running this file directly
instead trains the older, single-band precursor architecture described in
the manuscript's Discussion (the one that can't represent classes like
Humpback whose energy crosses the LF/HF sub-band boundary) -- kept here
only because `train_joint.py` needs the parser/defaults it defines, not as
a second recommended way to train SOMAD.

Usage (custom data location -- see README.md for the expected layout):

    python train_lf.py --data-root /path/to/data --output-dir /path/to/output/bigru_lf
"""

import argparse
import os

import common

DEFAULT_DATA_ROOT = "/scratch/disk3/a5278/bigru_aplose/data"  # override with --data-root
DEFAULT_OUTPUT_ROOT = "/scratch/disk3/a5278/bigru_aplose/output_v8"  # override with --output-dir


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                    help="Root folder containing the aplose/soi audio + annotation files, laid out flat "
                         "(see module docstring)")
    p.add_argument("--aplose-csv", action="append", default=None,
                    help="APLOSE-format annotation CSV -- repeatable, pass it more than once to combine "
                         "several sources (default when omitted entirely: the Gerlache export at "
                         "<data-root>/Gerlache_Strait_SN8059_-_2024_-_subset_1h_results (3).csv, plus "
                         "<data-root>/bluefin_library_annotations.csv if that file exists)")
    p.add_argument("--soi-csv", default=None,
                    help="SOI/AURAL annotation CSV (default: <data-root>/2024_soi_train/"
                         "annotation_aural_subset_2024_correct.csv)")
    p.add_argument("--hermine-dir", default=None,
                    help="Gerlache/hermine audio dir (default: <data-root>/hermine_SN8059_aplose_subset_2h)")
    p.add_argument("--soi-train-dir", default=None, help="SOI train audio dir (default: <data-root>/2024_soi_train)")
    p.add_argument("--soi-val-dir", default=None, help="SOI val audio dir (default: <data-root>/2024_soi_val)")
    p.add_argument("--bluefin-dir", default=None,
                    help="AcousticTrends_BlueFinLibrary audio dir, LF-only -- see common.resolve_audio_dirs "
                         "(default: <data-root>/bluefin_library_aplose if it exists)")
    p.add_argument("--bluefin-sites", nargs="*", default=None,
                    help="Restrict bluefin_library to only these site-years (e.g. ElephantIsland2013Aural "
                         "ElephantIsland2014). Default (omitted): every site-year bluefin_library has "
                         "converted data for. Pass with no arguments to explicitly select all. See "
                         "common.filter_bluefin_sites.")
    p.add_argument("--bluefin-hf-dir", default=None,
                    help="Joint band only: a folder of bluefin_library audio pre-resampled+dithered to the "
                         "HF sub-band's sample rate ahead of time. When set, PamSequenceDataset(band='joint') "
                         "loads a bluefin stem's HF sub-band audio from here instead of live-resampling its "
                         "normal 500 Hz file on every access -- see common.py's "
                         "PamSequenceDataset.bluefin_hf_index. Default (omitted): live resample+dither every "
                         "time (slower, but no precompute step needed first).")

    p.add_argument("--output-dir", default=os.path.join(DEFAULT_OUTPUT_ROOT, "bigru_lf"))
    p.add_argument("--run-name", default="bigru_lf")
    p.add_argument("--resume", default=None,
                    help="Path to a checkpoint to resume training from -- model/optimizer/scheduler/AMP-"
                         "scaler state, epoch number, best-val-F1-so-far, and early-stop patience counter are "
                         "all restored, so training continues from exactly where it left off rather than "
                         "restarting at epoch 1. Use <output-dir>/latest_checkpoint.pth (saved at the end of "
                         "EVERY epoch, unlike best_model.pth which only updates on a new best val F1 -- "
                         "resuming from best_model.pth would silently discard however many non-improving "
                         "epochs happened after it). Refuses to resume if the checkpoint's band/label list "
                         "doesn't match this run's -- e.g. after a BAND_CONFIG freq_bins change, that's a "
                         "different architecture, not a resumable one.")

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=30)
    p.add_argument("--num-workers", type=int, default=8,
                    help="DataLoader worker PROCESSES for audio decode/spectrogram preprocessing (default: "
                         "8). Each worker is capped to a single PyTorch CPU thread internally (see common.py's "
                         "_worker_init_single_thread) so --num-workers processes doesn't ALSO multiply by "
                         "PyTorch's own per-process thread count -- on a CPU-only run especially, that "
                         "combination can oversubscribe the machine's real core count badly enough to make a "
                         "'parallel' run slower than a single-worker one. Set this to roughly your machine's "
                         "(or your job's actual CPU allocation's, on a shared/HPC server -- not necessarily "
                         "the whole physical machine's core count) core count for CPU inference/training; "
                         "the main process's own thread count for the model's forward/backward pass is a "
                         "separate knob -- see OMP_NUM_THREADS / torch.set_num_threads, unaffected by this "
                         "flag.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=12,
                    help="Early-stopping patience in epochs (checkpoint saved is always the best-val-F1 "
                         "epoch regardless of patience; this just controls how long training keeps looking "
                         "for a better one before giving up).")
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--amp", action="store_true", help="Use mixed precision (recommended on any CUDA GPU).")
    p.add_argument("--use-attention", action="store_true", default=True,
                    help="Add a multi-head self-attention block on top of the BiGRU (default: on, this is "
                         "what SOMAD uses). Use --no-attention for a plain BiGRU instead.")
    p.add_argument("--no-attention", action="store_false", dest="use_attention",
                    help="Turn off the attention block (plain BiGRU).")
    # Only meaningful for band="joint" (see BAND_CONFIG["joint"] and
    # PamSequenceDataset/ImprovedCNNBiGRU); ignored for lf/hf, which only
    # ever have one sub-band to begin with.
    #   stack         -- concatenate LF+HF on the frequency axis into one
    #                    image, single joint min-max normalize after.
    #   stack_normfix -- same concat, but each sub-band is min-max
    #                    normalized independently before stacking.
    #   channels      -- both sub-bands resized to the same frequency
    #                    resolution and stacked as 2 input channels instead
    #                    of extra frequency rows.
    #   twotower      -- SOMAD's architecture: fully separate CNN stem per
    #                    sub-band, fused only after each tower's own pooled
    #                    features are flattened, right before the GRU.
    p.add_argument("--fusion-mode", choices=["stack", "stack_normfix", "channels", "twotower"],
                    default="stack",
                    help="How band='joint' combines its HF and LF sub-bands (default: stack -- pass "
                         "--fusion-mode twotower to train/reproduce SOMAD's actual architecture). No effect "
                         "for band='lf'/'hf'.")
    return p


def resolve_defaults(args):
    """Fill in unset paths from --data-root, assuming a flat layout (no
    "aplose_training" subfolder, everything directly under data-root).

    --aplose-csv is action="append" (a list, or None if never passed at
    all) -- when None, defaults to [Gerlache export], plus
    bluefin_library_annotations.csv IF PRESENT under data-root (unlike the
    other defaults below, this one can't unconditionally assume the file
    exists: unlike hermine_dir/soi_*_dir, which are directories that
    build_audio_index simply globs empty if absent, load_aplose_annotations
    hard-errors on a missing CSV path)."""
    root = args.data_root
    if args.aplose_csv is None:
        args.aplose_csv = [os.path.join(root, "Gerlache_Strait_SN8059_-_2024_-_subset_1h_results (3).csv")]
        bluefin_csv = os.path.join(root, "bluefin_library_annotations.csv")
        if os.path.exists(bluefin_csv):
            args.aplose_csv.append(bluefin_csv)
    if args.soi_csv is None:
        args.soi_csv = os.path.join(root, "2024_soi_train", "annotation_aural_subset_2024_correct.csv")
    if args.hermine_dir is None:
        args.hermine_dir = os.path.join(root, "hermine_SN8059_aplose_subset_2h")
    if args.soi_train_dir is None:
        args.soi_train_dir = os.path.join(root, "2024_soi_train")
    if args.soi_val_dir is None:
        args.soi_val_dir = os.path.join(root, "2024_soi_val")
    if args.bluefin_dir is None:
        args.bluefin_dir = os.path.join(root, "bluefin_library_aplose")
    return args


if __name__ == "__main__":
    args = resolve_defaults(build_argparser().parse_args())
    common.run_training("lf", args)
