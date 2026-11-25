# -*- coding: utf-8 -*-
# AI-assisted code: developed with Claude Code (Anthropic); reviewed and tested by the SOMAD authors.
"""Apply a trained LF or HF checkpoint to every WAV file in a folder (e.g.
the full APLOSE/hermine subset, not just the val split) and save the raw
per-0.5s-bin model output for every class, for every file, into ONE
netCDF (xarray) file covering the whole dataset.

The raw score is still the source of truth (kept in full, nothing lossy
happens to it), but for convenience this also bakes in each class's
calibrated threshold and the resulting per-bin "above threshold" boolean
-- both cheap to store and cheap to recompute, so if you ever want to
explore a different threshold, scores_to_events.py can override it without
re-running the (expensive, GPU) model pass, using the raw `scores` this
file still has.

Reuses the exact same windowing/spectrogram pipeline training used
(PamSequenceDataset, with an empty annotation table -- no labels needed for
inference) and the same continuous per-file stitching event-level eval
uses (common.run_inference_continuous): raw model output (logits) stitched
across overlapping windows, max score per bin.

Output: a single netCDF file (xarray.Dataset) with:
  dims: file, label, bin
  coords: file (stem), label (class name), bin (index),
          time_s(bin) = bin * label_interval, start_time(file) = datetime64
          (from each file's own embedded timestamp; NaT if unparseable)
  data vars:
    scores          (file, label, bin) float32 -- raw logits, NaN-padded
                     past each file's own duration (files can differ in
                     length; NaN always compares False, so above_threshold
                     is automatically 0 in the padded region too)
    duration        (file,) float32 seconds
    threshold       (label,) float32 -- from --thresholds-csv, logit space
    above_threshold (file, label, bin) int8 (0/1) -- scores > threshold,
                     nothing else (no smoothing/merging)
  attrs: band, label_interval

Run once per band (LF classes and HF classes need separate checkpoints).

**Always pass the *continuous*-calibrated thresholds CSV**
(thresholds_<band>_continuous.csv), not the plain window-calibrated one --
this script's scoring is continuous (run_inference_continuous, max-stitched
across overlapping windows), and continuous scores run systematically
higher than window ones, so a window-calibrated threshold applied here is
measurably too permissive. See README's "v3 changes" #9/#11.

Usage (nautilus, in Docker -- see run_nautilus_inference.sh):

    python run_inference.py \
        --checkpoint /output/bigru_lf/best_model.pth \
        --thresholds-csv /output/eval_lf/thresholds_lf_continuous.csv \
        --audio-dir /data/hermine_SN8059_aplose_subset_2h \
        --out /output/scores_lf.nc
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import xarray as xr
from tqdm import tqdm

import common
from somad_args import build_argparser, resolve_defaults


def main():
    p = build_argparser()
    p.add_argument("--band", choices=["cqt"], default="cqt", help="v17 has one band, cqt (default)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--thresholds-csv", required=True,
                    help="thresholds_<band>.csv from predict_eval.py -- its best_threshold per class gets "
                         "baked into the output netCDF alongside the raw scores")
    p.add_argument("--audio-dir", default=None,
                    help="Folder of WAVs to run inference over (default: <data-root>/hermine_SN8059_aplose_subset_2h "
                         "-- the full APLOSE/hermine subset, not just its val split)")
    p.add_argument("--out", required=True, help="Output netCDF path (e.g. scores_lf.nc)")
    p.add_argument("--single-process", action="store_true",
                    help="Use common.run_inference_continuous_singleprocess instead of the default "
                         "DataLoader-based common.run_inference_continuous -- no worker processes, no "
                         "--num-workers/thread-oversubscription tuning, and a bad window is skipped with a "
                         "warning instead of a worker-process crash taking down the whole run. Simpler and "
                         "easier to reason about for a CPU-only deployment run; costs the decode/model-forward "
                         "overlap a >0-worker DataLoader gives you, which mainly matters if audio decode "
                         "(not model compute) is your actual bottleneck. --num-workers is ignored when this "
                         "is set.")
    p.add_argument("--no-overlap", action="store_true",
                    help="Cut windows back-to-back (snippet_overlap=1) instead of the default 50%% hop -- "
                         "must match whatever predict_eval.py used to calibrate the --thresholds-csv you're "
                         "passing (see its --no-overlap help for why). Since hermine files are exactly 60s "
                         "== snippet_length_s, this is usually a no-op for the default --audio-dir (each "
                         "file already gets exactly one window either way) -- it mainly matters if you point "
                         "--audio-dir at longer recordings.")
    p.add_argument("--filename-time-format", default=None,
                    help="strptime format for extracting each file's start time from its filename -- "
                         "overriding the project's default %%Y_%%m_%%d_%%H_%%M_%%S convention (the AURAL/SOI "
                         "mooring's own naming, so that one needs no flag at all) -- for deploying over audio "
                         "that isn't named the way this project's own training data is. E.g. SoundTrap-style "
                         "moorings (<serial>.<YYMMDDHHMMSS>.wav, e.g. '8712.250127120004.wav' -> "
                         "2025-01-27 12:00:04): pass '%%y%%m%%d%%H%%M%%S' -- the leading serial number doesn't "
                         "confuse the search, it's unanchored. Supported directives: %%Y %%y %%m "
                         "%%d %%H %%M %%S. Test a candidate format against your actual files first with "
                         "check_wav_timestamps.py --format (same parsing logic, no GPU/Docker model run "
                         "needed just to check this). Files that don't match get a blank (NaT) start_time "
                         "in the output netCDF, not a crash.")
    args = resolve_defaults(p.parse_args())

    band = args.band
    if args.no_overlap:
        common.BAND_CONFIG[band]["snippet_overlap"] = 1
        print(f"--no-overlap: cutting {band} windows back-to-back "
              f"(snippet_length_s={common.BAND_CONFIG[band]['snippet_length_s']}s, no overlap)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        # Visibility into the two separate CPU-parallelism knobs -- how many
        # DataLoader worker PROCESSES (--num-workers, each capped to 1
        # PyTorch thread internally, see common._worker_init_single_thread)
        # vs. how many threads the MAIN process's own model forward pass
        # uses (torch.get_num_threads(), driven by OMP_NUM_THREADS if set,
        # otherwise PyTorch's own auto-detected default). Printed so an
        # oversubscription/undersubscription mismatch against the machine's
        # (or your job's actual allocation's, on a shared server) real core
        # count is visible up front, not something to guess at from how
        # fast/slow a run turns out to be.
        print(f"CPU inference: os.cpu_count()={os.cpu_count()}, --num-workers={args.num_workers} worker "
              f"process(es) (1 torch thread each), main process torch.get_num_threads()="
              f"{torch.get_num_threads()} (set OMP_NUM_THREADS before launching to change this)")

    audio_dir = args.audio_dir or os.path.join(args.data_root, "hermine_SN8059_aplose_subset_2h")
    audio_index = common.build_audio_index([audio_dir])
    stems = sorted(audio_index.keys())
    print(f"Found {len(stems)} audio files under {audio_dir}")
    if not stems:
        raise SystemExit(f"No .wav files found under {audio_dir}")

    # Architecture (band_config incl. the CQT settings, use_attention) and
    # label order come from the checkpoint, not from today's common.py.
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("band", band) != band:
        print(f"WARNING: checkpoint was trained as band={checkpoint.get('band')!r}, "
              f"but --band={band!r} was requested")
    # v13: evaluate with the class set the checkpoint was TRAINED with (matters once
    # classes have been added to class_config.json after this checkpoint was made)
    if checkpoint.get('class_config'):
        common.configure_classes(checkpoint['class_config'])
    label_list = checkpoint.get("model_labels", common.BAND_LABELS[band])

    # No annotations needed for pure inference -- PamSequenceDataset's
    # labeling machinery just no-ops on an empty table (every window comes
    # back with an all-zero label, which run_inference_continuous ignores
    # -- we only use its "scores" output here).
    df_anno_empty = pd.DataFrame(columns=["file", "start_datetime", "end_datetime", "raw_label",
                                           "source", "label", "stem", "t_1_s", "t_2_s"])
    # full_coverage=True: deployment scoring only (never training/eval --
    # see PamSequenceDataset's own docstring) -- adds one extra window
    # anchored to each file's true end when the regular hop-spaced grid
    # would otherwise leave a real gap there (confirmed on real files: the
    # last several seconds to ~30s of a file left permanently NaN
    # otherwise). The file's true beginning is already always covered by
    # the grid's first window (always starts at 0), so no equivalent fix
    # is needed there.
    dataset = common.PamSequenceDataset(stems, audio_index, df_anno_empty, band=band, augment=False,
                                         full_coverage=True,
                                         bluefin_hf_dir=getattr(args, "bluefin_hf_dir", None))

    model = common.load_model_from_checkpoint(checkpoint, device)   # v17: CQT single tower
    print("Checkpoint epoch:", checkpoint.get("epoch"), "val F1 at save time:", checkpoint.get("best_f1"),
          "band_config:", checkpoint["band_config"])

    if "continuous" not in os.path.basename(args.thresholds_csv):
        print(f"WARNING: {args.thresholds_csv} doesn't look like a continuous-calibrated thresholds CSV "
              f"(filename has no 'continuous'). This script always scores with run_inference_continuous "
              f"(max-stitched across overlapping windows), and continuous scores run systematically higher "
              f"than window ones -- a window-calibrated threshold applied here is measurably too permissive. "
              f"Pass thresholds_{band}_continuous.csv from predict_eval.py unless you specifically intend "
              f"otherwise.")

    thr_df = pd.read_csv(args.thresholds_csv).set_index("label")
    thresholds = np.array([thr_df.loc[lb, "best_threshold"] for lb in label_list], dtype=np.float32)
    print("Using thresholds (logit space):", dict(zip(label_list, thresholds)))
    nan_classes = [lb for lb, t in zip(label_list, thresholds) if np.isnan(t)]
    if nan_classes:
        print(f"WARNING: no calibrated threshold (no scored validation positives) for {nan_classes} -- "
              f"these classes are never flagged above_threshold; their raw scores are still written.")

    if args.single_process:
        print("--single-process: running without DataLoader/worker processes "
              f"(main process torch.get_num_threads()={torch.get_num_threads()})")
        continuous = common.run_inference_continuous_singleprocess(model, dataset, device,
                                                                     batch_size=args.batch_size)
    else:
        continuous = common.run_inference_continuous(model, dataset, device,
                                                       batch_size=args.batch_size, num_workers=args.num_workers)

    print("Assembling combined netCDF...")
    stems_sorted = sorted(continuous.keys())
    max_bins = max(d["scores"].shape[1] for d in continuous.values())
    num_classes = len(label_list)

    scores_arr = np.full((len(stems_sorted), num_classes, max_bins), np.nan, dtype=np.float32)
    duration_arr = np.zeros(len(stems_sorted), dtype=np.float32)
    start_time_arr = np.empty(len(stems_sorted), dtype="datetime64[ns]")

    for i, stem in enumerate(tqdm(stems_sorted, desc="Stacking per-file scores")):
        d = continuous[stem]
        n_bins_i = d["scores"].shape[1]
        # Bins no window actually covered are left at a -1e4 sentinel in
        # d["scores"], not a real model output -- full_coverage=True above
        # (PamSequenceDataset) means this should now be rare (an extra
        # window closes the file-tail gap that used to leave the last
        # several seconds to ~30s of a file uncovered every time), but a
        # sub-bin rounding remainder can still occasionally leave one, so
        # this masking stays regardless. Mask those to NaN before they enter
        # the netCDF, same as the ragged-length padding beyond n_bins_i
        # below -- otherwise a leftover -1e4 poisons any nanmin/nanmax over
        # "scores" (e.g. plot_file_detections.py's per-file colorscale) the
        # same way it broke predict_eval.py's threshold search.
        # above_threshold is unaffected either way (NaN and -1e4 both
        # compare False), only the raw score value was wrong.
        scores_i = d["scores"].copy()
        scores_i[:, ~d["covered"]] = np.nan
        scores_arr[i, :, :n_bins_i] = scores_i
        duration_arr[i] = d["duration"]
        try:
            start_time_arr[i] = np.datetime64(common.gettime(stem, fmt=args.filename_time_format))
        except ValueError:
            start_time_arr[i] = np.datetime64("NaT")

    # NaN always compares False, so the padded region beyond each file's
    # own duration automatically comes out 0 here too -- no special-casing.
    above_threshold_arr = (scores_arr > thresholds[None, :, None]).astype(np.int8)

    ds = xr.Dataset(
        data_vars=dict(
            scores=(["file", "label", "bin"], scores_arr),
            duration=(["file"], duration_arr),
            threshold=(["label"], thresholds),
            above_threshold=(["file", "label", "bin"], above_threshold_arr),
        ),
        coords=dict(
            file=stems_sorted,
            label=label_list,
            bin=np.arange(max_bins),
            time_s=("bin", np.arange(max_bins) * dataset.label_interval),
            start_time=("file", start_time_arr),
        ),
        attrs=dict(band=band, label_interval=dataset.label_interval),
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    encoding = {"scores": {"zlib": True, "complevel": 4}, "above_threshold": {"zlib": True, "complevel": 4}}
    ds.to_netcdf(args.out, encoding=encoding)

    print(f"\nWrote {len(stems_sorted)} files' scores to {args.out} "
          f"({os.path.getsize(args.out) / 1e6:.1f} MB)")
    print(f"Total bins above threshold per class:")
    print(pd.Series(above_threshold_arr.sum(axis=(0, 2)), index=label_list))


if __name__ == "__main__":
    main()
