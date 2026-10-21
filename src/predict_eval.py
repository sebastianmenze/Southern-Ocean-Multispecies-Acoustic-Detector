# -*- coding: utf-8 -*-
"""Evaluate a trained LF or HF BiGRU checkpoint on its validation split:
runs inference over the val set, finds the best-F1 decision threshold per
class from a precision/recall curve, and saves plots + a thresholds CSV.

v3: calibrates thresholds TWO ways and reports both, rather than silently
picking one -- see the "window" vs "continuous" note in main() for why.

Usage (nautilus, v2 output tree -- see README.md):

    python predict_eval.py --band lf --checkpoint /scratch/disk3/a5278/bigru_aplose/output_v5/bigru_lf/best_model.pth
    python predict_eval.py --band hf --checkpoint /scratch/disk3/a5278/bigru_aplose/output_v5/bigru_hf/best_model.pth \
        --output-dir /scratch/disk3/a5278/bigru_aplose/output_v5/eval_hf

Prefer `./run_nautilus_eval.sh {lf|hf}` over calling this directly --
it sets `--shm-size=16g`, which this needs when run in Docker (the default
container shm is too small for the DataLoader workers and fails with a
misleading "No space left on device").
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

import common
from train_lf import build_argparser, resolve_defaults


def fit_thresholds(y_true_2d: np.ndarray, y_pred_2d: np.ndarray, label_list, curve_points: int = 500):
    """Per-class best-F1 threshold + average precision from a PR curve,
    given pooled (n_samples, n_classes) arrays. Returns (thresholds_best,
    f1_best, avg_precision, curve_df) -- curve_df has the full
    precision/recall/fscore-vs-threshold sweep per class, for
    plotting/saving."""
    thresholds_best, f1_best, avg_precision = [], [], []
    curve_df = pd.DataFrame([])
    for i_class, lb in enumerate(label_list):
        y_true = y_true_2d[:, i_class]
        y_pred = y_pred_2d[:, i_class]

        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        # sklearn: precision[i]/recall[i] correspond to thresholds[i] for
        # i in 0..len(thresholds)-1; precision[-1]=1/recall[-1]=0 is a
        # boundary point (threshold=+inf) with no corresponding entry in
        # `thresholds`, so it's precision[:-1]/recall[:-1] that lines up
        # with `thresholds`, not precision[1:]/recall[1:] (off-by-one --
        # pairs each threshold with the next, higher one's precision/recall
        # instead of its own).
        t = np.linspace(thresholds.min(), thresholds.max(), curve_points)
        prec_i = np.interp(t, thresholds, precision[:-1])
        rec_i = np.interp(t, thresholds, recall[:-1])
        fscore = (2 * prec_i * rec_i) / (prec_i + rec_i + 1e-12)

        curve_df[f"thresholds_{i_class}"] = t
        curve_df[f"precision_{i_class}"] = prec_i
        curve_df[f"recall_{i_class}"] = rec_i
        curve_df[f"fscore_{i_class}"] = fscore

        ix = np.argmax(fscore)
        thresholds_best.append(float(t[ix]))
        f1_best.append(float(fscore[ix]))
        avg_precision.append(float(average_precision_score(y_true, y_pred)))

    return thresholds_best, f1_best, avg_precision, curve_df


def plot_pr_curves(curve_df, thresholds_best, f1_best, label_list, rare_classes, title, out_path):
    plt.figure(figsize=(7, 6))
    for i_class, lb in enumerate(label_list):
        flag = " [LOW N]" if lb in rare_classes else ""
        rec_i = curve_df[f"recall_{i_class}"]
        prec_i = curve_df[f"precision_{i_class}"]
        fscore = curve_df[f"fscore_{i_class}"]
        ix = int(np.argmax(fscore))
        plt.plot(rec_i, prec_i, "-", label=lb + flag)
        plt.plot(rec_i[ix], prec_i[ix], ".k")
    plt.grid()
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def per_file_frame_metrics(continuous: dict, label_list, thresholds) -> pd.DataFrame:
    """Frame-level precision/recall/F1 broken down per validation file per
    class (not pooled across the whole val set), from the same continuous
    per-file timelines event scoring uses. Lets you tell "this class is
    just hard" apart from "this class's score is being dragged down by a
    handful of unusual files"."""
    rows = []
    for stem, d in continuous.items():
        for i_class, lb in enumerate(label_list):
            scores = d["scores"][i_class]
            true = d["true"][i_class]
            pred = (scores > thresholds[i_class]).astype(int)
            true_i = true.astype(int)
            tp = int(((pred == 1) & (true_i == 1)).sum())
            fp = int(((pred == 1) & (true_i == 0)).sum())
            fn = int(((pred == 0) & (true_i == 1)).sum())
            precision = tp / (tp + fp) if (tp + fp) else np.nan
            recall = tp / (tp + fn) if (tp + fn) else np.nan
            # NaN only when precision or recall is itself undefined (no
            # predicted/true positives at all in this file); when both are
            # legitimately 0 (predictions exist but none correct), F1 is 0
            # by convention, not NaN.
            if np.isnan(precision) or np.isnan(recall):
                f1 = np.nan
            elif precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            rows.append(dict(stem=stem, label=lb, duration_s=round(d["duration"], 1),
                              n_true_bins=int(true_i.sum()), tp=tp, fp=fp, fn=fn,
                              precision=precision, recall=recall, f1=f1))
    return pd.DataFrame(rows)


def main():
    p = build_argparser()
    p.add_argument("--band", choices=["lf", "hf", "joint"], required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--no-overlap", action="store_true",
                    help="Cut windows back-to-back (snippet_overlap=1) instead of the default 50%% hop -- "
                         "e.g. non-overlapping 60s chunks for LF/HF, since snippet_length_s=60 for both. "
                         "Doesn't change the model or its 60s/121-timestep input shape, only how many "
                         "windows a file gets cut into and where they start. With no overlap, 'window' and "
                         "'continuous' pooling become the same underlying scores (no window ever covers "
                         "another's bins, so there's nothing to max-stitch), which sidesteps the whole "
                         "overlap-stitching/coverage-sentinel bug class this pipeline has had (see README's "
                         "'v3 changes' #9/#11/#12) at the cost of fewer, less-overlapping training/eval "
                         "samples per file. Diagnostic/comparison mode, not a replacement for the default.")
    p.add_argument("--exclude-bluefin", action="store_true",
                    help="Restrict eval to the real APLOSE/hermine + SOI/AURAL data only, dropping "
                         "AcousticTrends_BlueFinLibrary from dataset construction (its annotation CSV and "
                         "--bluefin-dir audio). A checkpoint trained WITH bluefin has a val_files.txt that "
                         "still lists bluefin stems (named after their original site, e.g. "
                         "'..._kerguelen2005_...', not containing 'bluefin' -- so they can't be filtered by "
                         "name alone); those are dropped from the val split too since they'd otherwise "
                         "KeyError against the now-restricted audio_index.")
    args = resolve_defaults(p.parse_args())

    band = args.band
    if args.no_overlap:
        common.BAND_CONFIG[band]["snippet_overlap"] = 1
        print(f"--no-overlap: cutting {band} windows back-to-back "
              f"(snippet_length_s={common.BAND_CONFIG[band]['snippet_length_s']}s, no overlap)")
    if args.exclude_bluefin:
        args.aplose_csv = [c for c in args.aplose_csv if "bluefin" not in os.path.basename(c).lower()]
        args.bluefin_dir = None
        print(f"--exclude-bluefin: aplose_csv restricted to {args.aplose_csv}, bluefin_dir dropped")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    audio_dirs = common.resolve_audio_dirs(args, band)
    df_anno, audio_index = common.build_dataset_table(args.aplose_csv, args.soi_csv, audio_dirs)

    val_files_path = os.path.join(os.path.dirname(args.checkpoint), "val_files.txt")
    if os.path.exists(val_files_path):
        with open(val_files_path) as f:
            val_stems = [l.strip() for l in f if l.strip()]
    else:
        _, val_stems = common.get_split(args, audio_index, df_anno)

    if args.exclude_bluefin:
        n_before = len(val_stems)
        val_stems = [s for s in val_stems if s in audio_index]
        print(f"--exclude-bluefin: dropped {n_before - len(val_stems)} bluefin val file(s) from the saved "
              f"split, {len(val_stems)} real APLOSE/SOI val file(s) remain")

    # Read architecture/label metadata back from the checkpoint rather than
    # from today's common.py defaults, so eval always matches whatever the
    # checkpoint was actually trained with (band_config, use_attention,
    # label order, fusion_mode can all drift over time as the pipeline
    # evolves). Loaded before val_set so the dataset is built with the
    # SAME fusion_mode the checkpoint was trained on -- "channels"/
    # "twotower" produce a differently-shaped input than the "stack"
    # default, so getting this wrong crashes (or silently mismatches) the
    # very first inference call.
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("band", band) != band:
        print(f"WARNING: checkpoint was trained as band={checkpoint.get('band')!r}, "
              f"but --band={band!r} was requested")
    # v13: evaluate with the class set the checkpoint was TRAINED with (matters once
    # classes have been added to class_config.json after this checkpoint was made)
    if checkpoint.get('class_config'):
        common.configure_classes(checkpoint['class_config'])
    label_list = checkpoint.get("model_labels", common.BAND_LABELS[band])
    band_config = checkpoint.get("band_config", common.BAND_CONFIG[band])
    use_attention = checkpoint.get("use_attention", False)
    fusion_mode = checkpoint.get("fusion_mode", "stack")
    rare_classes = set(checkpoint.get("rare_classes", []))

    val_set = common.PamSequenceDataset(val_stems, audio_index, df_anno, band=band, augment=False,
                                         bluefin_hf_dir=getattr(args, "bluefin_hf_dir", None),
                                         fusion_mode=fusion_mode)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    in_channels = 2 if (band == "joint" and fusion_mode == "channels") else 1
    model = common.ImprovedCNNBiGRU(
        in_channels=in_channels,
        num_classes=len(label_list),
        timesteps=common.timesteps_for_band(band),
        fsteps=band_config["freq_bins"],
        hidden_size=band_config["hidden_size"],
        gru_layers=band_config.get("gru_layers", 3),
        use_attention=use_attention,
        pooled_freq_rows=band_config.get("pooled_freq_rows", 32),
        fusion_mode=fusion_mode,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Checkpoint epoch:", checkpoint.get("epoch"), "val F1 at save time:", checkpoint.get("best_f1"),
          "use_attention:", use_attention)
    if rare_classes:
        print(f"NOTE: {sorted(rare_classes)} had < {common.MIN_CLASS_COUNT_WARNING} training instances -- "
              f"their thresholds/F1 below are not reliable.")

    os.makedirs(args.output_dir, exist_ok=True)

    # -- "window" calibration: pool each overlapping window's own raw
    # score independently (no dedup across the 50%-overlap) -----------
    label_pred, label_true = [], []
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Inference"):
            x = common._to_device(x, device)
            yp = model(x)
            for k in range(y.shape[0]):
                label_true.append(pd.DataFrame(y.numpy()[k, :, :].T))
                label_pred.append(pd.DataFrame(yp.detach().cpu().numpy()[k, :, :].T))
    label_true = pd.concat(label_true, ignore_index=True).values
    label_pred = pd.concat(label_pred, ignore_index=True).values

    thresholds_window, f1_window, ap_window, curve_window = fit_thresholds(label_true, label_pred, label_list)
    for i_class, lb in enumerate(label_list):
        flag = " [LOW N]" if lb in rare_classes else ""
        print(f"{lb}{flag}: [window] best_threshold={thresholds_window[i_class]:.4f} "
              f"f1={f1_window[i_class]:.4f} avg_precision={ap_window[i_class]:.4f}")
    plot_pr_curves(curve_window, thresholds_window, f1_window, label_list, rare_classes,
                    f"{band.upper()} model - precision/recall by class (window)",
                    os.path.join(args.output_dir, f"pr_{band}.png"))
    curve_window.to_csv(os.path.join(args.output_dir, f"df_pr_model_{band}.csv"), index=False)

    thresholds_df = pd.DataFrame({
        "label": label_list,
        "best_threshold": thresholds_window,
        "best_f1": f1_window,
        "avg_precision": ap_window,
        "low_sample_warning": [lb in rare_classes for lb in label_list],
    })
    thresholds_df.to_csv(os.path.join(args.output_dir, f"thresholds_{band}.csv"), index=False)

    # -- v3: continuous-timeline stitching, needed for event-level
    # metrics/per-file breakdown/run_inference.py's deployment scores
    # (PamSequenceDataset's windows overlap 50%, so concatenating raw
    # per-window predictions -- as the "window" pass above does -- would
    # double-count/fragment events; this dedups by taking the max score
    # per bin across whichever windows cover it). ----------------------
    print("\nRunning continuous inference for event-level metrics + per-file breakdown...")
    continuous = common.run_inference_continuous(model, val_set, device,
                                                   batch_size=args.batch_size, num_workers=args.num_workers)

    # -- "continuous" calibration: same PR-curve fit, but against the
    # max-stitched scores instead of raw per-window ones. Since
    # max(a, b) >= a, b, continuous scores run systematically higher than
    # window scores for the same bins -- a threshold picked on "window"
    # data is being applied, everywhere downstream, to a scoring
    # population it was never actually measured against. This second fit
    # is the one that's internally consistent with what event-level
    # metrics, per-file metrics, and run_inference.py's deployment output
    # actually threshold. Reported side by side with "window" rather than
    # silently replacing it -- compare both before deciding which
    # thresholds_<band>*.csv to actually deploy with.
    # Drop bins no window ever covered (file-tail rounding edge cases --
    # see run_inference_continuous's docstring) before pooling. Left in,
    # their -1e4 sentinel becomes the global min score for a class and
    # `fit_thresholds`' linspace(min, max, curve_points) grid gets stretched
    # ~2000x, squeezing the entire real decision region (a handful of
    # logits) into ~1 grid point and making the best-F1 threshold search
    # pick off a flat "predict everything" plateau instead of the real
    # boundary. average_precision_score is unaffected (sklearn computes it
    # off the true PR curve, not this interpolation), which is why AP can
    # look reasonable even when best_threshold/best_f1 are nonsense -- a
    # sign to check for exactly this if it happens again.
    cont_true = np.concatenate([d["true"][:, d["covered"]].T for d in continuous.values()], axis=0)
    cont_pred = np.concatenate([d["scores"][:, d["covered"]].T for d in continuous.values()], axis=0)
    thresholds_cont, f1_cont, ap_cont, curve_cont = fit_thresholds(cont_true, cont_pred, label_list)
    for i_class, lb in enumerate(label_list):
        flag = " [LOW N]" if lb in rare_classes else ""
        print(f"{lb}{flag}: [continuous] best_threshold={thresholds_cont[i_class]:.4f} "
              f"f1={f1_cont[i_class]:.4f} avg_precision={ap_cont[i_class]:.4f}")
    plot_pr_curves(curve_cont, thresholds_cont, f1_cont, label_list, rare_classes,
                    f"{band.upper()} model - precision/recall by class (continuous)",
                    os.path.join(args.output_dir, f"pr_{band}_continuous.png"))
    curve_cont.to_csv(os.path.join(args.output_dir, f"df_pr_model_{band}_continuous.csv"), index=False)

    thresholds_cont_df = pd.DataFrame({
        "label": label_list,
        "best_threshold": thresholds_cont,
        "best_f1": f1_cont,
        "avg_precision": ap_cont,
        "low_sample_warning": [lb in rare_classes for lb in label_list],
    })
    thresholds_cont_df.to_csv(os.path.join(args.output_dir, f"thresholds_{band}_continuous.csv"), index=False)

    comparison = pd.DataFrame({
        "label": label_list,
        "threshold_window": thresholds_window,
        "threshold_continuous": thresholds_cont,
        "threshold_delta": np.array(thresholds_cont) - np.array(thresholds_window),
        "f1_window": f1_window,
        "f1_continuous": f1_cont,
        "f1_delta": np.array(f1_cont) - np.array(f1_window),
    })
    comparison.to_csv(os.path.join(args.output_dir, f"threshold_calibration_comparison_{band}.csv"), index=False)
    print(f"\nWindow vs. continuous calibration (also saved to "
          f"threshold_calibration_comparison_{band}.csv):")
    print(comparison.to_string(index=False))

    # Event-level metrics under BOTH calibrations -- but both are always
    # scored against `continuous` (the stitched per-file timeline; there's
    # no separate "window" event timeline, events need one continuous
    # sequence per file to exist at all). The threshold picked FROM window
    # calibration is included too, but only as an explicit comparison
    # baseline -- continuous scores run systematically higher than window
    # scores (max-stitched over overlapping windows), so a window threshold
    # applied here is measurably too permissive (see #9/#11 in the
    # README's "v3 changes"). "continuous" thresholds are the correct,
    # internally-consistent pairing for this data and get the unsuffixed
    # filename; "window" thresholds applied to the same continuous scores
    # get an explicit suffix so it can't be mistaken for the recommended
    # output.
    for tag, thresholds_to_use in [("continuous", thresholds_cont), ("window", thresholds_window)]:
        event_rows = []
        for i_class, lb in enumerate(label_list):
            m = common.event_level_metrics(continuous, i_class, thresholds_to_use[i_class], val_set.label_interval)
            flag = " [LOW N]" if lb in rare_classes else ""
            print(f"{lb}{flag}: event-level [{tag}] precision={m['precision']:.3f} recall={m['recall']:.3f} "
                  f"f1={m['f1']:.3f} (tp={m['tp']} fp={m['fp']} fn={m['fn']})")
            event_rows.append(dict(label=lb, **m))
        suffix = "" if tag == "continuous" else "_window_threshold"
        pd.DataFrame(event_rows).to_csv(os.path.join(args.output_dir, f"event_metrics_{band}{suffix}.csv"), index=False)

    # Same reasoning for the per-file breakdown -- it's computed from the
    # same continuous timelines, so continuous-calibrated thresholds are
    # the default/unsuffixed output; window-calibrated is the explicitly
    # labeled comparison.
    for tag, thresholds_to_use in [("continuous", thresholds_cont), ("window", thresholds_window)]:
        per_file_df = per_file_frame_metrics(continuous, label_list, thresholds_to_use)
        suffix = "" if tag == "continuous" else "_window_threshold"
        out_path = os.path.join(args.output_dir, f"per_file_f1_{band}{suffix}.csv")
        per_file_df.to_csv(out_path, index=False)
        if tag == "continuous":
            # Files with at least one true positive instance, sorted
            # worst-F1-first, printed as a quick "what's dragging this
            # class down" pointer -- full breakdown (including files with
            # no instances of the class at all) is in the CSV.
            with_calls = per_file_df[per_file_df["n_true_bins"] > 0].sort_values("f1")
            if len(with_calls):
                print(f"\nWorst per-file F1 among files with >=1 true positive bin (see "
                      f"{out_path} for the full breakdown):")
                print(with_calls.head(10).to_string(index=False))

    # training curves, if the log csv is next to the checkpoint
    log_csv = os.path.join(os.path.dirname(args.checkpoint), f"bigru_{band}.csv")
    if os.path.exists(log_csv):
        log = pd.read_csv(log_csv)
        plt.figure(figsize=(6, 6))
        plt.subplot(211)
        plt.plot(log["epoch"], log["train_f1"], "-k", label="train")
        plt.plot(log["epoch"], log["val_f1"], "-r", label="val")
        plt.ylabel("F1")
        plt.legend()
        plt.grid()
        plt.subplot(212)
        plt.plot(log["epoch"], log["train_loss"], "-k", label="train")
        plt.plot(log["epoch"], log["val_loss"], "-r", label="val")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f"checkpoints_{band}.png"))

    print(f"Saved plots + thresholds to {args.output_dir}")


if __name__ == "__main__":
    main()
