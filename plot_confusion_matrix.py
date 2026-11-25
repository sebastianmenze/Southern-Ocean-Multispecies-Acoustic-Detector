# -*- coding: utf-8 -*-
# AI-assisted code: developed with Claude Code (Anthropic); reviewed and tested by the SOMAD authors.
"""Confusion matrix plots for a trained LF or HF checkpoint's validation
split. Two complementary views, since this is a multi-label problem
(several classes can be true/predicted in the very same 0.5s bin -- there
is no single "the" class per bin the way a classic single-label confusion
matrix assumes):

  1. confusion_matrix_per_class_<band>.png -- one 2x2 (TN/FP/FN/TP) matrix
     per class. The technically correct view: no information thrown away,
     directly matches how the model is actually trained/thresholded (each
     class's own independent binary decision).

  2. confusion_matrix_combined_<band>.png -- one (N+1)x(N+1) matrix across
     all classes plus a "Background" row/column, the classic "what gets
     confused with what" cross-class view. Built by collapsing each bin to
     its single highest-RAW-SCORE class above threshold (or Background if
     none crossed) for predictions -- raw score, not margin (score minus
     threshold): a class with a near-degenerate, unusually permissive
     threshold (e.g. Blue whale Z call, ~12 training instances) has an
     enormous margin on nearly every bin regardless of whether it's really
     the most confident prediction, which silently dominated every row's
     argmax before this was raw-score-based (confirmed on synthetic data:
     every row, including Background, argmaxed to the degenerate class at
     100%). Similarly collapses ground truth to its first true class (or
     Background if none) -- a simplification that discards information on
     the rare bin where multiple classes are genuinely simultaneously
     true/predicted, so treat this as a complementary summary of
     cross-class confusion, not the full accounting -- (1) above is that.
     Row-normalized (each row sums to ~100%) since raw counts would be
     totally dominated by the Background->Background cell.

--exclude-labels (default: Blue whale Z call) leaves a class out of BOTH
plots entirely, not just the combined one -- a degenerate-threshold class
still wins the argmax on nearly every bin where nothing else fires even
with raw-score tie-breaking (it's not a tie-breaking problem, it really
does cross its own threshold almost everywhere), drowning out the more
meaningful cross-class confusion in the combined plot; and its own
precision/recall are already known to be near-meaningless at this sample
size (see README's "Rare-class handling"), so there's little reason to
keep it cluttering the per-class grid either while it's this data-starved.
Pass --exclude-labels with no arguments to see every class in both plots.

Uses the SAME continuous-timeline stitching (common.run_inference_continuous)
and expects the same CONTINUOUS-calibrated thresholds as the rest of the
deployment pipeline (see README's "v3 changes" #9/#11/#12 for why --
window-calibrated thresholds are measurably too permissive against
continuous-stitched scores) -- pass thresholds_<band>_continuous.csv, not
the plain thresholds_<band>.csv.

Also saves the raw counts behind both plots as CSVs.

Usage (nautilus, in Docker -- see run_nautilus_confusion_matrix.sh):

    python plot_confusion_matrix.py \
        --checkpoint /output/bigru_lf/best_model.pth \
        --thresholds-csv /output/eval_lf/thresholds_lf_continuous.csv \
        --output-dir /output/confusion_lf
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from torch.utils.data import DataLoader

import common
import eval_scores
from somad_args import build_argparser, resolve_defaults


def plot_per_class(all_true, all_pred, label_list, out_dir, band, all_mask=None):
    """v19: `all_mask` (N, C, 1 = scored) -> each class's 2x2 only counts its scored bins."""
    if all_mask is None:
        cms = multilabel_confusion_matrix(all_true, all_pred)  # (C, 2, 2): [[TN, FP], [FN, TP]]
    else:
        cms = []
        for i in range(all_true.shape[1]):
            k = all_mask[:, i] > 0
            t, pr = all_true[k, i], all_pred[k, i]
            cms.append(np.array([[((t == 0) & (pr == 0)).sum(), ((t == 0) & (pr == 1)).sum()],
                                 [((t == 1) & (pr == 0)).sum(), ((t == 1) & (pr == 1)).sum()]]))

    n = len(label_list)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    rows = []
    for i, lb in enumerate(label_list):
        tn, fp, fn, tp = cms[i].ravel()
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = (2 * precision * recall / (precision + recall)
              if (tp + fp) and (tp + fn) and (precision + recall) else float("nan"))
        rows.append(dict(label=lb, tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                          precision=precision, recall=recall, f1=f1))

        ax = axes[i]
        mat = np.array([[tn, fp], [fn, tp]])
        im = ax.imshow(mat, cmap="Blues")
        for r in range(2):
            for c in range(2):
                frac = mat[r, c] / mat.sum() if mat.sum() else 0
                ax.text(c, r, f"{mat[r, c]:,}\n({frac:.1%})", ha="center", va="center",
                        color="white" if mat[r, c] > mat.max() / 2 else "black", fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pred: no", "Pred: yes"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["True: no", "True: yes"])
        f1_str = f"{f1:.3f}" if f1 == f1 else "n/a"
        ax.set_title(f"{lb}\nprecision={precision:.3f} recall={recall:.3f} f1={f1_str}", fontsize=9)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(f"{band.upper()} — per-class confusion matrix (validation split, 0.5s bins"
                 f"{', masked: only bins where the class was annotated' if all_mask is not None else ''})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"confusion_matrix_per_class_{band}.png"), dpi=110)
    plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, f"confusion_matrix_per_class_{band}.csv"), index=False)
    return df


def plot_combined(all_true, all_pred, all_scores, thresholds, label_list, out_dir, band):
    """`label_list`/`all_true`/`all_pred`/`all_scores`/`thresholds` should
    already have any --exclude-labels classes filtered out (main() does
    this once, upstream, for both plot_per_class and this function --
    see main()'s comment for why a degenerate-threshold class needs
    excluding here specifically)."""
    labels_ext = list(label_list) + ["Background"]
    bg_idx = len(label_list)

    # Tie-break on raw score, NOT margin (score - threshold): thresholds
    # are calibrated independently per class and can land on wildly
    # different absolute values (a near-degenerate low-N class can end up
    # with a threshold far below every other class's, e.g. -30 vs ~0 --
    # see README's "Rare-class handling"). That class's margin would then
    # be enormous for basically any bin, silently dominating every row's
    # argmax regardless of whether it's really the most confident
    # prediction -- confirmed on synthetic data mimicking exactly that
    # pattern: every single row, including Background, argmaxed to the
    # degenerate class at 100% before this fix. Raw score doesn't have
    # that per-class offset baked in, so it stays comparable across
    # classes trained under the same shared BCE loss.
    any_pred = all_pred.any(axis=1)
    pred_idx = np.where(any_pred, np.argmax(np.where(all_pred, all_scores, -np.inf), axis=1), bg_idx)

    any_true = all_true.any(axis=1)
    true_idx = np.where(any_true, np.argmax(all_true, axis=1), bg_idx)

    cm = confusion_matrix(true_idx, pred_idx, labels=range(len(labels_ext)))
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    n = len(labels_ext)
    fig, ax = plt.subplots(figsize=(1.1 * n + 2, 1.1 * n + 2))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    for r in range(n):
        for c in range(n):
            if cm[r, c] == 0:
                continue
            ax.text(c, r, f"{cm_norm[r, c]:.0%}\n({cm[r, c]:,})", ha="center", va="center",
                    color="white" if cm_norm[r, c] > 0.5 else "black", fontsize=8)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels_ext, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels_ext, fontsize=8)
    ax.set_xlabel("Predicted (highest raw score among classes above their own threshold, or Background)")
    ax.set_ylabel("True (first true class, or Background)")
    ax.set_title(f"{band.upper()} — combined confusion matrix (validation split, 0.5s bins, row-normalized)\n"
                 f"simplification: collapses each bin to one class -- see module docstring", fontsize=9)
    fig.colorbar(im, ax=ax, label="Fraction of row's true bins")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"confusion_matrix_combined_{band}.png"), dpi=110)
    plt.close(fig)

    pd.DataFrame(cm, index=labels_ext, columns=labels_ext).to_csv(
        os.path.join(out_dir, f"confusion_matrix_combined_{band}.csv"))


def main():
    p = build_argparser()
    p.add_argument("--band", choices=["cqt"], default="cqt", help="v17 has one band, cqt (default)")
    p.add_argument("--checkpoint", default=None, help="not needed with --load-scores")
    p.add_argument("--load-scores", default=None,
                    help="v19: scores .npz from predict_eval.py (no model run, no audio needed)")
    p.add_argument("--val-list", default=None, help="v19: only these file stems (a subset of the validation split)")
    p.add_argument("--no-mask", action="store_true",
                    help="v19: count masked bins (out-of-scope classes, uncertain annotations) like v18")
    p.add_argument("--thresholds-csv", required=True,
                    help="thresholds_<band>_continuous.csv from predict_eval.py -- always the continuous "
                         "one, see module docstring.")
    p.add_argument("--exclude-labels", nargs="*", default=["Blue whale Z call"],
                    help="Classes to leave out of BOTH plots entirely (default: Blue whale Z call -- a "
                         "near-degenerate low-N class's threshold can be so permissive it wins nearly "
                         "every bin's argmax in the combined plot, drowning out real cross-class confusion; "
                         "excluded from the per-class grid too by default so both plots reflect the same "
                         "class set). Pass with no arguments to include every class in both.")
    args = resolve_defaults(p.parse_args())

    band = args.band
    if "continuous" not in os.path.basename(args.thresholds_csv):
        print(f"WARNING: {args.thresholds_csv} doesn't look like a continuous-calibrated thresholds CSV -- "
              f"this script always scores continuously (run_inference_continuous), and a window-calibrated "
              f"threshold applied here is measurably too permissive. Pass thresholds_{band}_continuous.csv "
              f"unless you specifically intend otherwise.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    keep_stems = eval_scores.read_list(args.val_list) if args.val_list else None
    use_mask = not args.no_mask

    if args.load_scores:
        continuous, _, meta = eval_scores.load_scores(args.load_scores, keep_stems)
        label_list = meta["label_list"]
    else:
        if not args.checkpoint:
            raise SystemExit("ERROR: --checkpoint is required unless --load-scores is given")
        # common.resolve_audio_dirs (not a hand-rolled list): includes --bluefin-dir,
        # whose stems the checkpoint's val_files.txt can list.
        audio_dirs = common.resolve_audio_dirs(args, band)
        df_anno, audio_index = common.build_dataset_table(args.aplose_csv, args.soi_csv, audio_dirs)
        val_files_path = os.path.join(os.path.dirname(args.checkpoint), "val_files.txt")
        if keep_stems is not None:
            val_stems = keep_stems
        elif os.path.exists(val_files_path):
            val_stems = eval_scores.read_list(val_files_path)
        else:
            _, val_stems = common.get_split(args, audio_index, df_anno)
        val_stems = [s for s in val_stems if s in audio_index]
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        # v13: evaluate with the class set the checkpoint was TRAINED with
        if checkpoint.get('class_config'):
            common.configure_classes(checkpoint['class_config'])
        label_list = checkpoint.get("model_labels", common.BAND_LABELS[band])
        val_set = common.PamSequenceDataset(val_stems, audio_index, df_anno, band=band, augment=False,
                                             bluefin_hf_dir=getattr(args, "bluefin_hf_dir", None), return_mask=True)
        model = common.load_model_from_checkpoint(checkpoint, device)   # v17: CQT single tower
        print("Running continuous inference over the validation split...")
        continuous, _ = eval_scores.infer(model, val_set, device, batch_size=args.batch_size,
                                          num_workers=args.num_workers)

    thr_df = pd.read_csv(args.thresholds_csv).set_index("label")
    thresholds = np.array([thr_df.loc[lb, "best_threshold"] for lb in label_list], dtype=np.float32)
    print("Using thresholds (logit space):", dict(zip(label_list, thresholds)))
    print("Evaluation:", "MASKED" if use_mask else "UNMASKED (--no-mask)")

    # Only covered bins -- same reasoning as predict_eval.py's threshold
    # fit and run_inference.py's netCDF export: an uncovered bin's -1e4
    # sentinel score is not a real prediction and would corrupt the
    # combined matrix's argmax (see README's "v3 changes" #11).
    all_true, all_scores, all_mask = [], [], []
    for d in continuous.values():
        covered = d["covered"]
        all_true.append(d["true"][:, covered].T)
        all_scores.append(d["scores"][:, covered].T)
        all_mask.append(d["mask"][:, covered].T)
    all_true = np.concatenate(all_true, axis=0).astype(int)      # (N, C) 0/1
    all_scores = np.concatenate(all_scores, axis=0)               # (N, C) raw logits
    all_mask = np.concatenate(all_mask, axis=0) if use_mask else np.ones_like(all_scores)
    # a class without a calibrated threshold (no scored positives) is never predicted
    all_pred = (all_scores > np.nan_to_num(thresholds, nan=np.inf)[None, :]).astype(int)   # (N, C) 0/1
    # v19 masks: a masked class can be neither the truth nor the prediction of a bin
    all_true = all_true * (all_mask > 0)
    all_pred = all_pred * (all_mask > 0)
    all_scores = np.where(all_mask > 0, all_scores, -np.inf)
    print(f"{all_true.shape[0]:,} covered bins across {len(continuous)} validation files")

    # Applied once, upstream of both plots (not per-plot) -- a class with
    # a degenerate, unusually permissive threshold (e.g. Blue whale Z
    # call, ~12 training instances, threshold can land around -30 vs ~0
    # for every other class) wins nearly every bin's argmax in the
    # combined plot regardless of whether it's really the most confident
    # prediction, drowning out the cross-class confusion between classes
    # with a real signal -- and since its own precision/recall are already
    # known to be near-meaningless at this sample size (see README's
    # "Rare-class handling"), there's little reason to keep it in the
    # per-class grid either while it's this data-starved. Pass
    # --exclude-labels with no arguments to see every class in both plots.
    if args.exclude_labels:
        keep_idx = [i for i, lb in enumerate(label_list) if lb not in args.exclude_labels]
        excluded = [lb for lb in label_list if lb in args.exclude_labels]
        if excluded:
            print(f"\nExcluding {excluded} from both plots (--exclude-labels).")
        label_list = [label_list[i] for i in keep_idx]
        all_true = all_true[:, keep_idx]
        all_pred = all_pred[:, keep_idx]
        all_scores = all_scores[:, keep_idx]
        all_mask = all_mask[:, keep_idx]
        thresholds = thresholds[keep_idx]

    os.makedirs(args.output_dir, exist_ok=True)
    per_class_df = plot_per_class(all_true, all_pred, label_list, args.output_dir, band,
                                  all_mask=all_mask if use_mask else None)
    print("\nPer-class:")
    print(per_class_df.to_string(index=False))

    plot_combined(all_true, all_pred, all_scores, thresholds, label_list, args.output_dir, band)

    print(f"\nSaved confusion_matrix_per_class_{band}.png/.csv and "
          f"confusion_matrix_combined_{band}.png/.csv to {args.output_dir}")


if __name__ == "__main__":
    main()
