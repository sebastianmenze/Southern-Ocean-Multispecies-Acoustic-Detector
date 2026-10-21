# -*- coding: utf-8 -*-
"""Train the LF (low-frequency) BiGRU classifier. v2: see README.md's "v2
changes" section -- smaller GRU (hidden_size=192, 2 layers instead of
512/3), ReduceLROnPlateau instead of warm-restart cosine schedule, lower
early-stop patience, and waveform-level gain/noise augmentation. All aimed
at the overfitting the v1 LF run showed (train F1 -> 0.89, val F1 stuck
~0.47 from epoch ~25 on).

v4: the multi-head attention block on top of the BiGRU is ON by default
now (v3 had it off) -- see README.md's "v4 changes" section. Pass
--no-attention to turn it back off and compare.

v5: adds the AcousticTrends_BlueFinLibrary data (see README's "v5
changes" and convert_bluefin_library.py) -- one brand new class (Minke
whale) plus far more training instances for Blue whale AB/D/Z call and
Fin whale 20 Hz/downsweep, all LF-only. Blue whale A call and B call are
now merged into one "Blue whale AB call" label (both are part of the same
stereotyped call sequence, and B call alone had only 1 occurrence in the
original Gerlache/hermine data -- nowhere near enough to train separately).
--aplose-csv is now repeatable (pass it more than once to combine several
APLOSE-format sources); --bluefin-dir is the new source's (resampled/
dithered) audio. Both default to including bluefin_library_annotations.csv
/ bluefin_library_aplose/ automatically if found under --data-root, same
as the original Gerlache/hermine source.

v6: adds Humpback, Leopard seal, Weddell seal, Crabeater seal -- normally
HF-only classes, but each has substantial real energy under LF's ~250 Hz
ceiling (checked against every local annotation source's own
min_frequency column: Humpback 69.6% of instances, Leopard seal 44.5%,
Weddell seal 16.3%, Crabeater seal 11.5% -- vs. Orca's 1.1%, left HF-only).
Motivated by a real observed failure mode: the LF model was misclassifying
Humpback impulse calls as Fin whale downsweep, since LF training
previously had no "Humpback" output to assign that energy to at all. No
new annotation data or code changes needed beyond common.py's LF_LABELS --
the same hermine/soi sources already cover both bands' audio, and
PamSequenceDataset already keys everything off BAND_LABELS[band]
generically. See common.py's LF_LABELS comment for the full reasoning.

v7: LF_LABELS reverted to v5's original six (v6's addition above is NOT
part of this checkout's common.py) -- superseded by a different fix for
the same Humpback/Fin-whale-downsweep confusion: train_joint.py's new
single joint LF+HF model sees both bands' evidence for a class in the same
forward pass, instead of v6's two independent per-band decisions
reconciled afterward. See README's "v7 changes" for the full comparison.
Run train_joint.py instead if you want that fix; this script now trains
the plain 6-class LF-only model again, same as v5.

Classes (6): Fin whale 20 Hz, Fin whale downsweep, Blue whale AB call,
Blue whale D call, Blue whale Z call, Minke whale.

Usage (nautilus, data already synced -- see README.md):

    python train_lf.py

Usage (elsewhere, or a different data location):

    python train_lf.py --data-root /some/other/path --output-dir /output/bigru_lf

Default `--data-root` is the nautilus scratch layout:

    /scratch/disk3/a5278/bigru_aplose/data/
      2024_soi_train/
      2024_soi_val/
      hermine_SN8059_aplose_subset_2h/
      Gerlache_Strait_SN8059_-_2024_-_subset_1h_results (3).csv
      bluefin_library_aplose/                                      (v5)
      bluefin_library_annotations.csv                              (v5)

See README.md for the expected data layout.
"""

import argparse
import os

import common

DEFAULT_DATA_ROOT = "/scratch/disk3/a5278/bigru_aplose/data"  # shared with v1, unchanged
DEFAULT_OUTPUT_ROOT = "/scratch/disk3/a5278/bigru_aplose/output_v13"  # separate from earlier versions' output_v*/


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
    p.add_argument("--extra-audio-dir", action="append", default=None,
                    help="v13: additional flat audio folder(s) of .wav files -- repeatable. Used for the DCLDE "
                         "orca training set (<data-root>/somad_dclde_orca/audio is added automatically if it "
                         "exists). Which source a file belongs to, and so which classes it is annotated for, "
                         "comes from class_config.json (stem rules / folder name).")
    p.add_argument("--training-data", default=None,
                    help="v13: ONE consolidated folder built by build_training_data_v13.py "
                         "(<dir>/audio/*.wav at 32 kHz + <dir>/training_data_v13_aplose.csv). When given, "
                         "it replaces --aplose-csv/--soi-csv/--hermine-dir/--soi-*-dir/--bluefin-dir/"
                         "--extra-audio-dir entirely; the data source of every file (and so its class "
                         "scope / split role) comes from class_config.json's stem rules.")
    p.add_argument("--class-config", default=None,
                    help="v13: class + data-source definition (default: class_config.json next to common.py, "
                         "or $SOMAD_CLASS_CONFIG). To ADD A CLASS later, edit this file and retrain with "
                         "--resume-add-classes <old best_model.pth> -- see CHANGES_v13.md.")
    p.add_argument("--split-dir", default=None,
                    help="v13: folder with train_files.txt / val_files.txt (written by make_split.py or by a "
                         "previous run's --output-dir) to use INSTEAD of recomputing the split. Use it to keep "
                         "training, evaluation, plots and SimSiam pretraining on one identical split.")
    p.add_argument("--min-val-positives", type=int, default=common.MIN_VAL_POSITIVE_BINS,
                    help="v13: a class enters the validation selection score (masked macro AP) only if it has at "
                         f"least this many positive 0.5 s bins in the validation split (default: "
                         f"{common.MIN_VAL_POSITIVE_BINS}).")
    p.add_argument("--exclude-sources", nargs="+", default=None, metavar="GLOB",
                    help="pretrain_simsiam.py only (v13): data-source names (class_config.json) to leave out of "
                         "pretraining, e.g. 'bluefin_*' (upsampled 500 Hz audio).")
    p.add_argument("--exclude-stems-file", default=None,
                    help="v13, pretrain_simsiam.py only: text file of file stems (one per line, e.g. the "
                         "supervised run's val_files.txt) to leave OUT of self-supervised pretraining.")
    p.add_argument("--bluefin-dir", default=None,
                    help="AcousticTrends_BlueFinLibrary audio dir, LF-only -- see common.resolve_audio_dirs "
                         "(default: <data-root>/bluefin_library_aplose if it exists)")
    p.add_argument("--bluefin-sites", nargs="*", default=None,
                    help="v7: restrict bluefin_library to only these site-years (e.g. "
                         "ElephantIsland2013Aural ElephantIsland2014 -- must match convert_bluefin_library.py's "
                         "own SITES keys exactly, since that's what its output filenames embed). Default "
                         "(omitted): every site-year bluefin_library actually has converted data for. See "
                         "common.filter_bluefin_sites.")
    p.add_argument("--bluefin-hf-dir", default=None,
                    help="v8, joint band only: precompute_bluefin_hf_cache.py's output -- bluefin_library "
                         "files pre-resampled+dithered to HF's sample rate once, offline. When set, "
                         "PamSequenceDataset(band='joint') loads a bluefin stem's HF sub-band audio from here "
                         "instead of live-resampling its normal 500 Hz file on every access -- see "
                         "common.py's PamSequenceDataset.bluefin_hf_index. Default (omitted): live resample+ "
                         "dither every time, same as v7 (slower, but no precompute step needed first).")

    p.add_argument("--output-dir", default=os.path.join(DEFAULT_OUTPUT_ROOT, "bigru_lf"))
    p.add_argument("--run-name", default="bigru_lf")
    p.add_argument("--resume", default=None,
                    help="v8: path to a checkpoint to resume training from -- model/optimizer/scheduler/AMP-"
                         "scaler state, epoch number, best-val-F1-so-far, and early-stop patience counter are "
                         "all restored, so training continues from exactly where it left off rather than "
                         "restarting at epoch 1. Use <output-dir>/latest_checkpoint.pth (saved at the end of "
                         "EVERY epoch, unlike best_model.pth which only updates on a new best val F1 -- "
                         "resuming from best_model.pth would silently discard however many non-improving "
                         "epochs happened after it). Refuses to resume if the checkpoint's band/label list "
                         "doesn't match this run's EXACTLY -- e.g. after a BAND_CONFIG freq_bins change, or "
                         "ANY class added/removed, that's a different architecture/task, not a resumable "
                         "one. See --resume-add-classes if you specifically want to add classes.")
    p.add_argument("--resume-add-classes", default=None,
                    help="v12: path to an OLDER checkpoint trained on a SUBSET of this run's current "
                         "label list (e.g. you added a new class to JOINT_LABELS/LF_LABELS/HF_LABELS and "
                         "have new annotations for it) -- warm-starts a NEW training phase from it: the "
                         "CNN towers/BiGRU/attention load unchanged (none of those depend on num_classes), "
                         "the classifier's final layer is rebuilt at the new, larger size with old classes' "
                         "rows copied over BY NAME (robust to a new class being inserted in the middle of "
                         "the label list, not just appended at the end) and new classes' rows left at "
                         "random init. Optimizer/scheduler/epoch-count/best-val-F1/patience are NOT "
                         "restored (unlike --resume) -- this is a new training phase with a different loss "
                         "landscape, not a continuation, so it starts fresh from epoch 1 with warm-started "
                         "weights. Mutually exclusive with --resume. Refuses to load if anything besides "
                         "num_classes differs (hidden_size, gru_layers, pooled_freq_rows, use_attention, "
                         "fusion_mode, band) -- --resume-add-classes only handles a changed class count.")

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
    # v2: lower than v1's 15. The v1 LF run's val F1 plateaued by epoch ~25-30
    # and never improved again through epoch 65 -- that patience was mostly
    # burning GPU time on further overfitting, not finding a better model
    # (the best checkpoint is picked by best val_f1 regardless of patience,
    # so this mainly saves wall-clock time, paired with ReduceLROnPlateau's
    # own patience=5 getting a chance to try a lower LR first).
    p.add_argument("--patience", type=int, default=12)
    # v13: 0.30 (v11's 70/30). Applied to the random-eligible pool only -- DCLDE
    # (train_only) files and the forced-val Southern Ocean orca files are
    # handled by class_config.json roles, see common.split_files.
    p.add_argument("--val-fraction", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--amp", action="store_true", help="Use mixed precision (recommended on nautilus GPUs)")
    # v4: on by default (v3 had it off -- "likely redundant capacity given
    # how little data these rare classes have", but never got a decisive
    # A/B result either way). --no-attention reverts to a plain BiGRU, kept
    # as a flag to compare against.
    p.add_argument("--use-attention", action="store_true", default=True,
                    help="Add the multi-head attention block on top of the BiGRU (default: on). "
                         "Use --no-attention to turn it off.")
    p.add_argument("--no-attention", action="store_false", dest="use_attention",
                    help="Turn off the attention block (plain BiGRU, v3's default).")
    # v9: only meaningful for band="joint" (see BAND_CONFIG["joint"] and
    # PamSequenceDataset/ImprovedCNNBiGRU) -- ignored for lf/hf, which only
    # ever have one sub-band to begin with. Added after suspecting v7's
    # "stack" approach (HF spectrogram concatenated directly on top of LF's
    # along the frequency axis into one image) might be hurting the model:
    # (a) the CNN's first conv layer convolves right across that seam as if
    # adjacent rows were physically adjacent frequencies, which they aren't
    # (LF's ~250 Hz ceiling vs HF's ~16 kHz one), and (b) the post-stack
    # min-max normalization was global across both bands, so whichever band
    # had the wider raw log-power range could wash out the other's contrast.
    #   stack         -- v7/v8 baseline, unchanged: concat on freq axis,
    #                    single joint min-max normalize after.
    #   stack_normfix -- same concat, but each sub-band is min-max
    #                    normalized independently BEFORE stacking (fixes (b)
    #                    without touching the architecture at all).
    #   channels      -- both sub-bands resized to the same freq resolution
    #                    and stacked as 2 input channels instead of extra
    #                    frequency rows, so the first conv layer never mixes
    #                    across the LF/HF seam spatially (fixes (a)).
    #   twotower      -- fully separate CNN stem per sub-band, fused only
    #                    after each tower's own pooled features are
    #                    flattened, right before the GRU (fixes (a) and (b),
    #                    most invasive/most parameters).
    p.add_argument("--fusion-mode", choices=["stack", "stack_normfix", "channels", "twotower"],
                    default="twotower",
                    help="How band='joint' combines its HF and LF sub-bands (default: twotower, "
                         "v10's default -- this is the architecture actually deployed, see "
                         "output_v9_twotower_allsites from the previous version. 'stack' was the "
                         "original v7/v8 default; pass it explicitly if you want that behavior back "
                         "for comparison. No effect for band='lf'/'hf'.")
    p.add_argument("--pretrained-cnn-towers", default=None,
                    help="train_joint.py, fusion_mode='twotower' only: path to a best_cnn_towers.pth "
                         "from pretrain_simsiam.py -- loads its cnn_hf/cnn_lf weights into this run's "
                         "fresh model before supervised training starts. The BiGRU/attention/classifier "
                         "stay randomly initialized either way; only the CNN towers' starting point "
                         "changes. Refuses to load if the checkpoint's saved pooled_freq_rows/timesteps "
                         "don't match this run's BAND_CONFIG (a shape mismatch would fail anyway; "
                         "checked explicitly for a clearer message). Default (omitted): train the CNN "
                         "towers from scratch, same as before v10.")

    # v10: pretrain_simsiam.py only (self-supervised, no labels) -- ignored
    # by every other entrypoint. Kept in this shared parser rather than a
    # separate one so pretrain_simsiam.py can still reuse --epochs/
    # --batch-size/--num-workers/--lr/--weight-decay/--amp/--val-fraction/
    # --seed/--output-dir/--run-name/--bluefin-hf-dir unchanged, same as
    # every other entrypoint here.
    p.add_argument("--audio-dir", nargs="+", default=None,
                    help="pretrain_simsiam.py only: one or more folders of raw WAV files to pretrain "
                         "on -- no annotation CSVs needed at all, this is self-supervised. Point this "
                         "at a much larger pool than the APLOSE-labeled subset (e.g. a full multi-year "
                         "mooring archive) -- that's the entire reason to pretrain.")
    p.add_argument("--proj-dim", type=int, default=256,
                    help="pretrain_simsiam.py only: SimSiam projector/embedding dimension (default: "
                         "256).")
    p.add_argument("--pred-hidden", type=int, default=128,
                    help="pretrain_simsiam.py only: hidden size of SimSiam's predictor MLP (default: "
                         "128, i.e. a 256->128->256 bottleneck).")
    p.add_argument("--simsiam-crop-min-sec", type=float,
                    default=common.DEFAULT_SIMSIAM_CFG["crop_min_sec"],
                    help="pretrain_simsiam.py only (v13): each of the two views keeps a random contiguous "
                         "crop of at least this many seconds of the 60 s window (the rest is masked), so "
                         "the views overlap by >= 2*value-60 s but differ in content. 0 (or >= window "
                         "length) = both views are the full window (v12 behaviour). Default: "
                         f"{common.DEFAULT_SIMSIAM_CFG['crop_min_sec']}.")
    p.add_argument("--simsiam-noise-prob", type=float, default=common.DEFAULT_SIMSIAM_CFG["noise_prob"],
                    help="pretrain_simsiam.py only (v13): probability a view gets foreign background (a "
                         "pool window from another file) mixed in -- independent draw per view (v12: 0.5).")
    p.add_argument("--simsiam-snr-db", type=float, nargs=2, metavar=("LO", "HI"),
                    default=list(common.DEFAULT_SIMSIAM_CFG["snr_db"]),
                    help="pretrain_simsiam.py only (v13): SNR range in dB of that mixed-in background "
                         "(v12: 6 20). Below ~0 dB faint calls get buried -- don't go lower.")
    p.add_argument("--noise-pool-size", type=int, default=common.NOISE_POOL_SIZE,
                    help=f"Number of real background (call-free) windows sampled once per band/split "
                         f"for augment_waveform's noise-mixing augmentation (default: "
                         f"{common.NOISE_POOL_SIZE}). A larger pool gives broader coverage of real "
                         f"background conditions and, for self-supervised pretraining specifically, "
                         f"reduces the chance of two unrelated windows drawing the same pooled sample "
                         f"and becoming artificially similar (a shortcut SimSiam's loss can't "
                         f"distinguish from genuine acoustic similarity). Loaded as decoded waveforms "
                         f"at dataset construction, not streamed -- memory scales with this directly "
                         f"(~7.7 MB/window at HF's 32 kHz rate for a 60 s window).")
    return p


def resolve_defaults(args):
    """Fill in unset paths from --data-root, assuming the flat layout the
    data actually has on nautilus (see module docstring) -- no
    "aplose_training" subfolder, everything directly under data-root.

    v5: --aplose-csv is now action="append" (a list, or None if never
    passed at all) -- when None, defaults to [Gerlache export], plus
    bluefin_library_annotations.csv IF PRESENT under data-root (unlike the
    other defaults below, this one can't unconditionally assume the file
    exists: unlike hermine_dir/soi_*_dir, which are directories that
    build_audio_index simply globs empty if absent, load_aplose_annotations
    hard-errors on a missing CSV path -- so an older data-root synced
    before v5 without this new file must still work with no --aplose-csv
    given at all)."""
    root = args.data_root
    td = getattr(args, "training_data", None)
    if td:
        import glob as _g
        csvs = sorted(_g.glob(os.path.join(td, "*aplose*.csv")))
        if not csvs or not os.path.isdir(os.path.join(td, "audio")):
            raise SystemExit(f"ERROR: --training-data {td!r} needs audio/ and a *aplose*.csv "
                             f"(build_training_data_v13.py output)")
        args.aplose_csv, args.soi_csv = csvs, None
        args.hermine_dir = args.soi_train_dir = args.soi_val_dir = args.bluefin_dir = None
        args.extra_audio_dir = [os.path.join(td, "audio")]
        cc = os.path.join(td, "class_config.json")
        if getattr(args, "class_config", None) is None and os.path.exists(cc):
            args.class_config = cc
        if getattr(args, "class_config", None):
            common.configure_classes(args.class_config)
        return args
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
    # v13: add-on training sets built by build_somad_set.py (DCLDE orca) and
    # build_srw_set.py (Southern right whale): every <root>/somad_*/ with an
    # audio/ folder is added as an --extra-audio-dir, and its *aplose*.csv as
    # an --aplose-csv. Which classes each set is annotated for lives in
    # class_config.json (sources), not here.
    import glob as _glob
    auto_dirs, auto_csvs = [], []
    for d in sorted(_glob.glob(os.path.join(root, "somad_*"))):
        if os.path.isdir(os.path.join(d, "audio")):
            auto_dirs.append(os.path.join(d, "audio"))
            auto_csvs += sorted(_glob.glob(os.path.join(d, "*aplose*.csv")))
    if args.extra_audio_dir is None:
        args.extra_audio_dir = auto_dirs
    for c in auto_csvs:
        if c not in args.aplose_csv:
            args.aplose_csv.append(c)
    if getattr(args, "class_config", None):
        common.configure_classes(args.class_config)
    return args


if __name__ == "__main__":
    args = resolve_defaults(build_argparser().parse_args())
    common.run_training("lf", args)
