# -*- coding: utf-8 -*-
"""Plot what the v17 model actually sees: the 500 x 600 CQTFrontend input
(after resize + min-max, before SpecAugment) for a few annotated windows,
with each window's labels underneath. A sanity check before a long training
run -- replaces v13's preview_joint_input.py.

    python preview_cqt_input.py --training-data /data/training_data_v13 \
        --split-dir /output/split_v17 --output-dir /output/preview_cqt --n 12

Runs the frontend on the GPU if there is one (it is the same module the model
uses), otherwise on the CPU.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import common
from somad_args import build_argparser, resolve_defaults


def main():
    p = build_argparser()
    p.add_argument("--n", type=int, default=12, help="number of windows to plot (default 12)")
    p.add_argument("--split", choices=["train", "val"], default="train")
    p.add_argument("--plot-seed", type=int, default=0)
    args = resolve_defaults(p.parse_args())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = common.BAND_CONFIG["cqt"]

    df_anno, audio_index = common.build_dataset_table(
        args.aplose_csv, args.soi_csv, common.resolve_audio_dirs(args, "cqt"),
        bluefin_dir=getattr(args, "bluefin_dir", None), bluefin_sites=getattr(args, "bluefin_sites", None))
    train_stems, val_stems = common.get_split(args, audio_index, df_anno)
    stems = train_stems if args.split == "train" else val_stems
    ds = common.PamSequenceDataset(stems, audio_index, df_anno, band="cqt", augment=False,
                                   bluefin_hf_dir=getattr(args, "bluefin_hf_dir", None))
    frontend = common.CQTFrontend(cfg).to(device).eval()

    # prefer windows with at least one annotation, spread over classes
    rng = np.random.default_rng(args.plot_seed)
    has = np.where(ds.class_presence.any(axis=1))[0]
    pool = has if len(has) else np.arange(len(ds))
    picks = rng.choice(pool, size=min(args.n, len(pool)), replace=False)

    # y ticks: one per octave, at the resized row of each octave's first CQT bin
    K = len(frontend.freqs)
    bpo = cfg["cqt_bins_per_octave"]
    rows = (np.arange(0, K, bpo) + 0.5) * cfg["freq_bins"] / K - 0.5
    labels_hz = [f"{f:.0f}" if f >= 100 else f"{f:.1f}" for f in frontend.freqs[::bpo].cpu().numpy()]

    os.makedirs(args.output_dir, exist_ok=True)
    for idx in picks:
        x, y = ds[int(idx)]
        with torch.no_grad():
            img = frontend(x["wave"][None].to(device))[0, 0].cpu().numpy()
        row = ds.df_data.loc[idx]
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 7), height_ratios=[3, 1], sharex=True)
        a1.imshow(img, origin="lower", aspect="auto", cmap="magma", extent=[0, cfg["snippet_length_s"], 0, cfg["freq_bins"]])
        a1.set_yticks(rows)
        a1.set_yticklabels(labels_hz)
        a1.set_ylabel("Frequency (Hz, log)")
        a1.set_title(f"{os.path.basename(row['filename'])} @ {row['start_sec']:.0f}-{row['stop_sec']:.0f} s  "
                     f"(model input {cfg['freq_bins']} x {cfg['time_bins']}, CQT {cfg['cqt_octaves']} oct x "
                     f"{bpo}/oct, gamma {cfg['cqt_gamma_hz']:g} Hz)", fontsize=9)
        a2.imshow(y.numpy(), origin="lower", aspect="auto", cmap="Greys", interpolation="nearest",
                  extent=[0, cfg["snippet_length_s"], 0, y.shape[0]])
        a2.set_yticks(np.arange(y.shape[0]) + 0.5)
        a2.set_yticklabels(ds.model_labels, fontsize=6)
        a2.set_xlabel("Time (s)")
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, f"cqt_input_{int(idx):06d}.png"), dpi=100)
        plt.close(fig)
    print(f"Saved {len(picks)} previews to {args.output_dir}")


if __name__ == "__main__":
    main()
