# -*- coding: utf-8 -*-
"""Train the JOINT LF+HF twotower BiGRU classifier -- one model, two
independent per-sub-band CNN towers (see common.py's
BAND_CONFIG["joint"] and ImprovedCNNBiGRU's fusion_mode="twotower"
branch), fused only after each tower's own pooled features are flattened,
right before the GRU. Scores all 11 LF+HF classes at once instead of two
separate per-band models.

v10: this is the final architecture from v9's own evolution
(output_v9_twotower_allsites) promoted to the actual default for this
version, rather than something you had to opt into with
--fusion-mode twotower --bluefin-sites every time:
  - --fusion-mode defaults to "twotower" (train_lf.py's build_argparser),
    not v7/v8's original "stack". Pass --fusion-mode stack explicitly if
    you want the old single-CNN behavior back for comparison.
  - --bluefin-sites defaults to every site-year (this script no longer
    restricts it to ElephantIsland2013Aural/ElephantIsland2014 -- pass
    --bluefin-sites ElephantIsland2013Aural ElephantIsland2014 explicitly
    if you want the smaller, earlier training scope back).

v10 also adds optional self-supervised pretraining for the twotower CNN
stems (SimSiam, see common.run_simsiam_pretraining / pretrain_simsiam.py)
-- --pretrained-cnn-towers loads a pretrained encoder's weights into
model.cnn_hf/model.cnn_lf before supervised training starts. See
pretrain_simsiam.py's own docstring for how to run that first.

Classes (11): Fin whale 20 Hz, Fin whale downsweep, Blue whale AB call,
Blue whale D call, Blue whale Z call, Minke whale, Humpback, Leopard seal,
Weddell seal, Crabeater seal, Orca.

v8: two training-speed fixes for bluefin_library files running close to
an hour long (every window PamSequenceDataset produces from one used to
redo that whole file's decode+spectrogram from scratch):
  - Per-file in-memory caching inside PamSequenceDataset itself -- no flag,
    always on, see common.py's _waveform_cache/_clean_spec_cache.
  - --bluefin-hf-dir (optional): point this at precompute_bluefin_hf_cache.py's
    output to skip the live 500 Hz -> 32 kHz resample+dither for the HF
    sub-band entirely, instead of paying it on every access. Omit it (the
    default) to keep doing that live -- no precompute step required, just
    slower.

Usage (nautilus, data already synced -- see README.md):

    python train_joint.py

Usage (elsewhere, or a different data location):

    python train_joint.py --data-root /some/other/path --output-dir /output/bigru_joint

Usage (restrict back to the original 2-site-year scope, for comparison):

    python train_joint.py --bluefin-sites ElephantIsland2013Aural ElephantIsland2014

Usage (fine-tune from a SimSiam-pretrained pair of CNN towers):

    python train_joint.py --pretrained-cnn-towers /output/pretrain_simsiam/best_cnn_towers.pth

See README.md / train_lf.py's docstring for the expected data layout --
identical for joint, no separate data source needed (see BAND_CONFIG
["joint"]'s comment on why bluefin_dir is still included).
"""

import argparse
import os

from train_lf import build_argparser, resolve_defaults, DEFAULT_OUTPUT_ROOT
import common


if __name__ == "__main__":
    p = build_argparser()
    # Smaller than HF's 16: the joint spectrogram is taller (see
    # BAND_CONFIG["joint"]/["hf"]'s current freq_bins) AND every window
    # decodes/resamples audio TWICE (once per sub-band) on a cache miss,
    # roughly doubling per-sample audio I/O on top of the bigger tensor --
    # start conservative, raise it if GPU memory allows once you can see
    # real batch timing (v8's per-file caching should make this much less
    # of a concern than it was in v7).
    p.set_defaults(output_dir=os.path.join(DEFAULT_OUTPUT_ROOT, "bigru_joint"), run_name="bigru_joint",
                    batch_size=12)
    # v10: no bluefin_sites default override here at all -- build_argparser's
    # own --bluefin-sites default (None = every site-year, see
    # common.filter_bluefin_sites) now IS the default training scope,
    # matching output_v9_twotower_allsites. Pass --bluefin-sites explicitly
    # with site names if you want the smaller, earlier (v7-v9 default)
    # 2-site-year scope back.
    args = resolve_defaults(p.parse_args())
    common.run_training("joint", args)
