# -*- coding: utf-8 -*-
"""Train SOMAD: the joint LF+HF two-tower BiGRU classifier. One model, one
forward pass per 60 s window, scoring all 11 classes (Fin whale 20 Hz, Fin
whale downsweep, Blue whale AB call, Blue whale D call, Blue whale Z call,
Minke whale, Humpback, Leopard seal, Weddell seal, Crabeater seal, Orca)
jointly from both sub-bands -- see common.py's BAND_CONFIG["joint"] and
PamSequenceDataset.__getitem__'s band=="joint" branch, and README.md for
the architecture description.

Reproduce the published SOMAD checkpoint with:

    python train_joint.py --data-root /path/to/data --output-dir /path/to/output/bigru_joint \\
        --fusion-mode twotower --bluefin-sites

`--fusion-mode twotower` selects SOMAD's actual architecture (separate
convolutional tower per sub-band, fused after pooling); the other choices
(`stack`, `stack_normfix`, `channels`) are single-CNN alternatives used
only for the ablation comparisons in the manuscript. `--bluefin-sites`
with no arguments uses all 11 site-years of the public BlueFinLibrary --
this script's own default below restricts it to 2, so don't drop the flag
if you want SOMAD's actual training data scope.
"""

import argparse
import os

from train_lf import build_argparser, resolve_defaults, DEFAULT_OUTPUT_ROOT
import common


if __name__ == "__main__":
    p = build_argparser()
    # Conservative default: the joint model decodes/resamples audio TWICE per
    # window (once per sub-band) and has a taller spectrogram than HF alone,
    # so it needs a smaller batch than a single-band model at the same GPU
    # memory budget. Raise it if your GPU has headroom once you can see real
    # batch timing.
    p.set_defaults(output_dir=os.path.join(DEFAULT_OUTPUT_ROOT, "bigru_joint"), run_name="bigru_joint",
                    batch_size=12,
                    # Restricted to 2 site-years by default to keep a quick
                    # smoke-test run cheap. Pass --bluefin-sites with no
                    # arguments to use every site-year instead (SOMAD's own
                    # training used all 11 -- see README.md).
                    bluefin_sites=["ElephantIsland2013Aural", "ElephantIsland2014"])
    args = resolve_defaults(p.parse_args())
    common.run_training("joint", args)
