# -*- coding: utf-8 -*-
"""Train SOMAD v17: one variable-Q CQT spectrogram -> one CNN tower -> BiGRU,
scoring every class in class_config.json from 60 s windows of 32 kHz audio.

Input: CQT 11 octaves x 24 bins/octave (7.8 Hz - 16 kHz), gamma 3 Hz, resized
to 500 freq x 600 time bins -- common.BAND_CONFIG["cqt"].

Where the work happens:
  DataLoader workers (CPU): read the 64 s slice (60 s window + 2 s each side),
    resample + shaped dither only for files not already at 32 kHz (none with
    --training-data), waveform augmentation (gain + noise from the noise pool).
  Model (GPU): CQT, crop, power-pool to 600 time bins, log10, resize to 500
    freq bins, per-window min-max, SpecAugment (common.CQTFrontend), then the
    CNN-BiGRU.

Usage:

    python train_cqt.py --training-data /localscratch/a5278/training_data_v13 --amp
    python train_cqt.py --training-data ... --split-dir /localscratch/a5278/output_v17/split --amp

See somad_args.py for all options and CHANGES_v17.md for what changed.
"""

import os

from somad_args import build_argparser, resolve_defaults, DEFAULT_OUTPUT_ROOT
import common


if __name__ == "__main__":
    p = build_argparser()
    # Each window reaches the GPU as ~8 MB of float32 audio; CQTFrontend
    # processes BAND_CONFIG["cqt"]["cqt_chunk"] windows per CQT call.
    p.set_defaults(output_dir=os.path.join(DEFAULT_OUTPUT_ROOT, "bigru_cqt"), run_name="bigru_cqt",
                   batch_size=12)
    args = resolve_defaults(p.parse_args())
    common.run_training("cqt", args)
