# -*- coding: utf-8 -*-
"""Self-supervised pretraining for ImprovedCNNBiGRU's twotower CNN stems
(cnn_hf/cnn_lf) via SimSiam (Chen & He, 2021) -- masked-free, negative-pair-
free, no momentum/EMA teacher. Trains on RAW, UNLABELED audio: point
--audio-dir at whatever raw WAV archive(s) you have (a full multi-year
mooring deployment, not just the APLOSE-labeled subset used for supervised
training) and it learns directly from that, no annotation CSVs anywhere in
this.

Why SimSiam specifically (over BYOL, the other common no-negatives SSL
method): mechanically simpler -- no momentum encoder, no EMA decay
hyperparameter, one fewer moving part to get wrong, while relying on the
same core finding (stop-gradient + a predictor MLP alone is enough to
prevent the trivial constant-output collapse any non-contrastive SSL
method has to avoid). See common.py's SimSiamPretrainModel docstring for
the full mechanics.

What this pretrains, and what it doesn't: only the CNN towers (the
per-window "what does this spectrogram generally look like" feature
extractor), not the BiGRU/attention/classifier -- those stay randomly
initialized and are trained entirely from scratch during the normal
supervised run afterward. Expected to help most on your LOWEST-N classes
(the ones currently weakest -- Orca, Weddell seal, Blue whale Z call,
Minke whale), since those are exactly the cases where supervised training
alone has the least labeled signal to learn good CNN features from;
classes with abundant labels already (Blue whale AB call, Fin whale 20 Hz)
are not expected to move much.

Usage:
    python pretrain_simsiam.py --audio-dir /path/to/raw/archive1 /path/to/raw/archive2 \
        --output-dir /output/pretrain_simsiam --epochs 30 --amp

v13: pass --exclude-stems-file <split-dir>/val_files.txt (from make_split.py) so
the supervised validation audio never takes part in pretraining.

Then fine-tune on labels using the result:
    python train_joint.py --data-root /path/to/data \
        --pretrained-cnn-towers /output/pretrain_simsiam/best_cnn_towers.pth
"""
from train_lf import build_argparser
import common


if __name__ == "__main__":
    p = build_argparser()
    p.set_defaults(
        output_dir="./output_v13/pretrain_simsiam",
        run_name="pretrain_simsiam",
        # v10.2: was 32 -- confirmed on a real 12 GB card (Tesla P100) to
        # CUDA-OOM well before even finishing one batch. The "no sequence
        # model, cheap" reasoning that set 32 originally was wrong: SimSiam's
        # symmetrized loss needs the FULL gradient graph for BOTH views
        # through BOTH CNN towers at once (z.detach() only cuts gradient on
        # the loss-TARGET side, not on the forward pass that produces each
        # view's own prediction) -- effectively 2 views x 2 towers of
        # full-resolution activations (HF starts at 500x600 before any
        # pooling) resident simultaneously, nothing like a single forward
        # pass's cost. 8 is still just a starting point, not a measured-safe
        # number for every GPU -- watch nvidia-smi on the first run and
        # adjust (a 12 GB P100 needed as low as --batch-size 4 in practice).
        batch_size=8,
        num_workers=8,
        epochs=30,
        # v13: only a small slice is held out for the SimSiam val loss -- the 0.30
        # supervised split must NOT be used here (pretraining is unlabeled, more audio is better).
        val_fraction=0.10,
    )
    args = p.parse_args()
    if not args.audio_dir:
        raise SystemExit("ERROR: --audio-dir is required (one or more folders of raw WAV files)")
    common.run_simsiam_pretraining(args)
