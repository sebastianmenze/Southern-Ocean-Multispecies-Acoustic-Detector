# -*- coding: utf-8 -*-
"""v13: compute the train/val split ONCE and write it to disk, so training,
evaluation, plots and SimSiam pretraining all use the identical split.

  python make_split.py --data-root /data --out-dir /output/split_v13
  -> /output/split_v13/train_files.txt, val_files.txt, split_by_source.csv

Then:
  python pretrain_simsiam.py --audio-dir ... --exclude-stems-file /output/split_v13/val_files.txt
  python train_joint.py --split-dir /output/split_v13 --pretrained-cnn-towers ...

Rules (class_config.json): sources with split=train_only (the DCLDE Pacific
sets) are never validated on; files containing Southern Ocean Orca
annotations (sources' val_only_classes) are forced into validation; the
remaining random-source files are split 70/30 (--val-fraction), stratified
by source.
"""
import os

import pandas as pd

from train_lf import build_argparser, resolve_defaults
import common


def main():
    p = build_argparser()
    p.add_argument("--out-dir", required=True)
    args = resolve_defaults(p.parse_args())
    audio_dirs = common.resolve_audio_dirs(args, "joint")
    df_anno, audio_index = common.build_dataset_table(
        args.aplose_csv, args.soi_csv, audio_dirs,
        bluefin_dir=getattr(args, "bluefin_dir", None), bluefin_sites=getattr(args, "bluefin_sites", None))
    train, val = common.split_files(audio_index, args.val_fraction, args.seed, df_anno=df_anno)
    os.makedirs(args.out_dir, exist_ok=True)
    for name, lst in (("train_files.txt", train), ("val_files.txt", val)):
        with open(os.path.join(args.out_dir, name), "w") as f:
            f.write("\n".join(lst))
    src = common.stem_sources(audio_index)
    vs = set(val)
    rows = pd.DataFrame({"stem": list(audio_index), "source": [src[s] for s in audio_index],
                         "split": ["val" if s in vs else "train" for s in audio_index]})
    rows.groupby(["source", "split"]).size().unstack(fill_value=0).to_csv(
        os.path.join(args.out_dir, "split_by_source.csv"))
    print(f"wrote {args.out_dir}: {len(train)} train / {len(val)} val files")


if __name__ == "__main__":
    main()
