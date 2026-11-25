# -*- coding: utf-8 -*-
# AI-assisted code: developed with Claude Code (Anthropic); reviewed and tested by the SOMAD authors.
"""v13: compute the train/val split ONCE and write it to disk, so training,
evaluation and plots all use the identical split.

  python make_split.py --training-data /data/training_data_v13 --out-dir /output/split_v17
  -> /output/split_v17/train_files.txt, val_files.txt, split_by_source.csv

Then:
  python train_cqt.py --training-data /data/training_data_v13 --split-dir /output/split_v17

Rules (class_config.json): sources with split=train_only (the DCLDE Pacific
sets) are never validated on; files containing Southern Ocean Orca
annotations (sources' val_only_classes) are forced into validation; the
remaining random-source files are split 70/30 (--val-fraction), stratified
by source.
"""
import os

import pandas as pd

from somad_args import build_argparser, resolve_defaults
import common


def main():
    p = build_argparser()
    p.add_argument("--out-dir", required=True)
    args = resolve_defaults(p.parse_args())
    audio_dirs = common.resolve_audio_dirs(args, "cqt")
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
