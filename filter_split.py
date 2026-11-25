# AI-assisted code: developed with Claude Code (Anthropic); reviewed and tested by the SOMAD authors.
"""v18: restrict an existing train/val split (v17's) to the files whose source is not split:ignore under
this class_config.json, so v18 keeps v17's train/val assignment (no train/val leakage when warm-starting)."""
import argparse, os, sys
import common

ap = argparse.ArgumentParser()
ap.add_argument("--class-config", required=True)
ap.add_argument("--audio-dir", required=True)
ap.add_argument("--in-split", required=True)
ap.add_argument("--out-split", required=True)
a = ap.parse_args()
common.configure_classes(a.class_config)
idx = common.build_audio_index([a.audio_dir])
os.makedirs(a.out_split, exist_ok=True)
for name in ("train_files.txt", "val_files.txt"):
    stems = [l.strip() for l in open(os.path.join(a.in_split, name)) if l.strip()]
    keep = [s for s in stems if s in idx and common.CLASS_CFG.split_role(common.CLASS_CFG.source_of(s, idx[s])) != "ignore"]
    open(os.path.join(a.out_split, name), "w").write("\n".join(keep) + "\n")
    print(f"{name}: {len(stems)} -> {len(keep)}")
