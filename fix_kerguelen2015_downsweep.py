# -*- coding: utf-8 -*-
# AI-assisted code: developed with Claude Code (Anthropic); reviewed and tested by the SOMAD authors.
"""v19 data fix: remove the Kerguelen 2015 "Fin whale downsweep" annotations.

They are a copy of the Kerguelen 2014 downsweep selection table (same 344 boxes: identical
durations, frequency bounds and minute:second within the hour, placed on 2015 files), so they
do not mark real calls in the 2015 audio. The true 2015 downsweeps are unknown, so v19's
class_config.json also MASKS Fin whale downsweep for bluefin_kerguelen2015 (neither positive nor
negative) -- removing the rows here is clean-up; the mask is what keeps training correct.

Works on both the BlueFin source table (pam2025/data/bluefin_library_annotations.csv) and the
consolidated training table (training_data_v13/training_data_v13_aplose.csv). Writes a backup
<name>.pre_v19_fix.csv next to the file once; running it again is a no-op.

    python fix_kerguelen2015_downsweep.py <csv> [<csv> ...] [--dry-run]
"""
import argparse
import hashlib
import os
import shutil
import tempfile
import time

import pandas as pd

SITE = "kerguelen2015"
LABEL = "Fin whale downsweep"


def bad_rows(df: pd.DataFrame) -> pd.Series:
    names = df["filename"].astype(str)
    return names.str.contains(SITE, case=False) & (df["annotation"].astype(str) == LABEL)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    for path in args.csv:
        backup = os.path.splitext(path)[0] + ".pre_v19_fix.csv"
        # read from the backup if one exists (a previous run may have left `path` half-written)
        source = backup if os.path.exists(backup) else path
        df = _read_local_copy(source)
        bad = bad_rows(df)
        print(f"{path}: {len(df)} rows in {os.path.basename(source)}, {int(bad.sum())} {SITE} '{LABEL}' rows "
              f"in {df.loc[bad, 'filename'].nunique()} files")
        if args.dry_run:
            continue
        if not os.path.exists(backup):
            _copy_verified(path, backup)
            print(f"  backup: {backup}")
        _write_verified(df[~bad], path)
        print(f"  wrote {len(df) - int(bad.sum())} rows (verified)")


# Network drives (Z:) on this machine intermittently fail large writes/reads with
# "[Errno 22] Invalid argument". So: read via a local temp copy, write to a local temp file,
# copy it over with retries and check the result byte-for-byte.
def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_verified(src, dst, tries=5):
    for i in range(tries):
        try:
            shutil.copyfile(src, dst)
            if _md5(src) == _md5(dst):
                return
        except OSError as e:
            print(f"  copy {src} -> {dst} failed ({e}), retry {i + 1}/{tries}")
        time.sleep(2)
    raise SystemExit(f"ERROR: could not copy {src} -> {dst} intact")


def _read_local_copy(path):
    with tempfile.TemporaryDirectory() as tmp:
        local = os.path.join(tmp, "in.csv")
        _copy_verified(path, local)
        return pd.read_csv(local)


def _write_verified(df, path):
    with tempfile.TemporaryDirectory() as tmp:
        local = os.path.join(tmp, "out.csv")
        df.to_csv(local, index=False)
        _copy_verified(local, path)


if __name__ == "__main__":
    main()
