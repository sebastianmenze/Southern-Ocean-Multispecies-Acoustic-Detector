# AI-assisted code: developed with Claude Code (Anthropic); reviewed and tested by the SOMAD authors.
"""
Convert run_inference.py's combined netCDF (raw per-bin scores +
thresholds + above_threshold) into an event CSV (source_file, band,
predicted_label, window_start_s, window_end_s, start_datetime,
end_datetime, confidence), for detections_to_aplose_csv.py and
plot_detection_timeseries.py, which both want event-style rows.

By default, just run-length-encodes the netCDF's already-baked-in
above_threshold (no re-thresholding). Pass --thresholds-csv to override
with different per-class thresholds instead -- recomputed from the
netCDF's raw `scores`, so exploring a different threshold never needs
re-running the (expensive, GPU) model pass, only this.

Either way: ONLY thresholding, nothing else BY DEFAULT. No gap-bridging, no
minimum-duration filtering -- every bin that crosses the threshold counts.
Adjacent above-threshold bins get described as one continuous span rather
than one row per bin, but that's run-length-encoding a boolean array, not
smoothing it -- two touching positive bins were always one continuous
region, nothing gets bridged or dropped that wasn't already there. If you
want gap-bridging/duration filtering on top of this, that's a separate,
explicit step against this CSV -- not baked in here (compare
predict_eval.py's event-level metrics, which do use
smooth_binary_predictions -- deliberately not reused here).

Optional exception: --max-coverage drops (with a printed note, doesn't
silently disappear) any single event that spans more than that fraction of
its own file's duration. Off by default. A real call here is a brief
pulse, not an enduring tone, so an event covering e.g. 100% of a 60s file
with no on/off structure at all is a different kind of thing -- see
flag_suspicious_detections.py and README's "v3 changes" #14, where this
was traced (on a handful of real examples) to files containing strong
broadband noise interacting with PamSequenceDataset's per-window min-max
spectrogram normalization, not a narrowband call actually spanning the
whole file. Judgment call, hence opt-in rather than default: a genuinely
dense/continuous real calling bout would look the same to this filter.

Torch-free (xarray has no torch dependency) -- runs locally, no
Docker/GPU needed.

Usage:
    python scores_to_events.py --scores-nc scores_lf.nc --out detections_lf.csv
    python scores_to_events.py --scores-nc scores_lf.nc --thresholds-csv thresholds_lf_continuous.csv \
        --out detections_lf_continuous.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
import xarray as xr


def format_aplose_dt(dt_val: pd.Timestamp) -> str:
    """Match APLOSE's export datetime format -- same convention as
    common.py's format_aplose_dt, duplicated here to stay torch-free."""
    return dt_val.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt_val.microsecond // 1000:03d}+00:00"


def frames_to_events(binary_seq: np.ndarray, label_interval: float):
    """Run-length encode a 0/1 sequence into (start_s, stop_s) spans --
    a lossless description of contiguous positive runs, not smoothing."""
    events = []
    i = 0
    n = len(binary_seq)
    while i < n:
        if not binary_seq[i]:
            i += 1
            continue
        j = i
        while j < n and binary_seq[j]:
            j += 1
        events.append((i * label_interval, j * label_interval))
        i = j
    return events


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scores-nc", required=True, help="Combined netCDF from run_inference.py")
    p.add_argument("--thresholds-csv", default=None,
                    help="Override the netCDF's baked-in thresholds with these (label,best_threshold "
                         "columns, e.g. thresholds_<band>_continuous.csv from predict_eval.py). "
                         "Default: use the netCDF's own threshold/above_threshold as-is.")
    p.add_argument("--max-coverage", type=float, default=None,
                    help="Drop (don't emit as a row) any single event spanning more than this fraction of "
                         "its file's own duration (e.g. 0.95). Off by default -- see module docstring for "
                         "why this is a judgment call, not a default-on cleanup.")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    ds = xr.open_dataset(args.scores_nc)
    label_list = [str(x) for x in ds["label"].values]
    label_interval = float(ds.attrs["label_interval"])
    band = ds.attrs.get("band", "")
    stems = [str(x) for x in ds["file"].values]

    if args.thresholds_csv:
        thr_df = pd.read_csv(args.thresholds_csv).set_index("label")
        thresholds = np.array([thr_df.loc[lb, "best_threshold"] for lb in label_list])
        print(f"Overriding netCDF thresholds with {args.thresholds_csv}:", dict(zip(label_list, thresholds)))
    else:
        thresholds = ds["threshold"].values
        print("Using netCDF's own thresholds:", dict(zip(label_list, thresholds)))

    rows = []
    n_dropped_coverage = 0
    for stem in stems:
        file_ds = ds.sel(file=stem)
        duration_s = float(file_ds["duration"].values)
        n_valid = int(round(duration_s / label_interval)) + 1
        scores = file_ds["scores"].values[:, :n_valid]  # (label, bin), trim NaN padding
        above = scores > thresholds[:, None]

        start_time = file_ds["start_time"].values
        file_start_ts = pd.Timestamp(start_time, tz="UTC") if not pd.isna(start_time) else None

        for c, lb in enumerate(label_list):
            for start_s, stop_s in frames_to_events(above[c], label_interval):
                if args.max_coverage is not None and (stop_s - start_s) / duration_s > args.max_coverage:
                    n_dropped_coverage += 1
                    continue
                b1 = int(round(start_s / label_interval))
                b2 = int(round(stop_s / label_interval))
                peak_score = float(scores[c, b1:b2].max())
                confidence = 1 / (1 + np.exp(-peak_score))

                start_dt = end_dt = ""
                if file_start_ts is not None:
                    start_dt = format_aplose_dt(file_start_ts + pd.Timedelta(seconds=start_s))
                    end_dt = format_aplose_dt(file_start_ts + pd.Timedelta(seconds=stop_s))

                rows.append(dict(source_file=stem, band=band, predicted_label=lb,
                                  window_start_s=start_s, window_end_s=stop_s,
                                  start_datetime=start_dt, end_datetime=end_dt,
                                  confidence=confidence))

    if args.max_coverage is not None and n_dropped_coverage:
        print(f"Dropped {n_dropped_coverage} events covering > {args.max_coverage:.0%} of their file "
              f"(--max-coverage) -- these are excluded from the CSV below, not just flagged.")

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {len(out_df)} events to {args.out}")
    if len(out_df):
        print(out_df["predicted_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
