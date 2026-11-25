#!/usr/bin/env bash
# AI-assisted code: developed with Claude Code (Anthropic); reviewed and tested by the SOMAD authors.
# SOMAD v19 inference over a whole dataset on a GPU server: every .wav under AUDIO_DIR (recursive), in chunks.
#
#   1. list all WAVs under AUDIO_DIR (follows symlinks, recursive), split into chunks of CHUNK files
#   2. run_inference.py (v19 checkpoint, continuous-calibrated thresholds) per chunk -> chunks/chunk_NNNN.nc
#      (one netCDF per chunk: run_inference holds the whole file x label x bin array in memory, NaN-padded to the
#      longest file, so one netCDF for >10k files would need many GB; chunks keep it bounded)
#      chunks are dealt round-robin over GPUS; resumable: a chunk whose .nc exists is skipped
#   3. scores_to_events.py per chunk -> one merged events CSV (thresholded detections, run-length encoded)
#
# Usage (from the bundle folder /path/to/somad_v19):
#   ./infer_all.sh                                   # default: the whole v19 training corpus (train + val)
#   AUDIO_DIR=/data/soi_full NAME=soi_full GPUS="0 1" ./infer_all.sh
#   AUDIO_DIR=/path/soundtrap NAME=gerlache_full \
#       INFER_ARGS="--filename-time-format %y%m%d%H%M%S" ./infer_all.sh   # SoundTrap-style names
#   STAGE=events ./infer_all.sh                      # only step 3 (chunks already scored)
#
# Env (defaults in brackets):
#   AUDIO_DIR   [/data/training_data_v13/audio]   folder with WAVs (searched recursively)
#   NAME        [basename of AUDIO_DIR's parent, e.g. training_data_v13]
#   OUT         [/data/output_v19]
#   CHECKPOINT  [checkpoints/best_model.pth]
#   THRESHOLDS  [checkpoints/thresholds_cqt_continuous.csv]
#   WORK        [$OUT/inference_$NAME]
#   CHUNK [500] files per netCDF   GPUS ["0"]   BATCH [12]   WORKERS [8]
#   INFER_ARGS  [""]  extra run_inference.py arguments (e.g. --filename-time-format, --no-overlap)
#   STAGE       [all | inference | events]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
AUDIO_DIR="${AUDIO_DIR:-/data/training_data_v13/audio}"
AUDIO_DIR="${AUDIO_DIR%/}"
NAME="${NAME:-$(basename "$(dirname "$AUDIO_DIR")")}"
OUT="${OUT:-/data/output_v19}"
CHECKPOINT="${CHECKPOINT:-$HERE/checkpoints/best_model.pth}"
THRESHOLDS="${THRESHOLDS:-$HERE/checkpoints/thresholds_cqt_continuous.csv}"
WORK="${WORK:-$OUT/inference_$NAME}"
CHUNK="${CHUNK:-500}"
GPUS="${GPUS:-0}"
STAGE="${STAGE:-all}"
INFER_ARGS="${INFER_ARGS:-}"
PY="${PYTHON:-$HERE/.venv/bin/python}"

for f in "$CHECKPOINT" "$THRESHOLDS" "$PY"; do
    [[ -e "$f" ]] || { echo "ERROR: $f not found (see the env vars at the top of this script)" >&2; exit 1; }
done
[[ -d "$AUDIO_DIR" ]] || { echo "ERROR: AUDIO_DIR $AUDIO_DIR not found" >&2; exit 1; }
mkdir -p "$WORK/chunks" "$WORK/lists" "$WORK/logs"
ulimit -n 65536 2>/dev/null || ulimit -n 4096 2>/dev/null || true

# ---------------------------------------------------------------- 1. file list + chunks
if [[ ! -s "$WORK/all_files.txt" ]]; then
    find -L "$AUDIO_DIR" -type f \( -iname "*.wav" \) | sort > "$WORK/all_files.txt"
    dups=$(xargs -a "$WORK/all_files.txt" -d '\n' -n1 basename | sed 's/\.[wW][aA][vV]$//' | sort | uniq -d | head -5)
    if [[ -n "$dups" ]]; then
        echo "ERROR: duplicate file stems under $AUDIO_DIR (run_inference keys files by stem), e.g.:" >&2
        echo "$dups" >&2; rm -f "$WORK/all_files.txt"; exit 1
    fi
    rm -f "$WORK"/lists/chunk_*
    split -l "$CHUNK" -d -a 4 "$WORK/all_files.txt" "$WORK/lists/chunk_"
fi
N=$(wc -l < "$WORK/all_files.txt")
mapfile -t LISTS < <(ls "$WORK"/lists/chunk_* | sort)
echo "== $NAME: $N WAVs under $AUDIO_DIR in ${#LISTS[@]} chunks of <= $CHUNK -> $WORK =="

run_chunk() {   # run_chunk <list file> <gpu>
    local list="$1" gpu="$2" id; id=$(basename "$list")
    local nc="$WORK/chunks/$id.nc"
    if [[ -s "$nc" ]]; then echo "[$id] exists, skipping"; return 0; fi
    local d="$WORK/links/$id"
    rm -rf "$d"; mkdir -p "$d"
    while IFS= read -r f; do ln -s "$f" "$d/"; done < "$list"
    echo "[$id] $(wc -l < "$list") files on GPU $gpu"
    # shellcheck disable=SC2086
    if CUDA_VISIBLE_DEVICES="$gpu" "$PY" run_inference.py --checkpoint "$CHECKPOINT" --thresholds-csv "$THRESHOLDS" \
            --audio-dir "$d" --out "$nc.part" --batch-size "${BATCH:-12}" --num-workers "${WORKERS:-8}" $INFER_ARGS \
            > "$WORK/logs/inference_$id.log" 2>&1; then
        mv "$nc.part" "$nc"; rm -rf "$d"; echo "[$id] done"
    else
        echo "[$id] FAILED -- see $WORK/logs/inference_$id.log" >&2; return 1
    fi
}

# ---------------------------------------------------------------- 2. inference
if [[ "$STAGE" == all || "$STAGE" == inference ]]; then
    read -ra G <<< "$GPUS"
    echo "== inference over GPU(s) ${G[*]} (logs in $WORK/logs/) =="
    t0=$(date +%s)
    pids=()
    for gi in "${!G[@]}"; do
        (   # one sequential queue per GPU, chunks dealt round-robin
            for ci in "${!LISTS[@]}"; do
                if (( ci % ${#G[@]} == gi )); then run_chunk "${LISTS[$ci]}" "${G[$gi]}" || true; fi
            done
        ) &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || true; done
    missing=0
    for l in "${LISTS[@]}"; do [[ -s "$WORK/chunks/$(basename "$l").nc" ]] || { echo "missing: $(basename "$l")"; missing=1; }; done
    echo "== inference finished in $(( ($(date +%s) - t0) / 60 )) min =="
    if (( missing )); then echo "WARNING: some chunks failed (see above); rerun the script to retry only those"; fi
fi

# ---------------------------------------------------------------- 3. events
if [[ "$STAGE" == all || "$STAGE" == events ]]; then
    echo "== events (thresholded detections) -> $WORK/events_$NAME.csv =="
    mkdir -p "$WORK/events"
    shopt -s nullglob
    for nc in "$WORK"/chunks/chunk_*.nc; do
        ev="$WORK/events/$(basename "$nc" .nc).csv"
        [[ -s "$ev" ]] || "$PY" scores_to_events.py --scores-nc "$nc" --out "$ev" > "$WORK/logs/events_$(basename "$nc" .nc).log" 2>&1 \
            || echo "events FAILED for $nc (see logs/)" >&2
    done
    evs=("$WORK"/events/chunk_*.csv)
    if (( ${#evs[@]} )); then
        { head -n1 "${evs[0]}"; for e in "${evs[@]}"; do tail -n +2 "$e"; done; } > "$WORK/events_$NAME.csv"
        echo "$(( $(wc -l < "$WORK/events_$NAME.csv") - 1 )) events from ${#evs[@]} chunks"
        "$PY" - "$WORK/events_$NAME.csv" <<'PY' || true
import sys, pandas as pd
d = pd.read_csv(sys.argv[1])
col = "predicted_label" if "predicted_label" in d else d.columns[2]
print(d[col].value_counts().to_string())
PY
    fi
    echo "== done. Copy back: $WORK/events_$NAME.csv (and chunks/*.nc for the raw per-bin scores) =="
fi
