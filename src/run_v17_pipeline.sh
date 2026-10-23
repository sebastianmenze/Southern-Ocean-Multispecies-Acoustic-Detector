#!/usr/bin/env bash
# v17 pipeline (runs directly with python on the training server, no Docker):
#
#   split     fixed 70/30 train/val split -> $OUTPUT_DIR/split/{train,val}_files.txt
#   overview  classes x sources x labelled bins table -> $OUTPUT_DIR/overview/overview.md
#   preview   a few model inputs (500 x 600 CQT) as PNGs -> $OUTPUT_DIR/preview_cqt/
#   train     supervised training of the CQT single-tower model (class-masked loss,
#             masked macro-AP selection) -> $OUTPUT_DIR/bigru_cqt/
#   eval      predict_eval.py on the validation split (calibrated thresholds)
#             -> $OUTPUT_DIR/eval_cqt/thresholds_cqt_continuous.csv
#   all       split -> overview -> train -> eval
#
# Usage (from this directory, inside the python env that has torch etc.):
#   ./run_v17_pipeline.sh all
#   ./run_v17_pipeline.sh train --epochs 80            # extra args go to train_cqt.py
#   CUDA_VISIBLE_DEVICES=0 ./run_v17_pipeline.sh all   # pick the GPU
#
# ADD A CLASS LATER: add it to class_config.json, then
#   ADD_CLASSES_FROM=$OUTPUT_DIR/bigru_cqt/best_model.pth ./run_v17_pipeline.sh train
#
# Env vars:
#   TRAINING_DATA_DIR  consolidated 32 kHz folder from build_training_data_v13.py, containing
#                      audio/, training_data_v13_aplose.csv, class_config.json
#                      (default /localscratch/a5278/training_data_v13 -- same data as v13)
#   OUTPUT_DIR         (default /localscratch/a5278/output_v17)
#   PYTHON             interpreter (default: ./.venv from ./setup_env.sh, else python3)
#   ADD_CLASSES_FROM   path of an older v17 best_model.pth (warm start for a new class)
set -euo pipefail

STAGE="${1:-all}"; shift || true
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
TRAINING_DATA_DIR="${TRAINING_DATA_DIR:-/localscratch/a5278/training_data_v13}"
OUTPUT_DIR="${OUTPUT_DIR:-/localscratch/a5278/output_v17}"
if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x "$HERE/.venv/bin/python" ]]; then PYTHON="$HERE/.venv/bin/python"
    else PYTHON=python3; fi
fi

if [[ ! -d "$TRAINING_DATA_DIR/audio" ]] || ! compgen -G "$TRAINING_DATA_DIR/*aplose*.csv" >/dev/null; then
    echo "ERROR: $TRAINING_DATA_DIR must contain audio/ and a *aplose*.csv (set TRAINING_DATA_DIR)" >&2
    exit 1
fi
mkdir -p "$OUTPUT_DIR"

# class_config.json: the copy shipped inside the training folder wins, else the one next to this script
CLASS_CONFIG="$TRAINING_DATA_DIR/class_config.json"
[[ -f "$CLASS_CONFIG" ]] || CLASS_CONFIG="$HERE/class_config.json"

DATA_ARGS=(
    --training-data "$TRAINING_DATA_DIR"
    --class-config "$CLASS_CONFIG"
)

run() {  # run <script.py> <args...>
    "$PYTHON" "$@"
}

stage_split() {
    echo "== split (70/30; Southern Ocean orca -> val only; DCLDE -> train only) =="
    run make_split.py "${DATA_ARGS[@]}" --out-dir "$OUTPUT_DIR/split" --val-fraction 0.30
}
stage_overview() {
    echo "== overview =="
    run class_data_overview.py "${DATA_ARGS[@]}" --split-dir "$OUTPUT_DIR/split" --out-dir "$OUTPUT_DIR/overview"
}
stage_preview() {
    echo "== preview of the model input (CQT 500 x 600) =="
    run preview_cqt_input.py "${DATA_ARGS[@]}" --split-dir "$OUTPUT_DIR/split" \
        --output-dir "$OUTPUT_DIR/preview_cqt" "$@"
}
stage_train() {
    echo "== supervised training (CQT single tower, masked loss, masked macro-AP selection) =="
    local extra=()
    [[ -n "${ADD_CLASSES_FROM:-}" ]] && extra+=(--resume-add-classes "$ADD_CLASSES_FROM")
    run train_cqt.py "${DATA_ARGS[@]}" \
        --split-dir "$OUTPUT_DIR/split" --val-fraction 0.30 \
        --output-dir "$OUTPUT_DIR/bigru_cqt" --run-name bigru_cqt --amp \
        "${extra[@]}" "$@"
}
stage_eval() {
    echo "== eval =="
    run predict_eval.py "${DATA_ARGS[@]}" --split-dir "$OUTPUT_DIR/split" \
        --checkpoint "$OUTPUT_DIR/bigru_cqt/best_model.pth" \
        --output-dir "$OUTPUT_DIR/eval_cqt" --batch-size 12 "$@"
}

case "$STAGE" in
    split)    stage_split ;;
    overview) stage_overview ;;
    preview)  stage_preview "$@" ;;
    train)    stage_train "$@" ;;
    eval)     stage_eval "$@" ;;
    all)
        stage_split; stage_overview
        stage_train "$@"; stage_eval ;;
    *) echo "unknown stage '$STAGE' (split|overview|preview|train|eval|all)" >&2; exit 1 ;;
esac
echo "== done: $STAGE =="
