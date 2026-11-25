#!/usr/bin/env bash
# AI-assisted code: developed with Claude Code (Anthropic); reviewed and tested by the SOMAD authors.
# v19 pipeline (runs directly with python on the training server, no Docker):
#
#   split     70/30 train/val split -> $SPLIT_DIR/{train,val}_files.txt
#   overview  classes x sources x labelled bins table -> $OUTPUT_DIR/overview/overview.md
#   preview   a few model inputs (500 x 600 CQT) as PNGs -> $OUTPUT_DIR/preview_cqt/
#   train     CQT single tower, masked loss, masked macro-AP selection, noise-robust loss on the
#             BlueFin sources ($ROBUST_LOSS) -> $OUTPUT_DIR/bigru_cqt/
#   eval      MASKED predict_eval.py (out-of-scope classes and "uncertain" bins are not scored)
#             -> $OUTPUT_DIR/eval_cqt_masked/thresholds_cqt_continuous.csv
#   all       split -> overview -> train -> eval
#
# Usage (from this directory, inside the python env that has torch etc.):
#   ./run_v19_pipeline.sh all
#   ./run_v19_pipeline.sh train --epochs 80                       # extra args go to train_cqt.py
#   SPLIT_DIR=/path/v18/split WARM_START=/path/v18/weights/best_model.pth ./run_v19_pipeline.sh train
#   CHECKPOINT=/path/v18/weights/best_model.pth EVAL_DIR=/path/v18_eval_masked ./run_v19_pipeline.sh eval
#   ROBUST_LOSS=none ./run_v19_pipeline.sh train                  # plain v18 loss, for comparison
#
# Env vars:
#   TRAINING_DATA_DIR  consolidated 32 kHz folder from build_training_data_v13.py (audio/,
#                      training_data_v13_aplose.csv) -- run fix_kerguelen2015_downsweep.py on its csv
#                      first (default /data/training_data_v13)
#   OUTPUT_DIR         default /data/output_v19
#   SPLIT_DIR          default $OUTPUT_DIR/split (point it at the v18 release's split/ to keep v18's split)
#   ROBUST_LOSS        none | bootstrap (default) | gce   -- applied to --robust-sources (default bluefin_*)
#   WARM_START         checkpoint whose weights are copied by class name (fresh optimizer), e.g. v18/v17
#   CHECKPOINT         eval stage: checkpoint to evaluate (default $OUTPUT_DIR/bigru_cqt/best_model.pth)
#   EVAL_DIR           eval stage: output folder (default <checkpoint dir>/../eval_cqt_masked)
#   PYTHON             interpreter (default: ./.venv from ./setup_env.sh, else python3)
#   ADD_CLASSES_FROM   older best_model.pth with fewer classes (warm start for a new class)
# The class config is always class_config.json NEXT TO THIS SCRIPT (the copy inside the
# training-data folder is the older v13 one).
set -euo pipefail

STAGE="${1:-all}"; shift || true
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
TRAINING_DATA_DIR="${TRAINING_DATA_DIR:-/data/training_data_v13}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/output_v19}"
SPLIT_DIR="${SPLIT_DIR:-$OUTPUT_DIR/split}"
ROBUST_LOSS="${ROBUST_LOSS:-bootstrap}"
if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x "$HERE/.venv/bin/python" ]]; then PYTHON="$HERE/.venv/bin/python"
    else PYTHON=python3; fi
fi

if [[ ! -d "$TRAINING_DATA_DIR/audio" ]] || ! compgen -G "$TRAINING_DATA_DIR/*aplose*.csv" >/dev/null; then
    echo "ERROR: $TRAINING_DATA_DIR must contain audio/ and a *aplose*.csv (set TRAINING_DATA_DIR)" >&2
    exit 1
fi
mkdir -p "$OUTPUT_DIR"

CLASS_CONFIG="$HERE/class_config.json"

DATA_ARGS=(
    --training-data "$TRAINING_DATA_DIR"
    --class-config "$CLASS_CONFIG"
)

run() {  # run <script.py> <args...>
    "$PYTHON" "$@"
}

stage_split() {
    echo "== split (70/30) =="
    run make_split.py "${DATA_ARGS[@]}" --out-dir "$SPLIT_DIR" --val-fraction 0.30
}
stage_overview() {
    echo "== overview =="
    run class_data_overview.py "${DATA_ARGS[@]}" --split-dir "$SPLIT_DIR" --out-dir "$OUTPUT_DIR/overview"
}
stage_preview() {
    echo "== preview of the model input (CQT 500 x 600) =="
    run preview_cqt_input.py "${DATA_ARGS[@]}" --split-dir "$SPLIT_DIR" \
        --output-dir "$OUTPUT_DIR/preview_cqt" "$@"
}
stage_train() {
    echo "== supervised training (CQT single tower, masked loss, robust loss: $ROBUST_LOSS) =="
    local extra=(--robust-loss "$ROBUST_LOSS")
    [[ -n "${ADD_CLASSES_FROM:-}" ]] && extra+=(--resume-add-classes "$ADD_CLASSES_FROM")
    [[ -n "${WARM_START:-}" ]] && extra+=(--resume-add-classes "$WARM_START")
    run train_cqt.py "${DATA_ARGS[@]}" \
        --split-dir "$SPLIT_DIR" --val-fraction 0.30 \
        --output-dir "$OUTPUT_DIR/bigru_cqt" --run-name bigru_cqt --amp \
        "${extra[@]}" "$@"
}
stage_eval() {
    echo "== masked eval =="
    local ck="${CHECKPOINT:-$OUTPUT_DIR/bigru_cqt/best_model.pth}"
    local out="${EVAL_DIR:-$(dirname "$ck")/../eval_cqt_masked}"
    run predict_eval.py "${DATA_ARGS[@]}" --split-dir "$SPLIT_DIR" \
        --checkpoint "$ck" --output-dir "$out" --batch-size 12 "$@"
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
