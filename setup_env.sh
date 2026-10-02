#!/usr/bin/env bash
# One-time setup: create a Python virtualenv (.venv) with everything
# run_inference.py (and the other scripts here) need, no Docker/nautilus
# required. Run this once; run_inference.sh activates .venv on its own
# after that.
#
# Usage:
#   ./setup_env.sh                    # auto-detect Python + GPU
#   ./setup_env.sh --cpu              # force the smaller CPU-only torch build
#   PYTHON=python3.11 ./setup_env.sh  # force a specific interpreter

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

FORCE_CPU=0
[[ "${1:-}" == "--cpu" ]] && FORCE_CPU=1

# torch>=2.4.1 (see the pip install line below for why this is a MINIMUM,
# not an exact pin matching nn_training_aplose_v9's Dockerfile -- confirmed
# on a real server that an exact-version pin can simply not exist for a
# given Python). Even so, some servers' Python is old enough that even the
# newest AVAILABLE torch build fails to install -- search for a working
# interpreter by ACTUAL reported version, not by binary name alone --
# `python3` and `python3.x`-named binaries don't reliably correspond to
# the same version across systems (seen in practice: this project's own
# server has `python` -> 3.14 but `python3` -> 3.9, i.e. the versioned
# name convention doesn't even apply consistently here). Override with
# PYTHON=... if your compatible interpreter isn't found by any of these
# names (e.g. a pyenv/conda/module-system install).
find_python() {
    local candidates=(python3.13 python3.12 python3.11 python3.10 python3.9 python3 python)
    local c ver
    for c in "${candidates[@]}"; do
        if command -v "$c" >/dev/null 2>&1; then
            ver="$("$c" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null)" || continue
            case "$ver" in
                3.9|3.10|3.11|3.12|3.13) echo "$c"; return 0 ;;
            esac
        fi
    done
    return 1
}

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
    if PYTHON=$(find_python); then
        echo "Using $PYTHON (auto-detected)"
    else
        PYTHON="python3"
        echo "WARNING: none of python3.13/3.12/3.11/3.10/3.9 found on PATH. Falling back to" >&2
        echo "'$PYTHON' ($("$PYTHON" --version 2>&1)) -- if torch==2.9.1 has no wheel for" >&2
        echo "its version, the pip install step below will fail. Check for a compatible" >&2
        echo "interpreter (e.g. 'module avail python' on an HPC system, or a pyenv/conda" >&2
        echo "install) and re-run with, e.g.:" >&2
        echo "  PYTHON=python3.12 ./setup_env.sh" >&2
    fi
fi

echo "== creating virtualenv (.venv) with $PYTHON =="
"$PYTHON" -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip

if [[ "$FORCE_CPU" -eq 1 ]]; then
    echo "== installing CPU-only torch (forced via --cpu) =="
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    echo "== nvidia-smi found -- installing CUDA 12.8 torch build (matches the Dockerfile's cuda12.8 base) =="
    TORCH_INDEX="https://download.pytorch.org/whl/cu128"
else
    echo "== no NVIDIA GPU detected -- installing CPU-only torch build (pass --cpu to skip this check) =="
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
fi

# v9.1: NOT hard-pinned to exact patch versions (2.9.1/0.24.1/2.9.1, matching
# the Dockerfile) anymore -- confirmed as a real problem, not a hypothetical
# one: torch==2.9.1 has no +cpu wheel at all for Python 3.9 on a real server
# (anhur), capping out at 2.8.0+cpu there. A hard pin means this script's
# only recovery from a too-new pin is someone manually discovering and
# editing in a lower version by hand. `torch>=2.4.1` (the oldest version this
# project has ever required) with torchvision/torchaudio left completely
# unpinned instead lets pip pick whatever's the newest MUTUALLY COMPATIBLE
# triplet actually available for this interpreter+platform on the index --
# pip resolves torchvision/torchaudio's own metadata dependency on a
# matching torch version correctly, so this doesn't risk landing on a
# mismatched combination. Loading a checkpoint saved under a newer torch
# (2.9.1) with a somewhat older one is normally fine for INFERENCE
# specifically -- state_dict loading operates on plain tensor serialization,
# not version-sensitive internals -- unlike training, where exact-version
# reproducibility matters more.
pip install --index-url "$TORCH_INDEX" "torch>=2.4.1" torchvision torchaudio
pip install -r requirements.txt

echo
echo "== done =="
echo "If 'import soundfile' or 'import librosa' fails at runtime, this server is"
echo "likely missing the system libsndfile/ffmpeg packages (installed via apt in"
echo "nn_training_aplose_v9's Dockerfile, but that's a container, not this host) --"
echo "ask your admin for: libsndfile1 ffmpeg (Debian/Ubuntu) or libsndfile ffmpeg (RHEL/Fedora)."
echo "Activate this environment yourself with: source .venv/bin/activate"
