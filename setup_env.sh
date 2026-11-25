#!/usr/bin/env bash
# AI-assisted code: developed with Claude Code (Anthropic); reviewed and tested by the SOMAD authors.
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
    # v10.1: auto-detect Pascal-generation GPUs (Tesla P100 and similar,
    # compute capability 6.0/sm_60) and fall back to an OLDER CUDA-targeted
    # torch build for them -- confirmed as a real, repeatable failure, not
    # hypothetical: the newest cu128 torch wheels have DROPPED kernel
    # support for sm_60 entirely ("no kernel image is available for
    # execution on the device" at model init, only discovered after a full
    # training run was already launched on a real P100 server). cu118
    # wheels still ship sm_60 kernels, and a 580.x-series driver (new
    # enough to report CUDA 13.0 in nvidia-smi) is backward-compatible with
    # a cu118 runtime without issue. Matched by NAME ("P100") rather than
    # trying to query compute capability directly, since nvidia-smi doesn't
    # expose that without --query-gpu=compute_cap (not available on every
    # driver version) -- "P100" in the name is a reliable enough proxy for
    # this one known-bad generation; extend this list if another
    # Pascal-era card (P40, P4, ...) turns out to have the same problem.
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)"
    if [[ "$GPU_NAME" == *"P100"* ]]; then
        echo "== detected Pascal GPU ($GPU_NAME) -- cu128 torch has no sm_60 kernels, using cu118 instead =="
        TORCH_INDEX="https://download.pytorch.org/whl/cu118"
        TORCH_PIN='torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1'  # exact triplet known to ship sm_60 kernels
    else
        echo "== nvidia-smi found ($GPU_NAME) -- installing CUDA 12.8 torch build (matches the Dockerfile's cuda12.8 base) =="
        TORCH_INDEX="https://download.pytorch.org/whl/cu128"
        TORCH_PIN='torch>=2.4.1'
    fi
else
    echo "== no NVIDIA GPU detected -- installing CPU-only torch build (pass --cpu to skip this check) =="
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    TORCH_PIN='torch>=2.4.1'
fi

# v9.1: NOT hard-pinned to exact patch versions (2.9.1/0.24.1/2.9.1, matching
# the Dockerfile) by default -- confirmed as a real problem, not a
# hypothetical one: torch==2.9.1 has no +cpu wheel at all for Python 3.9 on
# a real server (anhur), capping out at 2.8.0+cpu there. A hard pin means
# this script's only recovery from a too-new pin is someone manually
# discovering and editing in a lower version by hand. `torch>=2.4.1` (the
# oldest version this project has ever required) with torchvision/
# torchaudio left completely unpinned instead lets pip pick whatever's the
# newest MUTUALLY COMPATIBLE triplet actually available for this
# interpreter+platform on the index. The one exception is $TORCH_PIN's
# Pascal branch above, which DOES pin an exact triplet -- there, the
# constraint isn't "newest available," it's "oldest build still known to
# ship sm_60 kernels at all," which an unpinned floor can't express.
pip install --index-url "$TORCH_INDEX" $TORCH_PIN
pip install -r requirements.txt

echo
echo "== done =="
echo "If 'import soundfile' or 'import librosa' fails at runtime, this server is"
echo "likely missing the system libsndfile/ffmpeg packages (installed via apt in"
echo "nn_training_aplose_v9's Dockerfile, but that's a container, not this host) --"
echo "ask your admin for: libsndfile1 ffmpeg (Debian/Ubuntu) or libsndfile ffmpeg (RHEL/Fedora)."
echo "Activate this environment yourself with: source .venv/bin/activate"
