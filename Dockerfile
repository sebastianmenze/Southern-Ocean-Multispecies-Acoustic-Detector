# GPU image for SOMAD (Southern Ocean Multispecies Acoustic Detector):
# training, evaluation, and deployment inference for the joint LF+HF
# two-tower BiGRU classifier. See README.md for the full model description,
# label set, and usage examples (both Docker and non-Docker).

FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

# torch 2.7.0+cu128 was the first stable PyTorch release to ship sm_120
# kernels (needed for Blackwell-generation GPUs, e.g. RTX 5090); pinned
# here to 2.9.1 for headroom on newer Blackwell parts too. If you're on an
# older GPU generation, any torch>=2.4.1 CUDA build will run the model fine
# -- see setup_env.sh for a non-Docker install that resolves this
# automatically for your own hardware.

# System deps for librosa/soundfile (libsndfile) and audio resampling.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
# torchvision/torchaudio matching the base image's torch+cuda build.
RUN pip install --no-cache-dir \
        torchvision==0.24.1 \
        torchaudio==2.9.1 \
    && pip install --no-cache-dir -r requirements.txt

COPY src/ ./

# /data holds the audio + annotation CSVs, /output holds checkpoints/logs,
# /checkpoints holds the released SOMAD checkpoint + calibrated thresholds
# -- all three are bind-mounted at `docker run` time, not baked into the
# image. See README.md for exact mount examples.
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python"]
CMD ["train_cqt.py", "--help"]
