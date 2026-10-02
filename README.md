<p align="center">
  <img src="somad_logo.png" alt="SOMAD logo" width="260"/>
</p>

<h1 align="center">SOMAD</h1>
<p align="center"><b>Southern Ocean Multispecies Acoustic Detector</b></p>

<p align="center">
A joint, dual-band CNN&ndash;BiGRU sequence tagger that scores eleven co-occurring
Southern Ocean marine mammal call types &mdash; four baleen whale call types,
Antarctic minke whale, humpback whale, four pinniped call types, and orca &mdash;
from one forward pass over a 60&nbsp;s, full-bandwidth acoustic scene.
</p>

---

## What SOMAD does differently

Most automatic call detectors score one species (or a handful of related
call types) at a time, from a single, fixed-bandwidth spectrogram. SOMAD
instead:

- **Scores all 11 classes jointly**, from one shared model and one forward
  pass, so classes explicitly compete for the same evidence instead of
  being decided in isolation. This is what resolves the systematic
  humpback&nbsp;/&nbsp;fin-whale-downsweep confusion that single-band models
  produce (measured: an earlier low-frequency-only model raised false
  downsweep detections ~19&times; during real humpback calling; SOMAD's
  rate for the same failure mode is ~12&times; lower &mdash; see
  `docs/`).
- **Combines two sub-bands with two different spectrogram representations**,
  each fed through its own convolutional tower before fusion: a
  **linear**-frequency 500&nbsp;Hz sub-band (0&ndash;250&nbsp;Hz, where most
  baleen whale calls carry their diagnostic structure and a mel-scale
  frontend would discard it) and a **mel**-scale 32&nbsp;kHz sub-band
  (broadband coverage for pinniped/odontocete calls and humpback song).
- **Analyses long (60&nbsp;s), unsegmented, full-bandwidth windows**, giving
  the classifier the same temporal and spectral context a human analyst
  uses when scanning a spectrogram, rather than a short, pre-cropped clip.
- **Extends to other bands/taxa without restructuring**: a further
  frequency sub-band is one more convolutional tower feeding the same
  shared classifier; a further species is one more output unit.

See the accompanying manuscript (citation below) for the full method,
results, and discussion.

## Performance (held-out validation split, frame-level, 0.5&nbsp;s bins)

| Class | Latin name | N (train) | N (val) | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|
| Fin whale 20 Hz call | *Balaenoptera physalus* | 20,238 | 2,639 | 0.64 | 0.71 | 0.67 |
| Fin whale downsweep | *B. physalus* | 6,148 | 1,221 | 0.43 | 0.46 | 0.45 |
| Blue whale AB call | *B. musculus intermedia* | 26,555 | 5,035 | 0.56 | 0.60 | 0.58 |
| Blue whale D call | *B. musculus intermedia* | 13,687 | 2,412 | 0.44 | 0.61 | 0.51 |
| Blue whale Z call | *B. musculus intermedia* | 2,148 | 376 | 0.16 | 0.33 | 0.21 |
| Antarctic minke whale | *B. bonaerensis* | 1,303 | 186 | 0.75 | 0.57 | 0.65 |
| Humpback whale | *Megaptera novaeangliae* | 2,702 | 379 | 0.83 | 0.85 | **0.84** |
| Leopard seal | *Hydrurga leptonyx* | 1,839 | 208 | 0.80 | 0.95 | **0.87** |
| Weddell seal | *Leptonychotes weddellii* | 127 | 14 | 0.60 | 0.61 | 0.60 |
| Crabeater seal | *Lobodon carcinophaga* | 576 | 84 | 0.68 | 0.79 | 0.73 |
| Orca | *Orcinus orca* | 246 | 15 | 0.27 | 0.40 | 0.33 |

Blue whale Z call and orca are the two most data-poor classes; their lower
F1 reflects sample size, not an architecture limitation. See
`docs/confusion_matrix_combined_joint.png` and
`docs/pr_joint_continuous_labeled.png` for the full confusion structure and
per-class precision&ndash;recall curves, and
`docs/example_F_val_4species_soi.png` for a worked example on a real,
held-out, multi-species window.

## Repository layout

```
somad/
├── src/                      training / evaluation / inference code
│   ├── common.py             shared data pipeline, model, losses
│   ├── train_lf.py           CLI argument parser (imported by train_joint.py)
│   ├── train_joint.py        entry point: trains the 11-class joint model
│   ├── predict_eval.py       calibrates thresholds, computes P/R/F1
│   ├── run_inference.py      applies a trained checkpoint to new audio
│   ├── scores_to_events.py   turns per-bin scores into discrete call events
│   └── plot_confusion_matrix.py
├── checkpoints/
│   ├── best_model.pth               the trained SOMAD checkpoint (not in
│   │                                 git -- see "Getting the checkpoint")
│   └── thresholds_joint_continuous.csv   calibrated per-class decision
│                                          thresholds (small, tracked in git)
├── docs/                      confusion matrix, PR curves, worked example
├── Dockerfile
├── requirements.txt
├── setup_env.sh               non-Docker environment setup (creates .venv)
└── somad_logo.png
```

## Installation

### Option A: Docker (recommended, matches the environment SOMAD was trained in)

```bash
docker build -t somad .
docker run --gpus all --rm -it \
    -v /path/to/your/data:/data \
    -v $(pwd)/checkpoints:/checkpoints \
    -v /path/to/your/output:/output \
    somad run_inference.py --help
```

Omit `--gpus all` to run on CPU. Inference is practical on CPU; training
is not.

### Option B: without Docker

```bash
cd somad
./setup_env.sh          # auto-detects Python 3.9-3.13 and GPU/CPU, creates .venv
# or: ./setup_env.sh --cpu       to force a CPU-only torch build
source .venv/bin/activate
```

`setup_env.sh` installs `torch>=2.4.1` plus a matching `torchvision`/
`torchaudio` from PyPI's CUDA or CPU wheel index, then
`pip install -r requirements.txt`. If `import soundfile` or `import librosa`
fails afterwards, install the system packages `libsndfile1 ffmpeg`
(Debian/Ubuntu; `libsndfile ffmpeg` on RHEL/Fedora) -- these are what the
Dockerfile installs via `apt-get` inside the container.

## Getting the checkpoint

This repository provides the source code. Trained weights
(`checkpoints/best_model.pth`, ~1.9 GB) are archived separately on
**Zenodo** — [DOI: 10.5281/zenodo.23099517](https://doi.org/10.5281/zenodo.23099517) —
which provides a permanent identifier cited in the manuscript's data
availability statement.

`checkpoints/thresholds_joint_continuous.csv` (the calibrated per-class
decision thresholds) is small enough to be tracked directly in the
repository.

Download the checkpoint once after cloning:

```bash
curl -L -o checkpoints/best_model.pth https://zenodo.org/records/23099517/files/best_model.pth
```

The downloaded file must be placed at `checkpoints/best_model.pth`, the
path expected by every training and inference command in this document.

## Running inference on new recordings

```bash
python src/run_inference.py \
    --band joint \
    --checkpoint checkpoints/best_model.pth \
    --thresholds-csv checkpoints/thresholds_joint_continuous.csv \
    --audio-dir /path/to/your/wav/files \
    --out scores_joint.nc
```

`--fusion-mode` is not a required argument here: `run_inference.py` reads
it directly from the checkpoint's saved metadata (`twotower`, matching
SOMAD's training configuration), eliminating the possibility of a mismatch
between model and inference settings.

The output is a single netCDF file containing, for each input recording:
raw per-class, per-0.5&nbsp;s-bin scores (logits), each class's calibrated
threshold, and the resulting above-threshold boolean. The raw scores are
retained without loss, so thresholds can be revised without rerunning the
model.
Convert the continuous score file into discrete call events with:

```bash
python src/scores_to_events.py --scores-nc scores_joint.nc --out events_joint.csv
```

## Training SOMAD from scratch (or retraining on your own data)

```bash
python src/train_joint.py \
    --data-root /path/to/your/data \
    --output-dir /path/to/output/bigru_joint \
    --fusion-mode twotower \
    --bluefin-sites
```

Notes:

- `--fusion-mode twotower` is required to reproduce SOMAD's architecture
  (two independent convolutional towers, one per sub-band, fused after
  pooling). The other supported values (`stack`, `stack_normfix`,
  `channels`) are single-CNN alternatives used only for the ablations
  discussed in the manuscript.
- `--bluefin-sites`, passed **with no arguments**, uses all 11
  circum-Antarctic site-years of the public [IWC-SORP AcousticTrends
  BlueFinLibrary](https://doi.org/10.26179/5e6056035c01b) (Miller et al.,
  2021). `train_joint.py`'s default restricts training to 2 site-years;
  the flag must be included to reproduce SOMAD's actual training data
  scope.
- Expected data layout (matching `--data-root`) and every other flag
  (`--epochs`, `--batch-size`, `--lr`, `--val-fraction`, `--seed`, ...) are
  documented in `src/train_lf.py --help` (shared argument parser) and in
  each script's own module docstring.
- Training needs a CUDA GPU in practice; `--amp` enables mixed precision
  and is recommended.

## Evaluating a retrained model / recalibrating thresholds

After training your own checkpoint, calibrate per-class decision thresholds
and get precision/recall/F1 (this is what produced
`checkpoints/thresholds_joint_continuous.csv` and the numbers in the
performance table above):

```bash
python src/predict_eval.py --band joint --checkpoint /path/to/your/best_model.pth
```

Regenerate the confusion matrix figures (per-class 2&times;2 and combined
cross-class views, like `docs/confusion_matrix_combined_joint.png`) with:

```bash
python src/plot_confusion_matrix.py --band joint \
    --checkpoint /path/to/your/best_model.pth \
    --thresholds-csv /path/to/your/thresholds_joint_continuous.csv \
    --output-dir confmat_joint
```

## Label set

Eleven classes, scored jointly: fin whale 20&nbsp;Hz call, fin whale
downsweep, blue whale AB call (Antarctic A and B calls merged), blue whale
D call, blue whale Z call, Antarctic minke whale (bioduck & downsweep
calls), humpback whale (song and single calls), leopard seal, Weddell seal,
crabeater seal, and orca (both clicks and whistles).

## Data availability

The IWC-SORP AcousticTrends BlueFinLibrary is public:
Miller et al. (2021), <https://doi.org/10.26179/5e6056035c01b>. The
opportunistic Gerlache Strait and South Orkney Islands annotation sets used
alongside it are available on request / at a DOI to be added on
publication.

## Citation

If you use SOMAD, please cite:

> Menze, S., Ahonen, H., & Krafft, B. A. (in review). SOMAD: a multi-species,
> multi-band neural network detector that resolves cross-species confusion
> in Southern Ocean passive acoustic monitoring.

Full citation details will be updated once the manuscript is accepted.

## License

MIT &mdash; see [LICENSE](LICENSE).
