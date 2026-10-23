# v17 changes (on top of v13)

v17 = v13 with **one input and one CNN tower**: a variable-Q CQT spectrogram of the full 32 kHz
band replaces v13's two sub-band spectrograms (mel 32 kHz + linear 500 Hz) and two CNN towers.
The spectrogram is computed **inside the model, on the GPU**. Classes, data, split, class
masking, loss, sampler, selection metric (masked macro AP), thresholds and the inference/plot
outputs are unchanged from v13.

v17 contains only this architecture. v13's lf/hf/joint bands, the stack/channels/twotower fusion
modes, SimSiam pretraining and the CPU spectrogram code are removed; use `../nn_training_aplose_v13`
for those (and for v13 checkpoints).

## 1. Input

| | v13 (joint, twotower) | v17 |
|---|---|---|
| spectrogram(s) | mel 8192-pt, 700 mels, 10 Hz-16 kHz -> 500 rows; linear 256-pt at 500 Hz -> 200 rows | CQT 11 octaves x 24 bins/octave, 7.8 Hz-16 kHz, gamma 3 Hz (264 bins) -> 500 rows |
| time bins | 600 | 600 (CQT power averaged into 0.1 s bins) |
| CNN towers | 2 | 1 |
| GRU input | 512 x 48 x 2 | 512 x 48 |
| parameters | ~171 M | ~94 M |
| computed in | DataLoader workers (CPU) | `CQTFrontend` in the model (GPU) |

`gamma` (variable-Q transform, as `librosa.vqt`) adds 3 Hz to every band's width. That only
changes the low bins: ~0.24 s window at 20 Hz instead of ~0.9 s for a plain CQT, so fin whale
20 Hz pulses keep their downsweep shape; above ~100 Hz the bins are an ordinary 24/octave CQT.
Settings: `common.BAND_CONFIG["cqt"]`. CPU benchmark of the model alone (batch 4): ~0.5x the
time of v13's two-tower model; ~0.4x the multiply-adds.

## 2. Where the work happens

* **DataLoader workers (CPU):** read the 64 s slice (the 60 s window + 2 s on each side; zeros past
  a file edge), resample + shaped dither only for audio not already at 32 kHz (none with
  `--training-data`; bluefin files otherwise, or use `--bluefin-hf-dir`), waveform augmentation
  (gain + noise from the per-split noise pool). Returns `{"wave": (1, 2048000), "aug": 0|2}`.
* **Model (GPU), `common.CQTFrontend`:** one CQT over the whole 64 s slice (one FFT block, so no
  block-edge artefacts; the circular wrap-around only touches the 2 s padding) -> |c|^2 -> crop
  the padding -> average power into 600 time bins -> log10 -> bilinear resize 264 -> 500 rows ->
  per-window min-max -> SpecAugment (training, `aug > 0` windows, same stripe rules as v13).
  No learnable parameters, no autograd, FFTs in float32 even under `--amp`. `cqt_chunk` (default
  4) windows per CQT call bound GPU memory (~0.5 GB per window).
* The CQT kernels are built like cqt-pytorch's (`archinetai/cqt-pytorch`, MIT) with gamma added
  (`common.cqt_kernels`, vendored so there is no extra dependency). They are non-persistent
  buffers rebuilt from the checkpoint's `band_config`, so checkpoints stay small.

## 3. Files

New: `train_cqt.py` (training entry point), `somad_args.py` (the shared argument parser, was
`train_lf.py`), `run_v17_pipeline.sh` (split -> overview -> train -> eval, plus `preview`),
`preview_cqt_input.py` (PNGs of the 500 x 600 model input), `tests_v17.py`.

Removed: `train_lf.py`, `train_hf.py`, `train_joint.py`, `pretrain_simsiam.py`,
`preview_joint_input.py`, `plot_training_spectrograms.py`, `plot_augmentation_demo.py`,
`tests_v13.py` (its tests that still apply are in `tests_v17.py`), `run_v13_pipeline.sh` and the
per-band Docker wrappers `run_nautilus_*.sh` (except the bluefin precompute one).

Changed: `common.py` (CQT band, dataset returns audio, `CQTFrontend`, single-tower model,
`build_model` / `load_model_from_checkpoint`, `cqt_spectrogram_db` for plots);
`predict_eval.py`, `run_inference.py`, `plot_confusion_matrix.py`, `plot_errors.py` rebuild the
model from the checkpoint with `load_model_from_checkpoint` (`--band` is optional, `cqt` only);
`plot_file_detections.py` / `plot_extreme_score_events.py` / `plot_monthly_file_detections.py`
show the model's CQT; `detections_to_aplose_csv.py` knows the cqt frequency range;
`run_inference.sh`, `Dockerfile`, `precompute_bluefin_hf_cache.py`, `make_split.py`,
`class_data_overview.py` updated. Output files are named `*_cqt*` (e.g.
`eval_cqt/thresholds_cqt_continuous.csv`).

## 4. Running it

```
./run_v17_pipeline.sh all                  # same TRAINING_DATA_DIR as v13; OUTPUT_DIR default .../output_v17
./run_v17_pipeline.sh preview              # look at a few model inputs first
./run_inference.sh --audio-dir /path/to/wavs --out-file scores.nc --model-dir ./model
python tests_v17.py [--smoke]              # CPU, synthetic data
```

`--batch-size` defaults to 12 in `train_cqt.py` (each window reaches the GPU as ~8 MB of audio);
raise it if GPU memory allows. `--num-workers` matters less than in v13 (no spectrograms on the
CPU) but still covers decode + augmentation.

## Not done / to watch

* No SimSiam pretraining for the single tower yet (v13's pretrained two towers do not fit).
* Not run on a GPU yet: tested on CPU only (`tests_v17.py --smoke`, and the full CLI chain
  split -> train -> eval -> inference -> plots on synthetic data). Check the first epoch's
  speed and GPU memory with `PAM_DEBUG_TIMING=1`.
* The input normalisation (log10 power, per-window min-max) matches v13; the CQT's dynamic range
  differs from the mel spectrogram's, so v13 thresholds/checkpoints do not carry over.
