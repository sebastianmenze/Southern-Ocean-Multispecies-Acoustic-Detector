# -*- coding: utf-8 -*-
# AI-assisted code: developed with Claude Code (Anthropic); reviewed and tested by the SOMAD authors.
"""v19: one inference pass over a validation set -> every evaluation output.

predict_eval.py used to run the model twice over the validation set (per-window pass for the "window"
threshold calibration, then run_inference_continuous for everything else) and plot_confusion_matrix.py a
third time. `infer()` does ONE pass and returns both, `save_scores()` / `load_scores()` keep them in one .npz
so thresholds, PR curves, event metrics, per-file tables, subset evaluations (--val-list) and confusion
matrices can all be recomputed without the model (`--load-scores` in both scripts).

Reads are retried (network drives on this machine fail sporadically with "[Errno 22] Invalid argument");
a window that still fails is skipped and reported, instead of ending a multi-hour run.
"""
import json
import time

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import common


class _RetryDataset(Dataset):
    def __init__(self, ds, tries=4):
        self.ds, self.tries = ds, tries

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        for k in range(self.tries):
            try:
                item = self.ds[i]
                if len(item) == 2:
                    item = (item[0], item[1], torch.ones_like(item[1]))
                return i, item
            except Exception as exc:  # noqa: BLE001 -- any read error: retry, then skip
                err = exc
                time.sleep(1 + 2 * k)
        print(f"WARNING: skipping window {i} (stem={self.ds.df_data.iloc[i]['stem']!r}) after {self.tries} tries: {err!r}")
        return i, None


def _collate(batch):
    batch = [b for b in batch if b[1] is not None]
    if not batch:
        return None
    idx = [b[0] for b in batch]
    xs = [b[1][0] for b in batch]
    x = {k: torch.stack([s[k] for s in xs]) for k in xs[0]}
    y = torch.stack([b[1][1] for b in batch])
    m = torch.stack([b[1][2] for b in batch])
    return idx, x, y, m


def _file_duration(path):
    for k in range(4):
        try:
            return sf.info(path).duration
        except Exception:  # noqa: BLE001
            time.sleep(1 + 2 * k)
    import librosa
    return librosa.get_duration(path=path)


@torch.no_grad()
def infer(model, dataset, device, batch_size=8, num_workers=2):
    """One pass. Returns (continuous, windows):
      continuous: dict[stem] -> {"duration", "scores" (C, n_bins), "true", "mask", "covered"} (max-stitched,
                  same contract as common.run_inference_continuous)
      windows:    dict with "stem" (N,), "start_sec" (N,), "true"/"pred"/"mask" (N, C, T) -- the raw per-window
                  outputs used by the "window" threshold calibration."""
    model.eval()
    li = dataset.label_interval
    C = dataset.num_classes
    continuous = {}
    for stem in dataset.df_data["stem"].unique():
        path = dataset.df_data.loc[dataset.df_data["stem"] == stem, "filename"].iloc[0]
        dur = _file_duration(path)
        n = int(dur / li) + 1
        continuous[stem] = {"duration": dur, "scores": np.full((C, n), -1e4, np.float32),
                            "true": np.zeros((C, n), np.float32), "mask": np.zeros((C, n), np.float32),
                            "covered": np.zeros(n, bool)}
    w_stem, w_start, w_true, w_pred, w_mask = [], [], [], [], []
    loader = DataLoader(_RetryDataset(dataset), batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        collate_fn=_collate,
                        worker_init_fn=common._worker_init_single_thread if num_workers > 0 else None)
    for batch in tqdm(loader, desc="Inference (single pass)"):
        if batch is None:
            continue
        idx, x, y, m = batch
        scores = model(common._to_device(x, device)).float().cpu().numpy()
        # copy: y/m come from the workers in shared memory (one file descriptor each); keeping numpy VIEWS of
        # them in w_true/w_mask kept every batch's buffers open -> "Too many open files" after ~470 batches
        y, m = y.numpy().copy(), m.numpy().copy()
        del x, batch
        for k, i in enumerate(idx):
            row = dataset.df_data.iloc[i]
            stem = row["stem"]
            w_stem.append(stem); w_start.append(float(row["start_sec"]))
            w_true.append(y[k]); w_pred.append(scores[k]); w_mask.append(m[k])
            d = continuous[stem]
            s0 = int(round(row["start_sec"] / li))
            e0 = min(s0 + scores.shape[2], d["scores"].shape[1])
            L = e0 - s0
            if L > 0:
                d["scores"][:, s0:e0] = np.maximum(d["scores"][:, s0:e0], scores[k, :, :L])
                d["true"][:, s0:e0] = np.maximum(d["true"][:, s0:e0], y[k, :, :L])
                d["mask"][:, s0:e0] = np.maximum(d["mask"][:, s0:e0], m[k, :, :L])
                d["covered"][s0:e0] = True
    windows = {"stem": np.array(w_stem), "start_sec": np.array(w_start, np.float32),
               "true": np.stack(w_true).astype(np.uint8), "pred": np.stack(w_pred).astype(np.float32),
               "mask": np.stack(w_mask).astype(np.uint8)}
    return continuous, windows


def save_scores(path, continuous, windows, meta: dict):
    stems = list(continuous)
    n_bins = np.array([continuous[s]["scores"].shape[1] for s in stems])
    cat = lambda k, dt: np.concatenate([continuous[s][k] for s in stems], axis=-1).astype(dt)  # noqa: E731
    np.savez_compressed(
        path, meta=json.dumps(meta), stems=np.array(stems), n_bins=n_bins,
        durations=np.array([continuous[s]["duration"] for s in stems]),
        scores=cat("scores", np.float32), true=cat("true", np.uint8), mask=cat("mask", np.uint8),
        covered=cat("covered", bool),
        w_stem=windows["stem"], w_start=windows["start_sec"], w_true=windows["true"], w_pred=windows["pred"],
        w_mask=windows["mask"])
    print(f"saved scores of {len(stems)} files / {len(windows['stem'])} windows to {path}")


def load_scores(path, stems_keep=None):
    """-> (continuous, windows, meta); `stems_keep` (iterable) restricts both to those files."""
    # NpzFile decompresses an array on EVERY z[key] access: read each one once, not once per file
    z = dict(np.load(path, allow_pickle=False))
    meta = json.loads(str(z["meta"]))
    keep = set(stems_keep) if stems_keep is not None else None
    scores, true, mask, covered = z["scores"], z["true"], z["mask"], z["covered"]
    continuous, off = {}, 0
    for stem, n, dur in zip(z["stems"], z["n_bins"], z["durations"]):
        sl = slice(off, off + n)
        off += n
        if keep is not None and stem not in keep:
            continue
        continuous[str(stem)] = {"duration": float(dur), "scores": scores[:, sl],
                                 "true": true[:, sl].astype(np.float32),
                                 "mask": mask[:, sl].astype(np.float32), "covered": covered[sl]}
    sel = np.ones(len(z["w_stem"]), bool) if keep is None else np.isin(z["w_stem"], list(keep))
    windows = {"stem": z["w_stem"][sel], "start_sec": z["w_start"][sel], "true": z["w_true"][sel],
               "pred": z["w_pred"][sel], "mask": z["w_mask"][sel]}
    if keep is not None:
        missing = keep - set(continuous)
        if missing:
            print(f"WARNING: {len(missing)} requested file(s) not in {path}")
    return continuous, windows, meta


def read_list(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]
