#!/usr/bin/env python
"""Block-wise prequential LABEL code on the 27-class ladder.

For one seed the ladder subsets are nested: S_500 ⊂ S_1000 ⊂ ... ⊂ S_23485.
Prequential code of the training labels given the texts:

    block 0 (first 500 comments): uniform code, log2(27) bits per label
    block k: train on S_{n_k}, encode the labels of S_{n_{k+1}} \\ S_{n_k}
             with -log2 P(e | x) from that model

Every model is turned into P(e | x):
  * MDL codes (generative): P(e|x) ∝ 2^(-L(x|M_e)) · P_train(e)   (Bayes)
  * LSTM (discriminative): softmax output.
"raw" uses these as they are; "calibrated" divides the logits by a temperature T
fitted on the validation split for each trained model (same for every model).

A final model trained on all 23,485 comments gives the in-sample "fit" cost of
the same labels (row with block_end = -1).

Usage:
    python scripts/experiments/prequential_labels_27.py results/prequential_labels/preq_all.csv
    python scripts/experiments/prequential_labels_27.py out.csv --models kn lstm --seeds 42
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--models", nargs="+", default=["kn", "add1", "shared", "lstm"])
ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
args = ap.parse_args()

import torch  # noqa: E402
from emotion_ait.single_label_27 import data as D  # noqa: E402
from emotion_ait.single_label_27.mdl import MDL27Classifier  # noqa: E402
from emotion_ait.single_label_27.shared_neural import SharedNeuralBigramMDL27  # noqa: E402
from emotion_ait.lstm_classifier import LSTMClassifier  # noqa: E402

K = 27
LN2 = math.log(2)
SIZES = [500, 1000, 2000, 4000, 8000, 12000, 16000, 23485]
VAL = D.load_single_label_27_dataframe("validation")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build(name, seed):
    if name == "kn":
        return MDL27Classifier(n=2, smoothing="kneser_ney", kn_backoff="class")
    if name == "add1":
        return MDL27Classifier(n=2, smoothing="laplace", k=1.0)
    if name == "shared":
        return SharedNeuralBigramMDL27(seed=seed, device=DEVICE)
    if name == "lstm":
        return LSTMClassifier(num_classes=K, max_epochs=10, patience=3, seed=seed, device=DEVICE)
    raise ValueError(name)


def logits(model, texts):
    """Natural-log unnormalised scores for P(e|x), shape (n, 27)."""
    if isinstance(model, LSTMClassifier):
        frame = pd.DataFrame({"text": list(texts), "label": [0] * len(texts)})
        data = model._dataset(frame)
        out = []
        model._model.eval()
        with torch.no_grad():
            for s in range(0, len(data), model.batch_size):
                x, lengths, _ = model._batch(data[s:s + model.batch_size], model.device)
                out.append(model._model(x, lengths).double().cpu().numpy())
        return np.vstack(out)
    bits = np.asarray(model.predict_scores(list(texts)), dtype=float)  # L(x|M_e), no prior
    counts = np.array([model.class_counts[c] if hasattr(model, "class_counts") and model.class_counts
                       else model.class_priors[c] for c in range(K)], dtype=float)
    prior = np.log(counts / counts.sum())
    return -bits * LN2 + prior


def label_bits(lg, y, T=1.0):
    z = lg / T
    z = z - z.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return -logp[np.arange(len(y)), y] / LN2


def fit_T(model):
    lg = logits(model, VAL.text.tolist())
    y = VAL.label.astype(int).to_numpy()
    grid = np.exp(np.linspace(np.log(0.05), np.log(200), 120))
    return float(grid[np.argmin([label_bits(lg, y, T).sum() for T in grid])])


Path(args.out).parent.mkdir(parents=True, exist_ok=True)
rows = []
for seed in args.seeds:
    subsets = {n: D.load_single_label_27_ladder_subset_dataframe(seed, n) for n in SIZES}
    ids = {n: set(df.example_id) for n, df in subsets.items()}
    for a, b in zip(SIZES, SIZES[1:]):
        assert ids[a] <= ids[b], "ladder not nested"
    full = subsets[SIZES[-1]]
    for name in args.models:
        t0 = time.time()
        raw = cal = 500 * math.log2(K)
        rows.append(dict(model=name, seed=seed, block_end=500, n_block=500, train_size=0,
                         bits_raw=raw, bits_cal=cal, acc=float("nan"), T=float("nan")))
        for a, b in zip(SIZES, SIZES[1:]):
            m = build(name, seed).fit(subsets[a], validation_data=VAL)
            block = subsets[b][~subsets[b].example_id.isin(ids[a])]
            lg = logits(m, block.text.tolist())
            y = block.label.astype(int).to_numpy()
            T = fit_T(m)
            br, bc = label_bits(lg, y).sum(), label_bits(lg, y, T).sum()
            rows.append(dict(model=name, seed=seed, block_end=b, n_block=len(block), train_size=a,
                             bits_raw=br, bits_cal=bc, acc=float((lg.argmax(1) == y).mean()), T=T))
        m = build(name, seed).fit(full, validation_data=VAL)
        lg = logits(m, full.text.tolist())
        y = full.label.astype(int).to_numpy()
        T = fit_T(m)
        rows.append(dict(model=name, seed=seed, block_end=-1, n_block=len(full), train_size=SIZES[-1],
                         bits_raw=label_bits(lg, y).sum(), bits_cal=label_bits(lg, y, T).sum(),
                         acc=float((lg.argmax(1) == y).mean()), T=T))
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"seed {seed} {name}: done in {time.time() - t0:.0f}s", flush=True)
print("ALL DONE")
