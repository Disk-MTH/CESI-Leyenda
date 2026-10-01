"""Métriques, graphiques et schéma d'architecture."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_binary(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Métriques de classification binaire (photo = classe positive)."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both = len(np.unique(y_true)) == 2
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if both else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def per_source_accuracy(meta: pd.DataFrame, y_prob, threshold: float = 0.5) -> pd.DataFrame:
    """Accuracy par classe source (Photo, Painting...). `meta` doit être aligné avec `y_prob`."""
    pred = (np.asarray(y_prob) >= threshold).astype(int)
    df = pd.DataFrame({"source": meta["source"].to_numpy(), "ok": pred == meta["label"].to_numpy()})
    out = df.groupby("source")["ok"].agg(n="size", accuracy="mean").reset_index()
    return out.sort_values("accuracy").reset_index(drop=True)


def diagnose_fit(hist: pd.DataFrame, low_acc: float = 0.90, gap_max: float = 0.05, drift_max: float = 0.05) -> dict:
    """Diagnostic biais/variance chiffré à partir de l'historique d'entraînement.

    * biais élevé (sous-apprentissage)  : accuracy d'entraînement finale < `low_acc`
    * variance élevée (sur-apprentissage) : écart accuracy train - val > `gap_max`
      OU val_loss finale supérieure de plus de `drift_max` à son minimum (la val_loss remonte)
    * sinon : compromis correct.
    Attention : pendant l'entraînement, dropout et augmentation sont actifs, ce qui rend l'accuracy
    d'entraînement pessimiste (l'écart train/val est alors sous-estimé).
    """
    last = hist.iloc[-1]
    gap = float(last["accuracy"] - last["val_accuracy"])
    best_epoch = int(hist["epoch"].iloc[int(hist["val_loss"].to_numpy().argmin())])
    drift = float(last["val_loss"] - hist["val_loss"].min())
    if last["accuracy"] < low_acc:
        verdict = "biais élevé (sous-apprentissage)"
    elif gap > gap_max or drift > drift_max:
        verdict = "variance élevée (sur-apprentissage)"
    else:
        verdict = "compromis biais/variance correct"
    return {
        "train_acc": float(last["accuracy"]), "val_acc": float(last["val_accuracy"]), "gap_acc": gap,
        "train_loss": float(last["loss"]), "val_loss": float(last["val_loss"]),
        "best_epoch": best_epoch, "val_loss_drift": drift, "verdict": verdict,
    }
