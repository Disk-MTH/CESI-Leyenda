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


def diagnose_fit(
    hist: pd.DataFrame,
    low_acc: float = 0.90,
    gap_max: float = 0.05,
    drift_max: float = 0.05,
    restored_best: bool = False,
) -> dict:
    """Diagnostic biais/variance chiffré à partir de l'historique d'entraînement.

    * biais élevé (sous-apprentissage)  : accuracy d'entraînement finale < `low_acc`
    * variance élevée (sur-apprentissage) : écart accuracy train - val > `gap_max`
      OU val_loss finale supérieure de plus de `drift_max` à son minimum (la val_loss remonte)
    * sinon : compromis correct.
    Avec `restored_best=True` (early stopping avec restauration des meilleurs poids), le diagnostic décrit
    les poids restaurés : on utilise la ligne de l'epoch de plus faible val_loss et la remontée est nulle.
    La clé `kept_epoch` indique l'epoch de la ligne utilisée.
    Attention : pendant l'entraînement, dropout et augmentation sont actifs, ce qui rend l'accuracy
    d'entraînement pessimiste (l'écart train/val est alors sous-estimé).
    """
    best_pos = int(hist["val_loss"].to_numpy().argmin())
    last = hist.iloc[best_pos] if restored_best else hist.iloc[-1]
    gap = float(last["accuracy"] - last["val_accuracy"])
    best_epoch = int(hist["epoch"].iloc[best_pos])
    drift = 0.0 if restored_best else float(last["val_loss"] - hist["val_loss"].min())
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
        "kept_epoch": int(last["epoch"]),
    }


def plot_history(hist: pd.DataFrame, title: str = ""):
    """Courbes loss (gauche) et accuracy (droite), entraînement vs validation, par epoch."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, metric, label in zip(axes, ["loss", "accuracy"], ["Erreur (binary cross-entropy)", "Accuracy"]):
        ax.plot(hist["epoch"], hist[metric], marker="o", ms=3, label="entraînement")
        ax.plot(hist["epoch"], hist[f"val_{metric}"], marker="o", ms=3, label="validation")
        ax.set_xlabel("epoch")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_title(f"{title} - erreur" if title else "Erreur")
    axes[1].set_title(f"{title} - accuracy" if title else "Accuracy")
    fig.tight_layout()
    return fig


def plot_confusion(y_true, y_prob, threshold: float = 0.5, title: str = ""):
    """Matrice de confusion annotée (lignes = réel, colonnes = prédit)."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    cm = confusion_matrix(np.asarray(y_true).astype(int), y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["non-photo", "photo"])
    ax.set_yticks([0, 1], ["non-photo", "photo"])
    ax.set_xlabel("prédit")
    ax.set_ylabel("réel")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    return fig


def plot_precision_recall(y_true, y_prob, title: str = ""):
    """Courbe précision/rappel pour discuter du choix du seuil."""
    p, r, _ = precision_recall_curve(np.asarray(y_true).astype(int), y_prob)
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    ax.plot(r, p)
    ax.set_xlabel("rappel (photos retrouvées)")
    ax.set_ylabel("précision (photos prédites qui en sont)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def show_errors(X, meta: pd.DataFrame, y_prob, kind: str = "fp", n: int = 8, threshold: float = 0.5):
    """Galerie des erreurs. kind='fp' : non-photos prédites photo ; 'fn' : photos manquées.

    Les plus « confiantes » (les plus graves) d'abord. Renvoie la figure (ou None s'il n'y a aucune erreur).
    """
    y_prob = np.asarray(y_prob)
    label = meta["label"].to_numpy()
    mask = (label == 0) & (y_prob >= threshold) if kind == "fp" else (label == 1) & (y_prob < threshold)
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return None
    idx = idx[np.argsort(-np.abs(y_prob[idx] - threshold))][:n]
    cols = min(4, len(idx))
    rows = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.2 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), idx):
        ax.imshow(X[i])
        ax.set_title(f"{meta['source'].iloc[i]}\nP(photo)={y_prob[i]:.2f}", fontsize=9)
    fig.tight_layout()
    return fig


_LAYER_COLORS = {
    "InputLayer": "#d9d9d9", "Rescaling": "#d9d9d9", "Conv2D": "#9ecae1", "BatchNormalization": "#c7e9c0",
    "Activation": "#fdd0a2", "MaxPooling2D": "#bcbddc", "GlobalAveragePooling2D": "#fcbba1",
    "Dense": "#fdae6b", "Dropout": "#f2f2a0", "RandomFlip": "#eeeeee", "RandomRotation": "#eeeeee",
    "RandomZoom": "#eeeeee", "RandomContrast": "#eeeeee",
}


def draw_architecture(model, title: str | None = None):
    """Schéma vertical du réseau dessiné avec matplotlib (aucun outil externe type graphviz requis).

    Une boîte par couche : nom, type, forme de sortie. Flèches entre couches.
    """
    layers_ = list(model.layers)
    n = len(layers_)
    box_h, gap = 0.6, 0.25
    fig, ax = plt.subplots(figsize=(6.2, 0.85 * n * 0.62 + 1.2))
    for k, layer in enumerate(layers_):
        y = (n - 1 - k) * (box_h + gap)
        kind = type(layer).__name__
        shape = tuple(layer.output.shape)[1:] if hasattr(layer, "output") else ""
        ax.add_patch(FancyBboxPatch((0, y), 5, box_h, boxstyle="round,pad=0.02",
                                    fc=_LAYER_COLORS.get(kind, "#ffffff"), ec="#444444"))
        ax.text(0.15, y + box_h / 2, f"{layer.name}  [{kind}]", va="center", fontsize=9, fontweight="bold")
        ax.text(4.85, y + box_h / 2, str(shape), va="center", ha="right", fontsize=9)
        if k < n - 1:
            ax.annotate("", xy=(2.5, y - gap + 0.02), xytext=(2.5, y), arrowprops=dict(arrowstyle="->", color="#444444"))
    ax.set_xlim(-0.2, 5.2)
    ax.set_ylim(-0.3, n * (box_h + gap))
    ax.axis("off")
    ax.set_title(title or f"Architecture : {model.name} ({model.count_params():,} paramètres)")
    fig.tight_layout()
    return fig
