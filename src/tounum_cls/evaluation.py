"""Metrics, plots and architecture diagram."""
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
    """Binary classification metrics (photo = positive class)."""
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
    """Accuracy per source class (Photo, Painting...). `meta` must be aligned with `y_prob`."""
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
    """Bias/variance diagnosis from the training history, with explicit thresholds.

    * high bias (underfitting): final training accuracy < `low_acc`
    * high variance (overfitting): train - val accuracy gap > `gap_max`
      OR final val_loss more than `drift_max` above its minimum (val_loss going back up)
    * otherwise: balanced.
    With `restored_best=True` (early stopping that restores the best weights), the diagnosis describes the
    restored weights: it reads the row of the lowest val_loss and the drift is zero.
    The `kept_epoch` key gives the epoch of the row that was read.
    Note: dropout and augmentation are active during training, so the training accuracy is pessimistic
    (and the train/val gap is underestimated).
    """
    best_pos = int(hist["val_loss"].to_numpy().argmin())
    last = hist.iloc[best_pos] if restored_best else hist.iloc[-1]
    gap = float(last["accuracy"] - last["val_accuracy"])
    best_epoch = int(hist["epoch"].iloc[best_pos])
    drift = 0.0 if restored_best else float(last["val_loss"] - hist["val_loss"].min())
    if last["accuracy"] < low_acc:
        verdict = "high bias (underfitting)"
    elif gap > gap_max or drift > drift_max:
        verdict = "high variance (overfitting)"
    else:
        verdict = "balanced (no clear bias or variance problem)"
    return {
        "train_acc": float(last["accuracy"]), "val_acc": float(last["val_accuracy"]), "gap_acc": gap,
        "train_loss": float(last["loss"]), "val_loss": float(last["val_loss"]),
        "best_epoch": best_epoch, "val_loss_drift": drift, "verdict": verdict,
        "kept_epoch": int(last["epoch"]),
    }


def plot_history(hist: pd.DataFrame, title: str = ""):
    """Loss (left) and accuracy (right) curves per epoch, training vs validation."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, metric, label in zip(axes, ["loss", "accuracy"], ["Loss (binary cross-entropy)", "Accuracy"]):
        ax.plot(hist["epoch"], hist[metric], marker="o", ms=3, label="training")
        ax.plot(hist["epoch"], hist[f"val_{metric}"], marker="o", ms=3, label="validation")
        ax.set_xlabel("epoch")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_title(f"{title}: loss" if title else "Loss")
    axes[1].set_title(f"{title}: accuracy" if title else "Accuracy")
    fig.tight_layout()
    return fig


def plot_confusion(y_true, y_prob, threshold: float = 0.5, title: str = ""):
    """Annotated confusion matrix (rows = true class, columns = predicted)."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    cm = confusion_matrix(np.asarray(y_true).astype(int), y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["non-photo", "photo"])
    ax.set_yticks([0, 1], ["non-photo", "photo"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    return fig


def plot_precision_recall(y_true, y_prob, title: str = ""):
    """Precision/recall curve, used to discuss the decision threshold."""
    p, r, _ = precision_recall_curve(np.asarray(y_true).astype(int), y_prob)
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    ax.plot(r, p)
    ax.set_xlabel("recall (share of photos found)")
    ax.set_ylabel("precision (share of predicted photos that are photos)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def show_errors(X, meta: pd.DataFrame, y_prob, kind: str = "fp", n: int = 8, threshold: float = 0.5):
    """Error gallery. kind='fp': non-photos predicted as photos; 'fn': missed photos.

    Most confident (worst) errors first. Returns the figure, or None if there is no error.
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
    """Vertical diagram of the network drawn with matplotlib (no graphviz needed).

    One box per layer (name, type, output shape), with arrows between layers.
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
    ax.set_title(title or f"Architecture: {model.name} ({model.count_params():,} parameters)")
    fig.tight_layout()
    return fig
