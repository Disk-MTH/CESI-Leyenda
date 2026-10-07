"""Plots for the captioning notebook: architecture and workflow diagrams, learning curves, examples, attention."""
from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from PIL import Image

from . import config

FROZEN, TRAINED, DATA = "#c6dbef", "#fdd0a2", "#e5e5e5"


def _box(ax, x, y, w, h, text, color, fontsize=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", fc=color, ec="#444"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def _arrow(ax, start, end, text="", color="#333", ls="-", pos=0.5, dy=0.0):
    """Arrow from `start` to `end`, with an optional label placed at `pos` (0 = start, 1 = end) along it."""
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color=color, ls=ls, lw=1.2))
    if text:
        x, y = start[0] + pos * (end[0] - start[0]), start[1] + pos * (end[1] - start[1])
        ax.text(x + 0.08, y + dy, text, fontsize=7.5, color=color, va="center")


def draw_architecture(vocab_size: int = config.VOCAB_SIZE, embed_dim: int = config.EMBED_DIM,
                      units: int = config.UNITS):
    """Schematic of the captioning model: frozen InceptionV3 encoder, trained projection, GRU decoder, attention."""
    fig, ax = plt.subplots(figsize=(13, 6.6))
    w, h = 2.7, 0.75
    x0, x1, x2 = 0.0, 4.4, 8.8
    # image branch (left column)
    _box(ax, x0, 5.2, w, h, f"image\nredimensionnée en {config.FEATURE_IMG_SIZE}x{config.FEATURE_IMG_SIZE}x3", DATA)
    _box(ax, x0, 3.9, w, h, "InceptionV3, poids ImageNet\ngelé, sans tête de classification", FROZEN)
    _box(ax, x0, 2.6, w, h, f"carte de features 8 x 8 x {config.FEATURE_DIM}\n= {config.N_REGIONS} régions x {config.FEATURE_DIM}", DATA)
    _box(ax, x0, 1.3, w, h, f"encodeur CNN : Dense {embed_dim}, ReLU\n(chaque région, entraîné)", TRAINED)
    _box(ax, x0, 0.0, w, h, f"moyenne des régions, normalisée\n-> Dense {units}, tanh", TRAINED)
    for y_from, y_to in ((5.2, 3.9), (3.9, 2.6), (2.6, 1.3), (1.3, 0.0)):
        _arrow(ax, (x0 + w / 2, y_from), (x0 + w / 2, y_to + h))
    ax.text(x0 + w / 2, 6.2, "calculé une fois, mis en cache", ha="center", fontsize=8, style="italic")
    # word branch (middle column)
    _box(ax, x1, 5.2, w, h, "légende déjà écrite\n(startseq a dog ...)", DATA)
    _box(ax, x1, 3.9, w, h, f"Embedding\n{vocab_size} mots -> {embed_dim}", TRAINED)
    _box(ax, x1, 2.6, w, h, f"GRU, {units} unités\n(un pas par mot)", TRAINED)
    _box(ax, x1, 1.3, w, h, "attention additive\npoids sur les 64 régions", TRAINED)
    for y_from, y_to in ((5.2, 3.9), (3.9, 2.6)):
        _arrow(ax, (x1 + w / 2, y_from), (x1 + w / 2, y_to + h))
    _arrow(ax, (x1 + w / 2, 2.6), (x1 + w / 2, 1.3 + h), "requête")
    _arrow(ax, (x0 + w, 0.0 + h / 2), (x1, 2.6 + 0.15), "état initial", pos=0.2)
    _arrow(ax, (x0 + w, 1.3 + h / 2), (x1, 1.3 + h / 2), "régions", pos=0.05, dy=0.13)
    # output column
    _box(ax, x2, 1.3, w, h, "concaténation\nsortie GRU + contexte", DATA)
    _box(ax, x2, 2.6, w, h, f"Dense {units}, ReLU, dropout\n-> Dense {vocab_size}", TRAINED)
    _box(ax, x2, 3.9, w, h, "softmax : probabilité\ndu mot suivant", DATA)
    _arrow(ax, (x1 + w, 1.3 + h / 2), (x2, 1.3 + h / 2), "contexte", pos=0.35, dy=0.13)
    _arrow(ax, (x1 + w, 2.6 + h / 2), (x2, 1.3 + h - 0.1))
    _arrow(ax, (x2 + w / 2, 1.3 + h), (x2 + w / 2, 2.6))
    _arrow(ax, (x2 + w / 2, 2.6 + h), (x2 + w / 2, 3.9))
    _arrow(ax, (x2 + w / 2, 3.9 + h), (x1 + w, 5.2 + h / 2), "le mot le plus probable est ajouté", color="#2a7f2a", ls="--")
    ax.legend(handles=[FancyBboxPatch((0, 0), 1, 1, fc=FROZEN, ec="#444", label="pré-entraîné, gelé"),
                       FancyBboxPatch((0, 0), 1, 1, fc=TRAINED, ec="#444", label="entraîné par nous"),
                       FancyBboxPatch((0, 0), 1, 1, fc=DATA, ec="#444", label="données")],
              loc="lower right", fontsize=8)
    ax.set_xlim(-0.3, 11.8)
    ax.set_ylim(-0.4, 6.5)
    ax.axis("off")
    ax.set_title("Modèle de captioning : encodeur CNN (InceptionV3) + décodeur RNN (GRU) avec attention")
    fig.tight_layout()
    return fig


def draw_workflow(noise_threshold: float | None = None):
    """The whole TouNum workflow, from a folder of images to a table of captions."""
    fig, ax = plt.subplots(figsize=(13, 3.6))
    w, h, y = 2.0, 0.95, 1.4
    steps = [
        ("dossier d'images\n(toute taille, jpg/png)", DATA),
        ("livrable 1\nfiltre photo\n(MobileNetV2 ou CNN)", TRAINED),
        ("livrable 2\nestimation du bruit,\ndébruitage si besoin", TRAINED),
        ("InceptionV3\n64 vecteurs de région", FROZEN),
        ("livrable 3\ndécodeur GRU\ndécodage glouton", TRAINED),
        ("tableau : fichier,\nP(photo), débruitée,\nlégende", DATA),
    ]
    for i, (text, color) in enumerate(steps):
        _box(ax, i * 2.35, y, w, h, text, color)
        if i:
            _arrow(ax, ((i - 1) * 2.35 + w, y + h / 2), (i * 2.35, y + h / 2))
    _box(ax, 2.35, 0.05, w, 0.55, "écartée, sans légende", DATA)
    _arrow(ax, (2.35 + w / 2, y), (2.35 + w / 2, 0.6), "pas une photo", color="#a33")
    if noise_threshold is None:
        rule = "débruitage si le bruit estimé est assez fort"
    elif noise_threshold <= 0:
        rule = "toutes les photos sont débruitées"
    elif not np.isfinite(noise_threshold):
        rule = "débruitage désactivé"
    else:
        rule = f"débruitage si sigma estimé >= {noise_threshold:.1f}"
    ax.text(2 * 2.35 + w / 2, y - 0.25, rule, ha="center", fontsize=8)
    ax.set_xlim(-0.2, 6 * 2.35)
    ax.set_ylim(0, 2.6)
    ax.axis("off")
    ax.set_title("Workflow TouNum : filtrage, prétraitement, captioning")
    fig.tight_layout()
    return fig


def plot_history(hist: pd.DataFrame, title: str = ""):
    """Loss, masked word accuracy and validation BLEU-4 per epoch."""
    panels = [("loss", "Perte (entropie croisée par mot)"), ("masked_accuracy", "Accuracy sur le mot suivant")]
    if "val_bleu4" in hist:
        panels.append(("val_bleu4", "BLEU-4 (échantillon de validation)"))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 3.6))
    for ax, (metric, label) in zip(axes, panels):
        if metric != "val_bleu4":
            ax.plot(hist["epoch"], hist[metric], marker="o", ms=3, label="entraînement")
            ax.plot(hist["epoch"], hist[f"val_{metric}"], marker="o", ms=3, label="validation")
            ax.legend()
        else:
            ax.plot(hist["epoch"], hist[metric], marker="o", ms=3, color="tab:green")
        best = int(hist.loc[hist["val_loss"].idxmin(), "epoch"])
        ax.axvline(best, color="grey", ls=":", lw=1)
        ax.set_xlabel("époque")
        ax.set_title(f"{title} : {label}" if title else label, fontsize=10)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def show_captions(images, predicted: list[str], references: list[list[str]] | None = None, cols: int = 3,
                  size: float = 3.6, titles: list[str] | None = None):
    """Images with their generated caption (bold) and, if given, the first two reference captions."""
    n = len(images)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(size * cols, size * rows * 1.05), squeeze=False)
    for i, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if i >= n:
            continue
        ax.imshow(images[i])
        text = "prédite : " + textwrap.fill(predicted[i], 36)
        if references is not None:
            text += "\n" + "\n".join("réf. : " + textwrap.fill(r, 36) for r in references[i][:2])
        ax.set_title(text, fontsize=8.5, loc="left")
        if titles:
            ax.text(0.01, 0.99, titles[i], transform=ax.transAxes, va="top", fontsize=8, color="white",
                    bbox=dict(fc="black", alpha=0.6, pad=2))
    fig.tight_layout()
    return fig


def show_attention(image: np.ndarray, caption: str, weights: np.ndarray, max_words: int = 11):
    """Where the model looks for each word: the 8 x 8 attention weights upsampled over the image."""
    words = caption.split()[:max_words]
    side = int(np.sqrt(weights.shape[-1]))
    img = np.asarray(Image.fromarray(image).resize((224, 224)))
    cols = 6
    rows = int(np.ceil((len(words) + 1) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.3 * cols, 2.5 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("image", fontsize=9)
    for k, word in enumerate(words, start=1):
        ax = axes.ravel()[k]
        att = weights[k - 1].reshape(side, side)                  # weights used to produce word k
        att = np.asarray(Image.fromarray((att / att.max() * 255).astype(np.uint8)).resize((224, 224), Image.BICUBIC))
        ax.imshow(img)
        ax.imshow(att, cmap="inferno", alpha=0.55)
        ax.set_title(word, fontsize=10)
    fig.tight_layout()
    return fig


def show_gallery(results: pd.DataFrame, n: int = 12, cols: int = 4, size: float = 3.0):
    """Pipeline output: each image with its decision (photo or not, denoised or not) and its caption."""
    from tounum_cls import data as cls_data

    shown = results.head(n)
    rows = max(1, int(np.ceil(len(shown) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(size * cols, size * rows * 1.05), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, r) in zip(axes.ravel(), shown.iterrows()):
        try:
            img = cls_data.read_rgb(r["path"])
        except Exception:
            continue
        ax.imshow(img)
        if r["is_photo"]:
            head = f"photo ({r['p_photo']:.2f})" + (", débruitée" if r["denoised"] else "")
            text = head + "\n" + textwrap.fill(r["caption"], 32)
        else:
            text = f"pas une photo ({r['p_photo']:.2f}) : sans légende"
        ax.set_title(f"{r['file']}\n{text}", fontsize=8, loc="left")
    fig.tight_layout()
    return fig
