"""Quality metrics, classical convolution filters, plots and architecture diagram."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib.patches import FancyBboxPatch
from scipy import ndimage


# --------------------------------------------------------------------------- metrics
def psnr(clean: np.ndarray, test: np.ndarray) -> np.ndarray:
    """PSNR in dB per image, for float images in [0, 1] of shape (n, h, w, 3) or (h, w, 3)."""
    return tf.image.psnr(np.asarray(clean, np.float32), np.asarray(test, np.float32), max_val=1.0).numpy()


def ssim(clean: np.ndarray, test: np.ndarray) -> np.ndarray:
    """SSIM per image, for float images in [0, 1] of shape (n, h, w, 3) or (h, w, 3)."""
    return tf.image.ssim(np.asarray(clean, np.float32), np.asarray(test, np.float32), max_val=1.0).numpy()


# --------------------------------------------------------------------------- classical filters
def mean_kernel(size: int) -> np.ndarray:
    """Box kernel: every pixel of the size x size window gets the same weight."""
    return np.full((size, size), 1.0 / size**2)


def gaussian_kernel(sigma: float) -> np.ndarray:
    """Gaussian kernel (radius 3 sigma): nearby pixels weigh more than distant ones."""
    radius = max(1, int(np.ceil(3 * sigma)))
    ax = np.arange(-radius, radius + 1)
    k = np.exp(-(ax[:, None] ** 2 + ax[None, :] ** 2) / (2 * sigma**2))
    return k / k.sum()


def convolve(images: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply a 2D convolution kernel to each channel of float images (n, h, w, 3) or (h, w, 3)."""
    k = kernel[None, :, :, None] if images.ndim == 4 else kernel[:, :, None]
    return ndimage.convolve(images, k, mode="reflect")


def median(images: np.ndarray, size: int) -> np.ndarray:
    """Median filter per channel (not a convolution: it replaces each pixel by the median of its window)."""
    footprint = (1, size, size, 1) if images.ndim == 4 else (size, size, 1)
    return ndimage.median_filter(images, size=footprint, mode="reflect")


CLASSICAL_FILTERS = {
    "mean 3x3": lambda x: convolve(x, mean_kernel(3)),
    "mean 5x5": lambda x: convolve(x, mean_kernel(5)),
    "Gaussian s=1": lambda x: convolve(x, gaussian_kernel(1.0)),
    "median 3x3": lambda x: median(x, 3),
    "median 5x5": lambda x: median(x, 5),
}


# --------------------------------------------------------------------------- plots
def plot_history(hist: pd.DataFrame, title: str = ""):
    """Loss (MSE) and PSNR curves per epoch, training vs validation."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, metric, label in zip(axes, ["loss", "psnr"], ["Loss (mean squared error)", "PSNR (dB)"]):
        ax.plot(hist["epoch"], hist[metric], marker="o", ms=3, label="training")
        ax.plot(hist["epoch"], hist[f"val_{metric}"], marker="o", ms=3, label="validation")
        ax.set_xlabel("epoch")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_title(f"{title}: loss" if title else "Loss")
    axes[1].set_title(f"{title}: PSNR" if title else "PSNR")
    axes[0].set_yscale("log")
    fig.tight_layout()
    return fig


def show_grid(columns: dict[str, list[np.ndarray]], row_labels: list[str] | None = None, size: float = 2.6):
    """Grid of images: one column per entry of `columns` (title -> list of images), one row per example."""
    titles = list(columns)
    n_rows = len(columns[titles[0]])
    fig, axes = plt.subplots(n_rows, len(titles), figsize=(size * len(titles), size * n_rows), squeeze=False)
    for c, title in enumerate(titles):
        for r, img in enumerate(columns[title]):
            ax = axes[r, c]
            ax.imshow(np.clip(img, 0, 1) if np.asarray(img).dtype != np.uint8 else img)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=10)
            if c == 0 and row_labels:
                ax.set_ylabel(row_labels[r], fontsize=9)
    fig.tight_layout()
    return fig


def draw_autoencoder(filters: tuple[int, ...], skips: bool, title: str = ""):
    """U-shaped diagram of the autoencoder: encoder going down, decoder going up, optional skip connections."""
    levels = len(filters)
    fig, ax = plt.subplots(figsize=(10, 1.25 * levels + 1.4))
    w, h = 2.6, 0.7
    for i, n in enumerate(filters):
        y = (levels - 1 - i) * 1.2
        res = "H x W" if i == 0 else f"H/{2**i} x W/{2**i}"
        if i < levels - 1:
            for x0, label, color in ((0.0, "encoder", "#9ecae1"), (7.4, "decoder", "#fdae6b")):
                ax.add_patch(FancyBboxPatch((x0, y), w, h, boxstyle="round,pad=0.03", fc=color, ec="#444"))
                ax.text(x0 + w / 2, y + h / 2, f"2 x Conv 3x3, {n} filters\n{res}", ha="center", va="center", fontsize=8.5)
            ax.annotate("", xy=(w / 2, y - 0.5), xytext=(w / 2, y), arrowprops=dict(arrowstyle="->"))
            ax.text(w / 2 + 0.1, y - 0.27, "MaxPool 2x2", fontsize=7.5)
            ax.annotate("", xy=(7.4 + w / 2, y), xytext=(7.4 + w / 2, y - 0.5), arrowprops=dict(arrowstyle="->"))
            ax.text(7.4 + w / 2 + 0.1, y - 0.27, "Conv2DTranspose 2x2, stride 2", fontsize=7.5)
            if skips:
                ax.annotate("", xy=(7.4, y + h / 2), xytext=(w, y + h / 2),
                            arrowprops=dict(arrowstyle="->", color="#2a7f2a", lw=1.5, ls="--"))
                ax.text(5.0, y + h / 2 + 0.08, "skip connection (concatenate)", ha="center", fontsize=8, color="#2a7f2a")
        else:
            ax.add_patch(FancyBboxPatch((3.7, y), w, h, boxstyle="round,pad=0.03", fc="#c7e9c0", ec="#444"))
            ax.text(3.7 + w / 2, y + h / 2, f"bottleneck\n2 x Conv 3x3, {n} filters\n{res}", ha="center", va="center", fontsize=8.5)
            ax.annotate("", xy=(3.7, y + h / 2), xytext=(w / 2, y + 1.2 - 0.5), arrowprops=dict(arrowstyle="->"))
            ax.annotate("", xy=(7.4 + w / 2, y + 1.2 - 0.5), xytext=(3.7 + w, y + h / 2), arrowprops=dict(arrowstyle="->"))
    top = (levels - 1) * 1.2 + h
    ax.text(w / 2, top + 0.3, "noisy image (H x W x 3)", ha="center", fontsize=9, fontweight="bold")
    ax.text(7.4 + w / 2, top + 0.3, "denoised image: Conv 1x1, 3 channels, sigmoid", ha="center", fontsize=9, fontweight="bold")
    ax.set_xlim(-0.3, 10.3)
    ax.set_ylim(-0.6, top + 0.7)
    ax.axis("off")
    ax.set_title(title)
    fig.tight_layout()
    return fig
