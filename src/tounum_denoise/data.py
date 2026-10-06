"""Loading the noisy dataset, building clean training patches and generating synthetic noise."""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.signal import convolve2d

from tounum_cls import data as cls_data

from . import config


# --------------------------------------------------------------------------- noisy dataset
def extract_noisy(zip_path: Path = config.NOISY_ZIP, dest: Path = config.NOISY_DIR) -> list[Path]:
    """Extract the noisy images of the archive (once) and return their sorted paths."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                target = dest / Path(name).name
                if not target.exists() or target.stat().st_size != zf.getinfo(name).file_size:
                    target.write_bytes(zf.read(name))
    return sorted(p for p in dest.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


def load_noisy(paths) -> tuple[list[np.ndarray], pd.DataFrame]:
    """Read the noisy images at full size. Returns (images, table with file, width, height, estimated noise)."""
    images = [cls_data.read_rgb(p) for p in paths]
    info = pd.DataFrame({
        "file": [Path(p).name for p in paths],
        "width": [im.shape[1] for im in images],
        "height": [im.shape[0] for im in images],
        "noise_sigma": [estimate_noise(im) for im in images],
    })
    return images, info


# --------------------------------------------------------------------------- clean training patches
def downscale(img: np.ndarray, short_side: int = config.CLEAN_SHORT_SIDE) -> np.ndarray:
    """Downscale (never upscale) so the short side equals `short_side`, keeping the aspect ratio."""
    h, w = img.shape[:2]
    scale = short_side / min(h, w)
    if scale >= 1:
        return img
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return np.asarray(Image.fromarray(img).resize(size, Image.LANCZOS))


def random_crops(img: np.ndarray, n: int, size: int, rng: np.random.Generator) -> np.ndarray:
    """`n` random (size, size, 3) crops of an image (images smaller than `size` are resized up first)."""
    h, w = img.shape[:2]
    if h < size or w < size:
        img = np.asarray(Image.fromarray(img).resize((max(w, size), max(h, size)), Image.BILINEAR))
        h, w = img.shape[:2]
    ys = rng.integers(0, h - size + 1, n)
    xs = rng.integers(0, w - size + 1, n)
    return np.stack([img[y:y + size, x:x + size] for y, x in zip(ys, xs)])


def build_patches(paths, crops_per_image: int = config.CROPS_PER_IMAGE, size: int = config.CROP_SIZE,
                  seed: int = config.SEED) -> np.ndarray:
    """Clean uint8 patches (n, size, size, 3) from a list of photo files (downscaled first)."""
    rng = np.random.default_rng(seed)
    out = [random_crops(downscale(cls_data.read_rgb(p)), crops_per_image, size, rng) for p in paths]
    return np.concatenate(out).astype(np.uint8)


def cached_patches(name: str, paths, **kwargs) -> np.ndarray:
    """build_patches with a cache in data/denoise_cache/<name>.npy (delete the file to rebuild it)."""
    path = config.CACHE_DIR / f"{name}.npy"
    if path.exists():
        return np.load(path)
    patches = build_patches(paths, **kwargs)
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, patches)
    return patches


# --------------------------------------------------------------------------- noise
def to_float(img) -> np.ndarray:
    """uint8 [0, 255] -> float32 [0, 1]."""
    return np.asarray(img, dtype=np.float32) / 255.0


def to_uint8(img) -> np.ndarray:
    """float [0, 1] -> uint8 [0, 255] (clipped and rounded)."""
    return np.clip(np.rint(np.asarray(img) * 255.0), 0, 255).astype(np.uint8)


def jpeg_roundtrip(img: np.ndarray, quality: int = config.JPEG_QUALITY) -> np.ndarray:
    """Encode a uint8 RGB image as JPEG (no chroma subsampling) and decode it back."""
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality, subsampling=0)
    return np.asarray(Image.open(buf).convert("RGB"))


def add_noise(clean: np.ndarray, sigma: float, rng: np.random.Generator,
              jpeg_quality: int | None = config.JPEG_QUALITY) -> np.ndarray:
    """Synthetic version of the TouNum noise for one uint8 RGB image: Gaussian noise of standard deviation
    `sigma` (0-255 scale) on every pixel and channel, clipping, then a JPEG re-encoding (None to skip it)."""
    noisy = np.asarray(clean, np.float32) + rng.standard_normal(clean.shape, dtype=np.float32) * sigma
    noisy = np.clip(np.rint(noisy), 0, 255).astype(np.uint8)
    return jpeg_roundtrip(noisy, jpeg_quality) if jpeg_quality else noisy


def make_dataset(clean_uint8: np.ndarray, batch_size: int = config.BATCH_SIZE, shuffle: bool = True,
                 sigma_range=config.SIGMA_RANGE, jpeg_quality: int = config.JPEG_QUALITY, seed: int = config.SEED):
    """tf.data pipeline of (noisy, clean) float32 pairs in [0, 1]. The noise level, the noise itself and a
    horizontal flip are drawn again at every epoch, so the network never sees the same noisy patch twice."""
    import tensorflow as tf

    ds = tf.data.Dataset.from_tensor_slices(clean_uint8)
    if shuffle:
        ds = ds.shuffle(len(clean_uint8), seed=seed, reshuffle_each_iteration=True)
    lo, hi = sigma_range

    def _pair(img):
        clean = tf.image.random_flip_left_right(img)
        sigma = tf.random.uniform([], lo, hi)
        noisy = tf.cast(clean, tf.float32) + tf.random.normal(tf.shape(clean)) * sigma
        noisy = tf.cast(tf.clip_by_value(tf.round(noisy), 0, 255), tf.uint8)
        noisy = tf.io.decode_jpeg(tf.io.encode_jpeg(noisy, quality=jpeg_quality, chroma_downsampling=False), channels=3)
        return tf.cast(noisy, tf.float32) / 255.0, tf.cast(clean, tf.float32) / 255.0

    return ds.map(_pair, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def fixed_noisy_set(clean_uint8: np.ndarray, sigma_range=config.SIGMA_RANGE, seed: int = config.SEED):
    """Validation pairs with the noise drawn once (seeded), so validation scores can be compared between epochs
    and between models. Returns (noisy float, clean float, sigmas)."""
    rng = np.random.default_rng(seed)
    sigmas = rng.uniform(*sigma_range, len(clean_uint8))
    noisy = np.stack([add_noise(img, s, rng) for img, s in zip(clean_uint8, sigmas)])
    return to_float(noisy), to_float(clean_uint8), sigmas


# --------------------------------------------------------------------------- clean originals of the noisy images
def load_originals(noisy_paths, csv: Path = config.ORIGINALS_CSV, dest: Path = config.ORIGINALS_DIR) -> list[np.ndarray]:
    """Download (once) and read the clean COCO original of each noisy file, in the same order."""
    mapping = pd.read_csv(csv).set_index("file")
    dest.mkdir(parents=True, exist_ok=True)
    images = []
    for p in noisy_paths:
        row = mapping.loc[Path(p).name]
        target = dest / row["coco_file"]
        if not target.exists():
            with urllib.request.urlopen(f"{config.COCO_URL}/{row['coco_file']}", timeout=60) as resp:
                target.write_bytes(resp.read())
        images.append(cls_data.read_rgb(target))
    return images


# --------------------------------------------------------------------------- noise level estimation
_IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)


def estimate_noise(img: np.ndarray) -> float:
    """Estimate the standard deviation of Gaussian noise (0-255 scale, per channel) without a clean reference.

    Immerkaer's method (1996): a 3x3 convolution kernel that cancels smooth image content (it is the
    difference of two Laplacians), so what is left is mostly noise. The estimate is computed on each channel
    and averaged. Textured images get a higher estimate than their true noise level.
    """
    arr = np.asarray(img, dtype=np.float64)
    channels = [arr] if arr.ndim == 2 else [arr[..., c] for c in range(arr.shape[-1])]
    h, w = channels[0].shape
    sigmas = [np.sqrt(np.pi / 2) * np.abs(convolve2d(c, _IMMERKAER, mode="valid")).sum() / (6 * (w - 2) * (h - 2))
              for c in channels]
    return float(np.mean(sigmas))
