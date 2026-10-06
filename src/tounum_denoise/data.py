"""Loading the noisy dataset, building clean training patches and generating synthetic noise."""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
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
# Full-range RGB <-> YCbCr conversion of the JPEG standard: Y is the brightness, Cb and Cr the colour differences.
RGB_TO_YCBCR = np.array([[0.299, 0.587, 0.114],
                         [-0.168736, -0.331264, 0.5],
                         [0.5, -0.418688, -0.081312]], dtype=np.float32)
YCBCR_TO_RGB = np.linalg.inv(RGB_TO_YCBCR).astype(np.float32)


def noise_kernel(blur: float = config.NOISE_BLUR) -> np.ndarray:
    """3x3 Gaussian kernel that blurs the noise, scaled so that blurred white noise keeps a standard deviation of 1."""
    ax = np.arange(-1, 2)
    k = np.exp(-(ax[:, None] ** 2 + ax[None, :] ** 2) / (2 * blur**2))
    return (k / np.sqrt((k**2).sum())).astype(np.float32)


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
              jpeg_quality: int | None = config.JPEG_QUALITY, blur: float = config.NOISE_BLUR) -> np.ndarray:
    """Synthetic version of the TouNum noise for one uint8 RGB image: Gaussian noise of standard deviation `sigma`
    (0-255 scale) on the colour channels Cb and Cr, brightness Y untouched, blurred by a 3x3 Gaussian kernel
    (`blur` = 0 for white noise), then clipping and a JPEG re-encoding (`jpeg_quality` = None to skip it)."""
    noise = rng.standard_normal(clean.shape[:2] + (2,), dtype=np.float32)
    if blur:
        noise = np.stack([ndimage.convolve(noise[..., c], noise_kernel(blur), mode="mirror") for c in range(2)], -1)
    ycc = np.asarray(clean, np.float32) @ RGB_TO_YCBCR.T
    ycc[..., 1:] += noise * sigma
    noisy = np.clip(np.rint(ycc @ YCBCR_TO_RGB.T), 0, 255).astype(np.uint8)
    return jpeg_roundtrip(noisy, jpeg_quality) if jpeg_quality else noisy


def make_dataset(clean_uint8: np.ndarray, batch_size: int = config.BATCH_SIZE, shuffle: bool = True,
                 sigma_range=config.SIGMA_RANGE, jpeg_quality: int = config.JPEG_QUALITY, seed: int = config.SEED):
    """tf.data pipeline of (noisy, clean) float32 pairs in [0, 1], with the same noise as `add_noise`. The noise
    level, the noise itself and a horizontal flip are drawn again at every epoch, so the network never sees the
    same noisy patch twice."""
    import tensorflow as tf

    ds = tf.data.Dataset.from_tensor_slices(clean_uint8)
    if shuffle:
        ds = ds.shuffle(len(clean_uint8), seed=seed, reshuffle_each_iteration=True)
    lo, hi = sigma_range
    to_ycc, to_rgb = tf.constant(RGB_TO_YCBCR.T), tf.constant(YCBCR_TO_RGB.T)
    kernel = tf.constant(np.tile(noise_kernel()[:, :, None, None], (1, 1, 2, 1)))     # same blur on Cb and Cr

    def _pair(img):
        clean = tf.image.random_flip_left_right(img)
        sigma = tf.random.uniform([], lo, hi)
        noise = tf.random.normal(tf.concat([tf.shape(clean)[:2], [2]], 0))
        noise = tf.pad(noise, [[1, 1], [1, 1], [0, 0]], mode="REFLECT")[None]
        noise = tf.nn.depthwise_conv2d(noise, kernel, [1, 1, 1, 1], "VALID")[0] * sigma
        ycc = tf.tensordot(tf.cast(clean, tf.float32), to_ycc, axes=1)
        ycc += tf.pad(noise, [[0, 0], [0, 0], [1, 0]])                                  # Y is left untouched
        noisy = tf.cast(tf.clip_by_value(tf.round(tf.tensordot(ycc, to_rgb, axes=1)), 0, 255), tf.uint8)
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


# --------------------------------------------------------------------------- real pairs
def split_real_pairs(files: list[str], n_val: int = config.REAL_VAL, n_test: int = config.REAL_TEST,
                     seed: int = config.SEED) -> dict[str, list[str]]:
    """Random split of the noisy file names into 'finetune', 'val' and 'test' (sorted names in each part)."""
    order = np.random.default_rng(seed).permutation(len(files))
    parts = {"test": order[:n_test], "val": order[n_test:n_test + n_val], "finetune": order[n_test + n_val:]}
    return {k: sorted(files[i] for i in v) for k, v in parts.items()}


def pair_crops(noisy_images, clean_images, n: int, size: int = config.CROP_SIZE,
               seed: int = config.SEED) -> tuple[np.ndarray, np.ndarray]:
    """`n` random aligned (size, size) crops from each (noisy, clean) image pair -> two uint8 arrays."""
    rng = np.random.default_rng(seed)
    noisy_out, clean_out = [], []
    for noisy, clean in zip(noisy_images, clean_images):
        h, w = noisy.shape[:2]
        for y, x in zip(rng.integers(0, h - size + 1, n), rng.integers(0, w - size + 1, n)):
            noisy_out.append(noisy[y:y + size, x:x + size])
            clean_out.append(clean[y:y + size, x:x + size])
    return np.stack(noisy_out), np.stack(clean_out)


def make_pair_dataset(noisy_uint8: np.ndarray, clean_uint8: np.ndarray, batch_size: int = config.BATCH_SIZE,
                      seed: int = config.SEED):
    """tf.data pipeline of fixed (noisy, clean) float32 pairs in [0, 1], shuffled, with random horizontal flips."""
    import tensorflow as tf

    def _flip(noisy, clean):
        flip = tf.random.uniform([]) < 0.5
        noisy, clean = tf.cast(noisy, tf.float32) / 255.0, tf.cast(clean, tf.float32) / 255.0
        return tf.cond(flip, lambda: (noisy[:, ::-1], clean[:, ::-1]), lambda: (noisy, clean))

    ds = tf.data.Dataset.from_tensor_slices((noisy_uint8, clean_uint8))
    ds = ds.shuffle(len(noisy_uint8), seed=seed, reshuffle_each_iteration=True)
    return ds.map(_flip, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


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
