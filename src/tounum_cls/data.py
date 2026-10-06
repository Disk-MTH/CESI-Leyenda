"""Inventory, sampling, extraction, loading and splitting of the images."""
from __future__ import annotations

import os
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from . import config

# Some paintings are very large (up to 68 Mpx in the archives). Pillow's limit is raised above the largest
# image while still protecting against decompression bombs.
Image.MAX_IMAGE_PIXELS = 200_000_000


# --------------------------------------------------------------------------- inventory
def list_zip_images(zip_path: str | Path) -> list[str]:
    """Sorted image names in an archive (jpg/jpeg/png only).

    Skips folders, `desktop.ini`, `__MACOSX` entries and hidden files.
    """
    names = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            base = os.path.basename(name)
            if not base or name.endswith("/"):
                continue
            if "__MACOSX" in name or base.startswith("."):
                continue
            if base.lower().endswith(config.IMAGE_EXTENSIONS):
                names.append(name)
    return sorted(names)


def build_inventory(zip_dir: str | Path = config.ZIP_DIR) -> dict[str, list[str]]:
    """{class: [file names in the archive]} for the 5 archives."""
    return {c: list_zip_images(Path(zip_dir) / z) for c, z in config.ZIP_FILES.items()}


# --------------------------------------------------------------------------- sampling
def sample_dataset(
    inventory: dict[str, list[str]],
    pool_counts: dict[str, int],
    ext_counts: dict[str, int | None],
    seed: int = config.SEED,
) -> pd.DataFrame:
    """Draw, for each class, a modelling pool and an extended test set that do not overlap.

    Columns: member (name in the zip), source (class), label (1 = photo), subset ('pool' | 'ext').
    `ext_counts[c] = None` means every image of the class left after the pool.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for source, members in inventory.items():
        n_pool = pool_counts[source]
        n_ext = ext_counts[source]
        n_ext_eff = len(members) - n_pool if n_ext is None else n_ext
        if n_pool + n_ext_eff > len(members):
            raise ValueError(
                f"{source}: {n_pool}+{n_ext_eff} images requested but only {len(members)} available"
            )
        order = rng.permutation(len(members))
        label = int(source == config.POSITIVE_CLASS)
        for rank, i in enumerate(order[: n_pool + n_ext_eff]):
            rows.append(
                {
                    "member": members[i],
                    "source": source,
                    "label": label,
                    "subset": "pool" if rank < n_pool else "ext",
                }
            )
    return pd.DataFrame(rows)


def extract_files(
    sample: pd.DataFrame,
    zip_dir: str | Path = config.ZIP_DIR,
    raw_dir: str | Path = config.RAW_DIR,
) -> pd.DataFrame:
    """Extract only the sampled files to raw_dir/<source>/<name> and add the `path` column.

    Idempotent: a file already on disk is not rewritten unless its size differs from the archive
    (interrupted extraction or truncated file).
    """
    raw_dir = Path(raw_dir)
    out = sample.copy()
    paths = pd.Series("", index=out.index, dtype=object)
    for source, grp in out.groupby("source"):
        dest_dir = raw_dir / source
        dest_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(Path(zip_dir) / config.ZIP_FILES[source]) as zf:
            for idx, member in grp["member"].items():
                dest = dest_dir / os.path.basename(member)
                if not dest.exists() or dest.stat().st_size != zf.getinfo(member).file_size:
                    dest.write_bytes(zf.read(member))
                paths.loc[idx] = str(dest)
    out["path"] = paths
    return out


# --------------------------------------------------------------------------- loading
def read_rgb(path: str | Path) -> np.ndarray:
    """Read an image with ImageIO at its original size, as (height, width, 3) uint8 RGB.

    Handles grayscale (2D or 1 channel), RGBA (alpha dropped) and 16-bit depth.
    """
    arr = np.asarray(iio.imread(path))
    if arr.dtype != np.uint8:
        arr = (arr.astype(np.float64) / (65535.0 if arr.dtype == np.uint16 else 1.0)) * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] == 2:                       # gray + alpha
        arr = np.repeat(arr[..., :1], 3, axis=-1)
    elif arr.shape[-1] == 4:                       # RGBA
        arr = arr[..., :3]
    return np.ascontiguousarray(arr)


def load_image(path: str | Path, size: int = config.IMG_SIZE) -> np.ndarray:
    """Load an image with ImageIO and return it as (size, size, 3) uint8 RGB.

    The bilinear resize does not keep the aspect ratio: the image is stretched.
    """
    return resize(read_rgb(path), size)


def resize(img: np.ndarray, size: int = config.IMG_SIZE) -> np.ndarray:
    """Resize a uint8 RGB image to (size, size, 3) with bilinear interpolation (aspect ratio not kept)."""
    return np.asarray(Image.fromarray(img).resize((size, size), Image.BILINEAR), dtype=np.uint8)


def load_images(
    meta: pd.DataFrame, size: int = config.IMG_SIZE, workers: int | None = None
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load every image in meta['path'] (in parallel).

    Returns (X, meta_ok): unreadable images are dropped (and reported with a print).
    X.shape == (len(meta_ok), size, size, 3), dtype uint8, aligned row by row with meta_ok.
    """
    def _safe(path):
        try:
            return load_image(path, size)
        except Exception as exc:  # corrupt image: skip it
            print(f"[WARN] unreadable image skipped: {path} ({type(exc).__name__}: {exc})")
            return None

    workers = workers or min(8, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_safe, meta["path"].tolist()))
    keep = [r is not None for r in results]
    meta_ok = meta[keep].reset_index(drop=True)
    X = np.stack([r for r in results if r is not None]).astype(np.uint8)
    return X, meta_ok


# --------------------------------------------------------------------------- splitting
def add_split(meta: pd.DataFrame, seed: int = config.SEED) -> pd.DataFrame:
    """Add the `split` column: train/val/test (70/15/15, stratified by source class) for the pool,
    'ext' for the extended test set."""
    meta = meta.copy()
    meta["split"] = "ext"
    pool_idx = meta.index[meta["subset"] == "pool"]
    train_idx, tmp_idx = train_test_split(
        pool_idx, test_size=0.30, stratify=meta.loc[pool_idx, "source"], random_state=seed
    )
    val_idx, test_idx = train_test_split(
        tmp_idx, test_size=0.50, stratify=meta.loc[tmp_idx, "source"], random_state=seed
    )
    meta.loc[train_idx, "split"] = "train"
    meta.loc[val_idx, "split"] = "val"
    meta.loc[test_idx, "split"] = "test"
    assert not (set(meta.loc[train_idx, "member"]) & set(meta.loc[test_idx, "member"]))
    return meta


def select(
    X: np.ndarray, meta: pd.DataFrame, split: str, sources: list[str] | None = None
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """(X, y, meta) subset of one split, optionally restricted to some source classes.

    Photos are always kept; `sources` lists only the NEGATIVE classes wanted.
    """
    mask = meta["split"] == split
    if sources is not None:
        mask &= meta["source"].isin([config.POSITIVE_CLASS, *sources])
    m = meta[mask]
    return X[m.index.to_numpy()], m["label"].to_numpy().astype("float32"), m.reset_index(drop=True)


def class_weights(y: np.ndarray) -> dict[int, float]:
    """'Balanced' weights: n / (2 * n_class). Offsets the class imbalance of stages 1 to 3."""
    y = np.asarray(y).astype(int)
    n = len(y)
    return {c: n / (2.0 * max(1, int((y == c).sum()))) for c in (0, 1)}


def make_dataset(X, y, batch_size=config.BATCH_SIZE, shuffle=False, seed=config.SEED):
    """tf.data.Dataset of (X uint8, y float32) batches; shuffle only for training."""
    import tensorflow as tf

    ds = tf.data.Dataset.from_tensor_slices((X, np.asarray(y, dtype="float32")))
    if shuffle:
        ds = ds.shuffle(len(X), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# --------------------------------------------------------------------------- orchestration
def prepare_dataset(force: bool = False) -> tuple[np.ndarray, pd.DataFrame]:
    """Full pipeline: inventory -> sampling -> extraction -> loading -> split -> cache.

    Later calls simply read data/cache/X.npy and data/cache/meta.csv back.
    Returns (X, meta); meta has the columns member, source, label, subset, path, split.
    `path` is stored relative to the project root in meta.csv (so the project can be moved)
    and returned as an absolute path.
    """
    x_path, m_path = config.CACHE_DIR / "X.npy", config.CACHE_DIR / "meta.csv"
    if x_path.exists() and m_path.exists() and not force:
        meta = pd.read_csv(m_path)
        meta["path"] = meta["path"].map(lambda p: str(config.PROJECT_ROOT / p))
        return np.load(x_path), meta
    inventory = build_inventory(config.ZIP_DIR)
    sample = sample_dataset(inventory, config.POOL_COUNTS, config.EXT_COUNTS)
    sample = extract_files(sample, config.ZIP_DIR, config.RAW_DIR)
    X, meta = load_images(sample)
    meta = add_split(meta)
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(x_path, X)
    to_save = meta.copy()
    to_save["path"] = to_save["path"].map(lambda p: os.path.relpath(p, config.PROJECT_ROOT))
    to_save.to_csv(m_path, index=False)
    return X, meta
