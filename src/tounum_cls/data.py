"""Inventaire, échantillonnage, extraction, chargement et découpage des images."""
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

# Certaines peintures sont très grandes : on désactive la protection anti "decompression bomb"
# de Pillow (les archives viennent de TouNum, source de confiance).
Image.MAX_IMAGE_PIXELS = None


# --------------------------------------------------------------------------- inventaire
def list_zip_images(zip_path: str | Path) -> list[str]:
    """Noms (triés) des images d'une archive : extensions jpg/jpeg/png uniquement.

    Ignore les dossiers, `desktop.ini`, les entrées `__MACOSX` et les fichiers cachés.
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
    """{classe: [noms de fichiers dans l'archive]} pour les 5 archives."""
    return {c: list_zip_images(Path(zip_dir) / z) for c, z in config.ZIP_FILES.items()}


# --------------------------------------------------------------------------- échantillonnage
def sample_dataset(
    inventory: dict[str, list[str]],
    pool_counts: dict[str, int],
    ext_counts: dict[str, int | None],
    seed: int = config.SEED,
) -> pd.DataFrame:
    """Tire, par classe, un pool de modélisation puis un test étendu **disjoints**.

    Colonnes : member (nom dans le zip), source (classe), label (1 = photo), subset ('pool' | 'ext').
    `ext_counts[c] = None` -> toutes les images restantes de la classe.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for source, members in inventory.items():
        n_pool = pool_counts[source]
        n_ext = ext_counts[source]
        n_ext_eff = len(members) - n_pool if n_ext is None else n_ext
        if n_pool + n_ext_eff > len(members):
            raise ValueError(
                f"{source}: {n_pool}+{n_ext_eff} images demandées mais seulement {len(members)} disponibles"
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
    """Extrait uniquement les fichiers du tirage vers raw_dir/<source>/<nom>. Ajoute la colonne `path`.

    Idempotent : un fichier déjà extrait n'est pas réécrit.
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
                if not dest.exists():
                    dest.write_bytes(zf.read(member))
                paths.loc[idx] = str(dest)
    out["path"] = paths
    return out


# --------------------------------------------------------------------------- chargement
def load_image(path: str | Path, size: int = config.IMG_SIZE) -> np.ndarray:
    """Charge une image avec ImageIO et la renvoie en (size, size, 3) uint8 RGB.

    Gère niveaux de gris (2D ou 1 canal), RGBA (alpha supprimé) et profondeur 16 bits.
    Le redimensionnement (bilinéaire) ne conserve pas le ratio : l'image est étirée.
    """
    arr = np.asarray(iio.imread(path))
    if arr.dtype != np.uint8:
        arr = (arr.astype(np.float64) / (65535.0 if arr.dtype == np.uint16 else 1.0)) * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] == 2:                       # gris + alpha
        arr = np.repeat(arr[..., :1], 3, axis=-1)
    elif arr.shape[-1] == 4:                       # RGBA
        arr = arr[..., :3]
    img = Image.fromarray(arr).resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def load_images(
    meta: pd.DataFrame, size: int = config.IMG_SIZE, workers: int | None = None
) -> tuple[np.ndarray, pd.DataFrame]:
    """Charge toutes les images de meta['path'] (en parallèle).

    Renvoie (X, meta_ok) : les images illisibles sont écartées (et signalées par un print).
    X.shape == (len(meta_ok), size, size, 3), dtype uint8, aligné ligne à ligne avec meta_ok.
    """
    def _safe(path):
        try:
            return load_image(path, size)
        except Exception as exc:  # image corrompue : on l'écarte
            print(f"[WARN] image illisible ignorée : {path} ({type(exc).__name__}: {exc})")
            return None

    workers = workers or min(8, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_safe, meta["path"].tolist()))
    keep = [r is not None for r in results]
    meta_ok = meta[keep].reset_index(drop=True)
    X = np.stack([r for r in results if r is not None]).astype(np.uint8)
    return X, meta_ok


# --------------------------------------------------------------------------- découpage
def add_split(meta: pd.DataFrame, seed: int = config.SEED) -> pd.DataFrame:
    """Ajoute la colonne `split` : train/val/test (70/15/15, stratifié par classe source) pour le pool,
    'ext' pour le test étendu."""
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
    """Sous-ensemble (X, y, meta) d'un split, éventuellement limité à certaines classes sources.

    Les photos sont toujours conservées ; `sources` ne liste que les classes NÉGATIVES voulues.
    """
    mask = meta["split"] == split
    if sources is not None:
        mask &= meta["source"].isin([config.POSITIVE_CLASS, *sources])
    m = meta[mask]
    return X[m.index.to_numpy()], m["label"].to_numpy().astype("float32"), m.reset_index(drop=True)


def class_weights(y: np.ndarray) -> dict[int, float]:
    """Poids 'balanced' : n / (2 * n_classe). Compense le déséquilibre des étapes 1 à 3."""
    y = np.asarray(y).astype(int)
    n = len(y)
    return {c: n / (2.0 * max(1, int((y == c).sum()))) for c in (0, 1)}


def make_dataset(X, y, batch_size=config.BATCH_SIZE, shuffle=False, seed=config.SEED):
    """tf.data.Dataset (X uint8, y float32) -> batchs ; shuffle uniquement pour l'entraînement."""
    import tensorflow as tf

    ds = tf.data.Dataset.from_tensor_slices((X, np.asarray(y, dtype="float32")))
    if shuffle:
        ds = ds.shuffle(len(X), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
