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
