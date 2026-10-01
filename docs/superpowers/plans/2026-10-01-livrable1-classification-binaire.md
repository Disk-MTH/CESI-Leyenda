# Livrable 1 — Classification binaire photo / non-photo : plan d'implémentation

> **Pour les agents d'exécution :** SOUS-COMPÉTENCE REQUISE : utiliser `superpowers:subagent-driven-development` (recommandé)
> ou `superpowers:executing-plans` pour exécuter ce plan tâche par tâche. Les étapes utilisent la syntaxe case à cocher (`- [ ]`).

**Objectif :** livrer un notebook Jupyter exécuté (`notebooks/livrable1_classification_binaire.ipynb`) qui entraîne et évalue un CNN
TensorFlow séparant les photos de tous les autres types d'images, avec le code, les tests et la documentation nécessaires pour que TouNum le relance et le maintienne.

**Architecture :** un petit paquet Python testé (`src/tounum_cls/` : `config`, `data`, `modeling`, `evaluation`) contient toute la logique non pédagogique ;
le notebook (écrit au format texte jupytext `.py`, converti en `.ipynb` puis exécuté) l'appelle, affiche le code du réseau et commente les résultats avec
des cellules « Lecture automatique » qui calculent leurs propres chiffres. Les images sont tirées des archives zip, redimensionnées en 128×128 RGB `uint8` et mises en cache (`data/cache`).

**Stack technique :** **uv** (environnement et verrouillage), Python 3.12, TensorFlow 2.21 / Keras 3 (CPU), scikit-learn, pandas, ImageIO + Pillow, matplotlib + seaborn, jupytext + nbconvert, pytest.

**Spec :** `docs/superpowers/specs/2026-10-01-livrable1-classification-binaire-design.md` (à lire en entier avant de commencer).

## Contraintes globales (valables pour TOUTES les tâches)

- Langue : tout le texte destiné au lecteur (notebook, README) est en **français** ; identifiants de code et messages de commit en anglais.
- **Environnement géré exclusivement par `uv`** (exigence explicite du client, pour éviter les conflits de paquets) : `pyproject.toml` + `uv.lock` versionnés, `.venv/` créé par `uv sync`, toutes les commandes via `uv run …`. **Interdit** : `pip install`, `python -m venv`, `conda`, activer le venv à la main. Ajouter une dépendance = `uv add <paquet>` (jamais d'édition manuelle de `uv.lock`).
- Python **3.12** (`requires-python = ">=3.12,<3.13"`, `.python-version` = `3.12`). **Ne jamais utiliser le Python 3.14 du système** (TensorFlow n'y est pas disponible).
- **CPU uniquement**, aucun GPU. TensorFlow ≥ 2.21 (Keras 3) : `import keras` (pas `tf.keras`).
- Graine unique `SEED = 42` (dans `config.py`). Ne jamais changer une graine, un `random_state` ou une valeur de `config.py` pour « faire marcher » un résultat.
- Images : 128×128, RGB, `uint8`. Label `1` = photo (`Photo`), label `0` = tout le reste. Normalisation ÷255 **dans le modèle**.
- Effectifs (fixés par la spec) : `POOL_COUNTS = Photo 3000, Painting 1200, Schematics 600, Text 600, Sketch 600` ; test étendu `EXT_COUNTS = 1000 par classe, Sketch = tout le reste (806)`.
- Découpage du pool : 70 % train / 15 % val / 15 % test, **stratifié par classe source**. Le jeu `test` n'est utilisé que pour la mesure finale ; la sélection du modèle se fait sur `val`.
- Le code TensorFlow du réseau (E1) doit être **visible dans le notebook** (affiché par `inspect.getsource`), avec le schéma, les paramètres, la perte et l'optimiseur.
- Le notebook ne contient **aucun chiffre de résultat écrit à la main** : les chiffres sont toujours calculés par les cellules « Lecture automatique ».
- Ne **jamais** modifier les archives `Dataset Livrable *.zip`, ne **jamais** les ajouter à git (7 Go), ni `data/`, ni `.venv/`.
- `pytest` doit passer à chaque fin de tâche. Si un test échoue : diagnostiquer le code, **ne pas affaiblir ni supprimer le test**.
- `Dataset Livrable 2.zip` (images bruitées) est hors périmètre.
- Les blocs de code de ce plan sont **vérifiés** : les recopier tels quels (y compris accents, commentaires et valeurs). Ne rien « améliorer » en cours de route.

## Pièges connus (à lire avant de commencer)

1. **TensorFlow est bavard** (messages CUDA / oneDNN / `absl`) : c'est normal et sans conséquence.
2. **Les entraînements n'affichent rien** (`verbose=0`) : une cellule peut tourner plusieurs minutes en silence. Normal.
3. **`bn_momentum=0.9` est voulu** (défaut Keras : 0.99). Avec 0.99, la validation reste ≈ 0.5 pendant plusieurs epochs. Ne pas y toucher.
4. **Le mode QUICK (`TOUNUM_QUICK=1`, 2 epochs) sert uniquement à vérifier que le notebook s'exécute.** Ses chiffres sont sans valeur : ne jamais les citer ni les comparer aux plages attendues de la tâche 13.
5. Le notebook se lance **depuis `notebooks/`** et avec **`uv run jupyter …`** : le script `scripts/run_notebook.sh` fait les deux (sinon le noyau n'aura pas TensorFlow).
6. Source de vérité du notebook : le fichier **`.py`** (format jupytext « percent »). Le `.ipynb` est régénéré à chaque exécution. Dans le `.py` : une cellule de code commence par `# %%`, une cellule Markdown par `# %% [markdown]` et chacune de ses lignes commence par `# `.
7. Les 6 archives sont à la racine du projet et le code les lit depuis là (`config.ZIP_DIR`). Elles ne doivent pas être déplacées.

---

## Cartographie spec → tâches

| Exigence de la spec | Tâche(s) |
|---|---|
| §2 Données, lecture des archives, fichiers parasites | 2, 3 |
| §3.1 Échantillonnage équilibré + test étendu disjoint | 2, 4 |
| §3.2 Chargement ImageIO, RGB, 128×128, cache | 3, 4 |
| §3.3 Découpage 70/15/15 stratifié | 3 |
| §3.4 CNN maison, perte, optimiseur | 5, 10 |
| §3.5 MobileNetV2 gelé de comparaison | 5, 12 |
| §3.6 Progression par difficulté (E5) | 9 |
| §3.7 Évaluation finale, seuil, erreurs | 6, 7, 12 |
| E1 code + schéma + paramètres + perte + optimiseur | 5, 7, 10 |
| E2 courbes loss/accuracy train vs test | 7, 10 |
| E3 analyse biais/variance | 6, 11 |
| E4 méthodes d'amélioration | 11 |
| §4 Livrables du dépôt (README, pyproject/uv.lock, .gitignore) ; §5 environnement uv | 1, 13 |
| §6 Critères d'acceptation | 13 |

---

### Task 1 : Environnement, squelette du projet et configuration

**Fichiers :**
- Créer : `.gitignore`, `pyproject.toml`, `.python-version`, `pytest.ini` (puis `uv.lock`, **généré** par `uv sync`)
- Créer : `src/tounum_cls/__init__.py`, `src/tounum_cls/config.py`
- Créer : `tests/conftest.py`, `tests/test_config.py`

**Interfaces :**
- Produit (utilisé par toutes les tâches suivantes) : le module `tounum_cls.config` avec les constantes `SEED`, `PROJECT_ROOT`, `ZIP_DIR`, `DATA_DIR`, `RAW_DIR`, `CACHE_DIR`,
  `IMG_SIZE`, `BATCH_SIZE`, `POSITIVE_CLASS`, `CLASSES`, `ZIP_FILES`, `IMAGE_EXTENSIONS`, `POOL_COUNTS`, `EXT_COUNTS`, `STAGES`.
- Produit : la fixture pytest `make_zip(name, n, folder=None, extra_png_gray=0)` qui fabrique une archive factice `Dataset Livrable 1 - {name}.zip` dans `tmp_path`.

- [ ] **Étape 1 : Initialiser git (si le dossier n'est pas déjà un dépôt) et créer `.gitignore` en premier**

Le dossier n'est pas un dépôt git au départ. Les archives font 7 Go : le `.gitignore` doit exister **avant** le premier `git add`.

**Créer `.gitignore`** — contenu exact :

```text
# Données et caches (volumineux, reconstruits par le notebook)
data/
*.zip
# Environnement et artefacts Python
.venv/
__pycache__/
*.pyc
.pytest_cache/
.ipynb_checkpoints/
*.egg-info/
```

Puis :

```bash
cd /home/diskmth/Dev/bloc_AI
git rev-parse --is-inside-work-tree 2>/dev/null || git init -b main
git status --short | head
```

Attendu : les `.zip` n'apparaissent **pas** dans `git status` ; `docs/` apparaît comme non suivi.

- [ ] **Étape 2 : Créer l'environnement avec uv (Python 3.12, dépendances épinglées, verrouillage)**

**Créer `pyproject.toml`** — contenu exact :

```toml
[project]
name = "tounum-cls"
version = "0.1.0"
description = "TouNum - classification binaire photo / non-photo (livrable 1)"
requires-python = ">=3.12,<3.13"
dependencies = [
    "tensorflow==2.21.0",
    "keras==3.15.1",
    "numpy==2.5.3",
    "pandas==3.0.6",
    "scikit-learn==1.9.1",
    "imageio==2.38.0",
    "pillow==12.3.0",
    "matplotlib==3.11.2",
    "seaborn==0.13.2",
    "jupytext==1.19.5",
    "nbconvert==7.17.1",
    "nbformat==5.11.1",
    "ipykernel==7.4.0",
]

[dependency-groups]
dev = ["pytest==9.1.1"]

[tool.uv]
package = false  # projet de scripts : le code de src/ est importé via sys.path / pytest.ini, pas installé
```

**Créer `.python-version`** — contenu exact :

```text
3.12
```

```bash
cd /home/diskmth/Dev/bloc_AI
uv --version                     # uv doit être installé (≥ 0.11) ; sinon : curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync                          # crée .venv/ (Python 3.12), installe les versions épinglées et écrit uv.lock ; ≈ 1 Go, quelques minutes la 1re fois
uv run python -c "import sys, tensorflow as tf, keras; print(sys.version.split()[0], tf.__version__, keras.__version__)"
ls uv.lock .venv/bin/python
```

Attendu : une ligne `3.12.x 2.21.0 3.15.1` (précédée éventuellement de messages CUDA/oneDNN, sans gravité) ; `uv.lock` et `.venv/bin/python` existent.
Si `uv sync` échoue sur une version introuvable, s'arrêter et signaler : ne pas relâcher les versions sans accord. Les `uv sync` suivants prennent ≈ 1 s.

- [ ] **Étape 3 : Créer `pytest.ini`, le paquet et la configuration**

**Créer `pytest.ini`** — contenu exact :

```ini
[pytest]
pythonpath = src
testpaths = tests
```

**Créer `src/tounum_cls/__init__.py`** — contenu exact :

```python
"""TouNum - Livrable 1 : classification binaire photo / non-photo."""
```

**Créer `src/tounum_cls/config.py`** — contenu exact :

```python
"""Constantes du projet. Tout ce qui est réglable se trouve ici."""
from pathlib import Path

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ZIP_DIR = PROJECT_ROOT                       # les archives sont à la racine du projet
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                   # images extraites : data/raw/<Classe>/<fichier>
CACHE_DIR = DATA_DIR / "cache"               # tableaux redimensionnés : X.npy + meta.csv

IMG_SIZE = 128
BATCH_SIZE = 64

POSITIVE_CLASS = "Photo"                     # label 1 ; toutes les autres classes = label 0
CLASSES = ["Photo", "Painting", "Schematics", "Text", "Sketch"]
ZIP_FILES = {c: f"Dataset Livrable 1 - {c}.zip" for c in CLASSES}
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Pool de modélisation (équilibré : 3000 photos / 3000 non-photos)
POOL_COUNTS = {"Photo": 3000, "Painting": 1200, "Schematics": 600, "Text": 600, "Sketch": 600}
# Test étendu, réaliste (déséquilibré). None = tout ce qui reste après le pool.
EXT_COUNTS = {"Photo": 1000, "Painting": 1000, "Schematics": 1000, "Text": 1000, "Sketch": None}

# Étapes de difficulté croissante : classes négatives opposées aux photos
STAGES = {
    "1_photo_vs_text": ["Text"],
    "2_+schematics": ["Text", "Schematics"],
    "3_+sketch": ["Text", "Schematics", "Sketch"],
    "4_+painting": ["Text", "Schematics", "Sketch", "Painting"],
}
```

**Créer `tests/conftest.py`** — contenu exact :

```python
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")  # pas d'affichage pendant les tests


def _img_bytes(fmt, mode="RGB", size=(40, 30), seed=0):
    rng = np.random.default_rng(seed)
    if mode == "L":
        arr = rng.integers(0, 256, size=(size[1], size[0]), dtype=np.uint8)
    elif mode == "RGBA":
        arr = rng.integers(0, 256, size=(size[1], size[0], 4), dtype=np.uint8)
    else:
        arr = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode).save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def make_zip(tmp_path):
    """Fabrique une archive {classe}.zip avec n jpg + parasites (desktop.ini, dossier, txt)."""

    def _make(name, n, folder=None, extra_png_gray=0):
        folder = folder or name
        path = tmp_path / f"Dataset Livrable 1 - {name}.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr(f"{folder}/", "")
            zf.writestr(f"{folder}/desktop.ini", "[.ShellClassInfo]")
            zf.writestr(f"{folder}/notes.txt", "hello")
            for i in range(n):
                zf.writestr(f"{folder}/{name.lower()}_{i:03d}.JPG", _img_bytes("JPEG", seed=i))
            for j in range(extra_png_gray):
                zf.writestr(f"{folder}/gray_{j:03d}.png", _img_bytes("PNG", mode="L", seed=100 + j))
        return path

    return _make
```

- [ ] **Étape 4 : Écrire le test de configuration**

**Créer `tests/test_config.py`** — contenu exact :

```python
from tounum_cls import config


def test_every_class_is_configured_everywhere():
    for mapping in (config.POOL_COUNTS, config.EXT_COUNTS, config.ZIP_FILES):
        assert set(mapping) == set(config.CLASSES)


def test_pool_is_balanced_photo_vs_non_photo():
    positives = config.POOL_COUNTS[config.POSITIVE_CLASS]
    negatives = sum(n for c, n in config.POOL_COUNTS.items() if c != config.POSITIVE_CLASS)
    assert positives == negatives


def test_stages_are_cumulative_and_end_with_all_negative_classes():
    previous = []
    for negatives in config.STAGES.values():
        assert set(previous) < set(negatives) or not previous
        previous = negatives
    assert set(previous) == set(config.CLASSES) - {config.POSITIVE_CLASS}
```

- [ ] **Étape 5 : Lancer les tests**

```bash
uv run pytest -q
```

Attendu : `3 passed`.

- [ ] **Étape 6 : Commit**

```bash
git add .gitignore pyproject.toml uv.lock .python-version pytest.ini src tests docs
git commit -m "chore: project skeleton, pinned environment and config"
```

Vérifier avec `git show --stat HEAD` qu'aucun `.zip`, ni `.venv/`, ni `data/` n'est dans le commit.

---

### Task 2 : Données — inventaire des archives, tirage reproductible, extraction sélective

**Fichiers :**
- Créer : `src/tounum_cls/data.py` (début du module)
- Créer : `tests/test_data.py` (début du fichier)

**Interfaces :**
- Consomme : `tounum_cls.config` (tâche 1) ; fixture `make_zip` (tâche 1).
- Produit :
  - `list_zip_images(zip_path: str | Path) -> list[str]` : noms d'images triés d'une archive (jpg/jpeg/png, hors dossiers, `desktop.ini`, `__MACOSX`, fichiers cachés).
  - `build_inventory(zip_dir=config.ZIP_DIR) -> dict[str, list[str]]` : `{classe: [noms]}` pour les 5 archives.
  - `sample_dataset(inventory, pool_counts, ext_counts, seed=config.SEED) -> pd.DataFrame` : colonnes `member, source, label, subset ('pool'|'ext')` ; pool et ext **disjoints** ; `ext_counts[c] = None` = tout le reste ; `ValueError` si pas assez d'images.
  - `extract_files(sample, zip_dir=config.ZIP_DIR, raw_dir=config.RAW_DIR) -> pd.DataFrame` : extrait seulement les fichiers tirés vers `raw_dir/<source>/<nom>`, ajoute la colonne `path`, idempotent.

- [ ] **Étape 1 : Écrire les tests (ils échouent car `data.py` n'existe pas)**

**Créer `tests/test_data.py`** — contenu exact :

```python
import os

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from tounum_cls import config, data


def test_list_zip_images_filters_parasites(make_zip):
    z = make_zip("Schematics", 5, extra_png_gray=2)
    names = data.list_zip_images(z)
    assert len(names) == 7
    assert all(n.lower().endswith((".jpg", ".png")) for n in names)
    assert not any("desktop.ini" in n or n.endswith("/") or n.endswith(".txt") for n in names)
    assert names == sorted(names)


def _inventory(n_each=20):
    return {c: [f"{c}/{c.lower()}_{i:03d}.jpg" for i in range(n_each)] for c in config.CLASSES}


def test_sample_dataset_disjoint_counts_and_labels():
    pool = {"Photo": 6, "Painting": 3, "Schematics": 2, "Text": 2, "Sketch": 2}
    ext = {"Photo": 4, "Painting": 4, "Schematics": 4, "Text": 4, "Sketch": None}
    df = data.sample_dataset(_inventory(), pool, ext, seed=1)
    assert df["member"].is_unique
    for c in config.CLASSES:
        sub = df[df["source"] == c]
        assert (sub["subset"] == "pool").sum() == pool[c]
        want_ext = 20 - pool[c] if ext[c] is None else ext[c]
        assert (sub["subset"] == "ext").sum() == want_ext
    assert set(df.loc[df["source"] == "Photo", "label"]) == {1}
    assert set(df.loc[df["source"] != "Photo", "label"]) == {0}
    # reproductible
    pd.testing.assert_frame_equal(df, data.sample_dataset(_inventory(), pool, ext, seed=1))


def test_sample_dataset_not_enough_images_raises():
    pool = {c: 15 for c in config.CLASSES}
    ext = {c: 15 for c in config.CLASSES}
    with pytest.raises(ValueError, match="disponibles"):
        data.sample_dataset(_inventory(20), pool, ext)


def test_extract_files_writes_only_selected_and_is_idempotent(make_zip, tmp_path):
    make_zip("Photo", 6)
    make_zip("Text", 6)
    for c in ("Painting", "Schematics", "Sketch"):
        make_zip(c, 6)
    inv = data.build_inventory(tmp_path)
    pool = {c: 2 for c in config.CLASSES}
    ext = {c: 1 for c in config.CLASSES}
    sample = data.sample_dataset(inv, pool, ext, seed=0)
    raw = tmp_path / "raw"
    out = data.extract_files(sample, tmp_path, raw)
    assert len(out) == 15
    assert all((raw / s).exists() for s in config.CLASSES)
    assert all(pd.Series(out["path"]).map(lambda p: os.path.exists(p)))
    assert len(list((raw / "Photo").glob("*"))) == 3
    data.extract_files(sample, tmp_path, raw)  # 2e appel sans erreur
```

- [ ] **Étape 2 : Lancer les tests et constater l'échec**

```bash
uv run pytest -q tests/test_data.py
```

Attendu : erreur de collecte `ImportError: cannot import name 'data' from 'tounum_cls'`.

- [ ] **Étape 3 : Écrire l'implémentation (début de `data.py`)**

Ce bloc contient **tous les imports du module** (y compris ceux qui ne servent qu'aux tâches 3 et 4) : c'est voulu.

**Créer `src/tounum_cls/data.py`** — contenu exact :

```python
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
```

- [ ] **Étape 4 : Relancer les tests**

```bash
uv run pytest -q tests/test_data.py
```

Attendu : `4 passed` (des `DeprecationWarning` d'imageio ne sont pas des erreurs).

- [ ] **Étape 5 : Commit**

```bash
git add src/tounum_cls/data.py tests/test_data.py
git commit -m "feat(data): zip inventory, reproducible disjoint sampling, selective extraction"
```

---

### Task 3 : Données — chargement des images, découpage, sélection, poids de classe, `tf.data`

**Fichiers :**
- Modifier : `src/tounum_cls/data.py` (ajout en fin de fichier)
- Modifier : `tests/test_data.py` (ajout en fin de fichier)

**Interfaces :**
- Consomme : `sample_dataset` / `extract_files` (tâche 2) pour les formes de colonnes (`member, source, label, subset, path`).
- Produit :
  - `load_image(path, size=config.IMG_SIZE) -> np.ndarray` : `(size, size, 3)` `uint8` RGB (gris, RGBA, 16 bits gérés ; étirement sans conserver le ratio).
  - `load_images(meta, size=config.IMG_SIZE, workers=None) -> (X, meta_ok)` : charge `meta["path"]` en parallèle, écarte les images illisibles (avec un `print` contenant « illisible »), `X` aligné ligne à ligne avec `meta_ok` (index `0..n-1`).
  - `add_split(meta, seed=config.SEED) -> pd.DataFrame` : ajoute `split` ∈ `train|val|test` (pool, 70/15/15 stratifié par `source`) ou `ext`.
  - `select(X, meta, split, sources=None) -> (X_sub, y_sub float32, meta_sub)` : les photos sont toujours gardées ; `sources` liste les classes **négatives** voulues.
  - `class_weights(y) -> {0: w0, 1: w1}` (poids « balanced » `n / (2 * n_classe)`).
  - `make_dataset(X, y, batch_size=config.BATCH_SIZE, shuffle=False, seed=config.SEED) -> tf.data.Dataset`.

- [ ] **Étape 1 : Ajouter les tests**

**Ajouter à la FIN de `tests/test_data.py`** (laisser deux lignes vides avant ce bloc ; ne rien modifier au-dessus) — contenu exact :

```python
@pytest.mark.parametrize("mode,fmt", [("RGB", "JPEG"), ("L", "PNG"), ("RGBA", "PNG")])
def test_load_image_always_rgb_uint8(tmp_path, mode, fmt):
    arr = np.random.default_rng(0).integers(0, 255, (50, 70) + (() if mode == "L" else (4 if mode == "RGBA" else 3,)), dtype=np.uint8)
    p = tmp_path / f"x.{fmt.lower()}"
    Image.fromarray(arr, mode).save(p)
    out = data.load_image(p, size=32)
    assert out.shape == (32, 32, 3) and out.dtype == np.uint8


def test_load_images_skips_corrupt(tmp_path, capsys):
    good = tmp_path / "ok.jpg"
    Image.fromarray(np.zeros((10, 10, 3), np.uint8)).save(good)
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    meta = pd.DataFrame({"path": [str(good), str(bad)], "label": [1, 0], "source": ["Photo", "Text"]})
    X, meta_ok = data.load_images(meta, size=16, workers=2)
    assert X.shape == (1, 16, 16, 3) and len(meta_ok) == 1
    assert "illisible" in capsys.readouterr().out


def _meta(n_per=40):
    rows = []
    for c in config.CLASSES:
        for i in range(n_per):
            rows.append({"member": f"{c}/{i}", "source": c, "label": int(c == "Photo"), "subset": "pool"})
    for i in range(10):
        rows.append({"member": f"ext/{i}", "source": "Text", "label": 0, "subset": "ext"})
    return pd.DataFrame(rows)


def test_add_split_proportions_stratified_and_disjoint():
    m = data.add_split(_meta())
    pool = m[m["subset"] == "pool"]
    assert (m.loc[m["subset"] == "ext", "split"] == "ext").all()
    counts = pool["split"].value_counts()
    assert counts["train"] == 140 and counts["val"] == 30 and counts["test"] == 30
    for c in config.CLASSES:  # stratifié par classe source
        assert (pool[pool["source"] == c]["split"] == "train").sum() == 28
    sets = [set(pool.loc[pool["split"] == s, "member"]) for s in ("train", "val", "test")]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])


def test_select_filters_split_and_sources():
    m = data.add_split(_meta())
    X = np.arange(len(m))[:, None, None, None] * np.ones((1, 4, 4, 3), np.uint8)
    Xs, ys, ms = data.select(X, m, "train", sources=["Text"])
    assert set(ms["source"]) == {"Photo", "Text"}
    assert len(Xs) == len(ys) == len(ms)
    assert (ms["label"].to_numpy() == ys).all()
    # X est bien aligné avec meta : la valeur du pixel = index de la ligne d'origine
    orig = m[(m["split"] == "train") & m["source"].isin(["Photo", "Text"])].index.to_numpy()
    assert (Xs[:, 0, 0, 0] == orig.astype(np.uint8)).all()
    Xa, _, ma = data.select(X, m, "ext")
    assert set(ma["split"]) == {"ext"}


def test_class_weights_balanced():
    w = data.class_weights(np.array([0] * 90 + [1] * 10))
    assert w[1] == pytest.approx(5.0) and w[0] == pytest.approx(100 / 180)


def test_make_dataset_batches():
    X = np.zeros((10, 8, 8, 3), np.uint8)
    y = np.arange(10) % 2
    ds = data.make_dataset(X, y, batch_size=4, shuffle=True)
    bx, by = next(iter(ds))
    assert tuple(bx.shape) == (4, 8, 8, 3) and by.dtype.name == "float32"
```

- [ ] **Étape 2 : Lancer les tests et constater l'échec**

```bash
uv run pytest -q tests/test_data.py
```

Attendu : `AttributeError: module 'tounum_cls.data' has no attribute 'load_image'` (ou similaire) sur les nouveaux tests ; les 4 anciens passent.

- [ ] **Étape 3 : Ajouter l'implémentation**

**Ajouter à la FIN de `src/tounum_cls/data.py`** (laisser deux lignes vides avant ce bloc ; ne rien modifier au-dessus) — contenu exact :

```python
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
```

- [ ] **Étape 4 : Relancer les tests**

```bash
uv run pytest -q tests/test_data.py
```

Attendu : `12 passed`.

- [ ] **Étape 5 : Commit**

```bash
git add src/tounum_cls/data.py tests/test_data.py
git commit -m "feat(data): image loading, stratified split, class weights and tf.data pipeline"
```

---

### Task 4 : Données — orchestration `prepare_dataset` + validation sur les vraies archives

**Fichiers :**
- Modifier : `src/tounum_cls/data.py` (ajout en fin de fichier)
- Modifier : `tests/test_data.py` (ajout en fin de fichier)

**Interfaces :**
- Consomme : toutes les fonctions des tâches 2 et 3 ; `config.ZIP_DIR / RAW_DIR / CACHE_DIR / PROJECT_ROOT / POOL_COUNTS / EXT_COUNTS` (lus **à l'appel**, ce qui permet de les « monkeypatcher » dans les tests).
- Produit : `prepare_dataset(force: bool = False) -> (X, meta)` : `X` `uint8 (N,128,128,3)`, `meta` avec colonnes `member, source, label, subset, path, split`. Écrit `data/cache/X.npy` et `data/cache/meta.csv` (chemins **relatifs** à la racine du projet dans le CSV, **absolus** dans le DataFrame renvoyé). 2ᵉ appel = lecture du cache.

- [ ] **Étape 1 : Ajouter le test de bout en bout (sur archives factices)**

**Ajouter à la FIN de `tests/test_data.py`** (laisser deux lignes vides avant ce bloc ; ne rien modifier au-dessus) — contenu exact :

```python
def test_prepare_dataset_end_to_end_and_cache(make_zip, tmp_path, monkeypatch):
    for c in config.CLASSES:
        make_zip(c, 30, extra_png_gray=2 if c == "Sketch" else 0)
    monkeypatch.setattr(config, "ZIP_DIR", tmp_path)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "data" / "raw")
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "data" / "cache")
    monkeypatch.setattr(config, "POOL_COUNTS", {c: 20 for c in config.CLASSES})
    monkeypatch.setattr(config, "EXT_COUNTS", {**{c: 3 for c in config.CLASSES}, "Sketch": None})
    X, meta = data.prepare_dataset()
    assert X.shape[0] == len(meta) == 5 * 20 + 4 * 3 + (32 - 20) and X.dtype == np.uint8
    assert X.shape[1:] == (128, 128, 3)
    assert set(meta["split"]) == {"train", "val", "test", "ext"}
    # 2e appel : lecture du cache, chemins ré-absolutisés, mêmes données
    X2, meta2 = data.prepare_dataset()
    assert (X2 == X).all() and meta2["path"].map(lambda p: os.path.exists(p)).all()
    assert not (tmp_path / "data" / "cache" / "meta.csv").read_text().count(str(tmp_path))
```

- [ ] **Étape 2 : Lancer le test et constater l'échec**

```bash
uv run pytest -q tests/test_data.py::test_prepare_dataset_end_to_end_and_cache
```

Attendu : `AttributeError: ... has no attribute 'prepare_dataset'`.

- [ ] **Étape 3 : Ajouter l'implémentation**

**Ajouter à la FIN de `src/tounum_cls/data.py`** (laisser deux lignes vides avant ce bloc ; ne rien modifier au-dessus) — contenu exact :

```python
# --------------------------------------------------------------------------- orchestration
def prepare_dataset(force: bool = False) -> tuple[np.ndarray, pd.DataFrame]:
    """Pipeline complet : inventaire -> tirage -> extraction -> chargement -> split -> cache.

    Au 2e appel, relit simplement data/cache/X.npy et data/cache/meta.csv.
    Renvoie (X, meta) ; meta a les colonnes member, source, label, subset, path, split.
    `path` est stocké relatif à la racine du projet dans meta.csv (le projet reste déplaçable)
    et renvoyé en absolu.
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
```

- [ ] **Étape 4 : Relancer toute la suite**

```bash
uv run pytest -q
```

Attendu : `16 passed` (3 de config + 13 de data).

- [ ] **Étape 5 : Valider sur les VRAIES archives (construit aussi le cache réutilisé par le notebook)**

```bash
cd /home/diskmth/Dev/bloc_AI
uv run python - <<'EOF'
import sys, time
sys.path.insert(0, "src")
from tounum_cls import data
t = time.time()
X, meta = data.prepare_dataset()
print("prepare_dataset :", round(time.time() - t), "s | X :", X.shape, X.dtype, round(X.nbytes / 1e6), "Mo")
print(meta.groupby(["subset", "source"]).size().unstack(0))
print(meta["split"].value_counts().to_dict())
EOF
```

Cette étape extrait ≈ 10 800 fichiers vers `data/raw/` (≈ 2 Go) puis les redimensionne : **compter quelques minutes** la première fois (≈ 30 s si les fichiers sont déjà extraits).
Résultat attendu (déterministe, `SEED = 42` ; `<n>` = durée, à l'alignement des colonnes près) :

```text
prepare_dataset : <n> s | X : (10806, 128, 128, 3) uint8 531 Mo
subset       ext  pool
source
Painting    1000  1200
Photo       1000  3000
Schematics  1000   600
Sketch       806   600
Text        1000   600
{'ext': 4806, 'train': 4200, 'test': 900, 'val': 900}
```

Aucune ligne `[WARN] image illisible` ne doit apparaître. Si les nombres diffèrent : s'arrêter et diagnostiquer (inventaire, `POOL_COUNTS`, `EXT_COUNTS`) ; ne rien ajuster pour « faire coïncider ».
Un second lancement de la même commande doit prendre ≈ 1 s (lecture du cache).

- [ ] **Étape 6 : Commit**

```bash
git add src/tounum_cls/data.py tests/test_data.py
git commit -m "feat(data): prepare_dataset orchestration with relative-path cache"
```
(`data/` est ignoré par git : le cache n'est pas commité.)

---

### Task 5 : Modèles — CNN maison, MobileNetV2 gelé, entraînement

**Fichiers :**
- Créer : `src/tounum_cls/modeling.py`
- Créer : `tests/test_modeling.py`

**Interfaces :**
- Consomme : `config` (tâche 1) ; `data.make_dataset` (tâche 3) dans les tests.
- Produit :
  - `_compile(model, learning_rate=1e-3) -> keras.Model` : Adam, `binary_crossentropy`, métriques `accuracy` et `auc` (le notebook en affiche le code : section 5).
  - `build_cnn(filters=(16,32,64,128), dense_units=64, dropout=0.0, l2=0.0, augmentation=False, img_size=128, learning_rate=1e-3, bn_momentum=0.9, name="cnn_photo") -> keras.Model` (≈ 106 481 paramètres avec les valeurs par défaut). Couches nommées `rescaling`, `conv1..convN`, `bn1..`, `relu1..`, `pool1..`, `gap`, `dense`, `dropout` (si `dropout>0`), `output`, et `aug_flip`, `aug_rotation`, `aug_zoom`, `aug_contrast` (si `augmentation=True`).
  - `build_mobilenet(img_size=128, weights="imagenet", name="mobilenetv2_gele") -> keras.Model` (base gelée ; `weights=None` pour les tests).
  - `train(model, train_ds, val_ds, epochs=15, class_weight=None, early_stopping=False, patience=4, verbose=2) -> pd.DataFrame` : colonnes `epoch, loss, accuracy, auc, val_loss, val_accuracy, val_auc`.
  - `predict_proba(model, X, batch_size=config.BATCH_SIZE) -> np.ndarray` (1D, P(photo)).

- [ ] **Étape 1 : Écrire les tests**

**Créer `tests/test_modeling.py`** — contenu exact :

```python
import keras
import numpy as np

from tounum_cls import data, modeling


def _toy(n=32, size=32, seed=0):
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n // 2), dtype="float32")
    X = rng.integers(0, 60, (n, size, size, 3), dtype=np.uint8)
    X[y == 1] += 150  # photos = images claires -> facilement séparables
    return X, y


def test_build_cnn_shapes_and_params():
    m = modeling.build_cnn(img_size=32)
    assert m.output_shape == (None, 1)
    assert [l.name for l in m.layers if l.name.startswith("conv")] == ["conv1", "conv2", "conv3", "conv4"]
    assert 90_000 < modeling.build_cnn().count_params() < 130_000   # ~106 k avec la config par défaut


def test_build_cnn_options_add_layers_and_l2():
    m = modeling.build_cnn(img_size=32, dropout=0.3, l2=1e-3, augmentation=True)
    names = [l.name for l in m.layers]
    assert "dropout" in names and "aug_flip" in names
    assert len(m.losses) > 0          # pénalités L2 enregistrées
    plain = modeling.build_cnn(img_size=32)
    assert "dropout" not in [l.name for l in plain.layers] and len(plain.losses) == 0


def test_cnn_learns_toy_problem():
    keras.utils.set_random_seed(0)
    X, y = _toy()
    ds = data.make_dataset(X, y, batch_size=8, shuffle=True)
    m = modeling.build_cnn(filters=(4, 8), dense_units=8, img_size=32)
    hist = modeling.train(m, ds, ds, epochs=40, verbose=0)  # 40 : les stats BatchNorm convergent lentement sur 4 pas/epoch
    assert list(hist.columns[:1]) == ["epoch"] and {"loss", "val_loss", "accuracy", "val_accuracy"} <= set(hist.columns)
    assert hist["val_accuracy"].iloc[-1] > 0.9
    p = modeling.predict_proba(m, X)
    assert p.shape == (32,) and ((p >= 0) & (p <= 1)).all()


def test_early_stopping_stops_before_max_epochs():
    keras.utils.set_random_seed(0)
    X, y = _toy()
    ds = data.make_dataset(X, y, batch_size=8)
    m = modeling.build_cnn(filters=(4,), dense_units=4, img_size=32, learning_rate=1e-9)  # n'apprend rien
    hist = modeling.train(m, ds, ds, epochs=30, early_stopping=True, patience=2, verbose=0)
    assert len(hist) < 30


def test_build_mobilenet_frozen():
    m = modeling.build_mobilenet(img_size=32, weights=None)
    assert m.output_shape == (None, 1)
    assert m.count_params() > 2_000_000
    assert 1_000 < sum(int(np.prod(w.shape)) for w in m.trainable_weights) < 3_000   # seule la tête apprend
```

- [ ] **Étape 2 : Lancer les tests et constater l'échec**

```bash
uv run pytest -q tests/test_modeling.py
```

Attendu : `ImportError: cannot import name 'modeling' from 'tounum_cls'`.

- [ ] **Étape 3 : Écrire l'implémentation**

**Créer `src/tounum_cls/modeling.py`** — contenu exact :

```python
"""Construction et entraînement des réseaux (Keras 3 / TensorFlow)."""
from __future__ import annotations

import keras
import pandas as pd
from keras import layers, regularizers

from . import config


def _compile(model: keras.Model, learning_rate: float = 1e-3) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


def build_cnn(
    filters: tuple[int, ...] = (16, 32, 64, 128),
    dense_units: int = 64,
    dropout: float = 0.0,
    l2: float = 0.0,
    augmentation: bool = False,
    img_size: int = config.IMG_SIZE,
    learning_rate: float = 1e-3,
    bn_momentum: float = 0.9,
    name: str = "cnn_photo",
) -> keras.Model:
    """CNN « maison » pour photo (1) vs non-photo (0).

    Entrée : images uint8 (img_size, img_size, 3) ; la normalisation /255 est DANS le modèle.
    Un bloc = Conv2D 3x3 (padding same) -> BatchNormalization -> ReLU -> MaxPooling2D 2x2.
    Tête : GlobalAveragePooling2D -> Dense(dense_units, relu) -> [Dropout] -> Dense(1, sigmoid).
    `augmentation=True` ajoute des transformations aléatoires, actives seulement à l'entraînement.
    `l2` pénalise les poids des Conv2D et de la Dense cachée.
    `bn_momentum=0.9` (au lieu de 0.99 par défaut) : les statistiques mobiles de BatchNorm se calibrent en
    quelques dizaines de pas, ce qui évite une validation erratique pendant les premières epochs.
    """
    reg = regularizers.l2(l2) if l2 > 0 else None
    inputs = keras.Input(shape=(img_size, img_size, 3), name="image")
    x = inputs
    if augmentation:
        x = layers.RandomFlip("horizontal", name="aug_flip")(x)
        x = layers.RandomRotation(0.05, name="aug_rotation")(x)
        x = layers.RandomZoom(0.1, name="aug_zoom")(x)
        x = layers.RandomContrast(0.1, name="aug_contrast")(x)
    x = layers.Rescaling(1.0 / 255, name="rescaling")(x)
    for i, n_filters in enumerate(filters, start=1):
        x = layers.Conv2D(n_filters, 3, padding="same", use_bias=False, kernel_regularizer=reg, name=f"conv{i}")(x)
        x = layers.BatchNormalization(momentum=bn_momentum, name=f"bn{i}")(x)
        x = layers.Activation("relu", name=f"relu{i}")(x)
        x = layers.MaxPooling2D(2, name=f"pool{i}")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(dense_units, activation="relu", kernel_regularizer=reg, name="dense")(x)
    if dropout > 0:
        x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
    return _compile(keras.Model(inputs, outputs, name=name), learning_rate)


def build_mobilenet(
    img_size: int = config.IMG_SIZE, weights: str | None = "imagenet", name: str = "mobilenetv2_gele"
) -> keras.Model:
    """MobileNetV2 gelé (extracteur de caractéristiques) + tête binaire. Sert de comparaison au CNN maison.

    Entrée : images uint8 ; mise à l'échelle [-1, 1] attendue par MobileNetV2 faite dans le modèle.
    `weights=None` (tests) évite le téléchargement des poids ImageNet.
    """
    base = keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3), include_top=False, weights=weights
    )
    base.trainable = False
    inputs = keras.Input(shape=(img_size, img_size, 3), name="image")
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="rescaling")(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.2, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
    return _compile(keras.Model(inputs, outputs, name=name))


def train(
    model: keras.Model,
    train_ds,
    val_ds,
    epochs: int = 15,
    class_weight: dict[int, float] | None = None,
    early_stopping: bool = False,
    patience: int = 4,
    verbose: int = 2,
) -> pd.DataFrame:
    """Entraîne le modèle et renvoie l'historique sous forme de DataFrame (une ligne par epoch).

    Colonnes : epoch, loss, accuracy, auc, val_loss, val_accuracy, val_auc.
    Avec early_stopping=True : arrêt après `patience` epochs sans amélioration de val_loss,
    et restauration des meilleurs poids.
    """
    callbacks = []
    if early_stopping:
        callbacks.append(
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
        )
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=callbacks,
        shuffle=False,  # le mélange est fait par tf.data ; évite un avertissement Keras
        verbose=verbose,
    )
    df = pd.DataFrame(hist.history)
    df.insert(0, "epoch", range(1, len(df) + 1))
    return df


def predict_proba(model: keras.Model, X, batch_size: int = config.BATCH_SIZE):
    """Probabilités P(photo) sous forme de vecteur 1D float."""
    return model.predict(X, batch_size=batch_size, verbose=0).ravel()
```

- [ ] **Étape 4 : Relancer les tests**

```bash
uv run pytest -q tests/test_modeling.py
```

Attendu : `5 passed` (≈ 10 s). Ces tests entraînent de minuscules réseaux sur des images synthétiques ; ils sont seedés et stables.
Le test `test_build_mobilenet_frozen` utilise `weights=None` : **aucun téléchargement**.

- [ ] **Étape 5 : Commit**

```bash
git add src/tounum_cls/modeling.py tests/test_modeling.py
git commit -m "feat(modeling): custom CNN, frozen MobileNetV2 baseline and training helpers"
```

---

### Task 6 : Évaluation — métriques et diagnostic biais/variance

**Fichiers :**
- Créer : `src/tounum_cls/evaluation.py` (début du module)
- Créer : `tests/test_evaluation.py` (début du fichier)

**Interfaces :**
- Consomme : `modeling.build_cnn` (tâche 5, uniquement dans un test de la tâche 7).
- Produit :
  - `evaluate_binary(y_true, y_prob, threshold=0.5) -> dict` : clés `accuracy, balanced_accuracy, precision, recall, f1, roc_auc, tn, fp, fn, tp` (`roc_auc = nan` si une seule classe).
  - `per_source_accuracy(meta, y_prob, threshold=0.5) -> DataFrame` : colonnes `source, n, accuracy`, trié du pire au meilleur ; `meta` doit avoir `source` et `label`, aligné avec `y_prob`.
  - `diagnose_fit(hist, low_acc=0.90, gap_max=0.05, drift_max=0.05) -> dict` : clés `train_acc, val_acc, gap_acc, train_loss, val_loss, best_epoch, val_loss_drift, verdict` ; `verdict` ∈ `"biais élevé (sous-apprentissage)"`, `"variance élevée (sur-apprentissage)"`, `"compromis biais/variance correct"`.

- [ ] **Étape 1 : Écrire les tests**

**Créer `tests/test_evaluation.py`** — contenu exact :

```python
import numpy as np
import pandas as pd
import pytest

from tounum_cls import evaluation, modeling


def test_evaluate_binary_values():
    y = np.array([1, 1, 0, 0, 1, 0])
    p = np.array([0.9, 0.4, 0.2, 0.7, 0.8, 0.1])
    r = evaluation.evaluate_binary(y, p)
    assert (r["tp"], r["fn"], r["fp"], r["tn"]) == (2, 1, 1, 2)
    assert r["accuracy"] == pytest.approx(4 / 6)
    assert r["precision"] == pytest.approx(2 / 3) and r["recall"] == pytest.approx(2 / 3)
    assert 0 <= r["roc_auc"] <= 1
    assert r["balanced_accuracy"] == pytest.approx((2 / 3 + 2 / 3) / 2)


def test_evaluate_binary_single_class_auc_nan():
    r = evaluation.evaluate_binary([0, 0], [0.1, 0.2])
    assert np.isnan(r["roc_auc"]) and r["accuracy"] == 1.0


def test_per_source_accuracy():
    meta = pd.DataFrame({"source": ["Photo", "Photo", "Text", "Text"], "label": [1, 1, 0, 0]})
    out = evaluation.per_source_accuracy(meta, [0.9, 0.2, 0.1, 0.1])
    acc = dict(zip(out["source"], out["accuracy"]))
    assert acc == {"Photo": 0.5, "Text": 1.0}
    assert out.iloc[0]["source"] == "Photo"  # trié du pire au meilleur


def _hist(train_acc, val_acc, val_loss):
    n = len(train_acc)
    return pd.DataFrame({"epoch": range(1, n + 1), "accuracy": train_acc, "val_accuracy": val_acc,
                         "loss": [0.1] * n, "val_loss": val_loss})


def test_diagnose_fit_three_regimes():
    under = evaluation.diagnose_fit(_hist([0.6, 0.7], [0.6, 0.7], [0.7, 0.6]))
    over = evaluation.diagnose_fit(_hist([0.9, 0.99], [0.9, 0.85], [0.3, 0.5]))
    good = evaluation.diagnose_fit(_hist([0.9, 0.97], [0.9, 0.96], [0.3, 0.2]))
    assert "biais" in under["verdict"]
    assert "variance" in over["verdict"] and over["best_epoch"] == 1 and over["gap_acc"] == pytest.approx(0.14)
    assert "correct" in good["verdict"] and good["best_epoch"] == 2
```

- [ ] **Étape 2 : Lancer les tests et constater l'échec**

```bash
uv run pytest -q tests/test_evaluation.py
```

Attendu : `ImportError: cannot import name 'evaluation' from 'tounum_cls'`.

- [ ] **Étape 3 : Écrire l'implémentation**

Le bloc contient **tous les imports du module** (matplotlib, sklearn…), y compris ceux qui ne servent qu'à la tâche 7 : c'est voulu.

**Créer `src/tounum_cls/evaluation.py`** — contenu exact :

```python
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
```

- [ ] **Étape 4 : Relancer les tests**

```bash
uv run pytest -q tests/test_evaluation.py
```

Attendu : `4 passed`.

- [ ] **Étape 5 : Commit**

```bash
git add src/tounum_cls/evaluation.py tests/test_evaluation.py
git commit -m "feat(evaluation): binary metrics, per-source accuracy and bias/variance diagnosis"
```

---

### Task 7 : Évaluation — graphiques (courbes, confusion, précision/rappel, erreurs) et schéma d'architecture

**Fichiers :**
- Modifier : `src/tounum_cls/evaluation.py` (ajout en fin de fichier)
- Modifier : `tests/test_evaluation.py` (ajout en fin de fichier)

**Interfaces :**
- Consomme : `modeling.build_cnn` (tâche 5).
- Produit (tous renvoient une `matplotlib.figure.Figure`, sauf `show_errors` qui renvoie `None` s'il n'y a aucune erreur de ce type) :
  - `plot_history(hist, title="")` : 2 graphiques (perte, accuracy), entraînement vs validation, d'après le DataFrame de `modeling.train`.
  - `plot_confusion(y_true, y_prob, threshold=0.5, title="")`.
  - `plot_precision_recall(y_true, y_prob, title="")`.
  - `show_errors(X, meta, y_prob, kind="fp"|"fn", n=8, threshold=0.5)` : galerie des erreurs les plus « confiantes ».
  - `draw_architecture(model, title=None)` : schéma vertical du réseau dessiné en matplotlib (aucun graphviz requis).

- [ ] **Étape 1 : Ajouter les tests**

**Ajouter à la FIN de `tests/test_evaluation.py`** (laisser deux lignes vides avant ce bloc ; ne rien modifier au-dessus) — contenu exact :

```python
def test_plots_return_figures():
    hist = pd.DataFrame({"epoch": [1, 2], "loss": [1, .5], "val_loss": [1, .7], "accuracy": [.5, .8], "val_accuracy": [.5, .7]})
    y = np.array([0, 1, 0, 1]); p = np.array([.1, .9, .6, .4])
    X = np.zeros((4, 8, 8, 3), np.uint8)
    meta = pd.DataFrame({"source": ["Text", "Photo", "Painting", "Photo"], "label": y})
    assert evaluation.plot_history(hist, "t") is not None
    assert evaluation.plot_confusion(y, p) is not None
    assert evaluation.plot_precision_recall(y, p) is not None
    assert evaluation.show_errors(X, meta, p, kind="fp") is not None
    assert evaluation.show_errors(X, meta, p, kind="fn") is not None
    assert evaluation.show_errors(X, meta, np.array([0, 1, 0, 1.0]), kind="fp") is None


def test_draw_architecture():
    m = modeling.build_cnn(img_size=32, dropout=0.2)
    assert evaluation.draw_architecture(m) is not None
```

- [ ] **Étape 2 : Lancer les tests et constater l'échec**

```bash
uv run pytest -q tests/test_evaluation.py
```

Attendu : `AttributeError: module 'tounum_cls.evaluation' has no attribute 'plot_history'` sur les 2 nouveaux tests.

- [ ] **Étape 3 : Ajouter l'implémentation**

**Ajouter à la FIN de `src/tounum_cls/evaluation.py`** (laisser deux lignes vides avant ce bloc ; ne rien modifier au-dessus) — contenu exact :

```python
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
```

- [ ] **Étape 4 : Relancer toute la suite**

```bash
uv run pytest -q
```

Attendu : `27 passed`.

- [ ] **Étape 5 : Commit**

```bash
git add src/tounum_cls/evaluation.py tests/test_evaluation.py
git commit -m "feat(evaluation): training curves, confusion matrix, PR curve, error gallery and architecture diagram"
```

---

### Task 8 : Notebook, partie A — contexte métier, environnement, données, découpage

**Fichiers :**
- Créer : `scripts/run_notebook.sh`
- Créer : `notebooks/livrable1_classification_binaire.py` (source jupytext ; le `.ipynb` est généré)
- Créer : `scripts/check_notebook.py`

**Interfaces :**
- Consomme : `config`, `data` (tâches 1 à 4) — le cache `data/cache` créé à la tâche 4 est réutilisé.
- Produit (variables de notebook utilisées par les parties suivantes) : `ROOT`, `QUICK`, `EPOCHS` (dict `stage, main, curve, improve, mobilenet`), `X`, `meta`, `get_split(negative_sources=None)`, `run(split, epochs, early_stopping=False, patience=4, **model_kwargs) -> (model, hist, seconds)`.

- [ ] **Étape 1 : Créer les deux scripts utilitaires**

**Créer `scripts/run_notebook.sh`** — contenu exact :

```bash
#!/usr/bin/env bash
# Régénère le .ipynb depuis la source jupytext (.py) puis l'exécute (sorties incluses). Environnement géré par uv.
# Usage : bash scripts/run_notebook.sh quick   # 2 epochs partout, pour vérifier que tout s'exécute
#         bash scripts/run_notebook.sh full    # exécution complète (livrable)
set -euo pipefail
cd "$(dirname "$0")/../notebooks"
NB=livrable1_classification_binaire
uv run jupytext --to ipynb "$NB.py"
if [ "${1:-full}" = "quick" ]; then export TOUNUM_QUICK=1; fi
uv run jupyter nbconvert --to notebook --execute --inplace "$NB.ipynb" --ExecutePreprocessor.timeout=7200
```

**Créer `scripts/check_notebook.py`** — contenu exact :

```python
"""Vérifie qu'un notebook exécuté est complet : exécuté sans erreur, sections et figures présentes.

Usage : python scripts/check_notebook.py notebooks/livrable1_classification_binaire.ipynb [--partial]
--partial : notebook en cours de construction -> vérifie seulement l'exécution sans erreur et affiche les comptes.
Code de sortie 0 si tout est conforme, 1 sinon.
"""
import sys

import nbformat

REQUIRED_HEADINGS = [f"## {i}." for i in range(0, 12)]
MIN_FIGURES = 13
MIN_AUTO_READINGS = 5          # cellules « Lecture automatique » (chiffres réels, pas de texte figé)


def check(path: str, partial: bool = False) -> list[str]:
    nb = nbformat.read(path, as_version=4)
    code = [c for c in nb.cells if c.cell_type == "code"]
    markdown = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    outputs = [o for c in code for o in c.get("outputs", [])]
    text = "\n".join(
        "".join(o.get("text", "")) + str(o.get("data", {}).get("text/markdown", "")) + str(o.get("data", {}).get("text/plain", ""))
        for o in outputs
    )
    problems = []
    n_fig = sum(1 for o in outputs if "image/png" in o.get("data", {}))
    if partial:
        if any(c.execution_count is None for c in code):
            problems.append("des cellules de code n'ont pas été exécutées")
        if any(o.output_type == "error" for o in outputs):
            problems.append("une cellule contient une erreur")
        print(f"{path}: {len(code)} cellules de code, {n_fig} figures")
        return problems
    for heading in REQUIRED_HEADINGS:
        if heading not in markdown:
            problems.append(f"section manquante : « {heading} »")
    if any(c.execution_count is None for c in code):
        problems.append("des cellules de code n'ont pas été exécutées")
    if any(o.output_type == "error" for o in outputs):
        problems.append("une cellule contient une erreur")
    if n_fig < MIN_FIGURES:
        problems.append(f"seulement {n_fig} figures (minimum {MIN_FIGURES})")
    if "Total params" not in text:
        problems.append("model.summary() absent (E1)")
    if "binary_crossentropy" not in text:
        problems.append("code de build_cnn absent des sorties (E1)")
    if text.count("Lecture automatique") < MIN_AUTO_READINGS:
        problems.append(f"moins de {MIN_AUTO_READINGS} cellules « Lecture automatique »")
    if "disjoints" not in text:
        problems.append("contrôle anti-fuite absent")
    print(f"{path}: {len(code)} cellules de code, {n_fig} figures, {text.count('Lecture automatique')} lectures automatiques")
    return problems


if __name__ == "__main__":
    issues = check(sys.argv[1], partial="--partial" in sys.argv)
    for issue in issues:
        print("  ✗", issue)
    print("OK" if not issues else f"{len(issues)} problème(s)")
    sys.exit(1 if issues else 0)
```

```bash
chmod +x scripts/run_notebook.sh
```

- [ ] **Étape 2 : Créer la partie A du notebook (sections 0 à 3 + fonctions utilitaires)**

Ce fichier contient les cellules « titre / contexte métier », « environnement », « les données » (inventaire, tailles, aperçu visuel, statistiques), « découpage » et les deux fonctions utilitaires du notebook.

**Créer `notebooks/livrable1_classification_binaire.py`** — contenu exact :

```python
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # TouNum — Livrable 1 : classification binaire *photo / non-photo*
#
# **Équipe Data Science CESI** · Prototype du workflow de catégorisation automatique d'images
#
# ---
# ## 0. Contexte métier et objectif
#
# **TouNum** numérise des documents (textes, images…) pour le compte de ses clients et veut proposer un service de
# **catégorisation et de légendage automatique** des images. Le workflow cible comporte trois modules :
#
# 1. **Tri** : ne garder que les *photographies* (ce livrable) ;
# 2. **Prétraitement** : nettoyer les images floues ou bruitées (livrable 2) ;
# 3. **Captioning** : générer une légende descriptive (livrable 3).
#
# Or, parmi les images déjà numérisées, beaucoup **ne sont pas des photos** (schémas, textes scannés, dessins,
# peintures). Les envoyer au module de légendage serait inutile et produirait des légendes absurdes.
# **Ce notebook construit le filtre d'entrée** : un réseau de neurones qui répond « photo » (1) ou « autre » (0).
#
# ### Ce que ce notebook démontre
# | Section | Contenu |
# |---|---|
# | 1 | Environnement et reproductibilité |
# | 2 | Les données : inventaire, exploration, échantillonnage |
# | 3 | Découpage train / validation / test |
# | 4 | **Progression par difficulté** : texte → schémas → dessins → peintures |
# | 5 | **Architecture du réseau** : schéma, paramètres, perte, optimiseur |
# | 6 | **Entraînement** : courbes d'erreur et d'accuracy (train vs validation) |
# | 7 | **Analyse biais / variance** (sous- et sur-apprentissage) |
# | 8 | **Améliorations** : augmentation, dropout, L2, early stopping |
# | 9 | Évaluation finale (test équilibré et test « réaliste »), erreurs, choix du seuil, mise en production |
# | 10 | Comparaison avec un réseau pré-entraîné (MobileNetV2) |
# | 11 | Conclusion, limites et suites |
#
# ### Comment relancer
# `Kernel → Restart & Run All`. Les images sont extraites des archives et mises en cache dans `data/` à la première
# exécution (quelques minutes), puis relues instantanément. `TOUNUM_QUICK=1` (variable d'environnement) lance une
# version très courte (2 epochs) pour tester que tout s'exécute.

# %% [markdown]
# ## 1. Environnement et reproductibilité

# %%
import os
import sys
import time
import inspect
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")      # moins de bruit TensorFlow
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))                   # le code réutilisable est dans src/tounum_cls

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from IPython.display import display, Markdown
import tensorflow as tf
import keras
import sklearn

from tounum_cls import config, data, modeling, evaluation

QUICK = os.environ.get("TOUNUM_QUICK") == "1"
# Nombre d'epochs par expérience (réduit en mode QUICK)
EPOCHS = {"stage": 8, "main": 20, "curve": 12, "improve": 25, "mobilenet": 8}
if QUICK:
    EPOCHS = {k: 2 for k in EPOCHS}

keras.utils.set_random_seed(config.SEED)                # fixe les graines Python, NumPy et TensorFlow
pd.set_option("display.precision", 3)
pd.set_option("display.width", 140)
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 100

print(f"Python {sys.version.split()[0]} | TensorFlow {tf.__version__} | Keras {keras.__version__} | "
      f"scikit-learn {sklearn.__version__} | pandas {pd.__version__} | numpy {np.__version__}")
print("GPU détecté :", tf.config.list_physical_devices("GPU") or "aucun (calcul sur CPU)")
print(f"Graine : {config.SEED} | mode QUICK : {QUICK} | epochs : {EPOCHS}")

# %% [markdown]
# > **Reproductibilité.** Toutes les sources d'aléa (tirage des images, découpage, initialisation des poids,
# > mélange des batchs) dépendent de la graine `SEED = 42`. Sur CPU, TensorFlow peut tout de même produire de très
# > petites variations d'une exécution à l'autre (ordre des opérations parallèles) : les conclusions ne changent pas,
# > mais les décimales peuvent bouger.

# %% [markdown]
# ## 2. Les données
#
# TouNum fournit cinq archives, **une par type d'image**. Le nom de l'archive est l'étiquette : il n'y a pas de fichier
# d'annotations. Notre cible binaire est : `Photo` → **1**, tout le reste → **0**.
#
# Les archives pèsent plusieurs Go. La fonction `prepare_dataset()` :
# 1. inventorie les images (extensions jpg/png ; on ignore par exemple le fichier parasite `desktop.ini`) ;
# 2. tire un échantillon **reproductible** et **disjoint** : un *pool de modélisation* équilibré et un *test étendu* ;
# 3. n'extrait que les fichiers tirés ;
# 4. les charge avec **ImageIO**, les convertit en **RGB** (les sketchs sont en niveaux de gris), les redimensionne en
#    **128×128** et les stocke en `uint8` dans un cache (`data/cache`).

# %%
t0 = time.perf_counter()
X, meta = data.prepare_dataset()
print(f"X : {X.shape}, dtype={X.dtype}, {X.nbytes / 1e6:.0f} Mo en mémoire | chargé en {time.perf_counter() - t0:.0f} s")

summary = meta.groupby(["source", "subset"]).size().unstack(fill_value=0)
summary["total tiré"] = summary.sum(axis=1)
display(summary.loc[config.CLASSES])

# %% [markdown]
# **Pourquoi un échantillon et pas les ~42 000 images ?** Nous calculons sur CPU. Un échantillon équilibré de
# 6 000 images (3 000 photos / 3 000 non-photos) suffit largement pour une tâche de ce type et permet de multiplier
# les expériences (comparaisons, régularisation) en un temps raisonnable. Les effectifs sont réglables dans
# `src/tounum_cls/config.py` (`POOL_COUNTS`, `EXT_COUNTS`).
#
# - Le **pool** est équilibré 50/50 ; les peintures y sont sur-représentées (1 200) car c'est la classe la plus difficile.
# - Le **test étendu** (`ext`) n'est **jamais** utilisé pour entraîner ni pour choisir un modèle. Il est *déséquilibré*
#   (≈ 21 % de photos) pour estimer les performances sur un fonds documentaire réel.

# %% [markdown]
# ### 2.1 Taille d'origine des images
# Les images n'ont ni la même taille ni le même format. On lit uniquement l'en-tête des fichiers (rapide).

# %%
def original_size(path):
    with Image.open(path) as im:
        return im.size  # (largeur, hauteur)

sizes = meta["path"].map(original_size)
meta["width"] = [s[0] for s in sizes]
meta["height"] = [s[1] for s in sizes]
meta["megapixels"] = meta["width"] * meta["height"] / 1e6
display(meta.groupby("source")[["width", "height", "megapixels"]].median().loc[config.CLASSES])
print("Toutes les images sont ramenées à 128×128 (étirement, ratio non conservé) : les formes très allongées "
      "(pages de texte A4) sont déformées, ce qui est acceptable pour une simple décision photo / non-photo.")

# %% [markdown]
# ### 2.2 Aperçu visuel (images telles que vues par le réseau, 128×128)

# %%
rng = np.random.default_rng(config.SEED)
fig, axes = plt.subplots(len(config.CLASSES), 8, figsize=(16, 2.1 * len(config.CLASSES)))
for r, source in enumerate(config.CLASSES):
    idx = rng.choice(meta.index[(meta["source"] == source) & (meta["subset"] == "pool")], 8, replace=False)
    for c, i in enumerate(idx):
        axes[r, c].imshow(X[i])
        axes[r, c].axis("off")
    axes[r, 0].set_title(source + (" (label 1)" if source == config.POSITIVE_CLASS else " (label 0)"),
                         loc="left", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2.3 Statistiques simples : ce qui distingue déjà les classes
# Deux descripteurs très simples : la **luminosité moyenne** et la **colorimétrie** (écart moyen entre canaux R, G, B :
# 0 pour une image en niveaux de gris). Ils donnent une intuition de la difficulté de chaque séparation.

# %%
pix = X.astype(np.float32)
meta["luminosite"] = pix.mean(axis=(1, 2, 3))
meta["colorimetrie"] = (np.abs(pix[..., 0] - pix[..., 1]) + np.abs(pix[..., 1] - pix[..., 2])).mean(axis=(1, 2))
del pix

fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
sns.boxplot(data=meta, x="source", y="luminosite", order=config.CLASSES, ax=axes[0])
axes[0].set_title("Luminosité moyenne (0 = noir, 255 = blanc)")
sns.boxplot(data=meta, x="source", y="colorimetrie", order=config.CLASSES, ax=axes[1])
axes[1].set_title("Colorimétrie (0 = niveaux de gris)")
for ax in axes:
    ax.set_xlabel("")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Lecture.** Les textes scannés sont très clairs (luminosité ≈ 244) et sans aucune couleur, les sketchs sont clairs
# (≈ 234) et en niveaux de gris, les schémas sont en général clairs mais très variables (certains sont sombres ou
# colorés) : un classifieur très simple (seuils sur ces deux grandeurs) en séparerait déjà une bonne part.
# En revanche, les **peintures** occupent les mêmes plages de luminosité (≈ 118 contre ≈ 112) et de couleur que les
# photos : elles ne se distingueront que par la **texture et la structure** de l'image, que seul un réseau de
# convolution peut exploiter. C'est pourquoi on avance par difficulté croissante (section 4).

# %% [markdown]
# ## 3. Découpage train / validation / test
#
# Le pool est découpé **de façon stratifiée par classe source** (et non seulement par label binaire), pour que chaque
# type d'image soit représenté dans les mêmes proportions dans les trois ensembles :
#
# | Ensemble | Part | Rôle |
# |---|---|---|
# | `train` | 70 % | apprendre les poids du réseau |
# | `val` (validation) | 15 % | suivre l'apprentissage epoch par epoch, régler / choisir le modèle. **C'est « l'erreur de test » des courbes de la section 6** |
# | `test` | 15 % | mesure finale, utilisée **une seule fois** à la fin (sections 4 et 9) |
# | `ext` | — | test étendu déséquilibré, hors pool |
#
# > Si l'on choisissait le modèle en regardant le jeu de test, on lui « fuiterait » de l'information et l'accuracy
# > affichée serait trop optimiste. D'où la séparation validation / test.

# %%
split_table = pd.crosstab(meta["source"], meta["split"]).loc[config.CLASSES, ["train", "val", "test", "ext"]]
split_table.loc["TOTAL"] = split_table.sum()
display(split_table)

# garde-fou anti-fuite : aucun fichier ne doit apparaître dans deux ensembles
members = {s: set(meta.loc[meta["split"] == s, "member"]) for s in ["train", "val", "test", "ext"]}
for a in members:
    for b in members:
        if a < b:
            assert not (members[a] & members[b]), f"fuite de données entre {a} et {b}"
print("OK : les ensembles train / val / test / ext sont deux à deux disjoints.")

# %% [markdown]
# Fonctions utilitaires du notebook : récupérer les données d'une étape et lancer un entraînement chronométré.

# %%
def get_split(negative_sources=None):
    """{'train': (X, y, meta), 'val': ..., 'test': ...}. Les photos sont toujours incluses ;
    `negative_sources` choisit les classes négatives (None = toutes)."""
    return {s: data.select(X, meta, s, negative_sources) for s in ("train", "val", "test")}


def run(split, epochs, early_stopping=False, patience=4, **model_kwargs):
    """Construit un CNN (`model_kwargs` -> modeling.build_cnn), l'entraîne et renvoie (modèle, historique, secondes)."""
    keras.utils.set_random_seed(config.SEED)
    Xtr, ytr, _ = split["train"]
    Xva, yva, _ = split["val"]
    model = modeling.build_cnn(**model_kwargs)
    t0 = time.perf_counter()
    hist = modeling.train(
        model,
        data.make_dataset(Xtr, ytr, shuffle=True),
        data.make_dataset(Xva, yva),
        epochs=epochs,
        class_weight=data.class_weights(ytr),
        early_stopping=early_stopping,
        patience=patience,
        verbose=0,
    )
    return model, hist, time.perf_counter() - t0
```

Méthode de contrôle utilisée dans toutes les tâches notebook :

```bash
cd /home/diskmth/Dev/bloc_AI
bash scripts/run_notebook.sh quick            # régénère le .ipynb depuis le .py puis l'exécute avec 2 epochs partout
uv run python scripts/check_notebook.py notebooks/livrable1_classification_binaire.ipynb --partial
```

- [ ] **Étape 3 : Exécuter en mode QUICK et contrôler**

Attendu : aucune erreur ; le script de contrôle affiche `OK` ; **2 figures** (aperçu 5×8, deux boîtes à moustaches). Dans le `.ipynb`, vérifier en particulier :
- `X : (10806, 128, 128, 3), dtype=uint8, 531 Mo en mémoire` ;
- tableau d'inventaire : Painting 1000/1200, Photo 1000/3000, Schematics 1000/600, Sketch 806/600, Text 1000/600 ;
- tableau de découpage : ligne `TOTAL` = train 4200, val 900, test 900, ext 4806 ;
- la ligne `OK : les ensembles train / val / test / ext sont deux à deux disjoints.`

Si une valeur diffère : revenir à la tâche 4 (le cache a-t-il été reconstruit avec les bons effectifs ? `rm -rf data/cache` puis relancer).

- [ ] **Étape 4 : Commit**

```bash
git add scripts notebooks/livrable1_classification_binaire.py
git commit -m "docs(notebook): business context, environment, data exploration and split (part A)"
```
(On ne commit pas le `.ipynb` partiel : il sera commité exécuté à la tâche 13.)

---

### Task 9 : Notebook, partie B — progression par difficulté (section 4)

**Fichiers :**
- Modifier : `notebooks/livrable1_classification_binaire.py` (ajout en fin de fichier)

**Interfaces :**
- Consomme : `get_split`, `run`, `EPOCHS`, `config.STAGES` (partie A) ; `modeling.predict_proba`, `evaluation.evaluate_binary`, `evaluation.per_source_accuracy`.
- Produit : `stage_df` (DataFrame indexé par étape), `stage_per_source`, `stage_models`.

- [ ] **Étape 1 : Ajouter la section 4**

**Ajouter à la FIN de `notebooks/livrable1_classification_binaire.py`** (laisser deux lignes vides avant ce bloc ; ne rien modifier au-dessus) — contenu exact :

```python
# %% [markdown]
# ## 4. Progression par difficulté
#
# Le sujet recommande de commencer par les images les plus faciles à distinguer des photos. On entraîne donc **le même
# réseau** (défini en section 5) sur quatre problèmes de difficulté croissante, en ajoutant à chaque étape un type
# d'image « négatif » :
#
# | Étape | Photos opposées à… |
# |---|---|
# | 1 | textes scannés |
# | 2 | textes + schémas |
# | 3 | textes + schémas + sketchs |
# | 4 | **textes + schémas + sketchs + peintures** (problème final) |
#
# Aux étapes 1 à 3, les photos sont plus nombreuses que les négatifs : on utilise des **poids de classe**
# (`class_weight`) et, pour juger honnêtement, la **balanced accuracy** (moyenne du rappel sur chaque classe) plutôt que
# l'accuracy brute. L'évaluation se fait sur le jeu `test` de l'étape (jamais vu à l'entraînement).

# %%
stage_rows, stage_per_source, stage_models = [], {}, {}
for stage, negatives in config.STAGES.items():
    split = get_split(negatives)
    model, hist, seconds = run(split, EPOCHS["stage"], early_stopping=True, patience=3)
    Xte, yte, mte = split["test"]
    proba = modeling.predict_proba(model, Xte)
    res = evaluation.evaluate_binary(yte, proba)
    per_src = evaluation.per_source_accuracy(mte, proba)
    worst = per_src.iloc[0]
    stage_rows.append({
        "étape": stage, "n_train": len(split["train"][1]), "epochs": len(hist), "durée (s)": round(seconds),
        "balanced_acc": res["balanced_accuracy"], "précision": res["precision"], "rappel": res["recall"],
        "F1": res["f1"], "ROC-AUC": res["roc_auc"], "classe la plus difficile": worst["source"],
        "accuracy sur cette classe": worst["accuracy"],
    })
    stage_per_source[stage] = per_src.set_index("source")["accuracy"]
    stage_models[stage] = model
stage_df = pd.DataFrame(stage_rows).set_index("étape")
display(stage_df)

# %%
fig, ax = plt.subplots(figsize=(9, 3.8))
pd.DataFrame(stage_per_source).loc[config.CLASSES].plot.bar(ax=ax, rot=0)
ax.set_ylim(0.5, 1.01)
ax.set_ylabel("accuracy sur le jeu test de l'étape")
ax.set_title("Accuracy par type d'image, selon l'étape de la progression")
ax.legend(title="étape", fontsize=8)
plt.tight_layout()
plt.show()

# %%
last = stage_df.iloc[-1]
first = stage_df.iloc[0]
display(Markdown(
    f"**Lecture automatique.** Étape 1 (photo vs texte) : balanced accuracy **{first['balanced_acc']:.3f}** ; "
    f"étape 4 (toutes classes, peintures comprises) : **{last['balanced_acc']:.3f}**, F1 **{last['F1']:.3f}**. "
    f"À l'étape 4, la classe la plus difficile est **{last['classe la plus difficile']}** "
    f"(accuracy {last['accuracy sur cette classe']:.3f}). "
    + ("Conforme à l'attente : les peintures réalistes, visuellement proches des photos, concentrent les erreurs."
       if last["classe la plus difficile"] == "Painting"
       else "Contrairement à l'attente, ce ne sont pas les peintures qui posent le plus de problèmes : à analyser.")
))
```

Méthode de contrôle utilisée dans toutes les tâches notebook :

```bash
cd /home/diskmth/Dev/bloc_AI
bash scripts/run_notebook.sh quick            # régénère le .ipynb depuis le .py puis l'exécute avec 2 epochs partout
uv run python scripts/check_notebook.py notebooks/livrable1_classification_binaire.ipynb --partial
```

- [ ] **Étape 2 : Exécuter en mode QUICK et contrôler**

Attendu : pas d'erreur ; `OK` du script ; **3 figures** au total (2 de la partie A + le diagramme en barres par étape). Le tableau `stage_df` a 4 lignes (`1_photo_vs_text` … `4_+painting`) et la colonne « classe la plus difficile » est renseignée.
Une cellule « **Lecture automatique.** … » s'affiche sous le tableau. (En QUICK, les chiffres n'ont aucune valeur.)

- [ ] **Étape 3 : Commit**

```bash
git add notebooks/livrable1_classification_binaire.py
git commit -m "docs(notebook): progression by difficulty across four stages (part B)"
```

---

### Task 10 : Notebook, partie C — architecture (section 5) et entraînement (section 6)

**Fichiers :**
- Modifier : `notebooks/livrable1_classification_binaire.py` (ajout en fin de fichier)

**Interfaces :**
- Consomme : `modeling.build_cnn`, `modeling._compile`, `evaluation.draw_architecture`, `evaluation.plot_history`, `evaluation.diagnose_fit`, `run`, `get_split`.
- Produit : `baseline` (modèle non entraîné, pour le schéma), `FINAL_SPLIT`, `Xtr, ytr, mtr`, `Xva, yva, mva`, `Xte, yte, mte`, `baseline_model`, `baseline_hist`, `baseline_sec`.

Cette partie couvre les exigences **E1** (code, schéma, paramètres, perte, optimiseur) et **E2** (courbes).

- [ ] **Étape 1 : Ajouter les sections 5 et 6**

**Ajouter à la FIN de `notebooks/livrable1_classification_binaire.py`** (laisser deux lignes vides avant ce bloc ; ne rien modifier au-dessus) — contenu exact :

```python
# %% [markdown]
# ## 5. Architecture du réseau
#
# On utilise un **réseau de neurones convolutif (CNN)** conçu pour ce problème. Le code ci-dessous est celui qui est
# réellement utilisé dans tout le notebook (`src/tounum_cls/modeling.py`).

# %%
print(inspect.getsource(modeling._compile))   # perte, optimiseur, métriques
print(inspect.getsource(modeling.build_cnn))  # architecture

# %%
baseline = modeling.build_cnn()
baseline.summary()

# %%
fig = evaluation.draw_architecture(baseline)
plt.show()

# %% [markdown]
# ### 5.1 Rôle de chaque couche
#
# | Couche | Rôle |
# |---|---|
# | `Rescaling(1/255)` | ramène les pixels de [0, 255] à [0, 1] (dans le modèle : l'inférence en production reçoit des images brutes) |
# | `Conv2D 3×3` ×4 (16→32→64→128 filtres) | détecte des motifs locaux : bords et textures (premiers blocs), formes plus complexes (derniers blocs). Les filtres sont **appris**, comme les filtres de convolution du livrable 2 mais sans les fixer à la main |
# | `BatchNormalization` | normalise les activations de chaque batch : apprentissage plus rapide et plus stable |
# | `ReLU` | non-linéarité `max(0, x)` |
# | `MaxPooling2D 2×2` | divise la résolution par 2 : moins de calculs, tolérance aux petits décalages |
# | `GlobalAveragePooling2D` | moyenne chaque carte de caractéristiques en un seul nombre : évite une énorme couche dense (donc peu de paramètres, moins de sur-apprentissage) |
# | `Dense(64, ReLU)` | combine les 128 caractéristiques |
# | `Dense(1, sigmoïde)` | sortie : probabilité *P(photo)* ∈ [0, 1] |
#
# ### 5.2 Paramètres du réseau

# %%
rows = []
for layer in baseline.layers:
    rows.append({
        "couche": layer.name, "type": type(layer).__name__,
        "forme de sortie": str(tuple(layer.output.shape)[1:]), "paramètres": layer.count_params(),
    })
param_table = pd.DataFrame(rows)
display(param_table[param_table["paramètres"] > 0])
trainable = int(sum(np.prod(w.shape) for w in baseline.trainable_weights))
non_trainable = int(sum(np.prod(w.shape) for w in baseline.non_trainable_weights))
print(f"Total : {baseline.count_params():,} paramètres — entraînables : {trainable:,} — "
      f"non entraînables (moyennes/variances de BatchNorm) : {non_trainable:,}")

# %% [markdown]
# ### 5.3 Fonction de perte
#
# Sortie binaire → **entropie croisée binaire** (*binary cross-entropy*). Pour une image de label réel $y \in \{0, 1\}$
# et une probabilité prédite $\hat{p}$ = P(photo) :
#
# $$\mathcal{L}(y, \hat{p}) = -\big[\, y \log \hat{p} + (1 - y)\log(1 - \hat{p}) \,\big]$$
#
# La perte est la moyenne sur le batch. Elle est faible quand le réseau est à la fois **juste et confiant**, et très
# élevée quand il est **confiant et faux** (une peinture annoncée « photo » à 99 %). C'est la fonction naturelle d'une
# sortie sigmoïde (maximum de vraisemblance d'une loi de Bernoulli) et elle donne de bons gradients, contrairement à
# l'erreur quadratique.
#
# ### 5.4 Algorithme d'optimisation
#
# **Adam** (taux d'apprentissage $10^{-3}$, $\beta_1 = 0{,}9$, $\beta_2 = 0{,}999$). Adam est une descente de gradient
# stochastique par mini-batchs de 64 images, qui adapte le pas de chaque paramètre à partir d'une moyenne mobile du
# gradient (inertie) et de son carré (échelle). C'est un choix par défaut robuste : il converge vite sans réglage
# fin du pas. Les métriques suivies sont l'**accuracy** (part d'images bien classées au seuil 0,5) et l'**AUC**
# (capacité à mieux classer une photo qu'une non-photo, indépendamment du seuil).
#
# ### 5.5 Pourquoi ce choix d'architecture ?
# - **CNN plutôt que réseau dense** : le partage des filtres exploite la structure 2D de l'image avec très peu de
#   paramètres (≈ 106 k, contre des millions pour un réseau dense sur 128×128×3 pixels).
# - **Petit réseau** : la tâche est une décision globale simple, et nous calculons sur CPU.
# - **Justification par les résultats** : sections 7 (le modèle n'est ni trop simple ni trop complexe), 8 et 10
#   (comparaison avec un réseau pré-entraîné bien plus gros).

# %% [markdown]
# ## 6. Entraînement et courbes d'apprentissage
#
# Entraînement du modèle de référence (*baseline*, sans régularisation) sur le problème **final** (étape 4 : photos
# contre toutes les autres classes). Les courbes comparent, à chaque epoch, l'**entraînement** et la **validation**
# (« erreur de test » du sujet) pour la perte et pour l'accuracy.

# %%
FINAL_SPLIT = get_split(None)
Xtr, ytr, mtr = FINAL_SPLIT["train"]
Xva, yva, mva = FINAL_SPLIT["val"]
Xte, yte, mte = FINAL_SPLIT["test"]
print(f"train {Xtr.shape[0]} | val {Xva.shape[0]} | test {Xte.shape[0]} | part de photos dans train : {ytr.mean():.1%}")

baseline_model, baseline_hist, baseline_sec = run(FINAL_SPLIT, EPOCHS["main"])
print(f"Entraînement terminé en {baseline_sec:.0f} s ({baseline_sec / len(baseline_hist):.1f} s/epoch)")
shown = sorted({i for i in (0, 1, 2, len(baseline_hist) // 2, len(baseline_hist) - 1) if i < len(baseline_hist)})
display(baseline_hist.round(4).set_index("epoch").iloc[shown])

# %%
evaluation.plot_history(baseline_hist, "Baseline")
plt.show()

# %%
diag = evaluation.diagnose_fit(baseline_hist)
display(Markdown(
    f"**Lecture automatique (baseline).** Accuracy finale : entraînement **{diag['train_acc']:.3f}**, "
    f"validation **{diag['val_acc']:.3f}** (écart {diag['gap_acc']:+.3f}). "
    f"Perte finale : entraînement {diag['train_loss']:.3f}, validation {diag['val_loss']:.3f}. "
    f"Meilleure perte de validation à l'epoch **{diag['best_epoch']}** sur {len(baseline_hist)} "
    f"(remontée depuis ce minimum : {diag['val_loss_drift']:+.3f}). Diagnostic : **{diag['verdict']}**."
))

# %% [markdown]
# **Comment lire ces courbes.**
# - Les premières epochs : la validation peut être instable (voire très mauvaise) car `BatchNormalization` utilise
#   des statistiques mobiles qui mettent quelques pas à se calibrer. Ce n'est pas un défaut du modèle.
# - Si la perte d'entraînement continue de baisser alors que celle de validation remonte, le réseau **mémorise**
#   (sur-apprentissage).
# - Si les deux courbes stagnent haut, le réseau est **trop simple** ou insuffisamment entraîné (sous-apprentissage).
```

Méthode de contrôle utilisée dans toutes les tâches notebook :

```bash
cd /home/diskmth/Dev/bloc_AI
bash scripts/run_notebook.sh quick            # régénère le .ipynb depuis le .py puis l'exécute avec 2 epochs partout
uv run python scripts/check_notebook.py notebooks/livrable1_classification_binaire.ipynb --partial
```

- [ ] **Étape 2 : Exécuter en mode QUICK et contrôler**

Attendu : pas d'erreur ; `OK` ; **5 figures** au total (+ schéma d'architecture, + courbes de la baseline). Dans le `.ipynb`, vérifier :
- la sortie de `inspect.getsource` contient `binary_crossentropy` et `Adam` (cellule 5) ;
- `model.summary()` se termine par `Total params: 106,481` ;
- le tableau des paramètres liste `conv1…conv4`, `bn1…bn4`, `dense`, `output` ; la ligne `Total : 106,481 paramètres — entraînables : 106,001 — non entraînables … : 480` s'affiche ;
- `train 4200 | val 900 | test 900 | part de photos dans train : 50.0%`.

- [ ] **Étape 3 : Commit**

```bash
git add notebooks/livrable1_classification_binaire.py
git commit -m "docs(notebook): architecture, loss, optimizer and baseline training curves (part C)"
```

---

### Task 11 : Notebook, partie D — analyse biais/variance (section 7) et améliorations (section 8)

**Fichiers :**
- Modifier : `notebooks/livrable1_classification_binaire.py` (ajout en fin de fichier)

**Interfaces :**
- Consomme : `baseline_model`, `baseline_hist`, `baseline_sec`, `FINAL_SPLIT`, `Xtr…`, `run`, `EPOCHS`, `evaluation.diagnose_fit`, `evaluation.plot_history`, `data.make_dataset`.
- Produit : `cap_df`, `curve_df`, `imp_df`, `imp_models`, `imp_hist`, `best_name`, `final_model`, `final_hist` (le **modèle retenu**, choisi sur la perte de **validation** uniquement).

Cette partie couvre **E3** (analyse biais/variance) et **E4** (méthodes d'amélioration : tableau + expériences augmentation, dropout, L2, early stopping).

- [ ] **Étape 1 : Ajouter les sections 7 et 8**

**Ajouter à la FIN de `notebooks/livrable1_classification_binaire.py`** (laisser deux lignes vides avant ce bloc ; ne rien modifier au-dessus) — contenu exact :

```python
# %% [markdown]
# ## 7. Analyse du compromis biais / variance
#
# **Rappel.**
# - **Biais élevé = sous-apprentissage** : le modèle est trop rigide pour capturer la structure des données ; l'erreur est
#   élevée *à la fois* sur l'entraînement et sur la validation.
# - **Variance élevée = sur-apprentissage** : le modèle s'adapte au bruit des données d'entraînement ; l'erreur
#   d'entraînement est faible mais celle de validation est nettement plus haute (grand écart).
# - L'objectif est le bon compromis : les deux erreurs sont basses et proches.
#
# Un modèle plus **complexe** réduit le biais mais augmente la variance. Nous le vérifions de deux façons.
#
# ### 7.1 Trois niveaux de capacité du modèle
# Même données, même procédure ; seule la taille du réseau change.

# %%
CAPACITY = {
    "trop simple": dict(filters=(4, 8), dense_units=4),
    "baseline": dict(),
    "trop complexe": dict(filters=(32, 64, 128, 256), dense_units=256),
}
cap_hist, cap_models, cap_rows = {"baseline": baseline_hist}, {"baseline": baseline_model}, []
for name, kwargs in CAPACITY.items():
    if name == "baseline":
        model, hist, sec = baseline_model, baseline_hist, baseline_sec
    else:
        model, hist, sec = run(FINAL_SPLIT, EPOCHS["main"], **kwargs)
        cap_hist[name], cap_models[name] = hist, model
    d = evaluation.diagnose_fit(hist)
    cap_rows.append({"modèle": name, "paramètres": model.count_params(), "acc train": d["train_acc"],
                     "acc val": d["val_acc"], "écart": d["gap_acc"], "loss val min": hist["val_loss"].min(),
                     "epoch du min": d["best_epoch"], "diagnostic": d["verdict"]})
cap_df = pd.DataFrame(cap_rows).set_index("modèle")
display(cap_df)

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
for name, hist in cap_hist.items():
    line, = axes[0].plot(hist["epoch"], hist["loss"], ls="--")
    axes[0].plot(hist["epoch"], hist["val_loss"], color=line.get_color(), label=name)
    line, = axes[1].plot(hist["epoch"], hist["accuracy"], ls="--")
    axes[1].plot(hist["epoch"], hist["val_accuracy"], color=line.get_color(), label=name)
axes[0].set_title("Perte (pointillé : train, plein : validation)")
axes[1].set_title("Accuracy (pointillé : train, plein : validation)")
for ax in axes:
    ax.set_xlabel("epoch")
    ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 7.2 Courbe d'apprentissage : que se passe-t-il avec plus de données ?
# On entraîne la baseline sur une fraction croissante du jeu d'entraînement (stratifiée par classe) et on compare
# l'accuracy finale d'entraînement et de validation. Un **grand écart qui se resserre avec plus de données**
# est le signe d'une variance qu'on peut réduire en collectant des images ; deux courbes **proches mais basses** signalent
# un biais que plus de données ne corrigera pas.

# %%
from sklearn.model_selection import train_test_split

curve_rows = []
for frac in (0.1, 0.25, 0.5, 1.0):
    if frac < 1.0:
        keep, _ = train_test_split(np.arange(len(ytr)), train_size=frac, stratify=mtr["source"], random_state=config.SEED)
    else:
        keep = np.arange(len(ytr))
    sub = {"train": (Xtr[keep], ytr[keep], mtr.iloc[keep]), "val": FINAL_SPLIT["val"], "test": FINAL_SPLIT["test"]}
    _, hist, _ = run(sub, EPOCHS["curve"])
    d = evaluation.diagnose_fit(hist)
    curve_rows.append({"fraction": frac, "n_train": len(keep), "acc train": d["train_acc"], "acc val": d["val_acc"]})
curve_df = pd.DataFrame(curve_rows).set_index("fraction")
display(curve_df)

fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot(curve_df["n_train"], curve_df["acc train"], marker="o", label="entraînement")
ax.plot(curve_df["n_train"], curve_df["acc val"], marker="o", label="validation")
ax.set_xlabel("nombre d'images d'entraînement")
ax.set_ylabel("accuracy (dernière epoch)")
ax.set_title("Courbe d'apprentissage (baseline)")
ax.legend()
plt.tight_layout()
plt.show()

# %%
small, big = curve_df.iloc[0], curve_df.iloc[-1]
display(Markdown(
    "**Lecture automatique.** "
    f"Capacité : le modèle « trop simple » ({cap_df.loc['trop simple', 'paramètres']:,} paramètres) atteint "
    f"{cap_df.loc['trop simple', 'acc val']:.3f} en validation, la baseline ({cap_df.loc['baseline', 'paramètres']:,}) "
    f"{cap_df.loc['baseline', 'acc val']:.3f} et le modèle « trop complexe » "
    f"({cap_df.loc['trop complexe', 'paramètres']:,}) {cap_df.loc['trop complexe', 'acc val']:.3f}. "
    f"Diagnostics : trop simple → *{cap_df.loc['trop simple', 'diagnostic']}* ; baseline → "
    f"*{cap_df.loc['baseline', 'diagnostic']}* ; trop complexe → *{cap_df.loc['trop complexe', 'diagnostic']}*. "
    f"Courbe d'apprentissage : avec {int(small['n_train'])} images, l'écart train − validation est "
    f"{small['acc train'] - small['acc val']:+.3f} ; avec {int(big['n_train'])} images, il est "
    f"{big['acc train'] - big['acc val']:+.3f}."
))

# %% [markdown]
# **Interprétation.** Un écart qui se réduit quand le nombre d'images augmente montre que l'on est dans un
# régime de **variance** : plus de données (ou une régularisation) aident. Un modèle dont l'accuracy d'entraînement
# elle-même est insuffisante est dans un régime de **biais** : il faut alors plus de capacité ou plus d'epochs, pas
# plus de données. Les valeurs affichées ci-dessus indiquent dans quel régime se trouve chaque configuration.
# La baseline retenue est un compromis : assez de paramètres (≈ 106 k) pour séparer les photos des peintures, sans
# la capacité excessive du gros modèle. Les techniques de la section suivante réduisent encore la variance.

# %% [markdown]
# ## 8. Améliorer le compromis biais / variance
#
# ### 8.1 Les méthodes disponibles
#
# | Méthode | Principe | Effet attendu | Dans ce notebook |
# |---|---|---|---|
# | **Plus de données** | davantage d'exemples = moins de mémorisation | ↓ variance | limité à l'échantillon (voir 7.2) |
# | **Augmentation de données** | transformations aléatoires (flip, rotation, zoom, contraste) à chaque epoch : le réseau ne voit jamais deux fois exactement la même image | ↓ variance | oui |
# | **Dropout** | éteint aléatoirement une fraction des neurones à chaque pas : empêche les co-adaptations | ↓ variance | oui (0,4) |
# | **Régularisation L2** (*weight decay*) | ajoute $\lambda \sum w^2$ à la perte : pénalise les gros poids, lisse le modèle | ↓ variance | oui ($\lambda = 10^{-4}$) |
# | **Early stopping** | arrête l'entraînement quand la perte de validation cesse de s'améliorer et restaure les meilleurs poids | ↓ variance | oui (patience 5) |
# | **Réduire la capacité** | moins de couches / de filtres | ↓ variance, ↑ biais | cf. 7.1 |
# | **BatchNormalization** | normalise les activations ; effet régularisant léger | stabilité | déjà dans le réseau |
# | **Plus de capacité / plus d'epochs** | plus de filtres, apprentissage plus long | ↓ biais | cf. 7.1 |
# | **Taux d'apprentissage adaptatif** | réduire le pas quand la validation stagne (`ReduceLROnPlateau`) | convergence plus fine | non retenu |
# | **Transfert d'apprentissage** | partir d'un réseau pré-entraîné (ImageNet) | ↓ biais et variance avec peu de données | section 10 |
#
# ### 8.2 Expériences
# Même problème (étape 4), même graine. On compare chaque technique **seule**, puis **combinées avec early stopping**.
# L'early stopping est évalué avec un budget d'epochs plus large (il choisira lui-même quand s'arrêter).

# %%
IMPROVEMENTS = {
    "baseline": dict(kwargs=dict(), epochs=EPOCHS["main"], es=False),
    "+ augmentation": dict(kwargs=dict(augmentation=True), epochs=EPOCHS["main"], es=False),
    "+ dropout 0,4": dict(kwargs=dict(dropout=0.4), epochs=EPOCHS["main"], es=False),
    "+ L2 1e-4": dict(kwargs=dict(l2=1e-4), epochs=EPOCHS["main"], es=False),
    "+ early stopping": dict(kwargs=dict(), epochs=EPOCHS["improve"], es=True),
    "tout combiné": dict(kwargs=dict(augmentation=True, dropout=0.4, l2=1e-4), epochs=EPOCHS["improve"], es=True),
}
imp_hist, imp_models, imp_rows = {}, {}, []
for name, cfg in IMPROVEMENTS.items():
    if name == "baseline":
        model, hist, sec = baseline_model, baseline_hist, baseline_sec
    else:
        model, hist, sec = run(FINAL_SPLIT, cfg["epochs"], early_stopping=cfg["es"], patience=5, **cfg["kwargs"])
    imp_hist[name], imp_models[name] = hist, model
    d = evaluation.diagnose_fit(hist)
    val_metrics = model.evaluate(data.make_dataset(Xva, yva), verbose=0, return_dict=True)  # poids FINAUX du modèle
    imp_rows.append({"configuration": name, "epochs faits": len(hist), "acc train": d["train_acc"],
                     "écart train−val": d["gap_acc"], "loss val (modèle final)": val_metrics["loss"],
                     "acc val (modèle final)": val_metrics["accuracy"], "diagnostic": d["verdict"],
                     "durée (s)": round(sec)})
imp_df = pd.DataFrame(imp_rows).set_index("configuration")
display(imp_df)

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for name, hist in imp_hist.items():
    axes[0].plot(hist["epoch"], hist["val_loss"], label=name)
    axes[1].plot(hist["epoch"], hist["val_accuracy"], label=name)
axes[0].set_title("Perte de validation")
axes[1].set_title("Accuracy de validation")
for ax in axes:
    ax.set_xlabel("epoch")
axes[0].set_ylim(top=min(1.5, axes[0].get_ylim()[1]))
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.show()

# %%
best_name = imp_df["loss val (modèle final)"].idxmin()
display(Markdown(
    f"**Sélection du modèle (sur la validation uniquement).** La configuration avec la plus faible perte de validation "
    f"est **« {best_name} »** (loss {imp_df.loc[best_name, 'loss val (modèle final)']:.4f}, "
    f"accuracy {imp_df.loc[best_name, 'acc val (modèle final)']:.3f}) ; la baseline obtient "
    f"{imp_df.loc['baseline', 'loss val (modèle final)']:.4f} / {imp_df.loc['baseline', 'acc val (modèle final)']:.3f}. "
    "Ce modèle est celui évalué sur le jeu de test à la section suivante."
))
final_model = imp_models[best_name]
final_hist = imp_hist[best_name]

# %% [markdown]
# **Courbes du modèle retenu** (entraînement vs validation, pour la perte et l'accuracy) :

# %%
evaluation.plot_history(final_hist, f"Modèle retenu ({best_name})")
plt.show()
d = evaluation.diagnose_fit(final_hist)
display(Markdown(
    f"Diagnostic du modèle retenu : accuracy entraînement {d['train_acc']:.3f}, validation {d['val_acc']:.3f}, "
    f"→ **{d['verdict']}**. Rappel : avec dropout et augmentation, l'accuracy d'entraînement est mesurée *avec* ces "
    "perturbations actives, donc plus basse que ce qu'elle serait en inférence."
))
```

Méthode de contrôle utilisée dans toutes les tâches notebook :

```bash
cd /home/diskmth/Dev/bloc_AI
bash scripts/run_notebook.sh quick            # régénère le .ipynb depuis le .py puis l'exécute avec 2 epochs partout
uv run python scripts/check_notebook.py notebooks/livrable1_classification_binaire.ipynb --partial
```

- [ ] **Étape 2 : Exécuter en mode QUICK et contrôler**

Attendu : pas d'erreur ; `OK` ; **9 figures** au total (+ comparaison des 3 capacités, + courbe d'apprentissage, + comparaison des améliorations, + courbes du modèle retenu).
`cap_df` a 3 lignes (`trop simple` = 485 paramètres, `baseline` = 106 481, `trop complexe` = 455 905) ; `curve_df` a 4 lignes (fractions 0.1, 0.25, 0.5, 1.0 ; n_train 420, 1050, 2100, 4200) ;
`imp_df` a 6 lignes (`baseline`, `+ augmentation`, `+ dropout 0,4`, `+ L2 1e-4`, `+ early stopping`, `tout combiné`). Les cellules « Lecture automatique » et « Sélection du modèle » s'affichent.

- [ ] **Étape 3 : Commit**

```bash
git add notebooks/livrable1_classification_binaire.py
git commit -m "docs(notebook): bias/variance analysis and regularization experiments (part D)"
```

---

### Task 12 : Notebook, partie E — évaluation finale (9), comparaison MobileNetV2 (10), conclusion (11)

**Fichiers :**
- Modifier : `notebooks/livrable1_classification_binaire.py` (ajout en fin de fichier)

**Interfaces :**
- Consomme : `final_model`, `best_name`, `imp_df`, `Xte, yte, mte`, `Xva, yva`, `Xtr, ytr`, `X`, `meta`, `evaluation.*`, `modeling.*`.
- Produit : `models/cnn_photo_final.keras` (modèle retenu sauvegardé et rechargé pour vérification), la fonction `predict_photo(model, paths, threshold=0.5)`, `compare_df`.

- [ ] **Étape 1 : Ajouter les sections 9, 10 et 11**

**Ajouter à la FIN de `notebooks/livrable1_classification_binaire.py`** (laisser deux lignes vides avant ce bloc ; ne rien modifier au-dessus) — contenu exact :

```python
# %% [markdown]
# ## 9. Évaluation finale
#
# On utilise maintenant le jeu **`test`** (équilibré, jamais vu) et le **test étendu `ext`** (déséquilibré, réaliste),
# **une seule fois**, avec le modèle retenu en section 8.

# %%
Xe, ye, me = data.select(X, meta, "ext")
results, probas = {}, {}
for label, (Xs, ys, ms) in {"test (équilibré)": (Xte, yte, mte), "ext (réaliste)": (Xe, ye, me)}.items():
    probas[label] = modeling.predict_proba(final_model, Xs)
    results[label] = evaluation.evaluate_binary(ys, probas[label])
res_df = pd.DataFrame(results).T
display(res_df[["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "tp", "fp", "fn", "tn"]])
print(f"Part de photos : test {yte.mean():.1%} | ext {ye.mean():.1%}")

# %%
for label, (Xs, ys, ms) in {"test (équilibré)": (Xte, yte, mte), "ext (réaliste)": (Xe, ye, me)}.items():
    evaluation.plot_confusion(ys, probas[label], title=f"Matrice de confusion — {label}")
    plt.show()

# %% [markdown]
# ### 9.1 Où le modèle se trompe-t-il ?
# Accuracy par type d'image (sur le test étendu, plus large) :

# %%
per_src_ext = evaluation.per_source_accuracy(me, probas["ext (réaliste)"])
display(per_src_ext)
fig, ax = plt.subplots(figsize=(6, 3))
sns.barplot(data=per_src_ext, x="source", y="accuracy", order=config.CLASSES, ax=ax)
ax.set_ylim(0.5, 1.0)
ax.set_xlabel("")
ax.set_title("Accuracy par type d'image — test étendu")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Faux positifs** : des non-photos prises pour des photos (les plus « sûres d'elles » d'abord). Ce sont les cas
# critiques pour TouNum, puisqu'ils polluent l'entrée du module de légendage.

# %%
fig = evaluation.show_errors(Xe, me, probas["ext (réaliste)"], kind="fp", n=8)
if fig is None:
    print("Aucun faux positif sur le test étendu.")
else:
    plt.show()

# %% [markdown]
# **Faux négatifs** : des photos rejetées (les plus « sûres d'elles » d'abord).

# %%
fig = evaluation.show_errors(Xe, me, probas["ext (réaliste)"], kind="fn", n=8)
if fig is None:
    print("Aucun faux négatif sur le test étendu.")
else:
    plt.show()

# %% [markdown]
# ### 9.2 Choix du seuil de décision
# Le seuil 0,5 est un défaut. Pour TouNum, **laisser passer une non-photo** (faux positif, baisse de précision) et
# **perdre une photo** (faux négatif, baisse du rappel) n'ont pas le même coût : le bon seuil est une décision
# métier. Le tableau ci-dessous aide à le choisir sur le test étendu, où les photos sont minoritaires.

# %%
rows = []
for thr in (0.1, 0.3, 0.5, 0.7, 0.9):
    r = evaluation.evaluate_binary(ye, probas["ext (réaliste)"], threshold=thr)
    rows.append({"seuil": thr, "précision": r["precision"], "rappel": r["recall"], "F1": r["f1"],
                 "faux positifs": r["fp"], "faux négatifs": r["fn"]})
display(pd.DataFrame(rows).set_index("seuil"))
evaluation.plot_precision_recall(ye, probas["ext (réaliste)"], "Précision / rappel — test étendu")
plt.show()

# %%
res_t, res_e = results["test (équilibré)"], results["ext (réaliste)"]
worst = per_src_ext.iloc[0]
display(Markdown(
    f"**Lecture automatique.** Test équilibré : accuracy **{res_t['accuracy']:.3f}**, F1 **{res_t['f1']:.3f}**, "
    f"ROC-AUC **{res_t['roc_auc']:.3f}**. Test étendu (photos ≈ {ye.mean():.0%}) : accuracy **{res_e['accuracy']:.3f}**, "
    f"précision **{res_e['precision']:.3f}**, rappel **{res_e['recall']:.3f}**, F1 **{res_e['f1']:.3f}**. "
    f"Le type d'image le moins bien classé est **{worst['source']}** (accuracy {worst['accuracy']:.3f}). "
    + ("La précision est plus basse sur le test étendu car les photos y sont minoritaires : à taux de faux positifs égal, "
       "les faux positifs pèsent plus lourd face à moins de vrais positifs. C'est la métrique à surveiller en production."
       if res_e["precision"] < res_t["precision"] - 0.01
       else "La précision reste stable malgré le déséquilibre des classes du test étendu.")
))

# %% [markdown]
# ### 9.3 Sauvegarde et utilisation en production
# Le modèle contient son propre prétraitement (`Rescaling`) : il suffit de lui fournir des images 128×128×3 en `uint8`.
# La fonction `data.load_image` fait la conversion (RGB + redimensionnement) depuis n'importe quel fichier jpg / png.

# %%
MODEL_PATH = config.PROJECT_ROOT / "models" / "cnn_photo_final.keras"
MODEL_PATH.parent.mkdir(exist_ok=True)
final_model.save(MODEL_PATH)
reloaded = keras.models.load_model(MODEL_PATH)
assert np.allclose(reloaded.predict(Xte[:32], verbose=0), final_model.predict(Xte[:32], verbose=0), atol=1e-5)
print(f"Modèle sauvegardé : {MODEL_PATH.relative_to(config.PROJECT_ROOT)} ({MODEL_PATH.stat().st_size / 1e6:.1f} Mo) — rechargement vérifié.")


def predict_photo(model, paths, threshold=0.5):
    """Classe des fichiers image : renvoie un DataFrame (fichier, P(photo), est_une_photo)."""
    batch = np.stack([data.load_image(p) for p in paths])
    proba = modeling.predict_proba(model, batch)
    return pd.DataFrame({"fichier": [Path(p).name for p in paths], "P(photo)": proba.round(3),
                         "est_une_photo": proba >= threshold})


demo = me.sample(8, random_state=config.SEED)
out = predict_photo(reloaded, demo["path"].tolist())
out["vérité"] = demo["source"].to_numpy()
display(out)

# %% [markdown]
# ## 10. Comparaison avec un réseau pré-entraîné (MobileNetV2)
#
# Pour **justifier le choix d'architecture par les résultats**, on le compare à **MobileNetV2** pré-entraîné sur ImageNet
# (1,4 million d'images), utilisé comme extracteur de caractéristiques **gelé** : seule la tête (`Dense` sigmoïde) est
# entraînée. C'est le *transfert d'apprentissage*. Les poids sont téléchargés depuis Internet à la première utilisation.

# %%
def summarize(model, name):
    row = {"modèle": name, "paramètres": model.count_params(),
           "entraînables": int(sum(np.prod(w.shape) for w in model.trainable_weights))}
    for label, (Xs, ys, ms) in {"test": (Xte, yte, mte), "ext": (Xe, ye, me)}.items():
        p = modeling.predict_proba(model, Xs)
        r = evaluation.evaluate_binary(ys, p)
        row[f"{label} accuracy"], row[f"{label} F1"], row[f"{label} AUC"] = r["accuracy"], r["f1"], r["roc_auc"]
        if label == "ext":
            acc = evaluation.per_source_accuracy(ms, p).set_index("source")["accuracy"]
            row["ext acc. Painting"] = acc.get("Painting", np.nan)
    return row


compare_rows = [summarize(final_model, f"CNN maison ({best_name})")]
try:
    mnet = modeling.build_mobilenet()
    keras.utils.set_random_seed(config.SEED)
    t0 = time.perf_counter()
    mnet_hist = modeling.train(mnet, data.make_dataset(Xtr, ytr, shuffle=True), data.make_dataset(Xva, yva),
                               epochs=EPOCHS["mobilenet"], class_weight=data.class_weights(ytr), verbose=0)
    mnet_sec = time.perf_counter() - t0
    row = summarize(mnet, "MobileNetV2 gelé + tête")
    row["durée d'entraînement (s)"] = round(mnet_sec)
    compare_rows.append(row)
    evaluation.plot_history(mnet_hist, "MobileNetV2")
    plt.show()
except Exception as exc:  # pas de réseau pour télécharger les poids ImageNet, par exemple
    print(f"Comparaison MobileNetV2 ignorée : {type(exc).__name__}: {exc}")
compare_rows[0]["durée d'entraînement (s)"] = round(imp_df.loc[best_name, "durée (s)"])
compare_df = pd.DataFrame(compare_rows).set_index("modèle")
display(compare_df)

# %%
if len(compare_df) == 2:
    a, b = compare_df.iloc[0], compare_df.iloc[1]
    gain = b["ext accuracy"] - a["ext accuracy"]
    ratio = b["paramètres"] / a["paramètres"]
    if gain > 0.02:
        verdict = (f"MobileNetV2 est nettement plus précis (+{gain:.3f} d'accuracy sur le test étendu) avec {ratio:.0f} fois plus "
                   "de paramètres. Si TouNum accepte un modèle plus lourd et la dépendance à des poids pré-entraînés, c'est le "
                   "meilleur candidat pour la production ; le CNN maison reste une base légère, autonome et entièrement maîtrisée.")
    elif gain < -0.02:
        verdict = (f"Le CNN maison fait mieux (+{-gain:.3f} d'accuracy sur le test étendu) avec {ratio:.0f} fois moins de "
                   "paramètres : il est retenu.")
    else:
        verdict = ("Performances comparables (écart < 0,02) : le CNN maison, plus léger et autonome, suffit.")
    display(Markdown(
        f"**Lecture automatique.** CNN maison : {int(a['paramètres']):,} paramètres, accuracy test {a['test accuracy']:.3f} / "
        f"ext {a['ext accuracy']:.3f}, peintures {a['ext acc. Painting']:.3f}. MobileNetV2 : {int(b['paramètres']):,} "
        f"paramètres ({int(b['entraînables']):,} entraînables), accuracy test {b['test accuracy']:.3f} / "
        f"ext {b['ext accuracy']:.3f}, peintures {b['ext acc. Painting']:.3f}. **{verdict}**"
    ))

# %% [markdown]
# ## 11. Conclusion, limites et suites
#
# ### Ce qui a été réalisé
# 1. Un **workflow reproductible de la donnée brute au modèle** : inventaire des archives, échantillonnage disjoint,
#    extraction sélective, normalisation (RGB, 128×128), cache, découpage stratifié.
# 2. Un **CNN de ≈ 106 k paramètres** entraîné par Adam sur l'entropie croisée binaire, validé par une progression
#    de difficulté croissante (section 4), et comparé à un réseau pré-entraîné (section 10, dont la conclusion
#    indique quel modèle retenir en production).
# 3. Une **analyse biais/variance** chiffrée (sections 6 à 8) et une comparaison de techniques de régularisation.
# 4. Une **évaluation honnête** sur un jeu de test jamais utilisé, équilibré et réaliste (section 9), avec analyse des
#    erreurs et choix du seuil.
# 5. Un **modèle sauvegardé** (`models/cnn_photo_final.keras`) et une fonction d'inférence (`predict_photo`).
#
# ### Limites et esprit critique
# - **Échantillon** : 6 000 images de modélisation sur ~42 000 disponibles ; les performances sur la totalité pourraient
#   être légèrement meilleures. Les effectifs se règlent dans `config.py`.
# - **Résolution 128×128 et étirement** : on perd les détails fins (utiles pour distinguer une peinture réaliste d'une
#   photo) et on déforme les pages allongées.
# - **Définition de « photo »** : l'étiquette dépend de l'archive d'origine. Une peinture hyper-réaliste ou une photo
#   fortement retouchée reste ambiguë, y compris pour un humain ; des erreurs d'étiquetage dans les archives plafonnent
#   l'accuracy atteignable.
# - **Généralisation** : le modèle est évalué sur des images de même provenance. Des images venant d'autres
#   sources (scanners différents, autres types de documents, autres époques) peuvent dégrader les résultats.
# - **Qualité d'image** : toutes les images d'entraînement sont de qualité correcte. Le livrable 2 (débruitage) devra
#   vérifier que le filtre reste performant sur des images bruitées ou floues.
# - **Reproductibilité** : de petites variations numériques sont possibles d'une machine à l'autre.
#
# ### Pistes pour la suite
# - Entraîner sur la totalité des archives et à plus haute résolution (GPU) ;
# - affiner les peintures réalistes (augmentation ciblée, *fine-tuning* de MobileNetV2) ;
# - ajuster le seuil avec TouNum selon le coût d'une erreur (section 9.2) ;
# - **brancher ce filtre en amont du prétraitement et du captioning** (livrables 2 et 3) et mesurer son effet de bout en bout ;
# - mettre en place un suivi en production (part de photos prédites, échantillon relu à la main) pour détecter une dérive des données.
```

Méthode de contrôle utilisée dans toutes les tâches notebook :

```bash
cd /home/diskmth/Dev/bloc_AI
bash scripts/run_notebook.sh quick            # régénère le .ipynb depuis le .py puis l'exécute avec 2 epochs partout
uv run python scripts/check_notebook.py notebooks/livrable1_classification_binaire.ipynb --partial
```

- [ ] **Étape 2 : Exécuter en mode QUICK, contrôler, puis lancer le contrôle complet**

Le contrôle `--partial` doit afficher `OK` et **16 figures** (±2 : une galerie d'erreurs est absente s'il n'y a aucune erreur du type correspondant). Ensuite, le contrôle **complet** :

```bash
uv run python scripts/check_notebook.py notebooks/livrable1_classification_binaire.ipynb
```

Attendu : `OK` (sections 0 à 11, ≥ 13 figures, ≥ 5 « Lecture automatique », `Total params`, `binary_crossentropy`, contrôle anti-fuite).
La section 10 télécharge les poids MobileNetV2 (ImageNet) : sans réseau, la cellule affiche `Comparaison MobileNetV2 ignorée : …` et le contrôle complet peut signaler une figure en moins ; ce n'est pas bloquant, le noter dans le compte rendu.
Vérifier aussi que `models/cnn_photo_final.keras` existe (≈ 1,4 Mo).

- [ ] **Étape 3 : Commit**

```bash
git add notebooks/livrable1_classification_binaire.py models
git commit -m "docs(notebook): final evaluation, MobileNetV2 comparison, conclusion and saved model (part E)"
```

---

### Task 13 : Exécution complète, contrôle d'acceptation, README et livraison

**Fichiers :**
- Créer : `README.md`
- Modifier : `notebooks/livrable1_classification_binaire.ipynb` (régénéré et exécuté en mode complet)
- Modifier : `pyproject.toml` / `uv.lock` (uniquement si une dépendance a dû être ajoutée avec `uv add`)

**Interfaces :**
- Consomme : tout ce qui précède.
- Produit : le notebook exécuté (livrable), le README, `models/cnn_photo_final.keras` mis à jour.

- [ ] **Étape 1 : Lancer l'exécution COMPLÈTE depuis un `data/` neuf (test du critère d'acceptation « depuis zéro »)**

```bash
cd /home/diskmth/Dev/bloc_AI
rm -rf data/cache                      # on garde data/raw (extraction déjà faite) : le cache est reconstruit
( time bash scripts/run_notebook.sh full ) > run_full.log 2>&1 &
```

Durée attendue : **≈ 45 à 60 minutes** sur un CPU 16 cœurs (≈ 11 s/epoch pour la baseline, ≈ 30 s/epoch pour le gros modèle). Surveiller avec `tail -3 run_full.log` ;
le log reste presque vide tant qu'aucune cellule ne produit de sortie sur le terminal (les sorties vont dans le notebook). **Ne pas interrompre.** Quand la ligne `real …` apparaît, noter la durée.
Si une erreur survient, elle s'affiche dans `run_full.log` : la corriger **dans le `.py`** (jamais dans le `.ipynb`) puis relancer l'étape.

- [ ] **Étape 2 : Contrôle d'acceptation automatique**

```bash
uv run python scripts/check_notebook.py notebooks/livrable1_classification_binaire.ipynb
```

Attendu : `OK`.

- [ ] **Étape 3 : Contrôle de plausibilité des résultats (lecture du notebook exécuté)**

Ouvrir les sorties du `.ipynb` (par exemple en extrayant les cellules « Lecture automatique ») et comparer aux **plages attendues** (mesurées sur la configuration de référence, `SEED = 42`, 20 epochs) :

| Élément | Plage attendue |
|---|---|
| Section 4, étape 1 (photo vs texte), balanced accuracy | ≥ 0,95 |
| Section 4, étape 4, classe la plus difficile | `Painting` |
| Section 6, baseline : accuracy de validation à l'epoch 1 | ≥ 0,70 (si ≈ 0,50 → `bn_momentum` modifié par erreur) |
| Section 6, baseline : accuracy train / val finale | train 0,90–0,97 ; val 0,85–0,93 |
| Section 7.1 « trop simple » (485 paramètres) | accuracy de validation 0,72–0,82 ; diagnostic « biais élevé » |
| Section 7.1 « trop complexe » (455 905 paramètres) | accuracy de validation 0,82–0,93 |
| Section 9, test équilibré, accuracy du modèle retenu | 0,85–0,95 |
| Section 9.1, type d'image le moins bien classé | `Painting` |
| Section 10, MobileNetV2 | accuracy test étendu 0,88–0,98 |

Si un résultat sort nettement de la plage (par exemple accuracy ≈ 0,50 partout, ou 1,00 partout) : **s'arrêter et diagnostiquer**, ne pas publier. Pistes : labels inversés (Photo doit être 1),
fuite entre ensembles (la cellule anti-fuite doit afficher OK), `bn_momentum`, graine, cache périmé (`rm -rf data/cache`), `QUICK` resté actif (`echo $TOUNUM_QUICK` doit être vide).
Une valeur légèrement hors plage est acceptable si les conclusions affichées par les cellules « Lecture automatique » restent cohérentes ; le signaler dans le compte rendu.

- [ ] **Étape 4 : Créer le README et y inscrire la durée mesurée**

**Créer `README.md`** — contenu exact :

````markdown
# TouNum — Livrable 1 : classification binaire photo / non-photo

Prototype d'un filtre d'entrée pour le workflow de légendage automatique d'images de TouNum : un réseau de neurones
(TensorFlow / Keras) qui distingue les **photographies** de tous les autres types d'images (peintures, schémas, dessins,
textes scannés).

**Livrable principal : `notebooks/livrable1_classification_binaire.ipynb`** (exécuté, avec ses sorties).

## Contenu du dépôt

| Chemin | Rôle |
|---|---|
| `notebooks/livrable1_classification_binaire.ipynb` | notebook exécuté (livrable) |
| `notebooks/livrable1_classification_binaire.py` | **source** du notebook (format jupytext « percent ») : c'est ce fichier qu'on édite |
| `src/tounum_cls/config.py` | tous les réglages : graine, effectifs d'échantillonnage, taille d'image, étapes de difficulté |
| `src/tounum_cls/data.py` | inventaire des archives, tirage, extraction, chargement, cache, découpage, `tf.data` |
| `src/tounum_cls/modeling.py` | CNN, MobileNetV2 gelé, entraînement, prédiction |
| `src/tounum_cls/evaluation.py` | métriques, diagnostic biais/variance, graphiques, schéma d'architecture |
| `tests/` | tests unitaires (`pytest`) sur données synthétiques |
| `pyproject.toml`, `uv.lock`, `.python-version` | dépendances épinglées et verrouillage complet (gérés par uv) |
| `scripts/run_notebook.sh` | régénère et exécute le notebook (`quick` ou `full`) |
| `scripts/check_notebook.py` | vérifie qu'un notebook exécuté est complet |
| `models/cnn_photo_final.keras` | modèle retenu, sauvegardé |
| `docs/superpowers/` | spécification et plan d'implémentation |
| `Dataset Livrable 1 - *.zip` | archives fournies par TouNum (non versionnées) |

## Installation

Prérequis : [**uv**](https://docs.astral.sh/uv/) (gestionnaire d'environnement ; `curl -LsSf https://astral.sh/uv/install.sh | sh`), ≈ 3 Go de disque pour `data/`, 8 Go de RAM.
Aucun GPU n'est nécessaire (tout est calibré pour un CPU). Python 3.12 est installé automatiquement par uv si besoin.

```bash
uv sync        # crée .venv/ avec les versions exactes verrouillées dans uv.lock
```

L'environnement est **entièrement géré par uv** (pas de `pip install`, pas de `python -m venv`) : `uv.lock` garantit les mêmes versions de paquets sur toutes les machines.
Toutes les commandes ci-dessous passent par `uv run`. Pour ajouter une dépendance : `uv add <paquet>` (met à jour `pyproject.toml` et `uv.lock` ; les commiter).

Placer les 5 archives `Dataset Livrable 1 - {Photo,Painting,Schematics,Text,Sketch}.zip` à la racine du projet.

## Utilisation

```bash
# 1. tests unitaires (≈ 15 s)
uv run pytest -q

# 2. vérification rapide que tout s'exécute (2 epochs partout)
bash scripts/run_notebook.sh quick

# 3. exécution complète = livrable (DUREE_MINUTES minutes sur un CPU 16 cœurs, mesuré)
bash scripts/run_notebook.sh full

# 4. contrôle de complétude du notebook exécuté
uv run python scripts/check_notebook.py notebooks/livrable1_classification_binaire.ipynb
```

À la première exécution, les images tirées sont extraites vers `data/raw/` et redimensionnées dans `data/cache/` (quelques minutes) ;
ensuite le cache est relu instantanément. Pour repartir de zéro : `rm -rf data/`.

## Utiliser le modèle en production

```python
import keras
import numpy as np
from tounum_cls import data, modeling

model = keras.models.load_model("models/cnn_photo_final.keras")
batch = np.stack([data.load_image(p) for p in ["a.jpg", "b.png"]])   # RGB 128x128 uint8 ; la normalisation est dans le modèle
p_photo = modeling.predict_proba(model, batch)                       # probabilité que l'image soit une photo
is_photo = p_photo >= 0.5                                            # seuil à fixer avec TouNum (voir notebook, section 9.2)
```

## Maintenance

- **Changer la taille de l'échantillon** : éditer `POOL_COUNTS` / `EXT_COUNTS` dans `src/tounum_cls/config.py`, supprimer `data/cache/`, relancer le notebook.
- **Ajouter un type d'image** : ajouter la classe dans `CLASSES`, `ZIP_FILES`, `POOL_COUNTS`, `EXT_COUNTS` et `STAGES` (config) ; les autres modules n'ont pas besoin d'être modifiés.
- **Modifier le notebook** : éditer le `.py`, puis `bash scripts/run_notebook.sh full`. Ne pas éditer le `.ipynb` à la main.
- **Reproductibilité** : graine `SEED = 42`. Sur CPU, TensorFlow peut produire de très petites variations numériques entre exécutions ou machines ; les conclusions ne changent pas.
- Les textes « Lecture automatique » du notebook calculent leurs chiffres à chaque exécution : ils restent justes après un ré-entraînement.

## Limites connues

Voir la section 11 du notebook (échantillon de 6 000 images, résolution 128×128, définition de « photo » dépendante de l'archive d'origine,
généralisation à d'autres sources non mesurée, images bruitées traitées au livrable 2).
````

Remplacer le texte `DUREE_MINUTES` par la durée mesurée à l'étape 1 (entier en minutes) :

```bash
sed -i 's/DUREE_MINUTES/<durée mesurée>/' README.md      # par exemple 52
grep -n "minutes sur un CPU" README.md
```

- [ ] **Étape 5 : Vérifier que la suite de tests passe toujours et que le dépôt est propre**

```bash
uv run pytest -q          # attendu : 27 passed
git status --short
```

`git status` ne doit lister que `README.md`, le `.ipynb`, `models/` et éventuellement `run_full.log` (à supprimer : `rm run_full.log`).
Aucun `.zip`, `data/` ni `.venv/` ne doit apparaître.

- [ ] **Étape 6 : Commit final**

```bash
git add README.md notebooks/livrable1_classification_binaire.ipynb models pyproject.toml uv.lock
git commit -m "docs: executed deliverable notebook, README and saved model"
git log --oneline
```

- [ ] **Étape 7 : Compte rendu**

Rédiger un court compte rendu (français) contenant : la durée d'exécution complète, le résultat de `check_notebook.py`, les 5 chiffres clés lus dans les cellules « Lecture automatique »
(accuracy test équilibré, accuracy / précision / rappel test étendu, accuracy sur les peintures, verdict de la comparaison MobileNetV2) et tout écart par rapport aux plages attendues.
**Ne pas inventer de chiffres : les copier depuis le notebook exécuté.**

---

## Auto-revue du plan

- **Couverture de la spec** : voir la cartographie en tête de document. E1 → tâches 5, 7, 10 ; E2 → 7, 10 ; E3 → 6, 11 ; E4 → 11 ; E5 → 9 ; critères d'acceptation → 13.
- **Écart assumé avec la spec §3.2** : les données sont lues depuis un cache `uint8` plutôt qu'en `tf.data` depuis les fichiers (documenté dans la spec).
- **Cohérence des signatures** : toutes les fonctions citées dans les blocs « Interfaces » sont définies dans la tâche indiquée et utilisées avec les mêmes noms dans le notebook
  (`get_split`, `run`, `modeling.build_cnn`, `modeling._compile`, `evaluation.diagnose_fit`, `data.prepare_dataset`…).
- **Aucun texte à compléter** : les seules valeurs saisies par l'exécutant sont la durée mesurée (`DUREE_MINUTES`) et le compte rendu final, tous deux issus de mesures.
