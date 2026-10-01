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
| `models/cnn_photo_final.keras` | CNN maison retenu, sauvegardé (le choix du modèle de production, CNN ou MobileNetV2, est discuté section 10 du notebook) |
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

# 2. vérification rapide que tout s'exécute (2 epochs partout) : écrit notebooks/livrable1_classification_binaire.quick.ipynb
#    (ignoré par git) et ne touche ni au livrable ni au modèle sauvegardé
bash scripts/run_notebook.sh quick

# 3. exécution complète = livrable (35 minutes sur un CPU 16 cœurs, mesuré)
bash scripts/run_notebook.sh full

# 4. contrôle de complétude du notebook exécuté
uv run python scripts/check_notebook.py notebooks/livrable1_classification_binaire.ipynb
```

À la première exécution, les images tirées sont extraites vers `data/raw/` et redimensionnées dans `data/cache/` (quelques minutes) ;
ensuite le cache est relu instantanément. Pour repartir de zéro : `rm -rf data/`.
Pour ouvrir le notebook de façon interactive : `uv run --with jupyterlab jupyter lab`.

## Utiliser le modèle en production

Lancer ce script avec `PYTHONPATH=src uv run python script.py` (ou garder la première ligne `sys.path.insert` ci-dessous) :
le paquet `tounum_cls` se trouve dans `src/`.

```python
import sys; sys.path.insert(0, "src")
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
- **Reproductibilité** : graine `SEED = 42`. Sur CPU, TensorFlow peut produire de très petites variations numériques entre exécutions ou machines ; les chiffres peuvent varier légèrement ; les diagnostics proches d'un seuil peuvent basculer.
- Seules les cellules « Lecture automatique » du notebook recalculent leurs chiffres à chaque exécution : le texte d'interprétation doit être relu après un ré-entraînement significatif.

## Limites connues

Voir la section 11 du notebook (échantillon de 6 000 images, résolution 128×128, définition de « photo » dépendante de l'archive d'origine,
généralisation à d'autres sources non mesurée, images bruitées traitées au livrable 2).
