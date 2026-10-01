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
