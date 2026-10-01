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
