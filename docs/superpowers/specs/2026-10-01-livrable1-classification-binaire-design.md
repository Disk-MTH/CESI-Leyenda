# Livrable 1 — Classification binaire photo / non-photo : spécification

Date : 2026-10-01 — Client : TouNum — Équipe : Data Science CESI

## 1. Objectif

Fournir un notebook Jupyter didactique et reproductible qui entraîne un réseau de neurones (TensorFlow/Keras)
à distinguer les **photographies** (classe positive, label `1`) de **tout autre type d'image** (label `0`) :
peintures, schémas/graphes, portraits dessinés (sketch), textes scannés.

Le notebook est la livraison. Il doit permettre à TouNum de comprendre, relancer et maintenir la solution
(contexte métier, code, justification des choix, statistiques, limites).

### Contenu obligatoire (consignes du livrable)

| # | Exigence | Section du notebook |
|---|----------|--------------------|
| E1 | Code TensorFlow + schéma de l'architecture + paramètres du réseau, fonction de perte, optimiseur | §5 |
| E2 | Graphique : erreur (loss) train/test + accuracy train/test au fil des epochs | §6 |
| E3 | Analyse des résultats, compromis biais/variance (sur/sous-apprentissage) | §7 |
| E4 | Description des méthodes d'amélioration du compromis (régularisation, dropout, early stopping…) | §8 |
| E5 | Progression des images faciles vers les difficiles (peintures réalistes en dernier) | §4 |

## 2. Données (dans le dossier du projet)

Cinq archives zip, une par type d'image. Pas d'étiquettes : le **nom de l'archive** est l'étiquette.

| Archive | Dossier interne | Images | Remarques vérifiées |
|---------|-----------------|--------|---------------------|
| `Dataset Livrable 1 - Photo.zip` | `Photo/` | 9 993 jpg | JPEG, tailles variables |
| `Dataset Livrable 1 - Painting.zip` | `Painting/` | 10 000 jpg | JPEG volumineux (~460 Ko en moyenne), certains très grands |
| `Dataset Livrable 1 - Schematics.zip` | `Schematics/` | 10 000 jpg + `desktop.ini` | fichier parasite `desktop.ini` à ignorer |
| `Dataset Livrable 1 - Text.zip` | `Text/` | 10 000 jpg | pages scannées ~595×842 |
| `Dataset Livrable 1 - Sketch.zip` | `Sketch/` | 606 jpg + 800 png (1 406 images) | **niveaux de gris**, PNG 1111×1111 : conversion RGB obligatoire |

`Dataset Livrable 2.zip` (images bruitées) est **hors périmètre** de ce livrable (livrable 2).

Règles de lecture : ne garder que les extensions `.jpg/.jpeg/.png` (insensible à la casse) ; ignorer les entrées de dossier et `desktop.ini`.

## 3. Décisions de conception

### 3.1 Échantillonnage équilibré (contrainte CPU, pas de GPU)

Les classes sont déséquilibrées par construction (1 classe positive contre 4 négatives). On construit :

* **Pool de modélisation** (équilibré 50/50 photo / non-photo), tirage aléatoire reproductible (`SEED = 42`) :
  `Photo` 3000 ; `Painting` 1200 ; `Schematics` 600 ; `Text` 600 ; `Sketch` 600 → 3000 positifs, 3000 négatifs.
  Les peintures sont sur-représentées parmi les négatifs car c'est la classe difficile.
* **Jeu de test étendu « réaliste »** (jamais vu à l'entraînement ni à la validation, **déséquilibré**) :
  1000 images par classe, tirées **hors du pool** ; `Sketch` : tout ce qui reste (1 406 − 600 = 806).
  Il sert à estimer les performances sur un fond documentaire réel où les photos sont minoritaires.
* Les effectifs sont paramétrés dans un dictionnaire de configuration (`POOL_COUNTS`, `EXT_COUNTS`) pour
  pouvoir être augmentés sans changer le code.

### 3.2 Prétraitement et cache

* Chargement avec **ImageIO** (`imageio.v3.imread`), conversion en **RGB** (gris → 3 canaux, suppression de l'alpha),
  redimensionnement **128×128** (Pillow, `Image.BILINEAR`, sans conservation du ratio : l'étirement est un choix assumé et documenté), stockage `uint8`.
* Les images redimensionnées sont stockées **une seule fois** dans un cache `.npz` (≈ 0,5 Go au total) pour que
  les expériences s'enchaînent vite sur CPU. Le redimensionnement se fait donc **hors boucle d'entraînement**.
  (Précision : c'est un écart assumé avec « pipeline tf.data depuis les fichiers » : à 128×128 en `uint8`,
  tout tient en RAM ; décoder 460 Ko de JPEG par image à chaque epoch serait inutilement coûteux.)
* Les fichiers sélectionnés sont **extraits sélectivement** de chaque zip vers `data/raw/<Classe>/` (jamais tout le zip).
* La normalisation (÷255) est faite **dans le modèle** (`Rescaling`), pas dans les données.
* Entrées `tf.data` construites depuis les tableaux NumPy : shuffle (train), batch 64, prefetch.

### 3.3 Découpage

* Pool : split **stratifié par classe source** (5 classes, pas seulement par label binaire) en
  **train 70 % / validation 15 % / test 15 %** (`sklearn.model_selection.train_test_split`, deux passes, `random_state=SEED`).
* Le nom « test » du sujet (courbes « erreur de test ») = jeu de **validation** pendant l'entraînement ;
  le jeu de **test** est utilisé **une seule fois** pour la mesure finale. Le notebook l'explique.

### 3.4 Modèle principal : CNN « maison »

Entrée 128×128×3 → `Rescaling(1/255)` → 4 blocs [`Conv2D 3×3` → `BatchNormalization` → `ReLU` → `MaxPooling2D 2×2`]
avec 16, 32, 64, 128 filtres → `GlobalAveragePooling2D` → `Dense(64, relu)` → (`Dropout` optionnel) → `Dense(1, sigmoid)`.

* Perte : `binary_crossentropy`. Optimiseur : `Adam(learning_rate=1e-3)`. Métriques : `accuracy`, `AUC`.
* Taille volontairement modeste (~100 k paramètres) : entraînable sur CPU, lisible.
* Variantes pour l'étude biais/variance (§7-8) : voir le plan (`build_cnn` paramétrable : nombre de filtres, dropout, L2, augmentation).

### 3.5 Modèle de comparaison : MobileNetV2 gelé

`keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=(128,128,3))` gelé,
`GlobalAveragePooling2D` + `Dense(1, sigmoid)`, entrée mise à l'échelle [-1, 1].
Sert à **justifier le choix d'architecture par les résultats** (§9) : coût/performance du CNN maison vs transfert.
Si le téléchargement des poids ImageNet échoue (pas de réseau), la cellule affiche un message et la comparaison est sautée
(le reste du notebook reste valide).

### 3.6 Progression par difficulté (E5)

Quatre étapes cumulatives, chacune avec **le même `build_cnn`**, la même procédure et la même graine ;
les photos du pool sont toujours opposées à des négatifs de plus en plus difficiles :

1. Photo vs **Text**
2. Photo vs **Text + Schematics**
3. Photo vs **Text + Schematics + Sketch**
4. Photo vs **toutes les classes** (dont **Painting**) → c'est le modèle final

Pour chaque étape : accuracy/F1/AUC sur le test de l'étape, et **accuracy par classe source**.
Le déséquilibre interne des étapes 1-3 est compensé par `class_weight` calculé sur le train.

### 3.7 Évaluation finale

Sur le test du pool (équilibré) **et** sur le test étendu (réaliste) : accuracy, précision, rappel, F1, ROC-AUC,
matrice de confusion, rapport par classe source, galerie des erreurs (en particulier peintures prises pour des photos).
Choix du seuil : 0,5 par défaut ; courbe précision/rappel et discussion du seuil (coût d'un faux « photo »
vs. d'une photo manquée).

## 4. Livrables du dépôt

```
bloc_AI/
├── README.md                               # installation + lancement + structure
├── pyproject.toml                          # dépendances (épinglées) — géré par uv
├── uv.lock                                 # verrouillage complet, versionné
├── .python-version                         # 3.12
├── .gitignore                              # data/, *.zip, .venv/, __pycache__, .ipynb_checkpoints
├── Dataset Livrable *.zip                  # (existant, non modifié)
├── src/tounum_cls/
│   ├── __init__.py
│   ├── config.py                           # constantes, effectifs, chemins, graine
│   ├── data.py                             # inventaire, échantillonnage, extraction, chargement/cache, split, tf.data
│   ├── modeling.py                         # build_cnn, build_mobilenet, entraînement, callbacks
│   └── evaluation.py                       # métriques, courbes, matrices de confusion, erreurs, schéma d'architecture
├── tests/                                  # pytest : data.py, modeling.py, evaluation.py (jeux de données synthétiques)
├── notebooks/
│   ├── livrable1_classification_binaire.py     # source jupytext (format percent), versionnable
│   └── livrable1_classification_binaire.ipynb  # notebook exécuté, sorties incluses (LIVRABLE)
└── docs/superpowers/{specs,plans}/
```

Le code TensorFlow du réseau (E1) est **écrit dans le notebook** (cellule d'architecture), pas caché dans `src/` :
`src/modeling.py` contient la fonction réutilisée par les expériences, et le notebook **l'affiche** (`inspect.getsource`)
et la commente. La logique non spécifique à la pédagogie (I/O, découpage, métriques) est dans `src/` et testée.

## 5. Environnement (géré par **uv**, exigence explicite du client)

Le venv est **créé et géré exclusivement par [uv](https://docs.astral.sh/uv/)** pour éviter tout conflit de paquets :

* `pyproject.toml` déclare `requires-python = ">=3.12,<3.13"` et les dépendances **épinglées** (`==`) :
  tensorflow 2.21.0 (Keras 3.15.1), numpy, pandas, scikit-learn, imageio, pillow, matplotlib, seaborn, jupytext, nbconvert,
  nbformat, ipykernel ; groupe de développement : pytest. `[tool.uv] package = false` (le code de `src/` n'est pas installé comme paquet ; il est importé via `sys.path` / `pytest.ini`).
* `uv.lock` (résolution complète, transitives comprises) est **versionné** : `uv sync` reconstruit le même environnement partout.
* `.python-version` = `3.12` ; `uv sync` crée `.venv/` à la racine (et télécharge Python 3.12 si nécessaire).
* Toutes les commandes passent par `uv run ...` (ex. `uv run pytest -q`). **Interdit** : `pip install`, `python -m venv`, `conda`. Ajouter une dépendance = `uv add <paquet>`.
* **CPU uniquement** (aucun GPU détecté). Python 3.14 (système) n'est pas utilisé : TensorFlow n'y est pas disponible.
* Vérifié le 2026-10-01 : `uv 0.11.16`, `uv sync` en ≈ 20 s, Python 3.12.13, TensorFlow 2.21.0, Keras 3.15.1.

## 6. Critères d'acceptation

1. `uv sync` depuis un dossier sans `.venv/` reconstruit l'environnement, puis `uv run pytest -q` passe sans erreur.
2. `jupytext --to ipynb` puis `jupyter nbconvert --execute` exécute le notebook **de bout en bout sans erreur**, depuis un `data/` vide
   (extraction + cache reconstruits), en un temps raisonnable sur CPU (cible < 45 min ; mesuré et noté dans le README).
3. Toutes les exigences E1 à E5 sont présentes et visibles dans le `.ipynb` exécuté (graphiques inclus).
4. Les commentaires d'analyse (§7, §9, conclusion) citent les **chiffres réellement obtenus** (pas de valeurs inventées).
5. Résultats reproductibles : relancer avec la même graine redonne des chiffres proches (l'ordre non déterministe de TF CPU peut
   causer de petites variations ; documenté).
6. Aucune donnée brute ni cache n'est commit-ée (`.gitignore`).

## 7. Hors périmètre

Débruitage / filtres de convolution (livrable 2), captioning (livrable 3), déploiement, GPU, recherche d'hyperparamètres automatique,
classification multi-classes des 5 types (on reste binaire).

## 8. Risques et parades

| Risque | Parade |
|--------|--------|
| Peintures réalistes confondues avec des photos | Étape 4 dédiée, sur-représentation des peintures, galerie d'erreurs, analyse, piste d'amélioration (augmentation, résolution, transfert) |
| Fuite de données (même image dans train et test) | Tirages disjoints par nom de fichier ; assertions dans le code (`assert set(a) & set(b) == set()`) |
| Image corrompue | `try/except` dans le chargement, comptée et journalisée, fichier écarté du cache |
| Entraînement trop long sur CPU | Taille 128×128, modèle ~100 k paramètres, effectifs paramétrables, cache |
| Poids ImageNet indisponibles | Comparaison MobileNetV2 optionnelle, message explicite |
| Biais de la liste d'images (ex. `desktop.ini`) | Filtre d'extension, test unitaire |
