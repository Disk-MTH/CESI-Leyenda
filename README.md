# TouNum : légendage automatique de photos numérisées

Équipe : Mathieu, Matthis et Julien.

Projet Leyenda, CESI. Ce dépôt contient le rendu de notre groupe pour TouNum, une entreprise de numérisation de
documents qui nous a demandé un prototype de légendage automatique d'images. Notre solution prend en entrée un
dossier d'images numérisées et produit un tableau donnant, pour chaque photo, une phrase qui la décrit. Le workflow
que nous avons conçu comporte trois modules, un par livrable :

1. **Filtrage** (livrable 1) : un classifieur garde les photos et écarte les peintures, schémas, dessins et textes
   scannés.
2. **Prétraitement** (livrable 2) : un autoencodeur convolutif (U-Net à sortie résiduelle), entraîné sur un bruit de
   couleur synthétique puis fine-tuné sur de vraies paires bruitée / propre mélangées à des photos propres et à du
   bruit synthétique, retire le bruit des photos qui en ont besoin.
3. **Captioning** (livrable 3) : un CNN pré-entraîné sur ImageNet (InceptionV3) encode la photo et un réseau
   récurrent (GRU) écrit la légende mot à mot. Nous comparons des versions avec et sans attention et gardons celle
   qui a le meilleur score de validation. Entraînement sur 80 000 images de MS COCO 2017.

Chaque livrable est un notebook Jupyter exécuté, qui explique notre démarche et présente les résultats :

| Livrable | Notebook | Modèles sauvegardés |
|---|---|---|
| 1. Classification binaire photo / non-photo | `notebooks/livrable1_classification_binaire.ipynb` | `models/photo_classifier.keras` (production, MobileNetV2), `models/photo_cnn.keras` (notre CNN) |
| 2. Traitement d'images : débruitage par autoencodeurs convolutifs | `notebooks/livrable2_traitement_images.ipynb` | `models/denoiser.keras`, `models/denoiser.json` |
| 3. Captioning d'images et workflow complet | `notebooks/livrable3_captioning.ipynb` | `models/captioner.keras`, `models/captioner.json` |

Les légendes produites sont en anglais, car MS COCO est annoté en anglais ; le livrable 3 en discute dans ses
limites.

## Organisation du dépôt

| Chemin | Contenu |
|---|---|
| `notebooks/*.ipynb` | notebooks exécutés (les livrables) |
| `notebooks/*.py` | sources des notebooks au format jupytext « percent » : c'est ce fichier qu'on modifie, jamais le `.ipynb` |
| `src/tounum_cls/` | livrable 1 : échantillonnage des archives, chargement des images, CNN et MobileNetV2, métriques et graphiques |
| `src/tounum_denoise/` | livrable 2 : dataset bruité, bruit synthétique, autoencodeurs, filtres classiques, estimation du bruit |
| `src/tounum_caption/` | livrable 3 : données COCO, features InceptionV3, modèle de captioning, BLEU, et le pipeline de bout en bout `pipeline.py` |
| `src/tounum_gpu.py` | permet à TensorFlow d'utiliser un GPU NVIDIA (voir « GPU » plus bas) |
| `tests/` | tests unitaires (`pytest`) sur des données synthétiques |
| `scripts/run_notebook.sh` | reconstruit et exécute un notebook (`quick` ou `full`) |
| `scripts/check_notebook.py` | vérifie qu'un notebook exécuté est complet |
| `scripts/caption_folder.py` | lance le workflow complet sur un dossier, en ligne de commande |
| `scripts/find_noisy_originals.py` | retrouve les originaux COCO propres des photos bruitées du livrable 2 (lancé une fois) |
| `pyproject.toml`, `uv.lock`, `.python-version` | dépendances figées, gérées par uv |
| `.vscode/` | tâches VS Code (tests, exécution des notebooks, vérifications, légendage d'un dossier) et configurations de débogage |
| `docs/` | notes de conception et plan d'implémentation rédigés pour le livrable 1 |
| `Dataset Livrable *.zip` | archives fournies par TouNum (non versionnées) |
| `data/` | images extraites, caches, échantillon COCO et features (non versionnés, reconstruits par les notebooks) |

## Installation

Prérequis : [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`), Python 3.12
(installé par uv si besoin), environ 50 Go d'espace disque libre pour `data/` (images COCO et leurs features
InceptionV3), et 48 Go de RAM pour le livrable 3 : ses features d'entraînement (21 Go) sont gardées en mémoire et le
notebook utilise environ 30 Go au maximum. Une connexion internet est nécessaire à la première exécution des
livrables 2 et 3 (images et légendes COCO, poids ImageNet).

```bash
uv sync        # crée .venv/ avec les versions exactes figées dans uv.lock
```

L'environnement est géré uniquement par uv : pas de `pip install`, pas de `python -m venv`. Toutes les commandes
passent par `uv run`. Pour ajouter une dépendance : `uv add <paquet>` (met à jour `pyproject.toml` et `uv.lock`, à
committer tous les deux).

Les archives fournies par TouNum se placent à la racine du projet : les cinq
`Dataset Livrable 1 - {Photo,Painting,Schematics,Text,Sketch}.zip` et `Dataset Livrable 2.zip`.

### GPU

Sous Linux x86_64, `uv sync` installe aussi les bibliothèques CUDA sous forme de paquets Python
(`tensorflow[and-cuda]`) : aucune installation de CUDA sur le système n'est nécessaire, seulement le pilote NVIDIA.
Sans GPU (ou sous macOS / Windows), TensorFlow tourne sur le CPU et tout fonctionne quand même, plus lentement.

Deux détails, tous deux gérés par `src/tounum_gpu.py`, que les notebooks, les tests et les scripts appellent avant
d'importer TensorFlow :

- Le wheel TensorFlow 2.21.0 n'a pas cuSOLVER dans son chemin de recherche des bibliothèques, et retombe alors sans
  prévenir sur le CPU. `tounum_gpu.setup()` charge cette bibliothèque en premier.
- TensorFlow 2.21 n'a pas de noyaux compilés pour les GPU les plus récents (RTX 50xx, compute capability 12.0). Le
  pilote les compile à la première exécution, ce qui prend quelques minutes une seule fois, et les garde dans un
  cache (`~/.nv/ComputeCache`) que `setup()` agrandit pour qu'ils ne soient pas recompilés.

Vérification : `uv run python -c "import sys; sys.path.insert(0, 'src'); import tounum_gpu; tounum_gpu.setup(); import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

## Exécuter les notebooks

Les notebooks dépendent les uns des autres : le livrable 2 réutilise le cache d'images et le classifieur du
livrable 1, et le livrable 3 réutilise les modèles des livrables 1 et 2. Il faut donc les lancer dans l'ordre.

```bash
uv run pytest -q                        # tests unitaires

bash scripts/run_notebook.sh 1 quick    # exécution courte (peu d'époques) : écrit seulement notebooks/*.quick.ipynb
bash scripts/run_notebook.sh 1 full     # exécution complète = le livrable (.ipynb + modèles sauvegardés)
bash scripts/run_notebook.sh 2 full
bash scripts/run_notebook.sh 3 full

uv run python scripts/check_notebook.py notebooks/livrable1_classification_binaire.ipynb   # vérification
```

Le mode quick ne touche jamais aux livrables ni à `models/` (ses modèles vont dans un dossier temporaire). Pour
ouvrir un notebook en mode interactif : `uv run --with jupyterlab jupyter lab`, ou l'ouvrir dans VS Code avec
l'interpréteur de `.venv`.

Sur une RTX 5090, une exécution complète prend quelques minutes pour le livrable 1 et jusqu'à une heure environ pour
les livrables 2 et 3. La première exécution du livrable 3 télécharge aussi environ 82 000 images COCO (une quinzaine
de minutes) et calcule leurs features. Sans GPU, les livrables 2 et 3 prennent de nombreuses heures ; on peut alors
réduire `N_TRAIN_IMAGES` dans `src/tounum_caption/config.py` et les époques en tête des notebooks.

## Légender un nouveau dossier

En ligne de commande :

```bash
uv run python scripts/caption_folder.py chemin/vers/dossier --out captions.csv
```

Depuis le notebook du livrable 3 : la dernière cellule est autonome. Il suffit de mettre le dossier dans
`INPUT_DIR` et de lancer cette cellule seule ; elle charge les modèles sauvegardés et affiche le tableau et une
galerie des résultats. C'est la cellule que nous utiliserons le jour de la soutenance sur le dataset fourni.

Depuis Python :

```python
import sys; sys.path.insert(0, "src")
import tounum_gpu; tounum_gpu.setup()
from tounum_caption.pipeline import CaptionPipeline

pipeline = CaptionPipeline.load()                 # dossier models/ des trois livrables
results = pipeline.run("chemin/vers/dossier")     # DataFrame pandas, une ligne par fichier image
# colonnes : file, path, p_photo, is_photo, noise_sigma, denoised, caption
```

Les images sont cherchées récursivement (jpg, jpeg, png). Les non-photos ne reçoivent pas de légende ; les fichiers
illisibles sont signalés dans la colonne `caption` au lieu d'interrompre l'exécution.

## Maintenance

- **Réglages** : ils se trouvent dans le `config.py` de chaque paquet (tailles d'échantillon, tailles d'image, plage
  de bruit, taille du vocabulaire...). Après avoir changé une taille d'échantillon, supprimer le cache
  correspondant (`data/cache/`, `data/denoise_cache/` ou `data/features/`) et relancer le notebook.
- **Modifier un notebook** : modifier le fichier `.py`, puis `bash scripts/run_notebook.sh <n> full`.
- **Chiffres des notebooks** : ils sont calculés. Les cellules qui commencent par « Résultats. » sont reconstruites
  à chaque exécution, et le texte fixe ne cite pas de chiffres qu'une nouvelle exécution pourrait changer.
- **Reproductibilité** : la graine `SEED = 42` est utilisée partout. TensorFlow peut tout de même donner des chiffres
  légèrement différents d'une exécution ou d'une machine à l'autre (surtout sur GPU), donc les dernières décimales
  peuvent bouger.
