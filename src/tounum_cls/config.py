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
