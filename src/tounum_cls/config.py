"""Project constants. Everything that can be tuned lives here."""
from pathlib import Path

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ZIP_DIR = PROJECT_ROOT                       # the archives sit at the project root
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                   # extracted images: data/raw/<Class>/<file>
CACHE_DIR = DATA_DIR / "cache"               # resized arrays: X.npy + meta.csv
MODELS_DIR = PROJECT_ROOT / "models"

IMG_SIZE = 128
BATCH_SIZE = 64

POSITIVE_CLASS = "Photo"                     # label 1; every other class is label 0
CLASSES = ["Photo", "Painting", "Schematics", "Text", "Sketch"]
ZIP_FILES = {c: f"Dataset Livrable 1 - {c}.zip" for c in CLASSES}
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Modelling pool (balanced: 3000 photos / 3000 non-photos)
POOL_COUNTS = {"Photo": 3000, "Painting": 1200, "Schematics": 600, "Text": 600, "Sketch": 600}
# Extended test set, closer to a real archive (imbalanced). None = everything left after the pool.
EXT_COUNTS = {"Photo": 1000, "Painting": 1000, "Schematics": 1000, "Text": 1000, "Sketch": None}

# Stages of increasing difficulty: negative classes put against the photos
STAGES = {
    "1_photo_vs_text": ["Text"],
    "2_+schematics": ["Text", "Schematics"],
    "3_+sketch": ["Text", "Schematics", "Sketch"],
    "4_+painting": ["Text", "Schematics", "Sketch", "Painting"],
}
