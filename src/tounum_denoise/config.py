"""Settings for the denoising module. Everything that can be tuned lives here."""
from pathlib import Path

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOISY_ZIP = PROJECT_ROOT / "Dataset Livrable 2.zip"    # 148 noisy RGB photos provided by TouNum
DATA_DIR = PROJECT_ROOT / "data"
NOISY_DIR = DATA_DIR / "noisy"                          # extracted noisy images
CACHE_DIR = DATA_DIR / "denoise_cache"                  # training crops (npy)
MODELS_DIR = PROJECT_ROOT / "models"

# Clean training images: the photos of deliverable 1 (train and val splits only, so the deliverable 1 test
# photos stay unseen and can be used for the case study).
CLEAN_SHORT_SIDE = 320          # photos are downscaled so their short side is 320 px before cropping
CROP_SIZE = 96                  # training patch size
CROPS_PER_IMAGE = 4

# Clean originals of the noisy images (found in COCO train2017, see scripts/find_noisy_originals.py)
ORIGINALS_CSV = PROJECT_ROOT / "src" / "tounum_denoise" / "noisy_originals.csv"
ORIGINALS_DIR = DATA_DIR / "noisy_originals"
COCO_URL = "http://images.cocodataset.org/train2017"

# Synthetic noise used for training, modelled on the real pairs: Gaussian noise drawn independently for each
# pixel and channel, sigma drawn uniformly in SIGMA_RANGE (0-255 scale), then a JPEG re-encoding at quality 95
# without chroma subsampling (the format of the noisy files).
SIGMA_RANGE = (0.0, 30.0)
JPEG_QUALITY = 95
SIGMA_LEVELS = (0, 5, 10, 20, 30)   # fixed levels used for evaluation

BATCH_SIZE = 32
