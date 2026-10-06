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
CLEAN_SHORT_SIDE = 480          # larger photos are downscaled to this short side; the usual 640x480 ones keep their resolution
CROP_SIZE = 96                  # training patch size
CROPS_PER_IMAGE = 8             # 4 was the practical limit on a CPU; the GPU makes 8 affordable

# Clean originals of the noisy images (found in COCO train2017, see scripts/find_noisy_originals.py)
ORIGINALS_CSV = PROJECT_ROOT / "src" / "tounum_denoise" / "noisy_originals.csv"
ORIGINALS_DIR = DATA_DIR / "noisy_originals"
COCO_URL = "http://images.cocodataset.org/train2017"

# Synthetic noise used for training, modelled on the real pairs (notebook, section 3): Gaussian noise on the two
# colour channels (Cb and Cr of the YCbCr space) only, brightness (Y) untouched, slightly blurred because the real
# noise is not independent from one pixel to the next. Its standard deviation (0-255 scale) is drawn uniformly in
# SIGMA_RANGE for each image, then the image is re-encoded as JPEG quality 95 without chroma subsampling (the format
# of the noisy files).
SIGMA_RANGE = (0.0, 20.0)
NOISE_BLUR = 0.5                    # standard deviation, in pixels, of the Gaussian blur applied to the noise
JPEG_QUALITY = 95
SIGMA_LEVELS = (0, 5, 10, 15, 20)   # fixed levels used for evaluation

BATCH_SIZE = 32

# Real (noisy, clean) pairs: the 148 TouNum images and their originals, split once (seeded) into fine-tuning,
# validation and test images. The test images are only used to measure the final models.
REAL_VAL, REAL_TEST = 20, 50                # the other 78 images are used for fine-tuning
REAL_CROPS_PER_IMAGE = 128                  # 96x96 (noisy, clean) crops per fine-tuning image
CLEAN_FT_PHOTOS = 1000                      # clean photos added to the fine-tuning (input = target), 10 crops each
FINE_TUNE_LEARNING_RATE = 1e-4              # 10 times lower than the first training, to adapt without forgetting
