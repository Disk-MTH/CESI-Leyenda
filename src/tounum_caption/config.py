"""Settings for the captioning module. Everything that can be tuned lives here."""
from pathlib import Path

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
COCO_DIR = DATA_DIR / "coco"                    # annotations/ + train2017/ + val2017/
FEATURES_DIR = DATA_DIR / "features"            # cached InceptionV3 features
MODELS_DIR = PROJECT_ROOT / "models"

COCO_URL = "http://images.cocodataset.org"
ANNOTATIONS_ZIP = "annotations/annotations_trainval2017.zip"

# Images sampled from COCO 2017. Train and validation come from train2017, the test set from val2017,
# so test images are never seen during training or model selection.
N_TRAIN_IMAGES = 80_000                         # 28,000 in a first CPU-sized version (see notebook, section 6)
N_VAL_IMAGES = 2_000
N_TEST_IMAGES = 2_000

# Image encoder: InceptionV3 pre-trained on ImageNet, last convolutional block (8 x 8 x 2048).
FEATURE_IMG_SIZE = 299
N_REGIONS = 64
FEATURE_DIM = 2048

# Text
START, END = "startseq", "endseq"
VOCAB_SIZE = 5_000                               # most frequent words; the rest map to [UNK]
MAX_LEN = 22                                     # tokens per caption, START and END included

# Model and training
EMBED_DIM = 256
UNITS = 512
BATCH_SIZE = 64
