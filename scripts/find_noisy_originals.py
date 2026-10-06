"""Find the clean COCO original of each noisy image of 'Dataset Livrable 2.zip' and write the mapping.

The noisy images are COCO train2017 photos (same sizes, consecutive ids). This script downloads every
train2017 image with an id below MAX_ID, compares 16x16 grayscale thumbnails and keeps the nearest one.
It only needs to run once: its output, src/tounum_denoise/noisy_originals.csv, is versioned.

Usage: PYTHONPATH=src uv run python scripts/find_noisy_originals.py
"""
import numpy as np
import pandas as pd
from PIL import Image

from tounum_caption import config as cap_config
from tounum_caption import data as cap_data
from tounum_denoise import config, data

MAX_ID = 3000
OUT = config.PROJECT_ROOT / "src" / "tounum_denoise" / "noisy_originals.csv"


def thumbnail(path) -> np.ndarray:
    with Image.open(path) as im:
        im.draft("L", (64, 64))
        t = np.asarray(im.convert("L").resize((16, 16), Image.BILINEAR), np.float32).ravel()
    return t - t.mean()


cap_data.download_annotations()
candidates = cap_data.load_captions("train2017").drop_duplicates("image_id")
candidates = candidates[candidates["image_id"] < MAX_ID].reset_index(drop=True)
candidates["path"] = [str(cap_config.COCO_DIR / "train2017" / f) for f in candidates["file_name"]]
failed = cap_data.download_images(candidates)
assert not failed, failed

noisy = data.extract_noisy()
N = np.stack([thumbnail(p) for p in noisy])
C = np.stack([thumbnail(p) for p in candidates["path"]])
dist = ((N[:, None, :] - C[None]) ** 2).mean(axis=2)
best = dist.argmin(axis=1)
out = pd.DataFrame({
    "file": [p.name for p in noisy],
    "coco_id": candidates["image_id"].to_numpy()[best],
    "coco_file": candidates["file_name"].to_numpy()[best],
    "thumbnail_distance": dist[np.arange(len(noisy)), best].round(3),
})
print(out["thumbnail_distance"].describe())
assert (out["thumbnail_distance"] < 5).all(), "some noisy images have no convincing match"
out.to_csv(OUT, index=False)
print(f"wrote {OUT} ({len(out)} rows)")
