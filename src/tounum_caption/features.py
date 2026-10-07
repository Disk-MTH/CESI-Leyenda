"""Image encoder: InceptionV3 pre-trained on ImageNet, used frozen to turn each image into 64 region vectors."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import keras
import numpy as np
from keras import layers

from tounum_cls import data as cls_data

from . import config


def build_encoder(weights: str | None = "imagenet") -> keras.Model:
    """InceptionV3 gelé, sans sa tête de classification.

    Entrée : images RGB uint8 (299, 299, 3) ; la mise à l'échelle [-1, 1] attendue par InceptionV3 est faite dans
    le modèle. Sortie : (64, 2048) float32, la grille 8 x 8 du dernier bloc convolutif aplatie en 64 régions.
    `weights=None` (tests) évite de télécharger les poids ImageNet.
    """
    size = config.FEATURE_IMG_SIZE
    base = keras.applications.InceptionV3(include_top=False, weights=weights, input_shape=(size, size, 3))
    base.trainable = False
    inputs = keras.Input(shape=(size, size, 3), name="image")
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="rescaling")(inputs)
    x = base(x, training=False)
    outputs = layers.Reshape((config.N_REGIONS, config.FEATURE_DIM), name="regions")(x)
    return keras.Model(inputs, outputs, name="inceptionv3_encoder")


def prepare(img: np.ndarray) -> np.ndarray:
    """uint8 RGB image of any size -> (299, 299, 3) uint8, stretched like in deliverable 1."""
    return cls_data.resize(img, config.FEATURE_IMG_SIZE)


def load_image(path: str | Path) -> np.ndarray:
    """Read an image file with ImageIO and prepare it for the encoder."""
    return prepare(cls_data.read_rgb(path))


def encode(encoder: keras.Model, images: np.ndarray, batch_size: int = 64) -> np.ndarray:
    """Region features (n, 64, 2048) float16 of a batch of prepared uint8 images."""
    return encoder.predict(np.asarray(images), batch_size=batch_size, verbose=0).astype(np.float16)


def extract_features(encoder: keras.Model, paths, out_path: Path, chunk: int = 1024, workers: int = 16) -> np.ndarray:
    """Encode every image file of `paths` and cache the result in `out_path` (.npy, float16).

    Later calls read the cache back (delete the file after changing the sample: only the number of images is
    checked). The array is written to a temporary file and renamed at the end, so an interrupted run never leaves a
    truncated cache behind.
    """
    out_path = Path(out_path)
    paths = [str(p) for p in paths]
    if out_path.exists():
        cached = np.load(out_path, mmap_mode="r")
        if len(cached) == len(paths):
            return np.load(out_path)
        out_path.unlink()                         # cache made for another sample
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.stem + ".part.npy")
    feats = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16,
                                      shape=(len(paths), config.N_REGIONS, config.FEATURE_DIM))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for start in range(0, len(paths), chunk):
            images = np.stack(list(ex.map(load_image, paths[start:start + chunk])))
            feats[start:start + len(images)] = encode(encoder, images)
    feats.flush()
    del feats
    tmp.rename(out_path)
    return np.load(out_path)
