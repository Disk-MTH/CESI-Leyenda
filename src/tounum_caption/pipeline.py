"""The whole TouNum workflow on a folder of images: photo filter (deliverable 1) -> denoising of the noisy photos
(deliverable 2) -> caption (deliverable 3).

    from tounum_caption.pipeline import CaptionPipeline
    pipeline = CaptionPipeline.load()            # models/ of the three deliverables
    results = pipeline.run("path/to/folder")     # one row per image file
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import keras
import numpy as np
import pandas as pd

from tounum_cls import data as cls_data, modeling as cls_modeling
from tounum_denoise import data as dn_data, modeling as dn_modeling

from . import config, features, model as cap_model

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def list_images(folder: str | Path) -> list[Path]:
    """Image files of a folder and its sub-folders (jpg/jpeg/png), sorted, hidden files skipped."""
    return sorted(p for p in Path(folder).rglob("*")
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith("."))


@dataclass
class CaptionPipeline:
    classifier: keras.Model             # deliverable 1: P(photo) from a 128x128 image
    denoiser: keras.Model               # deliverable 2: noisy -> clean, any image size
    noise_threshold: float              # deliverable 2: denoise when the estimated noise sigma is at least this
    encoder: keras.Model                # InceptionV3, 299x299 image -> 64 regions x 2048
    captioner: keras.Model              # deliverable 3: GRU decoder with attention
    vocab: list[str]
    max_len: int = config.MAX_LEN
    photo_threshold: float = 0.5
    timings: dict = field(default_factory=dict)

    @classmethod
    def load(cls, models_dir: str | Path = config.MODELS_DIR, encoder: keras.Model | None = None) -> "CaptionPipeline":
        """Load the saved models of the three deliverables (and InceptionV3, unless an encoder is given)."""
        models_dir = Path(models_dir)
        denoise_settings = json.loads((models_dir / "denoiser.json").read_text())
        caption_settings = json.loads((models_dir / "captioner.json").read_text())
        return cls(
            classifier=keras.models.load_model(models_dir / "photo_classifier.keras"),
            denoiser=keras.models.load_model(models_dir / "denoiser.keras"),
            noise_threshold=float(denoise_settings["noise_threshold"]),
            encoder=encoder or features.build_encoder(),
            captioner=keras.models.load_model(models_dir / "captioner.keras"),
            vocab=caption_settings["vocab"],
            max_len=int(caption_settings["max_len"]),
        )

    def run(self, inputs, chunk: int = 64) -> pd.DataFrame:
        """Lance le workflow sur un dossier (ou une liste de fichiers image).

        Renvoie une ligne par fichier : file, path, p_photo, is_photo, noise_sigma, denoised, caption.
        Les non-photos n'ont pas de légende ; les fichiers illisibles sont signalés dans la colonne `caption`.
        """
        paths = list_images(inputs) if isinstance(inputs, (str, Path)) else [Path(p) for p in inputs]
        rows, photo_feats, photo_rows = [], [], []
        t = {"lecture + filtre": 0.0, "débruitage": 0.0, "encodage": 0.0, "légende": 0.0}
        for start in range(0, len(paths), chunk):
            t0 = time.perf_counter()
            images, batch_rows = [], []
            for p in paths[start:start + chunk]:
                row = {"file": p.name, "path": str(p), "p_photo": np.nan, "is_photo": False,
                       "noise_sigma": np.nan, "denoised": False, "caption": ""}
                try:
                    images.append(cls_data.read_rgb(p))
                    batch_rows.append(row)
                except Exception as exc:          # fichier corrompu ou non pris en charge : signalé, pas bloquant
                    row["caption"] = f"[illisible : {type(exc).__name__}]"
                rows.append(row)
            if not images:
                continue
            proba = cls_modeling.predict_proba(self.classifier, np.stack([cls_data.resize(im) for im in images]))
            t["lecture + filtre"] += time.perf_counter() - t0
            prepared = []
            for img, row, p in zip(images, batch_rows, proba):
                row["p_photo"], row["is_photo"] = float(p), bool(p >= self.photo_threshold)
                if not row["is_photo"]:
                    continue
                t0 = time.perf_counter()
                row["noise_sigma"] = dn_data.estimate_noise(img)
                if row["noise_sigma"] >= self.noise_threshold:
                    img, row["denoised"] = dn_modeling.denoise(self.denoiser, img), True
                t["débruitage"] += time.perf_counter() - t0
                prepared.append(features.prepare(img))
                photo_rows.append(row)
            if prepared:
                t0 = time.perf_counter()
                photo_feats.append(features.encode(self.encoder, np.stack(prepared)))
                t["encodage"] += time.perf_counter() - t0
        if photo_rows:
            t0 = time.perf_counter()
            captions = cap_model.generate(self.captioner, np.concatenate(photo_feats), self.vocab, self.max_len)
            for row, caption in zip(photo_rows, captions):
                row["caption"] = caption
            t["légende"] += time.perf_counter() - t0
        self.timings = t
        return pd.DataFrame(rows, columns=["file", "path", "p_photo", "is_photo", "noise_sigma", "denoised", "caption"])
