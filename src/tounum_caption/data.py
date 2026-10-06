"""COCO download, sampling, caption cleaning and vocabulary."""
from __future__ import annotations

import json
import re
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


# --------------------------------------------------------------------------- download
def _download(url: str, dest: Path, retries: int = 3) -> bool:
    """Download `url` to `dest` through a temporary file, so an interrupted download never leaves a partial file."""
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for _ in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                tmp.write_bytes(resp.read())
            tmp.rename(dest)
            return True
        except OSError:
            continue
    tmp.unlink(missing_ok=True)
    return False


def download_annotations(coco_dir: Path = config.COCO_DIR) -> Path:
    """Download and unzip the COCO 2017 caption files (about 250 MB, done once)."""
    ann_dir = coco_dir / "annotations"
    if not (ann_dir / "captions_train2017.json").exists():
        zip_path = coco_dir / "annotations_trainval2017.zip"
        if not _download(f"{config.COCO_URL}/{config.ANNOTATIONS_ZIP}", zip_path):
            raise RuntimeError("could not download the COCO annotations")
        with zipfile.ZipFile(zip_path) as zf:
            for name in ("annotations/captions_train2017.json", "annotations/captions_val2017.json"):
                zf.extract(name, coco_dir)
    return ann_dir


def load_captions(coco_split: str, coco_dir: Path = config.COCO_DIR) -> pd.DataFrame:
    """One row per caption: image_id, file_name, coco_split, caption (raw text)."""
    with open(coco_dir / "annotations" / f"captions_{coco_split}.json") as f:
        raw = json.load(f)
    images = pd.DataFrame(raw["images"])[["id", "file_name"]].rename(columns={"id": "image_id"})
    caps = pd.DataFrame(raw["annotations"])[["image_id", "caption"]]
    out = caps.merge(images, on="image_id")
    out["coco_split"] = coco_split
    return out.sort_values(["image_id", "caption"]).reset_index(drop=True)


def sample_images(
    n_train: int = config.N_TRAIN_IMAGES,
    n_val: int = config.N_VAL_IMAGES,
    n_test: int = config.N_TEST_IMAGES,
    seed: int = config.SEED,
    coco_dir: Path = config.COCO_DIR,
) -> pd.DataFrame:
    """Reproducible image sample. One row per image: image_id, file_name, coco_split, split, path.

    train and val are drawn without overlap from train2017, test from val2017.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for coco_split, parts in (("train2017", [("train", n_train), ("val", n_val)]), ("val2017", [("test", n_test)])):
        images = load_captions(coco_split, coco_dir).drop_duplicates("image_id")   # already sorted by image_id
        order = rng.permutation(len(images))
        start = 0
        for split, n in parts:
            for i in order[start:start + n]:
                row = images.iloc[i]
                rows.append({"image_id": int(row.image_id), "file_name": row.file_name,
                             "coco_split": coco_split, "split": split})
            start += n
    out = pd.DataFrame(rows)
    out["path"] = [str(coco_dir / s / f) for s, f in zip(out["coco_split"], out["file_name"])]
    return out


def exclude_images(images: pd.DataFrame, image_ids) -> pd.DataFrame:
    """Drop some images from a sample without drawing it again (the other images keep their split)."""
    return images[~images["image_id"].isin(set(image_ids))].reset_index(drop=True)


def captions_for(images: pd.DataFrame, coco_dir: Path = config.COCO_DIR) -> pd.DataFrame:
    """Captions of the sampled images, one row per caption.

    Columns: image_id, caption (raw), split, image_index (row of the image within its split, which is also its
    row in the feature array of that split) and clean (cleaned caption with the start and end markers).
    """
    caps = pd.concat([load_captions(s, coco_dir) for s in sorted(images["coco_split"].unique())])
    index = images[["image_id", "split"]].copy()
    index["image_index"] = index.groupby("split").cumcount()
    out = caps[["image_id", "caption"]].merge(index, on="image_id")
    out["clean"] = out["caption"].map(clean_caption)
    return out.sort_values(["split", "image_index", "caption"]).reset_index(drop=True)


def references(captions: pd.DataFrame, split: str) -> list[list[str]]:
    """Reference captions of every image of a split (cleaned, without the markers), in feature order."""
    sub = captions[captions["split"] == split]
    return [[" ".join(c.split()[1:-1]) for c in grp] for _, grp in sub.groupby("image_index")["clean"]]


def download_images(images: pd.DataFrame, workers: int = 16) -> list[str]:
    """Download the sampled images one by one (parallel, skips files already on disk).

    Returns the paths that could not be downloaded.
    """
    def _one(row):
        ok = _download(f"{config.COCO_URL}/{row.coco_split}/{row.file_name}", Path(row.path))
        return None if ok else row.path

    with ThreadPoolExecutor(max_workers=workers) as ex:
        failed = [p for p in ex.map(_one, images.itertuples(index=False)) if p]
    return failed


# --------------------------------------------------------------------------- text
_NON_ALPHA = re.compile(r"[^a-z ]+")
_SPACES = re.compile(r"\s+")


def clean_caption(text: str) -> str:
    """Lowercase, keep letters only, add the start and end markers.

    >>> clean_caption("A man riding a wave on top of a surfboard.")
    'startseq a man riding a wave on top of a surfboard endseq'
    """
    text = _NON_ALPHA.sub(" ", text.lower().replace("-", " "))
    text = _SPACES.sub(" ", text).strip()
    return f"{config.START} {text} {config.END}"


def build_vocabulary(captions, vocab_size: int = config.VOCAB_SIZE) -> list[str]:
    """Vocabulary from cleaned training captions, most frequent first.

    Index 0 is padding ('') and index 1 is the unknown word ('[UNK]'), as in keras TextVectorization.
    """
    counts: dict[str, int] = {}
    for cap in captions:
        for word in cap.split():
            counts[word] = counts.get(word, 0) + 1
    words = sorted(counts, key=lambda w: (-counts[w], w))[: vocab_size - 2]
    return ["", "[UNK]", *words]


def encode(captions, vocab: list[str], max_len: int = config.MAX_LEN) -> np.ndarray:
    """Cleaned captions -> int32 array (n, max_len), zero padded. Long captions are cut but keep END."""
    index = {w: i for i, w in enumerate(vocab)}
    out = np.zeros((len(captions), max_len), dtype=np.int32)
    for row, cap in enumerate(captions):
        ids = [index.get(w, 1) for w in cap.split()]
        if len(ids) > max_len:
            ids = ids[: max_len - 1] + [index[config.END]]
        out[row, : len(ids)] = ids
    return out


def decode(ids, vocab: list[str]) -> str:
    """Token ids -> text, without padding and markers, stopping at END."""
    words = []
    for i in ids:
        word = vocab[int(i)]
        if word == config.END:
            break
        if word and word != config.START:
            words.append(word)
    return " ".join(words)
