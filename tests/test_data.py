import os

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from tounum_cls import config, data


def test_list_zip_images_filters_parasites(make_zip):
    z = make_zip("Schematics", 5, extra_png_gray=2)
    names = data.list_zip_images(z)
    assert len(names) == 7
    assert all(n.lower().endswith((".jpg", ".png")) for n in names)
    assert not any("desktop.ini" in n or n.endswith("/") or n.endswith(".txt") for n in names)
    assert names == sorted(names)


def _inventory(n_each=20):
    return {c: [f"{c}/{c.lower()}_{i:03d}.jpg" for i in range(n_each)] for c in config.CLASSES}


def test_sample_dataset_disjoint_counts_and_labels():
    pool = {"Photo": 6, "Painting": 3, "Schematics": 2, "Text": 2, "Sketch": 2}
    ext = {"Photo": 4, "Painting": 4, "Schematics": 4, "Text": 4, "Sketch": None}
    df = data.sample_dataset(_inventory(), pool, ext, seed=1)
    assert df["member"].is_unique
    for c in config.CLASSES:
        sub = df[df["source"] == c]
        assert (sub["subset"] == "pool").sum() == pool[c]
        want_ext = 20 - pool[c] if ext[c] is None else ext[c]
        assert (sub["subset"] == "ext").sum() == want_ext
    assert set(df.loc[df["source"] == "Photo", "label"]) == {1}
    assert set(df.loc[df["source"] != "Photo", "label"]) == {0}
    # reproductible
    pd.testing.assert_frame_equal(df, data.sample_dataset(_inventory(), pool, ext, seed=1))


def test_sample_dataset_not_enough_images_raises():
    pool = {c: 15 for c in config.CLASSES}
    ext = {c: 15 for c in config.CLASSES}
    with pytest.raises(ValueError, match="disponibles"):
        data.sample_dataset(_inventory(20), pool, ext)


def test_extract_files_writes_only_selected_and_is_idempotent(make_zip, tmp_path):
    make_zip("Photo", 6)
    make_zip("Text", 6)
    for c in ("Painting", "Schematics", "Sketch"):
        make_zip(c, 6)
    inv = data.build_inventory(tmp_path)
    pool = {c: 2 for c in config.CLASSES}
    ext = {c: 1 for c in config.CLASSES}
    sample = data.sample_dataset(inv, pool, ext, seed=0)
    raw = tmp_path / "raw"
    out = data.extract_files(sample, tmp_path, raw)
    assert len(out) == 15
    assert all((raw / s).exists() for s in config.CLASSES)
    assert all(pd.Series(out["path"]).map(lambda p: os.path.exists(p)))
    assert len(list((raw / "Photo").glob("*"))) == 3
    data.extract_files(sample, tmp_path, raw)  # 2e appel sans erreur
