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


@pytest.mark.parametrize("mode,fmt", [("RGB", "JPEG"), ("L", "PNG"), ("RGBA", "PNG")])
def test_load_image_always_rgb_uint8(tmp_path, mode, fmt):
    arr = np.random.default_rng(0).integers(0, 255, (50, 70) + (() if mode == "L" else (4 if mode == "RGBA" else 3,)), dtype=np.uint8)
    p = tmp_path / f"x.{fmt.lower()}"
    Image.fromarray(arr, mode).save(p)
    out = data.load_image(p, size=32)
    assert out.shape == (32, 32, 3) and out.dtype == np.uint8


def test_load_images_skips_corrupt(tmp_path, capsys):
    good = tmp_path / "ok.jpg"
    Image.fromarray(np.zeros((10, 10, 3), np.uint8)).save(good)
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    meta = pd.DataFrame({"path": [str(good), str(bad)], "label": [1, 0], "source": ["Photo", "Text"]})
    X, meta_ok = data.load_images(meta, size=16, workers=2)
    assert X.shape == (1, 16, 16, 3) and len(meta_ok) == 1
    assert "illisible" in capsys.readouterr().out


def _meta(n_per=40):
    rows = []
    for c in config.CLASSES:
        for i in range(n_per):
            rows.append({"member": f"{c}/{i}", "source": c, "label": int(c == "Photo"), "subset": "pool"})
    for i in range(10):
        rows.append({"member": f"ext/{i}", "source": "Text", "label": 0, "subset": "ext"})
    return pd.DataFrame(rows)


def test_add_split_proportions_stratified_and_disjoint():
    m = data.add_split(_meta())
    pool = m[m["subset"] == "pool"]
    assert (m.loc[m["subset"] == "ext", "split"] == "ext").all()
    counts = pool["split"].value_counts()
    assert counts["train"] == 140 and counts["val"] == 30 and counts["test"] == 30
    for c in config.CLASSES:  # stratifié par classe source
        assert (pool[pool["source"] == c]["split"] == "train").sum() == 28
    sets = [set(pool.loc[pool["split"] == s, "member"]) for s in ("train", "val", "test")]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])


def test_select_filters_split_and_sources():
    m = data.add_split(_meta())
    X = np.arange(len(m))[:, None, None, None] * np.ones((1, 4, 4, 3), np.uint8)
    Xs, ys, ms = data.select(X, m, "train", sources=["Text"])
    assert set(ms["source"]) == {"Photo", "Text"}
    assert len(Xs) == len(ys) == len(ms)
    assert (ms["label"].to_numpy() == ys).all()
    # X est bien aligné avec meta : la valeur du pixel = index de la ligne d'origine
    orig = m[(m["split"] == "train") & m["source"].isin(["Photo", "Text"])].index.to_numpy()
    assert (Xs[:, 0, 0, 0] == orig.astype(np.uint8)).all()
    Xa, _, ma = data.select(X, m, "ext")
    assert set(ma["split"]) == {"ext"}


def test_class_weights_balanced():
    w = data.class_weights(np.array([0] * 90 + [1] * 10))
    assert w[1] == pytest.approx(5.0) and w[0] == pytest.approx(100 / 180)


def test_make_dataset_batches():
    X = np.zeros((10, 8, 8, 3), np.uint8)
    y = np.arange(10) % 2
    ds = data.make_dataset(X, y, batch_size=4, shuffle=True)
    bx, by = next(iter(ds))
    assert tuple(bx.shape) == (4, 8, 8, 3) and by.dtype.name == "float32"


def test_prepare_dataset_end_to_end_and_cache(make_zip, tmp_path, monkeypatch):
    for c in config.CLASSES:
        make_zip(c, 30, extra_png_gray=2 if c == "Sketch" else 0)
    monkeypatch.setattr(config, "ZIP_DIR", tmp_path)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "data" / "raw")
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "data" / "cache")
    monkeypatch.setattr(config, "POOL_COUNTS", {c: 20 for c in config.CLASSES})
    monkeypatch.setattr(config, "EXT_COUNTS", {**{c: 3 for c in config.CLASSES}, "Sketch": None})
    X, meta = data.prepare_dataset()
    assert X.shape[0] == len(meta) == 5 * 20 + 4 * 3 + (32 - 20) and X.dtype == np.uint8
    assert X.shape[1:] == (128, 128, 3)
    assert set(meta["split"]) == {"train", "val", "test", "ext"}
    # 2e appel : lecture du cache, chemins ré-absolutisés, mêmes données
    X2, meta2 = data.prepare_dataset()
    assert (X2 == X).all() and meta2["path"].map(lambda p: os.path.exists(p)).all()
    assert not (tmp_path / "data" / "cache" / "meta.csv").read_text().count(str(tmp_path))
