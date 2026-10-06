import zipfile

import keras
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from tounum_denoise import data, evaluation, modeling


def _photo(h=60, w=80, seed=0):
    """Smooth synthetic 'photo': gradients plus a few soft blobs."""
    yy, xx = np.mgrid[0:h, 0:w]
    rng = np.random.default_rng(seed)
    img = np.stack([xx / w * 200, yy / h * 200, (xx + yy) / (h + w) * 200], -1)
    for _ in range(3):
        cy, cx = rng.integers(0, h), rng.integers(0, w)
        img += 40 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 100)[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


def test_extract_noisy_only_images(tmp_path):
    z = tmp_path / "noisy.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("Dataset/", "")
        zf.writestr("Dataset/notes.txt", "x")
        for i in range(3):
            path = tmp_path / f"n{i}.jpg"
            Image.fromarray(_photo(seed=i)).save(path)
            zf.write(path, f"Dataset/noisy_{i:03d}.jpg")
    paths = data.extract_noisy(z, tmp_path / "out")
    assert [p.name for p in paths] == ["noisy_000.jpg", "noisy_001.jpg", "noisy_002.jpg"]
    images, info = data.load_noisy(paths)
    assert len(images) == 3 and images[0].shape == (60, 80, 3)
    assert list(info.columns) == ["file", "width", "height", "noise_sigma"]


def test_downscale_keeps_ratio_and_never_upscales():
    img = np.zeros((400, 600, 3), np.uint8)
    assert data.downscale(img, 200).shape == (200, 300, 3)
    assert data.downscale(img, 1000).shape == (400, 600, 3)


def test_random_crops_shapes_and_small_images():
    rng = np.random.default_rng(0)
    assert data.random_crops(_photo(), 5, 32, rng).shape == (5, 32, 32, 3)
    assert data.random_crops(_photo(20, 30), 2, 32, rng).shape == (2, 32, 32, 3)


def test_build_patches_from_files(tmp_path):
    paths = []
    for i in range(2):
        p = tmp_path / f"p{i}.png"
        Image.fromarray(_photo(seed=i)).save(p)
        paths.append(p)
    patches = data.build_patches(paths, crops_per_image=3, size=16)
    assert patches.shape == (6, 16, 16, 3) and patches.dtype == np.uint8


def test_add_noise_level_and_identity():
    rng = np.random.default_rng(0)
    clean = np.full((64, 64, 3), 128, np.uint8)
    assert (data.add_noise(clean, 0.0, rng, jpeg_quality=None) == clean).all()
    noisy = data.add_noise(clean, 10.0, rng, jpeg_quality=None).astype(float)
    assert noisy.std() == pytest.approx(10.0, rel=0.1)
    assert data.add_noise(clean, 10.0, rng).shape == clean.shape   # with the JPEG step


def test_make_dataset_pairs_in_range():
    patches = np.stack([_photo(32, 32, seed=i) for i in range(6)])
    noisy, clean = next(iter(data.make_dataset(patches, batch_size=4)))
    assert tuple(noisy.shape) == (4, 32, 32, 3) and tuple(clean.shape) == (4, 32, 32, 3)
    assert float(noisy.numpy().min()) >= 0 and float(noisy.numpy().max()) <= 1


def test_fixed_noisy_set_is_reproducible():
    patches = np.stack([_photo(32, 32, seed=i) for i in range(4)])
    a = data.fixed_noisy_set(patches, seed=1)
    b = data.fixed_noisy_set(patches, seed=1)
    assert np.array_equal(a[0], b[0]) and len(a[2]) == 4


def test_estimate_noise_tracks_sigma():
    rng = np.random.default_rng(0)
    flat = np.full((200, 200, 3), 120, np.uint8)
    est = [data.estimate_noise(data.add_noise(flat, s, rng, jpeg_quality=None)) for s in (5, 15)]
    assert est[0] == pytest.approx(5, rel=0.25) and est[1] == pytest.approx(15, rel=0.15)
    assert data.estimate_noise(flat) == 0.0


def test_kernels_sum_to_one_and_keep_flat_images():
    assert evaluation.mean_kernel(3).sum() == pytest.approx(1)
    assert evaluation.gaussian_kernel(1.5).sum() == pytest.approx(1)
    flat = np.full((2, 20, 20, 3), 0.4, np.float32)
    for f in evaluation.CLASSICAL_FILTERS.values():
        assert np.allclose(f(flat), 0.4, atol=1e-6)


def test_psnr_ssim():
    x = data.to_float(_photo())
    assert evaluation.ssim(x, x) == pytest.approx(1.0)
    noisy = np.clip(x + 0.05, 0, 1)
    assert 20 < evaluation.psnr(x, noisy) < 30


@pytest.mark.parametrize("skips", [False, True])
def test_autoencoder_any_size_and_skips(skips):
    m = modeling.build_autoencoder(filters=(4, 8), skips=skips)
    assert any("skip" in l.name for l in m.layers) == skips
    out = modeling.denoise(m, _photo(37, 51))
    assert out.shape == (37, 51, 3) and out.dtype == np.uint8


def test_autoencoder_learns_and_history_columns():
    keras.utils.set_random_seed(0)
    patches = np.stack([_photo(32, 32, seed=i) for i in range(16)])
    noisy, clean, _ = data.fixed_noisy_set(patches)
    m = modeling.build_autoencoder(filters=(8, 16), skips=True)
    hist = modeling.train(m, data.make_dataset(patches, batch_size=8), (noisy, clean), epochs=15, verbose=0)
    assert {"epoch", "loss", "psnr", "ssim", "val_loss", "val_psnr", "val_ssim"} <= set(hist.columns)
    assert hist["val_loss"].min() < hist["val_loss"].iloc[0]


def test_save_and_reload(tmp_path):
    m = modeling.build_autoencoder(filters=(4, 8), skips=True)
    m.save(tmp_path / "d.keras")
    m2 = keras.models.load_model(tmp_path / "d.keras")
    x = data.to_float(_photo(32, 32))[None]
    assert np.allclose(m.predict(x, verbose=0), m2.predict(x, verbose=0), atol=1e-6)


def test_plots():
    hist = pd.DataFrame({"epoch": [1, 2], "loss": [0.1, 0.05], "val_loss": [0.1, 0.06], "psnr": [20, 25], "val_psnr": [20, 24]})
    assert evaluation.plot_history(hist) is not None
    assert evaluation.show_grid({"a": [_photo()], "b": [data.to_float(_photo())]}, ["x"]) is not None
    assert evaluation.draw_autoencoder((32, 64, 128), skips=True) is not None
