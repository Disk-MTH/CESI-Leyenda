import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")  # pas d'affichage pendant les tests


def _img_bytes(fmt, mode="RGB", size=(40, 30), seed=0):
    rng = np.random.default_rng(seed)
    if mode == "L":
        arr = rng.integers(0, 256, size=(size[1], size[0]), dtype=np.uint8)
    elif mode == "RGBA":
        arr = rng.integers(0, 256, size=(size[1], size[0], 4), dtype=np.uint8)
    else:
        arr = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode).save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def make_zip(tmp_path):
    """Fabrique une archive {classe}.zip avec n jpg + parasites (desktop.ini, dossier, txt)."""

    def _make(name, n, folder=None, extra_png_gray=0):
        folder = folder or name
        path = tmp_path / f"Dataset Livrable 1 - {name}.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr(f"{folder}/", "")
            zf.writestr(f"{folder}/desktop.ini", "[.ShellClassInfo]")
            zf.writestr(f"{folder}/notes.txt", "hello")
            for i in range(n):
                zf.writestr(f"{folder}/{name.lower()}_{i:03d}.JPG", _img_bytes("JPEG", seed=i))
            for j in range(extra_png_gray):
                zf.writestr(f"{folder}/gray_{j:03d}.png", _img_bytes("PNG", mode="L", seed=100 + j))
        return path

    return _make
