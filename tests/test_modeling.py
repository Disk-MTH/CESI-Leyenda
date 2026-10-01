import keras
import numpy as np

from tounum_cls import data, modeling


def _toy(n=32, size=32, seed=0):
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n // 2), dtype="float32")
    X = rng.integers(0, 60, (n, size, size, 3), dtype=np.uint8)
    X[y == 1] += 150  # photos = images claires -> facilement séparables
    return X, y


def test_build_cnn_shapes_and_params():
    m = modeling.build_cnn(img_size=32)
    assert m.output_shape == (None, 1)
    assert [l.name for l in m.layers if l.name.startswith("conv")] == ["conv1", "conv2", "conv3", "conv4"]
    assert 90_000 < modeling.build_cnn().count_params() < 130_000   # ~106 k avec la config par défaut


def test_build_cnn_options_add_layers_and_l2():
    m = modeling.build_cnn(img_size=32, dropout=0.3, l2=1e-3, augmentation=True)
    names = [l.name for l in m.layers]
    assert "dropout" in names and "aug_flip" in names
    assert len(m.losses) > 0          # pénalités L2 enregistrées
    plain = modeling.build_cnn(img_size=32)
    assert "dropout" not in [l.name for l in plain.layers] and len(plain.losses) == 0


def test_cnn_learns_toy_problem():
    keras.utils.set_random_seed(0)
    X, y = _toy()
    ds = data.make_dataset(X, y, batch_size=8, shuffle=True)
    m = modeling.build_cnn(filters=(4, 8), dense_units=8, img_size=32)
    hist = modeling.train(m, ds, ds, epochs=40, verbose=0)  # 40 : les stats BatchNorm convergent lentement sur 4 pas/epoch
    assert list(hist.columns[:1]) == ["epoch"] and {"loss", "val_loss", "accuracy", "val_accuracy"} <= set(hist.columns)
    assert hist["val_accuracy"].iloc[-1] > 0.9
    p = modeling.predict_proba(m, X)
    assert p.shape == (32,) and ((p >= 0) & (p <= 1)).all()


def test_early_stopping_stops_before_max_epochs():
    keras.utils.set_random_seed(0)
    X, y = _toy()
    ds = data.make_dataset(X, y, batch_size=8)
    m = modeling.build_cnn(filters=(4,), dense_units=4, img_size=32, learning_rate=1e-9)  # n'apprend rien
    hist = modeling.train(m, ds, ds, epochs=30, early_stopping=True, patience=2, verbose=0)
    assert len(hist) < 30


def test_build_mobilenet_frozen():
    m = modeling.build_mobilenet(img_size=32, weights=None)
    assert m.output_shape == (None, 1)
    assert m.count_params() > 2_000_000
    assert 1_000 < sum(int(np.prod(w.shape)) for w in m.trainable_weights) < 3_000   # seule la tête apprend
