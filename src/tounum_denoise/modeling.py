"""Convolutional autoencoders for denoising (Keras 3 / TensorFlow)."""
from __future__ import annotations

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers

from . import config, data

PAD_MULTIPLE = 8   # inputs are padded to a multiple of this so the pooling / upsampling sizes match


@keras.saving.register_keras_serializable(package="tounum")
def psnr(y_true, y_pred):
    """Peak signal-to-noise ratio in dB for images in [0, 1] (higher is better), from the mean squared error of the
    whole batch. A per-image PSNR would be infinite for a patch left exactly unchanged (a flat white area, which a
    residual model can return as is) and would make the epoch average infinite too."""
    mse = tf.reduce_mean(tf.square(tf.cast(y_true, tf.float32) - tf.cast(y_pred, tf.float32)))
    return 10.0 * tf.math.log(1.0 / tf.maximum(mse, 1e-10)) / tf.math.log(10.0)


@keras.saving.register_keras_serializable(package="tounum")
def ssim(y_true, y_pred):
    """Structural similarity for images in [0, 1] (1 = identical)."""
    return tf.image.ssim(y_true, y_pred, max_val=1.0)


def _conv(n_filters: int, name: str):
    return layers.Conv2D(n_filters, 3, padding="same", activation="relu", name=name)


def build_autoencoder(
    filters: tuple[int, ...] = (32, 64, 128),
    skips: bool = False,
    residual: bool = False,
    learning_rate: float = 1e-3,
    name: str | None = None,
) -> keras.Model:
    """Autoencodeur convolutif qui transforme une image RGB bruitée en image propre (float32 dans [0, 1]).

    Encodeur : à chaque niveau, deux Conv2D 3x3 + ReLU, puis MaxPooling2D 2x2 (le dernier niveau est le goulot
    d'étranglement, sans pooling). Décodeur : Conv2DTranspose 2x2 stride 2 (suréchantillonnage) puis deux Conv2D
    3x3, symétrique de l'encodeur. Sortie : Conv2D 1x1 à 3 canaux et sigmoïde.
    `skips=True` concatène chaque niveau de l'encodeur avec le niveau du décodeur de même résolution (style
    U-Net), pour que les détails fins n'aient pas à passer par le goulot basse résolution.
    `residual=True` fait prédire le bruit à la dernière Conv2D 1x1 au lieu de l'image propre ; la sortie est
    l'entrée bruitée moins ce bruit, ramenée dans [0, 1]. Laisser une image intacte demande alors seulement de
    prédire zéro.
    Le modèle est entièrement convolutif : il accepte toute hauteur et largeur multiples de 2**(niveaux-1).
    """
    name = name or ("residual_" if residual else "") + ("unet_autoencoder" if skips else "conv_autoencoder")
    inputs = keras.Input(shape=(None, None, 3), name="noisy")
    x, encoder_outputs = inputs, []
    for i, n in enumerate(filters[:-1], start=1):
        x = _conv(n, f"enc{i}_conv1")(x)
        x = _conv(n, f"enc{i}_conv2")(x)
        encoder_outputs.append(x)
        x = layers.MaxPooling2D(2, name=f"enc{i}_pool")(x)
    x = _conv(filters[-1], "bottleneck_conv1")(x)
    x = _conv(filters[-1], "bottleneck_conv2")(x)
    for i in range(len(filters) - 1, 0, -1):
        n = filters[i - 1]
        x = layers.Conv2DTranspose(n, 2, strides=2, padding="same", activation="relu", name=f"dec{i}_up")(x)
        if skips:
            x = layers.Concatenate(name=f"dec{i}_skip")([x, encoder_outputs[i - 1]])
        x = _conv(n, f"dec{i}_conv1")(x)
        x = _conv(n, f"dec{i}_conv2")(x)
    if residual:
        noise = layers.Conv2D(3, 1, name="noise")(x)
        x = layers.Subtract(name="noisy_minus_noise")([inputs, noise])
        outputs = layers.ReLU(max_value=1.0, name="clean")(x)          # clips the pixels to [0, 1]
    else:
        outputs = layers.Conv2D(3, 1, activation="sigmoid", name="clean")(x)
    model = keras.Model(inputs, outputs, name=name)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss="mse", metrics=[psnr, ssim])
    return model


def train(model: keras.Model, train_ds, val_data, epochs: int = 20, early_stopping: bool = True,
          patience: int = 4, verbose: int = 2) -> pd.DataFrame:
    """Train and return the history as a DataFrame (one row per epoch).

    Columns: epoch, loss, psnr, ssim, val_loss, val_psnr, val_ssim. With early stopping the best weights
    (lowest val_loss) are restored at the end.
    """
    callbacks = []
    if early_stopping:
        callbacks.append(keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True))
    hist = model.fit(train_ds, validation_data=val_data, epochs=epochs, callbacks=callbacks, shuffle=False, verbose=verbose)
    df = pd.DataFrame(hist.history)
    df.insert(0, "epoch", range(1, len(df) + 1))
    return df


def fine_tune(model: keras.Model, train_ds, val_data, epochs: int = 100,
              learning_rate: float = config.FINE_TUNE_LEARNING_RATE, patience: int = 8,
              verbose: int = 2) -> pd.DataFrame:
    """Continue training an already trained model on new pairs with a lower learning rate (best weights restored).

    The model is compiled again (new Adam state); it keeps its weights. Returns the history like `train`.
    """
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss="mse", metrics=[psnr, ssim])
    return train(model, train_ds, val_data, epochs=epochs, patience=patience, verbose=verbose)


def predict(model: keras.Model, noisy: np.ndarray, batch_size: int = 16) -> np.ndarray:
    """Denoise a batch of same-size float images in [0, 1] (sizes must already be multiples of PAD_MULTIPLE)."""
    return model.predict(noisy, batch_size=batch_size, verbose=0)


def denoise(model: keras.Model, img: np.ndarray) -> np.ndarray:
    """Denoise one uint8 RGB image of any size and return a uint8 image of the same size.

    The image is padded by reflection to a multiple of PAD_MULTIPLE, denoised, then cropped back. The model is
    called directly rather than through `predict`, which would compile a new function for every image size.
    """
    h, w = img.shape[:2]
    ph, pw = (-h) % PAD_MULTIPLE, (-w) % PAD_MULTIPLE
    x = np.pad(data.to_float(img), ((0, ph), (0, pw), (0, 0)), mode="reflect")
    y = keras.ops.convert_to_numpy(model(x[None], training=False))[0, :h, :w]
    return data.to_uint8(y)
