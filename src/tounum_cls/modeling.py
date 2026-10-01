"""Construction et entraînement des réseaux (Keras 3 / TensorFlow)."""
from __future__ import annotations

import keras
import pandas as pd
from keras import layers, regularizers

from . import config


def _compile(model: keras.Model, learning_rate: float = 1e-3) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


def build_cnn(
    filters: tuple[int, ...] = (16, 32, 64, 128),
    dense_units: int = 64,
    dropout: float = 0.0,
    l2: float = 0.0,
    augmentation: bool = False,
    img_size: int = config.IMG_SIZE,
    learning_rate: float = 1e-3,
    bn_momentum: float = 0.9,
    name: str = "cnn_photo",
) -> keras.Model:
    """CNN « maison » pour photo (1) vs non-photo (0).

    Entrée : images uint8 (img_size, img_size, 3) ; la normalisation /255 est DANS le modèle.
    Un bloc = Conv2D 3x3 (padding same) -> BatchNormalization -> ReLU -> MaxPooling2D 2x2.
    Tête : GlobalAveragePooling2D -> Dense(dense_units, relu) -> [Dropout] -> Dense(1, sigmoid).
    `augmentation=True` ajoute des transformations aléatoires, actives seulement à l'entraînement.
    `l2` pénalise les poids des Conv2D et de la Dense cachée.
    `bn_momentum=0.9` (au lieu de 0.99 par défaut) : les statistiques mobiles de BatchNorm se calibrent en
    quelques dizaines de pas, ce qui évite une validation erratique pendant les premières epochs.
    """
    reg = regularizers.l2(l2) if l2 > 0 else None
    inputs = keras.Input(shape=(img_size, img_size, 3), name="image")
    x = inputs
    if augmentation:
        x = layers.RandomFlip("horizontal", name="aug_flip")(x)
        x = layers.RandomRotation(0.05, name="aug_rotation")(x)
        x = layers.RandomZoom(0.1, name="aug_zoom")(x)
        x = layers.RandomContrast(0.1, name="aug_contrast")(x)
    x = layers.Rescaling(1.0 / 255, name="rescaling")(x)
    for i, n_filters in enumerate(filters, start=1):
        x = layers.Conv2D(n_filters, 3, padding="same", use_bias=False, kernel_regularizer=reg, name=f"conv{i}")(x)
        x = layers.BatchNormalization(momentum=bn_momentum, name=f"bn{i}")(x)
        x = layers.Activation("relu", name=f"relu{i}")(x)
        x = layers.MaxPooling2D(2, name=f"pool{i}")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(dense_units, activation="relu", kernel_regularizer=reg, name="dense")(x)
    if dropout > 0:
        x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
    return _compile(keras.Model(inputs, outputs, name=name), learning_rate)


def build_mobilenet(
    img_size: int = config.IMG_SIZE, weights: str | None = "imagenet", name: str = "mobilenetv2_gele"
) -> keras.Model:
    """MobileNetV2 gelé (extracteur de caractéristiques) + tête binaire. Sert de comparaison au CNN maison.

    Entrée : images uint8 ; mise à l'échelle [-1, 1] attendue par MobileNetV2 faite dans le modèle.
    `weights=None` (tests) évite le téléchargement des poids ImageNet.
    """
    base = keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3), include_top=False, weights=weights
    )
    base.trainable = False
    inputs = keras.Input(shape=(img_size, img_size, 3), name="image")
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="rescaling")(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.2, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
    return _compile(keras.Model(inputs, outputs, name=name))


def train(
    model: keras.Model,
    train_ds,
    val_ds,
    epochs: int = 15,
    class_weight: dict[int, float] | None = None,
    early_stopping: bool = False,
    patience: int = 4,
    verbose: int = 2,
) -> pd.DataFrame:
    """Entraîne le modèle et renvoie l'historique sous forme de DataFrame (une ligne par epoch).

    Colonnes : epoch, accuracy, auc, loss, val_accuracy, val_auc, val_loss.
    Avec early_stopping=True : arrêt après `patience` epochs sans amélioration de val_loss,
    et restauration des meilleurs poids.
    """
    callbacks = []
    if early_stopping:
        callbacks.append(
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
        )
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=callbacks,
        shuffle=False,  # le mélange est fait par tf.data ; évite un avertissement Keras
        verbose=verbose,
    )
    df = pd.DataFrame(hist.history)
    df.insert(0, "epoch", range(1, len(df) + 1))
    return df


def predict_proba(model: keras.Model, X, batch_size: int = config.BATCH_SIZE):
    """Probabilités P(photo) sous forme de vecteur 1D float."""
    return model.predict(X, batch_size=batch_size, verbose=0).ravel()
