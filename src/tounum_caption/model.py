"""Captioning network: InceptionV3 region features -> GRU decoder with attention (Keras 3 / TensorFlow)."""
from __future__ import annotations

import math
import warnings

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers

from . import config, data


@keras.saving.register_keras_serializable(package="tounum")
def masked_accuracy(y_true, y_pred):
    """Share of next words predicted correctly, padding positions (token 0) excluded."""
    y_true = tf.cast(y_true, tf.int32)
    correct = tf.cast(tf.equal(tf.argmax(y_pred, axis=-1, output_type=tf.int32), y_true), tf.float32)
    mask = tf.cast(y_true != 0, tf.float32)
    return tf.reduce_sum(correct * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)


def build_captioner(
    vocab_size: int = config.VOCAB_SIZE,
    attention: bool = True,
    embed_dim: int = config.EMBED_DIM,
    units: int = config.UNITS,
    dropout: float = 0.3,
    learning_rate: float = 1e-3,
    name: str | None = None,
) -> keras.Model:
    """Décodeur de légendes entraîné en teacher forcing : prédit le mot t+1 à partir de l'image et des mots 1..t.

    Entrées : `image_features` (64, 2048), les régions InceptionV3, et `tokens` (T,), la légende déjà écrite.
    Sortie : logits (T, vocab_size) du mot suivant à chaque position.
    Partie CNN : une couche Dense entraînable projette chaque région en `embed_dim` ; leur moyenne, normalisée,
    donne l'état initial du GRU. Sans la normalisation, la tanh de l'état initial peut saturer tôt dans
    l'entraînement : toutes les images reçoivent alors le même état et le modèle sans attention ignore l'image.
    Partie RNN : Embedding -> GRU. Avec `attention=True`, à chaque pas la sortie du GRU regarde les 64 régions
    projetées (attention additive) et la somme pondérée (contexte) lui est concaténée avant les couches de sortie.
    `attention=False` donne un modèle de type Show and Tell où l'image n'entre que par l'état initial.
    Le padding (token 0) n'est pas masqué dans le réseau : le GRU lit de gauche à droite, donc le padding placé
    après une légende ne peut pas changer les prédictions qui le précèdent, et la perte ignore ces positions.
    """
    name = name or ("captioner_attention" if attention else "captioner_no_attention")
    feats = keras.Input(shape=(config.N_REGIONS, config.FEATURE_DIM), name="image_features")
    tokens = keras.Input(shape=(None,), dtype="int32", name="tokens")

    regions = layers.Dense(embed_dim, activation="relu", name="cnn_encoder")(feats)
    pooled = layers.GlobalAveragePooling1D(name="image_mean")(regions)
    pooled = layers.LayerNormalization(name="image_norm")(pooled)      # keeps the tanh below out of saturation
    state = layers.Dense(units, activation="tanh", name="initial_state")(pooled)

    x = layers.Embedding(vocab_size, embed_dim, name="embedding")(tokens)
    x = layers.Dropout(dropout, name="embedding_dropout")(x)
    x = layers.GRU(units, return_sequences=True, name="gru")(x, initial_state=state)
    if attention:
        query = layers.Dense(embed_dim, name="attention_query")(x)
        context, _ = layers.AdditiveAttention(name="attention")([query, regions], return_attention_scores=True)
        x = layers.Concatenate(name="gru_and_context")([x, context])
    x = layers.Dense(units, activation="relu", name="hidden")(x)
    x = layers.Dropout(dropout, name="hidden_dropout")(x)
    logits = layers.Dense(vocab_size, name="next_word")(x)

    model = keras.Model({"image_features": feats, "tokens": tokens}, logits, name=name)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate, clipnorm=5.0),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=0),
        metrics=[masked_accuracy],
    )
    return model


def attention_model(captioner: keras.Model) -> keras.Model:
    """Model with the same inputs that returns the attention weights (T, 64) of each decoding step."""
    return keras.Model(captioner.inputs, captioner.get_layer("attention").output[1])


class CaptionBatches(keras.utils.PyDataset):
    """Teacher forcing batches: ({image_features, tokens[:-1]}, tokens[1:]).

    `features[i]` holds the regions of image i and `tokens[j]` an encoded caption of image `image_index[j]`.
    With `one_per_image=True` an epoch uses one caption per image, drawn again at every epoch: epochs stay short
    and all the captions of an image are seen over the epochs. Otherwise every caption is used, in a fixed order.
    """

    def __init__(self, features, tokens, image_index, batch_size: int = config.BATCH_SIZE,
                 one_per_image: bool = False, seed: int = config.SEED, **kwargs):
        super().__init__(**kwargs)
        self.features, self.tokens = features, np.asarray(tokens)
        self.image_index = np.asarray(image_index)
        self.batch_size, self.one_per_image = batch_size, one_per_image
        self.rng = np.random.default_rng(seed)
        self._draw()

    def _draw(self):
        if self.one_per_image:
            order = self.rng.permutation(len(self.tokens))
            _, first = np.unique(self.image_index[order], return_index=True)   # one random caption per image
            self.rows = self.rng.permutation(order[first])
        else:
            self.rows = np.arange(len(self.tokens))

    def __len__(self):
        return math.ceil(len(self.rows) / self.batch_size)

    def __getitem__(self, i):
        rows = self.rows[i * self.batch_size:(i + 1) * self.batch_size]
        tok = self.tokens[rows]
        feats = np.asarray(self.features[self.image_index[rows]], dtype=np.float32)
        return {"image_features": feats, "tokens": tok[:, :-1]}, tok[:, 1:]

    def on_epoch_end(self):
        self._draw()


def generate(model: keras.Model, features, vocab: list[str], max_len: int = config.MAX_LEN,
             batch_size: int = 500) -> list[str]:
    """Greedy decoding: start from START and add the most likely next word until END (or max_len words).

    Padding (index 0) and the unknown word (index 1) are never chosen: a caption with `[UNK]` is of no use.
    """
    start = vocab.index(config.START)
    captions = []
    for s in range(0, len(features), batch_size):
        feats = np.asarray(features[s:s + batch_size], dtype=np.float32)
        tokens = np.zeros((len(feats), max_len - 1), dtype=np.int32)
        tokens[:, 0] = start
        for t in range(max_len - 2):
            logits = model.predict_on_batch({"image_features": feats, "tokens": tokens})[:, t]
            logits[:, :2] = -np.inf
            tokens[:, t + 1] = np.argmax(logits, axis=-1)
        captions += [data.decode(row, vocab) for row in tokens]
    return captions


def attention_maps(att_model: keras.Model, features, captions: list[str], vocab: list[str],
                   max_len: int = config.MAX_LEN) -> np.ndarray:
    """Attention weights (n, max_len - 1, 64) used to produce each word of the given (generated) captions.

    `att_model` comes from `attention_model(captioner)`. Row t holds the weights used to predict word t + 1.
    """
    tokens = data.encode([f"{config.START} {c} {config.END}" for c in captions], vocab, max_len)[:, :-1]
    feats = np.asarray(features, dtype=np.float32)
    return att_model.predict({"image_features": feats, "tokens": tokens}, verbose=0)


def bleu(references: list[list[str]], hypotheses: list[str]) -> dict[str, float]:
    """Corpus BLEU-1 to BLEU-4 of generated captions against all the reference captions of each image."""
    from nltk.translate.bleu_score import corpus_bleu

    refs = [[r.split() for r in rs] for rs in references]
    hyps = [h.split() for h in hypotheses]
    with warnings.catch_warnings():          # nltk warns when a score is 0 (no 4-gram in common), early in training
        warnings.simplefilter("ignore", UserWarning)
        return {f"BLEU-{n}": corpus_bleu(refs, hyps, weights=tuple([1.0 / n] * n)) for n in range(1, 5)}


class BleuCallback(keras.callbacks.Callback):
    """Adds `val_bleu4` to the training history: BLEU-4 of greedy captions on a few validation images."""

    def __init__(self, features, references: list[list[str]], vocab: list[str]):
        super().__init__()
        self.features, self.references, self.vocab = features, references, vocab

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            logs["val_bleu4"] = bleu(self.references, generate(self.model, self.features, self.vocab))["BLEU-4"]


def train(model: keras.Model, train_batches, val_batches, epochs: int = 30, callbacks=(), patience: int = 5,
          verbose: int = 2) -> pd.DataFrame:
    """Train with early stopping on the validation loss (best weights restored) and return the history.

    Columns: epoch, loss, masked_accuracy, val_loss, val_masked_accuracy, plus what the callbacks add.
    """
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    hist = model.fit(train_batches, validation_data=val_batches, epochs=epochs,
                     callbacks=[*callbacks, stop], verbose=verbose)
    df = pd.DataFrame(hist.history)
    df.insert(0, "epoch", range(1, len(df) + 1))
    return df
