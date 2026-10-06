# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # TouNum, deliverable 1: telling photos apart from other images
#
# CESI Data Science team. Prototype of the automatic image captioning workflow.
#
# ---
# ## 0. Business context and goal
#
# TouNum digitises documents (text, images and so on) for its clients and wants to add an automatic
# captioning service for images. The target workflow has three modules:
#
# 1. **Filtering**: keep only the photographs (this deliverable);
# 2. **Pre-processing**: clean up noisy or blurry images (deliverable 2);
# 3. **Captioning**: write a descriptive caption for each photo (deliverable 3).
#
# Many of the images TouNum has already digitised are not photos: diagrams, scanned text, drawings, paintings.
# Sending them to the captioning module would waste compute and produce meaningless captions. This notebook builds
# the entry filter: a neural network that answers "photo" (1) or "other" (0).
#
# ### Outline
# | Section | Content |
# |---|---|
# | 1 | Environment and reproducibility |
# | 2 | The data: inventory, exploration, sampling |
# | 3 | Train / validation / test split |
# | 4 | Increasing difficulty: text, then diagrams, sketches and paintings |
# | 5 | Network architecture: diagram, parameters, loss, optimiser |
# | 6 | Training: loss and accuracy curves (training vs validation) |
# | 7 | Bias / variance analysis (underfitting and overfitting) |
# | 8 | Improvements: augmentation, dropout, L2, early stopping |
# | 9 | Final evaluation on a balanced and a realistic test set, errors, threshold |
# | 10 | Comparison with a pre-trained network (MobileNetV2) and choice of the production model |
# | 11 | Conclusion, limits and next steps |
#
# ### How to run it
# The notebook is meant to be run with `bash scripts/run_notebook.sh 1 full`, which rebuilds the `.ipynb` from this
# `.py` file and executes it. To open it interactively: `uv run --with jupyterlab jupyter lab`.
# On the first run, the sampled images are extracted from the archives and cached in `data/` (a few minutes);
# later runs read the cache. `bash scripts/run_notebook.sh 1 quick` runs a very short version (2 epochs) to check
# that everything executes. It writes a separate `.quick.ipynb` file and never touches the deliverable.

# %% [markdown]
# ## 1. Environment and reproducibility

# %%
import os
import sys
import time
import inspect
import tempfile
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")      # quieter TensorFlow logs
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))                   # reusable code lives in src/tounum_cls
import tounum_gpu
tounum_gpu.setup()                                      # use the NVIDIA GPU if there is one (src/tounum_gpu.py)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from IPython.display import display, Markdown
import tensorflow as tf
import keras
import sklearn

from tounum_cls import config, data, modeling, evaluation

QUICK = os.environ.get("TOUNUM_QUICK") == "1"
# Number of epochs for each experiment (reduced in QUICK mode)
EPOCHS = {"stage": 8, "main": 20, "curve": 12, "improve": 25, "mobilenet": 8}
if QUICK:
    EPOCHS = {k: 2 for k in EPOCHS}

keras.utils.set_random_seed(config.SEED)                # seeds Python, NumPy and TensorFlow
pd.set_option("display.precision", 3)
pd.set_option("display.width", 140)
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 100

print(f"Python {sys.version.split()[0]} | TensorFlow {tf.__version__} | Keras {keras.__version__} | "
      f"scikit-learn {sklearn.__version__} | pandas {pd.__version__} | numpy {np.__version__}")
print("GPU:", tf.config.list_physical_devices("GPU") or "none (running on CPU)")
print(f"Seed: {config.SEED} | QUICK mode: {QUICK} | epochs: {EPOCHS}")

# %% [markdown]
# Every source of randomness (image sampling, split, weight initialisation, batch shuffling) depends on the seed
# `SEED = 42`. On CPU, TensorFlow can still produce tiny differences between runs because parallel operations are
# not always summed in the same order. The last decimals may move, and a diagnosis that sits right on a threshold
# can flip.

# %% [markdown]
# ## 2. The data
#
# TouNum provided five archives, one per image type. The archive name is the label: there is no annotation file.
# Our binary target is `Photo` = **1** and everything else = **0**.
#
# The archives weigh several GB, so `prepare_dataset()` does not unpack them. It:
# 1. lists the images (jpg/png only, which skips stray files such as `desktop.ini`);
# 2. draws a reproducible sample made of two disjoint parts: a balanced modelling pool and an extended test set;
# 3. extracts only the sampled files;
# 4. loads them with **ImageIO**, converts them to **RGB** (the sketches are grayscale), resizes them to
#    **128×128** and stores them as `uint8` in a cache (`data/cache`).

# %%
t0 = time.perf_counter()
X, meta = data.prepare_dataset()
print(f"X: {X.shape}, dtype={X.dtype}, {X.nbytes / 1e6:.0f} MB in memory | loaded in {time.perf_counter() - t0:.0f} s")

summary = meta.groupby(["source", "subset"]).size().unstack(fill_value=0)
summary["total sampled"] = summary.sum(axis=1)
display(summary.loc[config.CLASSES])

# %% [markdown]
# **Why a sample rather than all 41,400 images?** Everything runs on a CPU. A balanced sample of 6,000 images
# (3,000 photos and 3,000 non-photos) is plenty for this kind of task and lets us run many experiments
# (comparisons, regularisation) in reasonable time. The counts can be changed in `src/tounum_cls/config.py`
# (`POOL_COUNTS`, `EXT_COUNTS`).
#
# - The **pool** is split 50/50 between photos and non-photos. Paintings get the largest share of the negatives
#   (1,200) because they are the hardest class.
# - The **extended test set** (`ext`) is never used to train, to pick a model or to set a threshold. It is
#   imbalanced on purpose (about 21% photos) to estimate performance on an archive that looks like real client data.

# %% [markdown]
# ### 2.1 Original image sizes
# The images come in many sizes and formats. Only the file headers are read here, which is fast.

# %%
def original_size(path):
    with Image.open(path) as im:
        return im.size  # (width, height)

sizes = meta["path"].map(original_size)
meta["width"] = [s[0] for s in sizes]
meta["height"] = [s[1] for s in sizes]
meta["megapixels"] = meta["width"] * meta["height"] / 1e6
display(meta.groupby("source")[["width", "height", "megapixels"]].median().loc[config.CLASSES])
print("Every image is resized to 128x128 without keeping the aspect ratio. Very elongated images (A4 text pages) "
      "get distorted, which is acceptable for a simple photo / non-photo decision.")

# %% [markdown]
# ### 2.2 What the network sees (128×128 images)

# %%
rng = np.random.default_rng(config.SEED)
fig, axes = plt.subplots(len(config.CLASSES), 8, figsize=(16, 2.1 * len(config.CLASSES)))
for r, source in enumerate(config.CLASSES):
    idx = rng.choice(meta.index[(meta["source"] == source) & (meta["subset"] == "pool")], 8, replace=False)
    for c, i in enumerate(idx):
        axes[r, c].imshow(X[i])
        axes[r, c].axis("off")
    axes[r, 0].set_title(source + (" (label 1)" if source == config.POSITIVE_CLASS else " (label 0)"),
                         loc="left", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2.3 Two simple statistics that already separate some classes
# Mean **brightness**, and **colourfulness** (mean difference between the R, G and B channels, which is 0 for a
# grayscale image). They give a feel for how hard each separation will be.

# %%
pix = X.astype(np.float32)
meta["brightness"] = pix.mean(axis=(1, 2, 3))
meta["colourfulness"] = (np.abs(pix[..., 0] - pix[..., 1]) + np.abs(pix[..., 1] - pix[..., 2])).mean(axis=(1, 2))
del pix

fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
sns.boxplot(data=meta, x="source", y="brightness", order=config.CLASSES, ax=axes[0])
axes[0].set_title("Mean brightness (0 = black, 255 = white)")
sns.boxplot(data=meta, x="source", y="colourfulness", order=config.CLASSES, ax=axes[1])
axes[1].set_title("Colourfulness (0 = grayscale)")
for ax in axes:
    ax.set_xlabel("")
plt.tight_layout()
plt.show()

# %% [markdown]
# Scanned text is very bright (about 244) with no colour at all, sketches are bright (about 234) and grayscale, and
# diagrams are usually bright but vary a lot (some are dark or colourful). A very simple classifier using thresholds
# on these two numbers would already separate a good share of them.
# Paintings are another story: their brightness (about 118 against 112 for photos) and colour ranges overlap with
# photos. They can only be told apart by texture and structure, which is exactly what a convolutional network can
# pick up. This is why section 4 moves from the easiest classes to the hardest.

# %% [markdown]
# ## 3. Train / validation / test split
#
# The pool is split with stratification on the **source class** (not only on the binary label), so every image
# type has the same share in all three sets:
#
# | Set | Share | Role |
# |---|---|---|
# | `train` | 70% | learn the network weights |
# | `val` (validation) | 15% | follow training epoch by epoch, tune and pick the model. This is the "test error" plotted in section 6 |
# | `test` | 15% | final measurement only: never used to train, to pick a model or to set a threshold (sections 4 and 9) |
# | `ext` | n/a | extended imbalanced test set, outside the pool, used the same way as `test` |
#
# Picking the model by looking at the test set would leak information into the choice and make the reported
# accuracy too optimistic. Hence the separate validation and test sets.

# %%
split_table = pd.crosstab(meta["source"], meta["split"]).loc[config.CLASSES, ["train", "val", "test", "ext"]]
split_table.loc["TOTAL"] = split_table.sum()
display(split_table)

# leakage check: no file may appear in two sets
members = {s: set(meta.loc[meta["split"] == s, "member"]) for s in ["train", "val", "test", "ext"]}
for a in members:
    for b in members:
        if a < b:
            assert not (members[a] & members[b]), f"data leak between {a} and {b}"
print("OK: train / val / test / ext are pairwise disjoint.")

# %% [markdown]
# Two helpers used throughout the notebook: fetch the data of one stage, and run a timed training.

# %%
def get_split(negative_sources=None):
    """{'train': (X, y, meta), 'val': ..., 'test': ...}. Photos are always included;
    `negative_sources` picks the negative classes (None = all of them)."""
    return {s: data.select(X, meta, s, negative_sources) for s in ("train", "val", "test")}


def run(split, epochs, early_stopping=False, patience=4, **model_kwargs):
    """Build a CNN (`model_kwargs` go to modeling.build_cnn), train it, return (model, history, seconds)."""
    keras.utils.set_random_seed(config.SEED)
    Xtr, ytr, _ = split["train"]
    Xva, yva, _ = split["val"]
    model = modeling.build_cnn(**model_kwargs)
    t0 = time.perf_counter()
    hist = modeling.train(
        model,
        data.make_dataset(Xtr, ytr, shuffle=True),
        data.make_dataset(Xva, yva),
        epochs=epochs,
        class_weight=data.class_weights(ytr),
        early_stopping=early_stopping,
        patience=patience,
        verbose=0,
    )
    return model, hist, time.perf_counter() - t0


# %% [markdown]
# ## 4. Increasing difficulty
#
# The brief suggests starting with the images that are easiest to tell apart from photos. We train **the same
# network** (described in section 5) on four problems of increasing difficulty, adding one negative image type at
# each stage:
#
# | Stage | Photos against |
# |---|---|
# | 1 | scanned text |
# | 2 | text + diagrams |
# | 3 | text + diagrams + sketches |
# | 4 | text + diagrams + sketches + **paintings** (the final problem) |
#
# In stages 1 to 3 there are more photos than negatives, so we use **class weights** (`class_weight`) and report the
# **balanced accuracy** (mean recall over the two classes) instead of the raw accuracy. Each stage is evaluated on its
# own `test` set, which training never sees.

# %%
stage_rows, stage_per_source, stage_models = [], {}, {}
for stage, negatives in config.STAGES.items():
    split = get_split(negatives)
    model, hist, seconds = run(split, EPOCHS["stage"], early_stopping=True, patience=3)
    Xte, yte, mte = split["test"]
    proba = modeling.predict_proba(model, Xte)
    res = evaluation.evaluate_binary(yte, proba)
    per_src = evaluation.per_source_accuracy(mte, proba)
    worst = per_src.iloc[0]
    stage_rows.append({
        "stage": stage, "n_train": len(split["train"][1]), "epochs": len(hist), "time (s)": round(seconds),
        "balanced_acc": res["balanced_accuracy"], "precision": res["precision"], "recall": res["recall"],
        "F1": res["f1"], "ROC-AUC": res["roc_auc"], "hardest class": worst["source"],
        "accuracy on that class": worst["accuracy"],
    })
    stage_per_source[stage] = per_src.set_index("source")["accuracy"]
    stage_models[stage] = model
stage_df = pd.DataFrame(stage_rows).set_index("stage")
display(stage_df)

# %%
fig, ax = plt.subplots(figsize=(9, 3.8))
pd.DataFrame(stage_per_source).loc[config.CLASSES].plot.bar(ax=ax, rot=0)
ax.set_ylim(0.5, 1.01)
ax.set_ylabel("accuracy on the stage test set")
ax.set_title("Accuracy per image type at each stage")
ax.legend(title="stage", fontsize=8)
plt.tight_layout()
plt.show()

# %%
last = stage_df.iloc[-1]
first = stage_df.iloc[0]
display(Markdown(
    f"**Results.** Stage 1 (photo vs text): balanced accuracy **{first['balanced_acc']:.3f}**. "
    f"Stage 4 (all classes, paintings included): **{last['balanced_acc']:.3f}**, F1 **{last['F1']:.3f}**. "
    f"At stage 4 the hardest class is **{last['hardest class']}** "
    f"(accuracy {last['accuracy on that class']:.3f}). "
    + ("This matches what we expected: realistic paintings look like photos and account for most errors."
       if last["hardest class"] == "Painting"
       else "Unexpectedly, paintings are not the main source of errors. This needs a closer look.")
))


# %% [markdown]
# ## 5. Network architecture
#
# We use a **convolutional neural network (CNN)** designed for this problem. The code printed below is the code
# actually used everywhere in this notebook (`src/tounum_cls/modeling.py`).

# %%
print(inspect.getsource(modeling._compile))   # loss, optimiser, metrics
print(inspect.getsource(modeling.build_cnn))  # architecture

# %%
baseline = modeling.build_cnn()
baseline.summary()

# %%
fig = evaluation.draw_architecture(baseline)
plt.show()

# %% [markdown]
# ### 5.1 What each layer does
#
# | Layer | Role |
# |---|---|
# | `Rescaling(1/255)` | maps pixels from [0, 255] to [0, 1]. It sits inside the model, so production code can feed raw images |
# | `Conv2D 3×3` ×4 (16, 32, 64, 128 filters) | detects local patterns: edges and textures in the first blocks, more complex shapes in the last ones. The filters are learned, unlike the hand-made convolution filters of deliverable 2 |
# | `BatchNormalization` | normalises the activations of each batch, which makes training faster and more stable |
# | `ReLU` | the non-linearity `max(0, x)` |
# | `MaxPooling2D 2×2` | halves the resolution: less computation and some tolerance to small shifts |
# | `GlobalAveragePooling2D` | averages each feature map into a single number. This avoids a huge dense layer, so the model has few parameters and overfits less |
# | `Dense(64, ReLU)` | combines the 128 features |
# | `Dense(1, sigmoid)` | output: the probability *P(photo)* in [0, 1] |
#
# ### 5.2 Network parameters

# %%
rows = []
for layer in baseline.layers:
    rows.append({
        "layer": layer.name, "type": type(layer).__name__,
        "output shape": str(tuple(layer.output.shape)[1:]), "parameters": layer.count_params(),
    })
param_table = pd.DataFrame(rows)
display(param_table[param_table["parameters"] > 0])
trainable = int(sum(np.prod(w.shape) for w in baseline.trainable_weights))
non_trainable = int(sum(np.prod(w.shape) for w in baseline.non_trainable_weights))
print(f"Total: {baseline.count_params():,} parameters, of which {trainable:,} trainable and "
      f"{non_trainable:,} non-trainable (BatchNorm moving means and variances)")

# %% [markdown]
# ### 5.3 Loss function
#
# The output is binary, so we use the **binary cross-entropy**. For an image with true label $y \in \{0, 1\}$ and a
# predicted probability $\hat{p}$ = P(photo):
#
# $$\mathcal{L}(y, \hat{p}) = -\big[\, y \log \hat{p} + (1 - y)\log(1 - \hat{p}) \,\big]$$
#
# The loss is averaged over the batch. It is small when the network is both right and confident, and very large
# when it is confidently wrong (a painting called "photo" with 99% confidence). It is the natural loss for a
# sigmoid output (maximum likelihood of a Bernoulli variable) and gives better gradients than the squared error.
#
# ### 5.4 Optimisation algorithm
#
# **Adam** (learning rate $10^{-3}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$). Adam is stochastic gradient descent on
# mini-batches (64 images here) that adapts the step of each parameter using a moving average of the gradient
# (momentum) and of its square (scale). It is a reliable default that converges quickly without fine-tuning the
# learning rate. We track the **accuracy** (share of images correctly classified at the 0.5 threshold) and the
# **AUC** (how well photos are ranked above non-photos, whatever the threshold).
#
# ### 5.5 Why this architecture?
# - **A CNN rather than a dense network**: sharing filters across the image uses its 2D structure with very few
#   parameters (about 106k, against millions for a dense network on 128×128×3 pixels).
# - **A small network**: the task is a single global decision and we train on CPU.
# - **Backed by results**: section 7 compares this network with a much smaller and a larger version, and section 10
#   with a pre-trained network of several million parameters.

# %% [markdown]
# ## 6. Training and learning curves
#
# We train the reference model (*baseline*, no regularisation) on the **final** problem (stage 4: photos against
# every other class). The curves compare training and validation (the "test error" of the brief) at each epoch,
# for the loss and for the accuracy.

# %%
FINAL_SPLIT = get_split(None)
Xtr, ytr, mtr = FINAL_SPLIT["train"]
Xva, yva, mva = FINAL_SPLIT["val"]
Xte, yte, mte = FINAL_SPLIT["test"]
print(f"train {Xtr.shape[0]} | val {Xva.shape[0]} | test {Xte.shape[0]} | photos in train: {ytr.mean():.1%}")

baseline_model, baseline_hist, baseline_sec = run(FINAL_SPLIT, EPOCHS["main"])
print(f"Training done in {baseline_sec:.0f} s ({baseline_sec / len(baseline_hist):.1f} s per epoch)")
shown = sorted({i for i in (0, 1, 2, len(baseline_hist) // 2, len(baseline_hist) - 1) if i < len(baseline_hist)})
display(baseline_hist.round(4).set_index("epoch").iloc[shown])

# %%
evaluation.plot_history(baseline_hist, "Baseline")
plt.show()

# %%
diag = evaluation.diagnose_fit(baseline_hist)
display(Markdown(
    f"**Results (baseline).** Final accuracy: training **{diag['train_acc']:.3f}**, "
    f"validation **{diag['val_acc']:.3f}** (gap {diag['gap_acc']:+.3f}). "
    f"Final loss: training {diag['train_loss']:.3f}, validation {diag['val_loss']:.3f}. "
    f"Lowest validation loss at epoch **{diag['best_epoch']}** out of {len(baseline_hist)} "
    f"(increase since that minimum: {diag['val_loss_drift']:+.3f}). Diagnosis: **{diag['verdict']}**."
))

# %% [markdown]
# **Reading the curves.**
# - In the first epochs the validation curve can jump around, sometimes badly, because `BatchNormalization` uses
#   moving statistics that need a few steps to settle. This is not a flaw of the model.
# - If the training loss keeps falling while the validation loss goes back up, the network is **memorising** the
#   training set (overfitting).
# - If both curves stay high, the network is **too simple** or not trained long enough (underfitting).


# %% [markdown]
# ## 7. Bias / variance analysis
#
# - **High bias = underfitting**: the model is too rigid to capture the structure of the data. The error is high on
#   both the training and the validation set.
# - **High variance = overfitting**: the model fits the noise of the training data. The training error is low but
#   the validation error is clearly higher (large gap).
# - What we want: both errors low and close to each other.
#
# A more complex model lowers the bias but raises the variance. We check this in two ways.
#
# The automatic diagnoses use fixed thresholds (training accuracy below 0.90 means bias; a train-validation gap
# above 0.05, or a validation loss more than 0.05 above its minimum, means variance). A case close to a threshold
# can flip from one run to the next.
#
# ### 7.1 Three model capacities
# Same data, same procedure; only the size of the network changes.

# %%
CAPACITY = {
    "too simple": dict(filters=(4, 8), dense_units=4),
    "baseline": dict(),
    "too complex": dict(filters=(32, 64, 128, 256), dense_units=256),
}
cap_hist, cap_models, cap_rows = {"baseline": baseline_hist}, {"baseline": baseline_model}, []
for name, kwargs in CAPACITY.items():
    if name == "baseline":
        model, hist, sec = baseline_model, baseline_hist, baseline_sec
    else:
        model, hist, sec = run(FINAL_SPLIT, EPOCHS["main"], **kwargs)
        cap_hist[name], cap_models[name] = hist, model
    d = evaluation.diagnose_fit(hist)
    cap_rows.append({"model": name, "parameters": model.count_params(), "train acc": d["train_acc"],
                     "val acc": d["val_acc"], "gap": d["gap_acc"], "min val loss": hist["val_loss"].min(),
                     "epoch of min": d["best_epoch"], "diagnosis": d["verdict"]})
cap_df = pd.DataFrame(cap_rows).set_index("model")
display(cap_df)

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
for name, hist in cap_hist.items():
    line, = axes[0].plot(hist["epoch"], hist["loss"], ls="--")
    axes[0].plot(hist["epoch"], hist["val_loss"], color=line.get_color(), label=name)
    line, = axes[1].plot(hist["epoch"], hist["accuracy"], ls="--")
    axes[1].plot(hist["epoch"], hist["val_accuracy"], color=line.get_color(), label=name)
axes[0].set_title("Loss (dashed: training, solid: validation)")
axes[1].set_title("Accuracy (dashed: training, solid: validation)")
for ax in axes:
    ax.set_xlabel("epoch")
    ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 7.2 Learning curve: what happens with more data?
# We train the baseline on a growing share of the training set (stratified by class) and compare the final training
# and validation accuracy. A **large gap that closes as data grows** points to variance, which more images would
# reduce. Two curves that are **close but low** point to bias, which more data will not fix.

# %%
from sklearn.model_selection import train_test_split

curve_rows = []
for frac in (0.1, 0.25, 0.5, 1.0):
    if frac < 1.0:
        keep, _ = train_test_split(np.arange(len(ytr)), train_size=frac, stratify=mtr["source"], random_state=config.SEED)
    else:
        keep = np.arange(len(ytr))
    sub = {"train": (Xtr[keep], ytr[keep], mtr.iloc[keep]), "val": FINAL_SPLIT["val"], "test": FINAL_SPLIT["test"]}
    _, hist, _ = run(sub, EPOCHS["curve"])
    d = evaluation.diagnose_fit(hist)
    curve_rows.append({"fraction": frac, "n_train": len(keep), "train acc": d["train_acc"], "val acc": d["val_acc"]})
curve_df = pd.DataFrame(curve_rows).set_index("fraction")
display(curve_df)

fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot(curve_df["n_train"], curve_df["train acc"], marker="o", label="training")
ax.plot(curve_df["n_train"], curve_df["val acc"], marker="o", label="validation")
ax.set_xlabel("number of training images")
ax.set_ylabel("accuracy (last epoch)")
ax.set_title("Learning curve (baseline)")
ax.legend()
plt.tight_layout()
plt.show()

# %%
small, big = curve_df.iloc[0], curve_df.iloc[-1]
delta_big = cap_df.loc["too complex", "val acc"] - cap_df.loc["baseline", "val acc"]
ratio_big = cap_df.loc["too complex", "parameters"] / cap_df.loc["baseline", "parameters"]
delta_small = cap_df.loc["too simple", "val acc"] - cap_df.loc["baseline", "val acc"]
capacity_note = (
    f" Multiplying the number of parameters by {ratio_big:.1f} changes the validation accuracy by {delta_big:+.3f}"
    + (", so the baseline already has enough capacity for this task." if abs(delta_big) < 0.01
       else (", so the larger model does better." if delta_big > 0 else ", so the larger model does worse."))
    + f" The \"too simple\" model is {delta_small:+.3f} away from the baseline."
)
display(Markdown(
    "**Results.** "
    f"Capacity: the \"too simple\" model ({cap_df.loc['too simple', 'parameters']:,} parameters) reaches "
    f"{cap_df.loc['too simple', 'val acc']:.3f} on validation, the baseline ({cap_df.loc['baseline', 'parameters']:,}) "
    f"{cap_df.loc['baseline', 'val acc']:.3f} and the \"too complex\" model "
    f"({cap_df.loc['too complex', 'parameters']:,}) {cap_df.loc['too complex', 'val acc']:.3f}. "
    f"Diagnoses: too simple, *{cap_df.loc['too simple', 'diagnosis']}*; baseline, "
    f"*{cap_df.loc['baseline', 'diagnosis']}*; too complex, *{cap_df.loc['too complex', 'diagnosis']}*. "
    f"Learning curve: with {int(small['n_train'])} images the train-validation gap is "
    f"{small['train acc'] - small['val acc']:+.3f}; with {int(big['n_train'])} images it is "
    f"{big['train acc'] - big['val acc']:+.3f}."
    + capacity_note
))

# %% [markdown]
# **Interpretation.** A gap that shrinks as the number of images grows means we are in a **variance** regime: more
# data, or regularisation, will help. A model whose training accuracy is itself too low is in a **bias** regime: it
# needs more capacity or more epochs, not more data. The numbers above show which regime each configuration is in.

# %% [markdown]
# ## 8. Improving the bias / variance trade-off
#
# ### 8.1 Available methods
#
# | Method | Idea | Expected effect | Used here |
# |---|---|---|---|
# | **More data** | more examples, less memorisation | lower variance | limited to the sample (see 7.2) |
# | **Data augmentation** | random transformations (flip, rotation, zoom, contrast) at every epoch, so the network never sees exactly the same image twice | lower variance | yes |
# | **Dropout** | randomly switches off a share of the neurons at each step, which prevents co-adaptation | lower variance | yes (0.4) |
# | **L2 regularisation** (*weight decay*) | adds $\lambda \sum w^2$ to the loss: large weights are penalised and the model gets smoother | lower variance | yes ($\lambda = 10^{-4}$) |
# | **Early stopping** | stops training when the validation loss stops improving and restores the best weights | lower variance | yes (patience 5) |
# | **Lower capacity** | fewer layers or filters | lower variance, higher bias | see 7.1 |
# | **BatchNormalization** | normalises activations; slight regularising effect | stability | already in the network |
# | **More capacity or epochs** | more filters, longer training | lower bias | see 7.1 |
# | **Adaptive learning rate** | lower the step when validation stalls (`ReduceLROnPlateau`) | finer convergence | not used |
# | **Transfer learning** | start from a network pre-trained on ImageNet | lower bias and variance with little data | section 10 |
#
# ### 8.2 Experiments
# Same problem (stage 4), same seed. Each technique is tried **alone**, then **all together** (with early stopping).
# Early stopping gets a larger epoch budget since it decides on its own when to stop.

# %%
IMPROVEMENTS = {
    "baseline": dict(kwargs=dict(), epochs=EPOCHS["main"], es=False),
    "+ augmentation": dict(kwargs=dict(augmentation=True), epochs=EPOCHS["main"], es=False),
    "+ dropout 0.4": dict(kwargs=dict(dropout=0.4), epochs=EPOCHS["main"], es=False),
    "+ L2 1e-4": dict(kwargs=dict(l2=1e-4), epochs=EPOCHS["main"], es=False),
    "+ early stopping": dict(kwargs=dict(), epochs=EPOCHS["improve"], es=True),
    "all combined": dict(kwargs=dict(augmentation=True, dropout=0.4, l2=1e-4), epochs=EPOCHS["improve"], es=True),
}
imp_hist, imp_models, imp_rows = {}, {}, []
for name, cfg in IMPROVEMENTS.items():
    if name == "baseline":
        model, hist, sec = baseline_model, baseline_hist, baseline_sec
    else:
        model, hist, sec = run(FINAL_SPLIT, cfg["epochs"], early_stopping=cfg["es"], patience=5, **cfg["kwargs"])
    imp_hist[name], imp_models[name] = hist, model
    d = evaluation.diagnose_fit(hist, restored_best=cfg["es"])  # early stopping: restored weights = best epoch
    val_metrics = model.evaluate(data.make_dataset(Xva, yva), verbose=0, return_dict=True)  # FINAL model weights
    imp_rows.append({"configuration": name, "epochs run": len(hist), "kept epoch": d["kept_epoch"],
                     "train acc (kept epoch)": d["train_acc"],
                     "train-val gap (kept epoch)": d["gap_acc"], "val loss (final model)": val_metrics["loss"],
                     "val acc (final model)": val_metrics["accuracy"], "diagnosis": d["verdict"],
                     "time (s)": round(sec)})
imp_df = pd.DataFrame(imp_rows).set_index("configuration")
display(imp_df)

# %%
base = imp_df.loc["baseline"]
others = imp_df.drop(index="baseline")
lower_loss = others.index[others["val loss (final model)"] < base["val loss (final model)"]].tolist()
higher_acc = others.index[others["val acc (final model)"] > base["val acc (final model)"]].tolist()
display(Markdown("**Results.** Compared with the baseline (validation loss "
    f"{base['val loss (final model)']:.3f}, accuracy {base['val acc (final model)']:.3f}), "
    f"lower validation loss: {', '.join(lower_loss) or 'none'}; "
    f"higher validation accuracy: {', '.join(higher_acc) or 'none'}. "
    + (f"The early-stopping configurations get {EPOCHS['improve']} epochs instead of {EPOCHS['main']}, "
       "so the budgets are not identical." if EPOCHS["improve"] != EPOCHS["main"] else "")))

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for name, hist in imp_hist.items():
    axes[0].plot(hist["epoch"], hist["val_loss"], label=name)
    axes[1].plot(hist["epoch"], hist["val_accuracy"], label=name)
axes[0].set_title("Validation loss")
axes[1].set_title("Validation accuracy")
for ax in axes:
    ax.set_xlabel("epoch")
axes[0].set_ylim(top=min(1.5, axes[0].get_ylim()[1]))
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.show()

# %%
best_name = imp_df["val loss (final model)"].idxmin()
display(Markdown(
    f"**Model selection (validation set only).** The configuration with the lowest validation loss is "
    f"**{best_name}** (loss {imp_df.loc[best_name, 'val loss (final model)']:.4f}, "
    f"accuracy {imp_df.loc[best_name, 'val acc (final model)']:.3f}); the baseline gets "
    f"{imp_df.loc['baseline', 'val loss (final model)']:.4f} / {imp_df.loc['baseline', 'val acc (final model)']:.3f}. "
    "This is the custom CNN evaluated on the test sets in the next section."
))
final_model = imp_models[best_name]
final_hist = imp_hist[best_name]

# %% [markdown]
# Curves of the selected model (training vs validation, loss and accuracy):

# %%
evaluation.plot_history(final_hist, f"Selected model ({best_name})")
plt.show()
d = evaluation.diagnose_fit(final_hist, restored_best=IMPROVEMENTS[best_name]["es"])
kept_kwargs = IMPROVEMENTS[best_name]["kwargs"]
reg_note = (" Note that with dropout and augmentation the training accuracy is measured with these perturbations "
            "switched on, so it is lower than it would be at inference time."
            if ("augmentation" in kept_kwargs or "dropout" in kept_kwargs) else "")
display(Markdown(
    f"Diagnosis of the selected model (kept epoch: {d['kept_epoch']}): training accuracy {d['train_acc']:.3f}, "
    f"validation {d['val_acc']:.3f}, so **{d['verdict']}**." + reg_note
))


# %% [markdown]
# ## 9. Final evaluation
#
# We now use the **`test`** set (balanced, never seen) and the **extended test set `ext`** (imbalanced, closer to a
# real archive) with the model selected in section 8. Both are used for measurement only, never to train, to pick a
# model or to set a threshold.

# %%
Xe, ye, me = data.select(X, meta, "ext")
results, probas = {}, {}
for label, (Xs, ys, ms) in {"test (balanced)": (Xte, yte, mte), "ext (realistic)": (Xe, ye, me)}.items():
    probas[label] = modeling.predict_proba(final_model, Xs)
    results[label] = evaluation.evaluate_binary(ys, probas[label])
res_df = pd.DataFrame(results).T
display(res_df[["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "tp", "fp", "fn", "tn"]])
print(f"Share of photos: test {yte.mean():.1%} | ext {ye.mean():.1%}")

# %%
for label, (Xs, ys, ms) in {"test (balanced)": (Xte, yte, mte), "ext (realistic)": (Xe, ye, me)}.items():
    evaluation.plot_confusion(ys, probas[label], title=f"Confusion matrix, {label}")
    plt.show()

# %% [markdown]
# ### 9.1 Where does the model go wrong?
# Accuracy per image type on the extended test set (the larger one):

# %%
per_src_ext = evaluation.per_source_accuracy(me, probas["ext (realistic)"])
display(per_src_ext)
fig, ax = plt.subplots(figsize=(6, 3))
sns.barplot(data=per_src_ext, x="source", y="accuracy", order=config.CLASSES, ax=ax)
ax.set_ylim(0.5, 1.0)
ax.set_xlabel("")
ax.set_title("Accuracy per image type, extended test set")
plt.tight_layout()
plt.show()

# %% [markdown]
# **False positives**: non-photos taken for photos, most confident first. These are the costly errors for TouNum
# because they end up in the captioning module.

# %%
fig = evaluation.show_errors(Xe, me, probas["ext (realistic)"], kind="fp", n=8)
if fig is None:
    print("No false positive on the extended test set.")
else:
    plt.show()

# %% [markdown]
# **False negatives**: photos that were rejected, most confident first.

# %%
fig = evaluation.show_errors(Xe, me, probas["ext (realistic)"], kind="fn", n=8)
if fig is None:
    print("No false negative on the extended test set.")
else:
    plt.show()

# %% [markdown]
# ### 9.2 Choosing the decision threshold
# 0.5 is only a default. For TouNum, **letting a non-photo through** (false positive, lower precision) and
# **losing a photo** (false negative, lower recall) do not cost the same, so the threshold is a business decision.
# The table below is descriptive: it shows how precision and recall move with the threshold on the extended test
# set, where photos are a minority. A production threshold should be set on validation data (or fresh data) and then
# confirmed, not tuned on this table, which is a measurement set.

# %%
rows = []
for thr in (0.1, 0.3, 0.5, 0.7, 0.9):
    r = evaluation.evaluate_binary(ye, probas["ext (realistic)"], threshold=thr)
    rows.append({"threshold": thr, "precision": r["precision"], "recall": r["recall"], "F1": r["f1"],
                 "false positives": r["fp"], "false negatives": r["fn"]})
display(pd.DataFrame(rows).set_index("threshold"))
evaluation.plot_precision_recall(ye, probas["ext (realistic)"], "Precision / recall, extended test set")
plt.show()

# %%
res_t, res_e = results["test (balanced)"], results["ext (realistic)"]
worst = per_src_ext.iloc[0]
display(Markdown(
    f"**Results.** Balanced test set: accuracy **{res_t['accuracy']:.3f}**, F1 **{res_t['f1']:.3f}**, "
    f"ROC-AUC **{res_t['roc_auc']:.3f}**. Extended test set (about {ye.mean():.0%} photos): accuracy "
    f"**{res_e['accuracy']:.3f}**, precision **{res_e['precision']:.3f}**, recall **{res_e['recall']:.3f}**, "
    f"F1 **{res_e['f1']:.3f}**. The worst classified image type is **{worst['source']}** "
    f"(accuracy {worst['accuracy']:.3f}). "
    + ("Precision is lower on the extended set because photos are a minority there: for the same false positive "
       "rate, false positives weigh more against fewer true positives. This is the metric to watch in production."
       if res_e["precision"] < res_t["precision"] - 0.01
       else "Precision holds up despite the class imbalance of the extended set.")
))

# %% [markdown]
# ### 9.3 Saving the custom CNN
# The model contains its own pre-processing (`Rescaling`), so it only needs 128×128×3 `uint8` images.
# `data.load_image` does the conversion (RGB and resizing) from any jpg or png file.

# %%
SAVE_DIR = Path(tempfile.gettempdir()) / "tounum_quick" if QUICK else config.MODELS_DIR   # QUICK never touches models/
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def save_and_check(model, filename):
    """Save a model, reload it and check that both give the same predictions."""
    path = SAVE_DIR / filename
    model.save(path)
    reloaded = keras.models.load_model(path)
    assert np.allclose(reloaded.predict(Xte[:32], verbose=0), model.predict(Xte[:32], verbose=0), atol=1e-5)
    shown = path.relative_to(config.PROJECT_ROOT) if path.is_relative_to(config.PROJECT_ROOT) else path
    print(f"Saved {shown} ({path.stat().st_size / 1e6:.1f} MB), reload checked.")
    return reloaded


save_and_check(final_model, "photo_cnn.keras")


def predict_photo(model, paths, threshold=0.5):
    """Classify image files. Returns a DataFrame (file, P(photo), is_photo)."""
    batch = np.stack([data.load_image(p) for p in paths])
    proba = modeling.predict_proba(model, batch)
    return pd.DataFrame({"file": [Path(p).name for p in paths], "P(photo)": proba.round(3),
                         "is_photo": proba >= threshold})


# %% [markdown]
# ## 10. Comparison with a pre-trained network (MobileNetV2)
#
# To put the custom architecture in perspective, we compare it with **MobileNetV2** pre-trained on ImageNet (over a
# million images), used as a **frozen** feature extractor: only the head (one sigmoid `Dense` layer) is trained.
# This is *transfer learning*. The ImageNet weights are downloaded on first use.

# %%
def summarize(model, name):
    row = {"model": name, "parameters": model.count_params(),
           "trainable": int(sum(np.prod(w.shape) for w in model.trainable_weights))}
    for label, (Xs, ys, ms) in {"val": (Xva, yva, mva), "test": (Xte, yte, mte), "ext": (Xe, ye, me)}.items():
        p = modeling.predict_proba(model, Xs)
        r = evaluation.evaluate_binary(ys, p)
        row[f"{label} accuracy"], row[f"{label} F1"], row[f"{label} AUC"] = r["accuracy"], r["f1"], r["roc_auc"]
        if label == "ext":
            acc = evaluation.per_source_accuracy(ms, p).set_index("source")["accuracy"]
            row["ext acc. Painting"] = acc.get("Painting", np.nan)
    return row


compare_rows = [summarize(final_model, f"custom CNN ({best_name})")]
mnet = None
try:
    mnet = modeling.build_mobilenet()
    keras.utils.set_random_seed(config.SEED)
    t0 = time.perf_counter()
    mnet_hist = modeling.train(mnet, data.make_dataset(Xtr, ytr, shuffle=True), data.make_dataset(Xva, yva),
                               epochs=EPOCHS["mobilenet"], class_weight=data.class_weights(ytr), verbose=0)
    mnet_sec = time.perf_counter() - t0
    row = summarize(mnet, "frozen MobileNetV2 + head")
    row["training time (s)"] = round(mnet_sec)
    compare_rows.append(row)
    evaluation.plot_history(mnet_hist, "MobileNetV2")
    plt.show()
except Exception as exc:  # e.g. no network access to download the ImageNet weights
    mnet = None
    print(f"MobileNetV2 comparison skipped: {type(exc).__name__}: {exc}")
compare_rows[0]["training time (s)"] = round(imp_df.loc[best_name, "time (s)"])
compare_df = pd.DataFrame(compare_rows).set_index("model")
display(compare_df)

# %%
production_model, production_name = final_model, "custom CNN"
if len(compare_df) == 2:
    a, b = compare_df.iloc[0], compare_df.iloc[1]
    gain = b["val accuracy"] - a["val accuracy"]  # the decision uses VALIDATION; test and ext are only reported
    ratio = b["parameters"] / a["parameters"]
    if gain > 0.02:
        production_model, production_name = mnet, "MobileNetV2"
        verdict = (f"MobileNetV2 has a better validation accuracy (+{gain:.3f}) with {ratio:.0f} times more "
                   "parameters, only about a thousand of which are trained. It becomes the production model. "
                   "The custom CNN stays available as a lighter model that does not depend on pre-trained weights.")
    elif gain < -0.02:
        verdict = (f"The custom CNN has a better validation accuracy (+{-gain:.3f}) with {ratio:.0f} times fewer "
                   "parameters. It stays the production model.")
    else:
        verdict = "Validation accuracy is about the same (gap under 0.02): the lighter custom CNN stays the production model."
    display(Markdown(
        f"**Results.** Custom CNN: {int(a['parameters']):,} parameters, accuracy {a['val accuracy']:.3f} (validation) / "
        f"{a['test accuracy']:.3f} (test) / {a['ext accuracy']:.3f} (ext), paintings {a['ext acc. Painting']:.3f}. "
        f"MobileNetV2: {int(b['parameters']):,} parameters ({int(b['trainable']):,} trainable), accuracy "
        f"{b['val accuracy']:.3f} / {b['test accuracy']:.3f} / {b['ext accuracy']:.3f}, paintings "
        f"{b['ext acc. Painting']:.3f}. {verdict} The decision is based on the validation set; the test and ext "
        "figures are reported for information."
    ))
else:
    verdict = "The MobileNetV2 comparison could not run, so the custom CNN is the production model."
    print(verdict)

# %% [markdown]
# ### 10.1 Saving the production model
# The production model is saved as `models/photo_classifier.keras`. This is the file the full pipeline of
# deliverable 3 loads. Like the custom CNN, it takes raw 128×128×3 `uint8` images.

# %%
classifier = save_and_check(production_model, "photo_classifier.keras")
demo = me.sample(8, random_state=config.SEED)
out = predict_photo(classifier, demo["path"].tolist())
out["true type"] = demo["source"].to_numpy()
display(out)

# %% [markdown]
# ## 11. Conclusion, limits and next steps
#
# ### Key figures
# The figures below are recomputed on every run from the results of the previous sections.

# %%
worst_src = per_src_ext.iloc[0]
display(Markdown(
    f"**Summary.** Selected custom CNN configuration (chosen on validation): **{best_name}**, "
    f"kept epoch {d['kept_epoch']}. Accuracy: validation **{imp_df.loc[best_name, 'val acc (final model)']:.3f}**, "
    f"test **{res_t['accuracy']:.3f}**, extended test **{res_e['accuracy']:.3f}**. "
    f"On the extended test set: precision **{res_e['precision']:.3f}**, recall **{res_e['recall']:.3f}**. "
    f"Worst classified image type (extended test): **{worst_src['source']}** (accuracy {worst_src['accuracy']:.3f}). "
    f"Production model: **{production_name}**. {verdict}"
))

# %% [markdown]
# ### What was built
# 1. A **reproducible workflow from raw archives to a trained model**: archive inventory, disjoint sampling,
#    selective extraction, normalisation (RGB, 128×128), cache, stratified split.
# 2. A **custom CNN of about 106k parameters** trained with Adam on the binary cross-entropy, checked on problems of
#    increasing difficulty (section 4) and compared with a pre-trained network (section 10).
# 3. A **bias/variance analysis** backed by numbers (sections 6 to 8) and a comparison of regularisation techniques.
# 4. An **evaluation on test sets that were never used for training or model selection**, one balanced and one
#    realistic (section 9), with an error analysis and a discussion of the threshold.
# 5. **Saved models**: `models/photo_cnn.keras` (custom CNN) and `models/photo_classifier.keras` (production model,
#    used by the pipeline of deliverable 3), plus an inference function (`predict_photo`).
#
# ### Limits
# - **Sample size**: 6,000 modelling images out of about 41,400. Training on everything could do slightly better.
#   The counts are set in `config.py`.
# - **128×128 resolution and stretching**: fine details are lost (they help tell a realistic painting from a photo)
#   and elongated pages are distorted.
# - **What counts as a "photo"**: the label comes from the source archive. A hyper-realistic painting or a heavily
#   edited photo stays ambiguous, even for a person, and labelling mistakes in the archives cap the reachable
#   accuracy.
# - **Generalisation**: the model is evaluated on images from the same sources. Images from other sources (other
#   scanners, other document types, other periods) may lower the results.
# - **Image quality**: all training images are reasonably clean. Deliverable 2 (denoising) checks how the filter
#   behaves on noisy images.
# - **Reproducibility**: small numerical differences are possible from one machine to another.
#
# ### Next steps
# - Train on all the archives at a higher resolution (with a GPU);
# - improve on realistic paintings (targeted augmentation, fine-tuning the last MobileNetV2 blocks);
# - set the threshold with TouNum based on the cost of each type of error (section 9.2);
# - plug this filter in front of the pre-processing and captioning modules (deliverables 2 and 3) and measure its
#   effect end to end;
# - monitor production (share of images predicted as photos, a sample checked by hand) to catch data drift.
