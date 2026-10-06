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
# # TouNum, deliverable 2: denoising photos with convolutional autoencoders
#
# CESI Data Science team. Prototype of the automatic image captioning workflow.
#
# ---
# ## 0. Business context and goal
#
# TouNum digitises documents on a production line, and image quality varies: some scans are noisy or blurry.
# The captioning module (deliverable 3) works on photos, so it is worth cleaning them before they reach it.
# This deliverable covers the pre-processing step of the workflow:
#
# 1. Filtering: keep only the photos (deliverable 1);
# 2. **Pre-processing: remove the noise from the photos (this deliverable)**;
# 3. Captioning (deliverable 3).
#
# TouNum provided 148 noisy RGB photos. We build a **convolutional autoencoder** that takes a noisy photo and
# returns a cleaned one, and we compare it with the classical convolution filters (mean, Gaussian, median).
#
# ### Outline
# | Section | Content |
# |---|---|
# | 1 | Environment |
# | 2 | Loading the TouNum dataset |
# | 3 | What kind of noise? Analysis with the clean originals |
# | 4 | Training data: clean photos + synthetic noise |
# | 5 | Baseline: classical convolution filters |
# | 6 | Convolutional autoencoders: architecture, loss, optimiser |
# | 7 | Training and learning curves |
# | 8 | Evaluation by noise level |
# | 9 | Case study: the 148 TouNum images, when to denoise, effect on the photo filter |
# | 10 | Conclusion, limits and next steps |
#
# ### How to run it
# `bash scripts/run_notebook.sh 2 full` rebuilds the `.ipynb` from this `.py` file and runs it (about 1h30 on a
# 16-core CPU, most of it training). It needs `Dataset Livrable 2.zip` and the deliverable 1 archive
# `Dataset Livrable 1 - Photo.zip` at the project root, and deliverable 1 must have been run once (its image cache
# and its saved classifier are reused). `bash scripts/run_notebook.sh 2 quick` runs a very short version to check
# that everything executes; it never touches the deliverable or the saved models.

# %% [markdown]
# ## 1. Environment

# %%
import io
import os
import sys
import json
import time
import inspect
import tempfile
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
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

from tounum_cls import config as cls_config, data as cls_data, modeling as cls_modeling
from tounum_denoise import config, data, modeling, evaluation

QUICK = os.environ.get("TOUNUM_QUICK") == "1"
EPOCHS = 1 if QUICK else 25

keras.utils.set_random_seed(config.SEED)
pd.set_option("display.precision", 3)
pd.set_option("display.width", 140)
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 100

print(f"Python {sys.version.split()[0]} | TensorFlow {tf.__version__} | Keras {keras.__version__} | numpy {np.__version__}")
print("GPU:", tf.config.list_physical_devices("GPU") or "none (running on CPU)")
print(f"Seed: {config.SEED} | QUICK mode: {QUICK} | epochs: {EPOCHS}")

# %% [markdown]
# ## 2. Loading the TouNum dataset
#
# `Dataset Livrable 2.zip` holds the noisy photos as JPEG files. `data.extract_noisy()` unpacks them once into
# `data/noisy/`, and `data.load_noisy()` reads them at full size with ImageIO (as RGB `uint8` arrays) and builds a
# table with their size and an estimate of their noise level (explained in section 3.3).

# %%
noisy_paths = data.extract_noisy()
noisy_images, noisy_info = data.load_noisy(noisy_paths)
print(f"{len(noisy_images)} images loaded from {config.NOISY_ZIP.name}")
display(noisy_info.head())
sizes = noisy_info.groupby(["width", "height"]).size().sort_values(ascending=False)
print("Most common sizes (width x height):", ", ".join(f"{w}x{h} ({n})" for (w, h), n in sizes.head(4).items()))

# %%
fig, axes = plt.subplots(2, 4, figsize=(16, 6.5))
for ax, i in zip(axes.ravel(), np.linspace(0, len(noisy_images) - 1, 8).astype(int)):
    ax.imshow(noisy_images[i])
    ax.set_title(f"{noisy_info.file[i]} ({noisy_info.width[i]}x{noisy_info.height[i]})", fontsize=9)
    ax.axis("off")
plt.suptitle("Some of the noisy photos provided by TouNum")
plt.tight_layout()
plt.show()

# %% [markdown]
# At this scale the noise is hard to see. It shows up when we zoom in on a flat area (sky, wall, skin):

# %%
def center_crop(img, size=96):
    h, w = img.shape[:2]
    return img[h // 2 - size // 2: h // 2 + size // 2, w // 2 - size // 2: w // 2 + size // 2]


worst = noisy_info["noise_sigma"].nlargest(4).index
fig = evaluation.show_grid({"noisy image, 96x96 centre crop": [center_crop(noisy_images[i]) for i in worst]},
                           [noisy_info.file[i] for i in worst], size=3)
plt.show()

# %% [markdown]
# ## 3. What kind of noise?
#
# ### 3.1 The clean originals
# These photos are 640 pixels wide and look like MS COCO pictures, the dataset we use for captioning in
# deliverable 3. Comparing small thumbnails of the noisy files with the COCO 2017 training images confirmed it:
# every noisy file has an exact match among the COCO images with ids 1308 to 2349
# (`scripts/find_noisy_originals.py`, run once, stored in `src/tounum_denoise/noisy_originals.csv`).
#
# This gives us something rare in a denoising project: the **clean ground truth** for the real noisy images.
# We use it in two ways:
# - to measure the real noise exactly, and build a training noise that looks like it (this section);
# - to score the denoisers on the real TouNum images (section 9), not only on synthetic noise.
#
# These 148 originals are **never used for training**. The denoisers learn from other photos (section 4).

# %%
originals = data.load_originals(noisy_paths)
assert all(o.shape == n.shape for o, n in zip(originals, noisy_images))
residuals = [n.astype(np.float32) - o.astype(np.float32) for n, o in zip(noisy_images, originals)]
noisy_info["true_sigma"] = [r.std() for r in residuals]
noisy_info["psnr_noisy"] = [evaluation.psnr(data.to_float(o), data.to_float(n)) for o, n in zip(originals, noisy_images)]
display(noisy_info[["true_sigma", "noise_sigma", "psnr_noisy"]].describe().T)

# %%
i = int(noisy_info["true_sigma"].idxmax())
r = residuals[i]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, img, title in zip(axes[:3], [center_crop(noisy_images[i]), center_crop(originals[i]),
                                     np.clip(center_crop(r) * 6 + 128, 0, 255).astype(np.uint8)],
                          ["noisy", "clean original", "difference x6 (grey = 0)"]):
    ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")
xs = np.linspace(-25, 25, 200)
axes[3].hist(r.ravel(), bins=101, range=(-25, 25), density=True, alpha=0.7, label="noisy - original")
axes[3].plot(xs, np.exp(-xs**2 / (2 * r.std() ** 2)) / (r.std() * np.sqrt(2 * np.pi)), label=f"Gaussian, sigma={r.std():.1f}")
axes[3].set_title(f"Noise distribution ({noisy_info.file[i]})")
axes[3].legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3.2 Characteristics of the real noise

# %%
def channel_corr(res):
    flat = res.reshape(-1, 3)
    c = np.corrcoef(flat.T)
    return np.mean([c[0, 1], c[0, 2], c[1, 2]])


def spatial_corr(res):
    g = res.mean(axis=-1)
    return np.corrcoef(g[:, :-1].ravel(), g[:, 1:].ravel())[0, 1]


def corr_with_brightness(res, orig):
    """Correlation between the local noise amplitude and the brightness (0 = noise does not depend on it)."""
    lum = orig.mean(axis=-1).ravel()
    amp = np.abs(res).mean(axis=-1).ravel()
    return np.corrcoef(lum, amp)[0, 1]


stats = pd.DataFrame({
    "mean (bias)": [r.mean() for r in residuals],
    "channel correlation": [channel_corr(r) for r in residuals],
    "neighbour correlation": [spatial_corr(r) for r in residuals],
    "correlation with brightness": [corr_with_brightness(r, o) for r, o in zip(residuals, originals)],
})
display(stats.describe().T[["mean", "25%", "50%", "75%"]])

def jpeg_settings(source):
    """Mean luminance quantisation step and chroma sampling factors of a JPEG file."""
    with Image.open(source) as im:
        return float(np.mean(im.quantization[0])), im.layer[1][1:3]


buf = io.BytesIO()
Image.fromarray(originals[0]).save(buf, format="JPEG", quality=95, subsampling=0)
file_settings = {jpeg_settings(p) for p in noisy_paths}
print(f"JPEG settings of the 148 noisy files (quantisation step, chroma sampling): {file_settings}")
print(f"JPEG settings of a re-encoding at quality 95 without chroma subsampling: {jpeg_settings(buf)}")

# %% [markdown]
# What the numbers say about the noise:
# - it has **zero mean** and a roughly **Gaussian** shape (the histogram above);
# - it is drawn **independently for each channel** (low correlation between R, G and B) and **each pixel** (low
#   correlation between neighbours), so it is white noise, not blotches;
# - its amplitude does **not depend on brightness**, which rules out photon (Poisson) noise;
# - its level **varies from one image to another**, from almost nothing to a standard deviation of about 10
#   (on a 0-255 scale);
# - the files were saved as **JPEG at quality 95 without chroma subsampling** (the quantisation table above is the
#   standard table scaled for quality 95), which adds a little compression error on top.
#
# **Training noise model**: Gaussian noise drawn independently for each pixel and channel, with a standard deviation
# drawn uniformly between 0 and 30 for each image, followed by a JPEG re-encoding at quality 95. The range goes well
# above the real maximum (about 10) so the model also handles noisier batches than the 148 samples.
#
# ### 3.3 Estimating the noise level without the original
# In production there is no clean original. To decide whether an image needs denoising we use **Immerkaer's
# estimator**: a 3×3 convolution kernel that cancels smooth image content (it is the difference of two Laplacians),
# so that mostly noise is left; its mean absolute response gives the noise standard deviation. Detailed textures
# (grass, fur) also pass through the kernel, so the estimate is too high on textured images.

# %%
print("Immerkaer kernel:\n", data._IMMERKAER.astype(int))
fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(noisy_info["true_sigma"], noisy_info["noise_sigma"], s=12)
lim = max(noisy_info["true_sigma"].max(), noisy_info["noise_sigma"].max()) * 1.05
ax.plot([0, lim], [0, lim], "k--", lw=1, label="perfect estimate")
ax.set_xlabel("true noise sigma (from the original)")
ax.set_ylabel("estimated sigma (no original)")
ax.set_title("Noise level estimation on the 148 images")
ax.legend()
plt.tight_layout()
plt.show()
corr = np.corrcoef(noisy_info["true_sigma"], noisy_info["noise_sigma"])[0, 1]
display(Markdown(
    f"**Results.** True noise level of the 148 images: median sigma **{noisy_info['true_sigma'].median():.1f}**, "
    f"maximum **{noisy_info['true_sigma'].max():.1f}**; PSNR of the noisy images against their originals: median "
    f"**{noisy_info['psnr_noisy'].median():.1f} dB** (from {noisy_info['psnr_noisy'].min():.1f} to "
    f"{noisy_info['psnr_noisy'].max():.1f} dB). The blind estimate is correlated with the true level "
    f"(r = {corr:.2f}) but overestimates it by {(noisy_info['noise_sigma'] - noisy_info['true_sigma']).median():.1f} "
    "on average because of textures. The noise is light overall: above 35 dB, a person barely sees it at "
    "normal viewing size."
))

# %% [markdown]
# ## 4. Training data: clean photos + synthetic noise
#
# An autoencoder learns denoising from pairs (noisy image, clean image). We do not have such pairs at scale, so we
# make them: we take clean photos and add the synthetic noise of section 3.
#
# - **Clean photos**: the photos of deliverable 1 (`Dataset Livrable 1 - Photo.zip`, already extracted and sampled
#   there). The training split of deliverable 1 is used for training, its validation split for validation. Its test
#   split stays unseen and is used in section 9.3.
# - **Patches**: each photo is downscaled so its short side is 320 pixels, then we cut 4 random 96×96 patches
#   from it. The network is fully convolutional, so it learns on small patches and then works on images of any size.
# - **Noise drawn on the fly**: in the training pipeline (`tf.data`), each patch gets a new noise level, a new noise
#   and a random horizontal flip at every epoch. The validation noise is drawn once with a fixed seed so validation
#   scores are comparable from one epoch (and one model) to the next.

# %%
meta = pd.read_csv(cls_config.CACHE_DIR / "meta.csv")
photos = meta[meta["source"] == cls_config.POSITIVE_CLASS]
photo_paths = {s: [cls_config.PROJECT_ROOT / p for p in photos.loc[photos["split"] == s, "path"]] for s in ("train", "val", "test")}
print({s: len(p) for s, p in photo_paths.items()}, "deliverable 1 photos per split")

t0 = time.perf_counter()
train_patches = data.cached_patches("train", photo_paths["train"])
val_patches = data.cached_patches("val", photo_paths["val"], crops_per_image=2, seed=config.SEED + 1)
if QUICK:
    train_patches, val_patches = train_patches[:640], val_patches[:128]
print(f"train patches {train_patches.shape}, val patches {val_patches.shape} ({time.perf_counter() - t0:.0f} s)")

train_ds = data.make_dataset(train_patches)
val_noisy, val_clean, val_sigma = data.fixed_noisy_set(val_patches)

# %%
rng = np.random.default_rng(config.SEED)
example = train_patches[7]
cols = {"clean": [example]}
for s in (5, 10, 20, 30):
    cols[f"sigma = {s}"] = [data.add_noise(example, s, rng)]
evaluation.show_grid(cols, size=2.6)
plt.suptitle("Synthetic noise on a training patch", y=1.02)
plt.show()

# %% [markdown]
# ## 5. Baseline: classical convolution filters
#
# Before training anything we need a reference point. The simplest denoisers are **convolution filters**: each
# output pixel is a weighted average of its neighbours, which averages out the noise but also blurs the edges.
# - **Mean filter**: all pixels of a k×k window have the same weight.
# - **Gaussian filter**: nearby pixels weigh more (Gaussian weights), so it blurs a little less for the same effect.
# - **Median filter** (not a convolution): takes the median of the window, which keeps edges sharper and is very
#   good against isolated outliers (salt-and-pepper noise).

# %%
print("3x3 mean kernel:\n", evaluation.mean_kernel(3).round(3))
print("Gaussian kernel, sigma = 1 (7x7):\n", evaluation.gaussian_kernel(1.0).round(3))

# %%
def score(name, pred, clean=val_clean, sigma=val_sigma):
    """Per-patch PSNR / SSIM of a denoising method on the validation set."""
    return pd.DataFrame({"method": name, "sigma": sigma, "psnr": evaluation.psnr(clean, pred),
                         "ssim": evaluation.ssim(clean, pred)})


val_scores = [score("noisy (no filter)", val_noisy)]
for name, f in evaluation.CLASSICAL_FILTERS.items():
    val_scores.append(score(name, f(val_noisy)))
filter_table = pd.concat(val_scores).groupby("method", sort=False)[["psnr", "ssim"]].mean()
display(filter_table)
best_filter = filter_table.drop(index="noisy (no filter)")["psnr"].idxmax()

# %% [markdown]
# **PSNR** (peak signal-to-noise ratio, in dB) measures the pixel error against the clean image:
# $\mathrm{PSNR} = 10 \log_{10}(1 / \mathrm{MSE})$ for pixels in [0, 1]. +3 dB means the squared error is halved.
# **SSIM** (structural similarity, from 0 to 1) compares local brightness, contrast and structure, and is closer
# to what a person perceives. Both are averaged over the validation patches, whose noise level is spread between 0
# and 30.

# %% [markdown]
# ## 6. Convolutional autoencoders
#
# An **autoencoder** is a network trained to output an image from an input image. The **encoder** compresses the
# image into a smaller representation (fewer pixels, more channels), and the **decoder** rebuilds a full-size image
# from it. For denoising, the input is the noisy image and the target is the clean one: the network has to learn
# what belongs to the image and what is noise.
#
# We compare two variants built by the same function:
# 1. **Plain convolutional autoencoder**: everything goes through the bottleneck, at a quarter of the resolution.
# 2. **Autoencoder with skip connections** (U-Net style): each encoder level is also passed directly to the
#    decoder level of the same resolution. Fine details (edges, textures) no longer have to squeeze through the
#    bottleneck, and the decoder can focus on removing the noise.

# %%
print(inspect.getsource(modeling.build_autoencoder))

# %%
VARIANTS = {"plain autoencoder": dict(skips=False), "autoencoder + skip connections": dict(skips=True)}
for name, kwargs in VARIANTS.items():
    evaluation.draw_autoencoder((32, 64, 128), title=f"{name}", **kwargs)
    plt.show()

# %%
keras.utils.set_random_seed(config.SEED)
modeling.build_autoencoder(skips=True).summary()

# %% [markdown]
# ### 6.1 Parameters, loss and optimiser
#
# | Item | Choice | Why |
# |---|---|---|
# | Input / output | RGB image, float values in [0, 1], any size (multiple of 4) | fully convolutional, no dense layer |
# | Convolutions | 3×3, ReLU, `same` padding, 32 / 64 / 128 filters | standard building block; the receptive field grows at each level |
# | Downsampling / upsampling | MaxPooling 2×2 / Conv2DTranspose 2×2 stride 2 | halves / doubles the resolution, the transposed convolution learns how to upsample |
# | Output | Conv 1×1, 3 channels, sigmoid | keeps the pixels in [0, 1] |
# | **Loss** | **mean squared error** $\frac{1}{N}\sum (\hat{x} - x)^2$ between output and clean image | directly linked to PSNR, which is what we want to raise |
# | **Optimiser** | **Adam**, learning rate $10^{-3}$, batches of 32 patches | reliable default, no learning rate tuning needed |
# | Regularisation | early stopping (patience 4) on the validation loss, best weights restored | stops when validation stops improving |
# | Tracked metrics | PSNR and SSIM | the quality measures defined in section 5 |
#
# The two models have about the same number of parameters (420k and 467k): the skip connections only add channels
# to the first convolution of each decoder level.

# %% [markdown]
# ## 7. Training and learning curves

# %%
histories, models_, times = {}, {}, {}
for name, kwargs in VARIANTS.items():
    keras.utils.set_random_seed(config.SEED)
    model = modeling.build_autoencoder(**kwargs)
    t0 = time.perf_counter()
    histories[name] = modeling.train(model, train_ds, (val_noisy, val_clean), epochs=EPOCHS, patience=4, verbose=0)
    times[name] = time.perf_counter() - t0
    models_[name] = model
    print(f"{name}: {len(histories[name])} epochs in {times[name] / 60:.1f} min")

# %%
for name, hist in histories.items():
    evaluation.plot_history(hist, name)
    plt.show()

# %%
rows = []
for name, hist in histories.items():
    best = hist.loc[hist["val_loss"].idxmin()]
    rows.append({"model": name, "parameters": models_[name].count_params(), "epochs run": len(hist),
                 "best epoch": int(best["epoch"]), "train PSNR": best["psnr"], "val PSNR": best["val_psnr"],
                 "val SSIM": best["val_ssim"], "train - val PSNR gap": best["psnr"] - best["val_psnr"],
                 "time (min)": times[name] / 60})
train_table = pd.DataFrame(rows).set_index("model")
display(train_table)
chosen = train_table["val PSNR"].idxmax()
denoiser = models_[chosen]
gap = train_table.loc[chosen, "train - val PSNR gap"]
display(Markdown(
    f"**Results.** Best validation PSNR: **{chosen}** with **{train_table.loc[chosen, 'val PSNR']:.2f} dB** "
    f"(SSIM {train_table.loc[chosen, 'val SSIM']:.3f}), against "
    + ", ".join(f"{train_table.loc[n, 'val PSNR']:.2f} dB for the {n}" for n in train_table.index if n != chosen)
    + f" and {filter_table.loc[best_filter, 'psnr']:.2f} dB for the best classical filter ({best_filter}). "
    f"The training and validation PSNR of the selected model differ by {gap:+.2f} dB"
    + (": no sign of overfitting, the model could probably use more capacity or more epochs (bias side)."
       if abs(gap) < 0.5 else
       (": training is clearly better than validation, a sign of overfitting (variance side)." if gap > 0
        else ": validation is better than training. Training PSNR is averaged over the epoch while the weights "
             "are still changing, and the training noise is redrawn at every epoch, so this is not a problem."))
))

# %% [markdown]
# **Reading the curves.** The training noise is new at every epoch, so the network cannot memorise noisy patches:
# with about 8,000 patches seen under always different noise, overfitting is unlikely and the train and validation
# curves stay close. If a gap appeared, the usual remedies would be more photos, more augmentation (rotations,
# colour changes) or weight decay. The plain autoencoder levels off lower: it is limited by its bottleneck, which
# throws away fine detail (a bias problem that more data would not fix). Skip connections remove that limit.

# %% [markdown]
# ## 8. Evaluation by noise level
#
# Denoising is easy when there is little noise and hard when there is a lot, so a single average hides a lot. We
# now add noise at fixed levels to the validation patches and compare every method at each level. Level 0 checks
# that a clean image is not damaged.

# %%
def all_methods(noisy):
    """Outputs of every method on a batch of noisy float patches."""
    outs = {"noisy (no filter)": noisy, best_filter: evaluation.CLASSICAL_FILTERS[best_filter](noisy)}
    for name, model in models_.items():
        outs[name] = modeling.predict(model, noisy)
    return outs


level_rows, level_examples = [], {}
for s in config.SIGMA_LEVELS:
    rng = np.random.default_rng(config.SEED + s)
    noisy = data.to_float(np.stack([data.add_noise(p, s, rng) for p in val_patches]))
    for name, out in all_methods(noisy).items():
        level_rows.append({"sigma": s, "method": name, "PSNR": evaluation.psnr(val_clean, out).mean(),
                           "SSIM": evaluation.ssim(val_clean, out).mean()})
        level_examples[(s, name)] = out[11]
level_df = pd.DataFrame(level_rows)
display(level_df.pivot(index="method", columns="sigma", values="PSNR").round(2))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for name, grp in level_df.groupby("method", sort=False):
    axes[0].plot(grp["sigma"], grp["PSNR"], marker="o", label=name)
    axes[1].plot(grp["sigma"], grp["SSIM"], marker="o", label=name)
axes[0].set_ylabel("PSNR (dB)")
axes[1].set_ylabel("SSIM")
for ax in axes:
    ax.set_xlabel("noise sigma (0-255 scale)")
axes[0].legend(fontsize=8)
plt.suptitle("Quality after denoising, by noise level (validation patches)")
plt.tight_layout()
plt.show()

# %%
methods = list(all_methods(val_noisy[:1]).keys())
cols = {"clean": [val_patches[11]] * 2}
for name in methods:
    cols[name] = [level_examples[(10, name)], level_examples[(30, name)]]
evaluation.show_grid(cols, ["sigma = 10", "sigma = 30"], size=2.5)
plt.show()

# %%
pivot = level_df.pivot(index="method", columns="sigma", values="PSNR")
gain = pivot.loc[chosen] - pivot.loc["noisy (no filter)"]
display(Markdown(
    f"**Results.** Gain of the {chosen} over the noisy input: "
    + ", ".join(f"{gain[s]:+.1f} dB at sigma {s}" for s in config.SIGMA_LEVELS) + ". "
    f"The best classical filter ({best_filter}) gives "
    + ", ".join(f"{pivot.loc[best_filter, s] - pivot.loc['noisy (no filter)', s]:+.1f} dB" for s in config.SIGMA_LEVELS)
    + ". "
    + ("On clean images (sigma 0) the autoencoder costs a little quality: like any denoiser, it slightly smooths "
       "images that had nothing to remove. This is why section 9.2 decides when to apply it."
       if gain[0] < 0 else "Even on clean images (sigma 0) the autoencoder does not lower the quality.")
))

# %% [markdown]
# ## 9. Case study
#
# ### 9.1 The 148 TouNum images, scored against their originals
# This is the real test: images the models never saw, with the real noise, scored against the clean originals
# found in section 3. Each image is denoised at full size (the network is fully convolutional).

# %%
t0 = time.perf_counter()
denoised = [modeling.denoise(denoiser, img) for img in noisy_images]
filtered = [data.to_uint8(evaluation.CLASSICAL_FILTERS[best_filter](data.to_float(img))) for img in noisy_images]
print(f"148 images denoised in {time.perf_counter() - t0:.0f} s")

case = noisy_info[["file", "true_sigma", "noise_sigma", "psnr_noisy"]].copy()
case["psnr_denoised"] = [evaluation.psnr(data.to_float(o), data.to_float(d)) for o, d in zip(originals, denoised)]
case["psnr_filter"] = [evaluation.psnr(data.to_float(o), data.to_float(f)) for o, f in zip(originals, filtered)]
case["ssim_noisy"] = [evaluation.ssim(data.to_float(o), data.to_float(n)) for o, n in zip(originals, noisy_images)]
case["ssim_denoised"] = [evaluation.ssim(data.to_float(o), data.to_float(d)) for o, d in zip(originals, denoised)]
case["gain"] = case["psnr_denoised"] - case["psnr_noisy"]
display(case[["psnr_noisy", "psnr_filter", "psnr_denoised", "ssim_noisy", "ssim_denoised", "gain"]].describe().T[["mean", "min", "50%", "max"]])

# %%
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.scatter(case["true_sigma"], case["gain"], s=14, label=chosen)
ax.scatter(case["true_sigma"], case["psnr_filter"] - case["psnr_noisy"], s=14, alpha=0.6, label=best_filter)
ax.axhline(0, color="k", lw=1)
ax.set_xlabel("true noise sigma of the image")
ax.set_ylabel("PSNR gain over the noisy image (dB)")
ax.set_title("Gain on the 148 TouNum images")
ax.legend()
plt.tight_layout()
plt.show()

# %%
show = case["true_sigma"].nlargest(3).index.tolist() + [int(case["true_sigma"].idxmin())]
evaluation.show_grid({
    "noisy": [center_crop(noisy_images[i], 128) for i in show],
    best_filter: [center_crop(filtered[i], 128) for i in show],
    chosen: [center_crop(denoised[i], 128) for i in show],
    "clean original": [center_crop(originals[i], 128) for i in show],
}, [f"{case.file[i]}, sigma {case.true_sigma[i]:.1f}" for i in show], size=3)
plt.show()

# %%
improved = (case["gain"] > 0).mean()
display(Markdown(
    f"**Results.** On the 148 real images, the {chosen} brings the mean PSNR from **{case['psnr_noisy'].mean():.2f}** "
    f"to **{case['psnr_denoised'].mean():.2f} dB** ({case['gain'].mean():+.2f} dB) and the mean SSIM from "
    f"{case['ssim_noisy'].mean():.3f} to {case['ssim_denoised'].mean():.3f}. It improves **{improved:.0%}** of the "
    f"images. The best classical filter ({best_filter}) reaches {case['psnr_filter'].mean():.2f} dB. "
    f"The gain grows with the noise level: images with sigma above 5 gain "
    f"{case.loc[case['true_sigma'] > 5, 'gain'].mean():+.2f} dB on average, images below 2 gain "
    f"{case.loc[case['true_sigma'] < 2, 'gain'].mean():+.2f} dB."
))

# %% [markdown]
# ### 9.2 When should the pipeline denoise?
# A denoiser cannot add information: on an image that is already clean it can only smooth details away. The
# pipeline should therefore denoise only the images that need it. In production there is no original, so the
# decision has to rely on the **blind estimate** of section 3.3.
#
# We set the threshold on the **validation patches** (not on the 148 images, which stay a test set): for each
# patch we compare the PSNR before and after denoising, and look at the gain as a function of the estimated
# noise level. The threshold is the lowest estimated level from which the median gain stays positive.

# %%
val_denoised = modeling.predict(denoiser, val_noisy)
val_gain = evaluation.psnr(val_clean, val_denoised) - evaluation.psnr(val_clean, val_noisy)
val_est = np.array([data.estimate_noise(data.to_uint8(x)) for x in val_noisy])
bins = np.arange(0, 41, 2)
binned = pd.DataFrame({"estimated sigma": pd.cut(val_est, bins, right=False), "gain": val_gain})
by_bin = binned.groupby("estimated sigma", observed=True)["gain"].agg(["median", "size"])
by_bin = by_bin[by_bin["size"] >= 5]                     # ignore nearly empty bins
edges = [iv.left for iv in by_bin.index]
positive = (by_bin["median"] > 0).to_numpy()
# lowest level from which every bin has a positive median gain
NOISE_THRESHOLD = float(next((edges[k] for k in range(len(edges)) if positive[k:].all()), bins[-1]))

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.scatter(val_est, val_gain, s=6, alpha=0.4)
ax.axhline(0, color="k", lw=1)
ax.axvline(NOISE_THRESHOLD, color="tab:red", ls="--", label=f"threshold = {NOISE_THRESHOLD:.0f}")
ax.set_xlabel("estimated noise sigma (blind)")
ax.set_ylabel("PSNR gain (dB)")
ax.set_title("Gain vs estimated noise, validation patches")
ax.legend()
plt.tight_layout()
plt.show()

# check the rule on the 148 TouNum images (test set)
apply = case["noise_sigma"] >= NOISE_THRESHOLD
case["psnr_rule"] = np.where(apply, case["psnr_denoised"], case["psnr_noisy"])
display(Markdown(
    f"**Results.** Threshold set on validation: denoise when the estimated sigma is at least "
    f"**{NOISE_THRESHOLD:.0f}**. On the 148 TouNum images the rule denoises {apply.sum()} images and leaves "
    f"{(~apply).sum()} untouched. Mean PSNR: always denoise {case['psnr_denoised'].mean():.2f} dB, rule "
    f"{case['psnr_rule'].mean():.2f} dB, never denoise {case['psnr_noisy'].mean():.2f} dB. Images made worse: "
    f"{(case['gain'] < 0).sum()} when always denoising, {((case['gain'] < 0) & apply).sum()} with the rule."
))

# %% [markdown]
# ### 9.3 Does noise hurt the photo filter of deliverable 1?
# The pipeline classifies images **before** denoising them (only photos are denoised). So the photo filter must
# itself cope with noisy inputs. We take the test split of deliverable 1 (photos and non-photos never used here),
# add synthetic noise at increasing levels, and measure the accuracy of the production classifier saved by
# deliverable 1, with and without denoising first. To keep this fast, we use 60 images per type and downscale
# them so their short side is 320 pixels before adding the noise.

# %%
classifier_path = cls_config.MODELS_DIR / "photo_classifier.keras"
if QUICK and not classifier_path.exists():          # deliverable 1 only ran in quick mode
    classifier_path = Path(tempfile.gettempdir()) / "tounum_quick" / "photo_classifier.keras"
classifier = keras.models.load_model(classifier_path)
test_meta = meta[meta["split"] == "test"].groupby("source").sample(n=10 if QUICK else 60, random_state=config.SEED)
test_full = [data.downscale(cls_data.read_rgb(cls_config.PROJECT_ROOT / p)) for p in test_meta["path"]]
labels = test_meta["label"].to_numpy()

acc_rows = []
for s in (0, 10, 20, 30):
    rng = np.random.default_rng(config.SEED + 100 + s)
    noisy_set = [data.add_noise(img, s, rng) if s else img for img in test_full]
    for variant, imgs in (("noisy", noisy_set), ("denoised first", [modeling.denoise(denoiser, im) for im in noisy_set])):
        proba = cls_modeling.predict_proba(classifier, np.stack([cls_data.resize(im) for im in imgs]))
        pred = proba >= 0.5
        acc_rows.append({"sigma": s, "input": variant, "accuracy": (pred == labels).mean(),
                         "photos kept": pred[labels == 1].mean(), "non-photos rejected": (~pred[labels == 0]).mean()})
acc_df = pd.DataFrame(acc_rows)
display(acc_df.pivot(index="input", columns="sigma", values="accuracy").round(3))

# %%
acc0 = acc_df.query("sigma == 0 and input == 'noisy'")["accuracy"].iloc[0]
acc30 = acc_df.query("sigma == 30 and input == 'noisy'")["accuracy"].iloc[0]
acc30d = acc_df.query("sigma == 30 and input == 'denoised first'")["accuracy"].iloc[0]
display(Markdown(
    f"**Results.** On {len(labels)} test images ({labels.sum()} photos), the classifier accuracy goes from "
    f"**{acc0:.3f}** on the original images to **{acc30:.3f}** at sigma 30, and to **{acc30d:.3f}** at sigma 30 when "
    "the images are denoised first. "
    + ("The photo filter barely notices the noise: images are shrunk to 128×128 before classification, and the "
       "shrinking averages most of the noise away. Classifying before denoising is therefore safe."
       if acc0 - acc30 < 0.03 else
       "Strong noise does hurt the photo filter. For very noisy archives it would be worth denoising every image "
       "before classifying it, or training the classifier with noise augmentation.")
))

# %% [markdown]
# ### 9.4 Saving the denoiser

# %%
SAVE_DIR = Path(tempfile.gettempdir()) / "tounum_quick" if QUICK else config.MODELS_DIR
SAVE_DIR.mkdir(parents=True, exist_ok=True)
denoiser.save(SAVE_DIR / "denoiser.keras")
settings = {"model": chosen, "noise_threshold": NOISE_THRESHOLD, "pad_multiple": modeling.PAD_MULTIPLE}
(SAVE_DIR / "denoiser.json").write_text(json.dumps(settings, indent=2))
reloaded = keras.models.load_model(SAVE_DIR / "denoiser.keras")
assert np.array_equal(modeling.denoise(reloaded, noisy_images[0]), denoised[0])
shown = SAVE_DIR.relative_to(config.PROJECT_ROOT) if SAVE_DIR.is_relative_to(config.PROJECT_ROOT) else SAVE_DIR
print(f"Saved {shown}/denoiser.keras ({(SAVE_DIR / 'denoiser.keras').stat().st_size / 1e6:.1f} MB) and "
      f"{shown}/denoiser.json {settings}, reload checked.")

# %% [markdown]
# Usage in production (this is what the pipeline of deliverable 3 does):
# ```python
# from tounum_denoise import data, modeling
# denoiser = keras.models.load_model("models/denoiser.keras")
# settings = json.load(open("models/denoiser.json"))
# img = cls_data.read_rgb("photo.jpg")                       # uint8 RGB, any size
# if data.estimate_noise(img) >= settings["noise_threshold"]:
#     img = modeling.denoise(denoiser, img)                  # uint8 RGB, same size
# ```

# %% [markdown]
# ## 10. Conclusion, limits and next steps

# %%
display(Markdown(
    f"**Summary.** Retained model: **{chosen}** ({denoiser.count_params():,} parameters). Validation PSNR "
    f"{train_table.loc[chosen, 'val PSNR']:.2f} dB on synthetic noise (sigma 0 to 30). On the 148 TouNum images: "
    f"{case['psnr_noisy'].mean():.2f} dB before, {case['psnr_denoised'].mean():.2f} dB after "
    f"({improved:.0%} of images improved), against {case['psnr_filter'].mean():.2f} dB for the best classical filter. "
    f"Pipeline rule: denoise when the estimated noise sigma is at least {NOISE_THRESHOLD:.0f}."
))

# %% [markdown]
# ### What was built
# 1. Loading code for the TouNum dataset, and the discovery of the **clean originals** of the 148 noisy images,
#    which turned a qualitative exercise into a measured one.
# 2. An **analysis of the real noise** (Gaussian, white, per channel, level varying by image, JPEG quality 95) and a
#    synthetic noise built to match it.
# 3. A **comparison** between classical convolution filters, a plain convolutional autoencoder and an autoencoder
#    with skip connections, by noise level, on validation patches and on the real images.
# 4. A **decision rule** for the pipeline (denoise only when the estimated noise is high enough), set on validation
#    data and checked on the real images.
# 5. A check that the **photo filter of deliverable 1** still works on noisy images.
# 6. A **saved model** (`models/denoiser.keras`) and its settings (`models/denoiser.json`).
#
# ### Limits
# - **Light real noise**: the 148 images are only lightly noisy (median PSNR around 38 dB), so the measurable gain is
#   small in absolute terms. The model is trained for much stronger noise as well (up to sigma 30).
# - **One type of noise**: the model learned Gaussian noise plus JPEG. Other degradations seen in digitisation
#   (blur, scanner streaks, moiré, heavy JPEG blocks, dust) are not handled and would need their own training noise.
# - **Blind noise estimate**: Immerkaer's estimator overestimates the noise of textured images, so the rule may
#   denoise some clean but detailed photos. A learned estimator would be more precise.
# - **PSNR and SSIM are not the end goal**: what matters for TouNum is the caption. Deliverable 3 measures the effect
#   of noise and denoising on caption quality.
# - **CPU training**: about 8,000 patches and 25 epochs at most. More photos and longer training would raise the
#   ceiling, in particular for the plain autoencoder, which is limited by its bottleneck.
#
# ### Next steps
# - Add blur to the training degradations (deblurring), since the brief mentions blurry scans too;
# - collect a few real (noisy, clean) pairs from TouNum's own scanners to fine-tune the model on their actual noise;
# - try a residual formulation (the network predicts the noise and subtracts it), known to train faster;
# - measure the effect on captions end to end (deliverable 3).
