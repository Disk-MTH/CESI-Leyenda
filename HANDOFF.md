# Handoff: resume work on the GPU server

Written 2026-10-06 when the work moved from a 16-core CPU laptop (no GPU) to a server with an RTX 5090 and an i9.
Delete this file once all three deliverables are done.

## The user's request (original, in French, summarised)

"Check that deliverable 1 is good, then do deliverables 2 and 3. The deliverables' wording must be clean, easy to
understand and must not sound AI-made. No em dashes. Deliverables and README in English. Code and comments in
English. Ask me if you have questions." The full CESI brief (Projet Leyenda / TouNum) was pasted in the chat; the key
requirements are copied at the bottom of this file.

## Decisions the user already made (do not ask again)

- COCO for deliverable 3: **official COCO 2017**, sampled images downloaded one by one (not the 18 GB archive).
- Photo classifier used by the full pipeline: **MobileNetV2** (frozen, from deliverable 1). Deliverable 1 now saves
  it as `models/photo_classifier.keras` (production model) and the custom CNN as `models/photo_cnn.keras`.
- Git: work on branch **`deliverables-2-3`**, commit after each milestone, **never push**.
- Python env: **uv only** (`uv sync`, `uv run`, `uv add`), never pip/venv. Python 3.12.

## Writing rules for every deliverable (notebooks, README)

- English, plain and direct, no marketing words, not too much bold, no em dash (U+2014). `scripts/check_notebook.py`
  fails if an em dash is found.
- Every number quoted in prose must come from a computed cell. Computed result cells start with `**Results.**`
  (the checker counts them). Static markdown must not state results that a run could contradict.
- Notebooks are edited as jupytext `.py` files (`notebooks/*.py`), never the `.ipynb` by hand.
  `bash scripts/run_notebook.sh <1|2|3> quick|full` rebuilds and executes them.
  `uv run python scripts/check_notebook.py notebooks/<name>.ipynb [--partial]` checks the result.

## State at handoff

| Item | State |
|---|---|
| Deliverable 1 code, tests, notebook source | Translated to English (`notebooks/deliverable1_photo_classification.py`), MobileNetV2 saved as production model. Quick run passed. **Full run not done yet** (the `.ipynb` currently has no outputs). |
| `scripts/run_notebook.sh`, `scripts/check_notebook.py` | Generic for the 3 deliverables. |
| Deliverable 2 package `src/tounum_denoise/` | Written, 15 tests pass (`tests/test_denoise.py`). |
| Deliverable 2 notebook `notebooks/deliverable2_denoising.py` | Written, **never executed yet** (not even quick). |
| Deliverable 3 package `src/tounum_caption/` | Only `config.py` and `data.py` (download, sampling, caption cleaning, vocabulary). No features/model/pipeline yet, no tests yet. |
| Deliverable 3 notebook | Not started. |
| README.md | Still the old French deliverable 1 README. Must be rewritten in English for the 3 deliverables. |
| `.vscode/tasks.json`, `.vscode/launch.json` | Still French labels and old notebook names. Translate and update. |
| Data in the zip | 5 deliverable 1 archives + `Dataset Livrable 2.zip` at the root; `data/coco/` (COCO 2017 caption annotations + sampled images: 28,000 train + 2,000 val from train2017, 2,000 test from val2017, + all train2017 images with id < 3000). |

## Findings to reuse

- Deliverable 1 results (old CPU run): custom CNN test accuracy 0.909, MobileNetV2 0.954, paintings are the hard
  class. The full run took 35 min on CPU.
- **The 148 noisy images of deliverable 2 are COCO train2017 photos** (ids 1308 to 2349). Their clean originals were
  found by thumbnail matching (`scripts/find_noisy_originals.py`), the mapping is versioned in
  `src/tounum_denoise/noisy_originals.csv`, and `data.load_originals()` downloads them. This gives real ground truth.
- Real noise measured on the 148 pairs: zero mean, roughly Gaussian, independent per pixel and per channel, does not
  depend on brightness, sigma from 0.6 to 9.75 (mean 3.5) on the 0-255 scale, PSNR noisy vs clean 28 to 52 dB
  (mean 38). Files re-saved as JPEG quality 95 without chroma subsampling. Training noise = Gaussian sigma U(0, 30)
  + JPEG q95 re-encoding (`config.SIGMA_RANGE`, `config.JPEG_QUALITY`).
- Deliverable 2 denoiser training on CPU: about 0.3 s per batch of 32 patches 96x96. On the GPU it should be much
  faster, so the epoch count / patch count in the notebook can be raised if it helps.
- **To do in deliverable 3**: drop the 148 noisy-image originals (ids in `noisy_originals.csv`) from the captioning
  train/val sample after sampling (do not re-sample, the images are already downloaded), so the pipeline demo on the
  noisy images does not show memorised captions.

## Plan for deliverable 3 (agreed design, not yet coded)

- Image encoder: **InceptionV3** pre-trained on ImageNet, 299x299 input, last conv block 8x8x2048 = 64 regions x 2048.
  Features extracted once and cached as float16 in `data/features/` (memmap). Same features for train/val/test.
- Text: `data.clean_caption` (lowercase, letters only, `startseq ... endseq`), vocabulary of the 5,000 most frequent
  training words, max 22 tokens (`config.py`).
- Model (Keras functional, built-in layers only so it saves/loads without custom objects):
  image regions -> Dense(256, relu) (trainable CNN-side encoder) ; mean of regions -> Dense(512, tanh) = initial GRU
  state ; tokens -> Embedding(5000, 256, mask_zero) -> GRU(512, return_sequences) ; AdditiveAttention(query = GRU
  outputs, keys/values = projected regions) -> concat -> Dense -> logits over the vocabulary. Teacher forcing, loss =
  `SparseCategoricalCrossentropy(from_logits=True, ignore_class=0)`, Adam, early stopping on val loss.
  Optional ablation: same model without attention (Show and Tell style) to justify the attention.
- One random caption per image per epoch (keeps epochs short); validation on all val captions.
- Curves: train/val loss and masked token accuracy per epoch, plus BLEU-4 on a few hundred val images per epoch
  (callback with greedy decoding). Final BLEU-1..4 on the 2,000 test images (add `nltk` with `uv add nltk` for
  `corpus_bleu`). Show test examples with predicted vs reference captions and attention maps.
- Case study: caption quality (BLEU) on clean vs noisy vs denoised test images, to show the value of deliverable 2.
- Full pipeline (`src/tounum_caption/pipeline.py` + `scripts/caption_folder.py`): folder of images -> photo
  classifier (`models/photo_classifier.keras`) -> denoiser if estimated noise >= threshold (`models/denoiser.keras`,
  `models/denoiser.json`) -> InceptionV3 features -> captioner (`models/captioner.keras` + vocabulary JSON) ->
  table/gallery. The brief says that on presentation day only the last cell of the captioning notebook should need
  changing (the input folder), so the notebook must end with a cell `INPUT_DIR = ...` that runs the whole pipeline.
- Notebook sections 0 to 10 (the checker expects `## 0.` to `## 10.` and the strings `Total params`, `BLEU-4`,
  `captioner.keras`).

## Order of work on the server

1. `uv sync`, then check that TensorFlow sees the GPU (see the setup commands given in the chat). The RTX 5090 is a
   Blackwell card: if `tensorflow==2.21.0` does not see it, try `tensorflow[and-cuda]` and check the TF release notes
   for Blackwell (sm_120) support with ctx7 docs. Do not break the CPU path.
2. `uv run pytest -q` (45 tests expected: 30 deliverable 1 + 15 deliverable 2).
3. Deliverable 1: `bash scripts/run_notebook.sh 1 full`, check, read the outputs, make sure the prose still matches
   the numbers (GPU runs give slightly different numbers), commit.
4. Deliverable 2: quick run, fix errors, full run, check the static prose against the results (section 7 "Reading
   the curves" claims the plain autoencoder levels off lower), commit.
5. Deliverable 3: features, model, tests, notebook, pipeline, full run, commit.
6. README in English (all three deliverables, install, GPU notes, how to run the pipeline on a new folder),
   `.vscode` files, final check of all three notebooks, commit.

## Brief requirements (copied from the CESI subject)

- Stack: Python, scikit-learn, TensorFlow, pandas for datasets, ImageIO to load images, NumPy, Matplotlib.
- Fully automated solution (data to captions), readable as Jupyter notebooks, so TouNum can put it in production and
  maintain it. Notebooks must present the business context and prove the model's relevance rigorously.
- Deliverable 1 (binary classification): TensorFlow code + architecture diagram; parameters, loss and optimiser
  explained; training/test loss and accuracy curves; bias/variance analysis; methods to improve it (regularisation,
  dropout, early stopping...).
- Deliverable 2 (image processing): denoising with convolutional autoencoders; notebook with the loading code, the
  denoising code on a subset of noisy images with explanations, and a case study. RGB images.
- Deliverable 3 (captioning on MS COCO): CNN encodes images, RNN predicts the caption word by word; images first
  pre-processed by a CNN pre-trained for classification. Notebook with: complete schematic architecture stating the
  CNN used, a short description of image and text pre-processing, code of the CNN and RNN parts, training curves,
  some test examples. It also presents the whole solution (deliverables 1 + 2 + 3) as a reproducible workflow.
- Presentation day: 20 min talk + live demo on a dataset given on the day. Image pre-processing (pre-trained model
  and associated steps) must be ready beforehand; only the last cell of the captioning network should need adapting.
