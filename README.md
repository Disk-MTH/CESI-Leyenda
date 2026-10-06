# TouNum: automatic captioning of digitised photos

Prototype of the automatic image captioning workflow for TouNum, a company that digitises documents. A folder of
digitised images goes in; a table comes out with, for every photo, a sentence that describes it. The workflow has
three modules, one per deliverable:

1. **Filtering** (deliverable 1): a classifier keeps the photos and sets aside paintings, diagrams, sketches and
   scanned text.
2. **Pre-processing** (deliverable 2): a convolutional autoencoder (U-Net with a residual output), trained on
   synthetic colour noise and fine-tuned on real noisy / clean pairs, removes the noise from the photos that need it.
3. **Captioning** (deliverable 3): a CNN pre-trained on ImageNet (InceptionV3) encodes the photo and a recurrent
   network (GRU) writes the caption word by word. Versions with and without attention are compared and the one
   with the better validation score is kept. Trained on 80,000 MS COCO 2017 images.

Each deliverable is an executed Jupyter notebook that explains the approach and shows the results:

| Deliverable | Notebook | Saved models |
|---|---|---|
| 1. Photo / non-photo classification | `notebooks/deliverable1_photo_classification.ipynb` | `models/photo_classifier.keras` (production, MobileNetV2), `models/photo_cnn.keras` (custom CNN) |
| 2. Denoising with convolutional autoencoders | `notebooks/deliverable2_denoising.ipynb` | `models/denoiser.keras`, `models/denoiser.json` |
| 3. Captioning and the complete workflow | `notebooks/deliverable3_captioning.ipynb` | `models/captioner.keras`, `models/captioner.json` |

## Repository layout

| Path | Content |
|---|---|
| `notebooks/*.ipynb` | executed notebooks (the deliverables) |
| `notebooks/*.py` | notebook sources in jupytext "percent" format: edit these, never the `.ipynb` |
| `src/tounum_cls/` | deliverable 1: archive sampling, image loading, CNN and MobileNetV2, metrics and plots |
| `src/tounum_denoise/` | deliverable 2: noisy dataset, synthetic noise, autoencoders, classical filters, noise estimate |
| `src/tounum_caption/` | deliverable 3: COCO data, InceptionV3 features, captioning model, BLEU, the end-to-end `pipeline.py` |
| `src/tounum_gpu.py` | lets TensorFlow use an NVIDIA GPU (see "GPU" below) |
| `tests/` | unit tests (`pytest`) on synthetic data |
| `scripts/run_notebook.sh` | rebuilds and runs a notebook (`quick` or `full`) |
| `scripts/check_notebook.py` | checks that an executed notebook is complete |
| `scripts/caption_folder.py` | runs the complete workflow on a folder from the command line |
| `scripts/find_noisy_originals.py` | finds the clean COCO originals of the deliverable 2 noisy photos (run once) |
| `pyproject.toml`, `uv.lock`, `.python-version` | pinned dependencies, managed by uv |
| `.vscode/` | VS Code tasks (tests, notebook runs, checks, captioning a folder) and debug configurations |
| `docs/` | design notes and implementation plan written for deliverable 1 |
| `Dataset Livrable *.zip` | archives provided by TouNum (not versioned) |
| `data/` | extracted images, caches, COCO sample and features (not versioned, rebuilt by the notebooks) |

## Installation

Requirements: [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`), Python 3.12
(installed by uv if needed), about 50 GB of free disk space for `data/` (COCO images and their InceptionV3
features), and 48 GB of RAM for deliverable 3: its training features (21 GB) are kept in memory and the notebook
uses about 30 GB at its peak. An internet
connection is needed on the first run of deliverables 2 and 3 (COCO images and captions, ImageNet weights).

```bash
uv sync        # creates .venv/ with the exact versions locked in uv.lock
```

The environment is managed by uv only: no `pip install`, no `python -m venv`. Every command goes through `uv run`.
To add a dependency: `uv add <package>` (updates `pyproject.toml` and `uv.lock`, commit both).

Put the archives provided by TouNum at the project root: the five `Dataset Livrable 1 - {Photo,Painting,Schematics,Text,Sketch}.zip`
and `Dataset Livrable 2.zip`.

### GPU

On Linux x86_64, `uv sync` also installs the CUDA libraries as Python packages (`tensorflow[and-cuda]`), so no
system-wide CUDA install is needed, only the NVIDIA driver. Without a GPU (or on macOS / Windows) TensorFlow runs on
the CPU and everything still works, only slower.

Two details, both handled by `src/tounum_gpu.py`, which the notebooks, tests and scripts call before importing
TensorFlow:

- The TensorFlow 2.21.0 wheel misses cuSOLVER in its library search path, and then silently falls back to the CPU.
  `tounum_gpu.setup()` loads that library first.
- TensorFlow 2.21 has no compiled kernels for the most recent GPUs (RTX 50xx, compute capability 12.0). The driver
  compiles them on the first run, which takes a few minutes once, and keeps them in a cache (`~/.nv/ComputeCache`)
  that `setup()` enlarges so they are not compiled again.

Check: `uv run python -c "import sys; sys.path.insert(0, 'src'); import tounum_gpu; tounum_gpu.setup(); import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

## Running the notebooks

The notebooks depend on each other: deliverable 2 reuses the image cache and the classifier of deliverable 1, and
deliverable 3 reuses the models of deliverables 1 and 2. Run them in order.

```bash
uv run pytest -q                        # unit tests

bash scripts/run_notebook.sh 1 quick    # short run (few epochs): writes notebooks/*.quick.ipynb only
bash scripts/run_notebook.sh 1 full     # full run = the deliverable (.ipynb + saved models)
bash scripts/run_notebook.sh 2 full
bash scripts/run_notebook.sh 3 full

uv run python scripts/check_notebook.py notebooks/deliverable1_photo_classification.ipynb   # completeness check
```

The quick mode never touches the deliverables or `models/` (its models go to a temporary folder). To open a
notebook interactively: `uv run --with jupyterlab jupyter lab`, or open it in VS Code with the `.venv` interpreter.

On an RTX 5090, a full run takes minutes for deliverable 1 and up to about an hour for deliverables 2 and 3. The
first run of deliverable 3 also downloads about 82,000 COCO images (around 15 minutes) and computes their features.
Without a GPU, deliverables 2 and 3 take many hours; reduce `N_TRAIN_IMAGES` in `src/tounum_caption/config.py` and
the epochs at the top of the notebooks if needed.

## Captioning a new folder

From the command line:

```bash
uv run python scripts/caption_folder.py path/to/folder --out captions.csv
```

From the deliverable 3 notebook: the last cell is self-contained. Set `INPUT_DIR` to the folder and run that cell
only; it loads the saved models and shows the table and a gallery of the results.

From Python:

```python
import sys; sys.path.insert(0, "src")
import tounum_gpu; tounum_gpu.setup()
from tounum_caption.pipeline import CaptionPipeline

pipeline = CaptionPipeline.load()                 # models/ of the three deliverables
results = pipeline.run("path/to/folder")          # pandas DataFrame, one row per image file
# columns: file, path, p_photo, is_photo, noise_sigma, denoised, caption
```

Images are searched recursively (jpg, jpeg, png). Non-photos get no caption; unreadable files are reported in the
`caption` column instead of stopping the run.

## Maintenance

- **Settings** live in each package's `config.py` (sample sizes, image sizes, noise range, vocabulary size...).
  After changing a sample size, delete the matching cache (`data/cache/`, `data/denoise_cache/` or `data/features/`)
  and run the notebook again.
- **Editing a notebook**: edit the `.py` file, then `bash scripts/run_notebook.sh <n> full`.
- **Numbers in the notebooks** are computed: the cells starting with "Results." are rebuilt on every run, and the
  static text does not quote figures that a new run could change.
- **Reproducibility**: seed `SEED = 42` everywhere. TensorFlow can still give slightly different numbers from one
  run or machine to another (more so on a GPU), so the last decimals may move.
