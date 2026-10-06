"""Run the whole TouNum workflow on a folder of images: photo filter -> denoising if needed -> caption.

Usage: uv run python scripts/caption_folder.py <folder> [--out captions.csv] [--models models]
Prints one line per image and writes the full table as CSV.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import tounum_gpu  # noqa: E402

tounum_gpu.setup()

from tounum_caption.pipeline import CaptionPipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path, help="folder of images (searched recursively)")
    parser.add_argument("--out", type=Path, default=Path("captions.csv"), help="CSV file to write")
    parser.add_argument("--models", type=Path, default=ROOT / "models", help="folder with the saved models")
    args = parser.parse_args()

    results = CaptionPipeline.load(args.models).run(args.folder)
    for _, r in results.iterrows():
        if r["is_photo"]:
            print(f"{r['file']}: {r['caption']}" + (" (denoised)" if r["denoised"] else ""))
        else:
            print(f"{r['file']}: {r['caption'] or 'not a photo, no caption'}")
    results.to_csv(args.out, index=False)
    print(f"{len(results)} images, {int(results['is_photo'].sum())} photos captioned -> {args.out}")


if __name__ == "__main__":
    main()
