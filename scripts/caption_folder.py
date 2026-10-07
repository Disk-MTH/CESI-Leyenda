"""Lance tout le workflow TouNum sur un dossier d'images : filtre photo -> débruitage si besoin -> légende.

Usage : uv run python scripts/caption_folder.py <dossier> [--out captions.csv] [--models models]
Affiche une ligne par image et écrit le tableau complet en CSV.
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
    parser.add_argument("folder", type=Path, help="dossier d'images (parcouru récursivement)")
    parser.add_argument("--out", type=Path, default=Path("captions.csv"), help="fichier CSV à écrire")
    parser.add_argument("--models", type=Path, default=ROOT / "models", help="dossier des modèles sauvegardés")
    args = parser.parse_args()

    results = CaptionPipeline.load(args.models).run(args.folder)
    for _, r in results.iterrows():
        if r["is_photo"]:
            print(f"{r['file']}: {r['caption']}" + (" (débruitée)" if r["denoised"] else ""))
        else:
            print(f"{r['file']}: {r['caption'] or 'pas une photo, sans légende'}")
    results.to_csv(args.out, index=False)
    print(f"{len(results)} images, {int(results['is_photo'].sum())} photos légendées -> {args.out}")


if __name__ == "__main__":
    main()
