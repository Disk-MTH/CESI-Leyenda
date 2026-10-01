#!/usr/bin/env bash
# Régénère le .ipynb depuis la source jupytext (.py) puis l'exécute (sorties incluses). Environnement géré par uv.
# Usage : bash scripts/run_notebook.sh quick   # 2 epochs partout -> écrit notebooks/*.quick.ipynb (jamais le livrable)
#         bash scripts/run_notebook.sh full    # exécution complète = livrable (ipynb + models/cnn_photo_final.keras)
set -euo pipefail
cd "$(dirname "$0")/../notebooks"
NB=livrable1_classification_binaire
case "${1:-}" in
  quick)
    export TOUNUM_QUICK=1
    uv run jupytext --to ipynb -o "$NB.quick.ipynb" "$NB.py"
    uv run jupyter nbconvert --to notebook --execute --inplace "$NB.quick.ipynb" --ExecutePreprocessor.timeout=7200
    ;;
  full)
    unset TOUNUM_QUICK
    uv run jupytext --to ipynb "$NB.py"
    uv run jupyter nbconvert --to notebook --execute --inplace "$NB.ipynb" --ExecutePreprocessor.timeout=7200
    ;;
  *)
    echo "usage: bash scripts/run_notebook.sh quick|full" >&2
    exit 2
    ;;
esac
