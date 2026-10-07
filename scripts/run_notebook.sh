#!/usr/bin/env bash
# Rebuild a deliverable notebook (.ipynb) from its jupytext source (.py) and execute it, outputs included.
# Usage: bash scripts/run_notebook.sh <1|2|3> quick   # short run (few epochs) -> writes notebooks/*.quick.ipynb only
#        bash scripts/run_notebook.sh <1|2|3> full    # full run = the deliverable (.ipynb + saved models)
set -euo pipefail
cd "$(dirname "$0")/../notebooks"
case "${1:-}" in
  1) NB=livrable1_classification_binaire ;;
  2) NB=livrable2_traitement_images ;;
  3) NB=livrable3_captioning ;;
  *) echo "usage: bash scripts/run_notebook.sh <1|2|3> quick|full" >&2; exit 2 ;;
esac
case "${2:-}" in
  quick)
    export TOUNUM_QUICK=1
    uv run jupytext --to ipynb -o "$NB.quick.ipynb" "$NB.py"
    uv run jupyter nbconvert --to notebook --execute --inplace "$NB.quick.ipynb" --ExecutePreprocessor.timeout=-1
    ;;
  full)
    unset TOUNUM_QUICK
    uv run jupytext --to ipynb "$NB.py"
    uv run jupyter nbconvert --to notebook --execute --inplace "$NB.ipynb" --ExecutePreprocessor.timeout=-1
    ;;
  *)
    echo "usage: bash scripts/run_notebook.sh <1|2|3> quick|full" >&2
    exit 2
    ;;
esac
