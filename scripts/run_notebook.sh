#!/usr/bin/env bash
# Régénère le .ipynb depuis la source jupytext (.py) puis l'exécute (sorties incluses). Environnement géré par uv.
# Usage : bash scripts/run_notebook.sh quick   # 2 epochs partout, pour vérifier que tout s'exécute
#         bash scripts/run_notebook.sh full    # exécution complète (livrable)
set -euo pipefail
cd "$(dirname "$0")/../notebooks"
NB=livrable1_classification_binaire
uv run jupytext --to ipynb "$NB.py"
if [ "${1:-full}" = "quick" ]; then export TOUNUM_QUICK=1; fi
uv run jupyter nbconvert --to notebook --execute --inplace "$NB.ipynb" --ExecutePreprocessor.timeout=7200
