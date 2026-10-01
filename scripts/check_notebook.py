"""Vérifie qu'un notebook exécuté est complet : exécuté sans erreur, sections et figures présentes.

Usage : python scripts/check_notebook.py notebooks/livrable1_classification_binaire.ipynb [--partial]
--partial : notebook en cours de construction -> vérifie seulement l'exécution sans erreur et affiche les comptes.
Code de sortie 0 si tout est conforme, 1 sinon.
"""
import sys

import nbformat

REQUIRED_HEADINGS = [f"## {i}." for i in range(0, 12)]
MIN_FIGURES = 13
MIN_AUTO_READINGS = 5          # cellules « Lecture automatique » (chiffres réels, pas de texte figé)


def check(path: str, partial: bool = False) -> list[str]:
    nb = nbformat.read(path, as_version=4)
    code = [c for c in nb.cells if c.cell_type == "code"]
    markdown = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    outputs = [o for c in code for o in c.get("outputs", [])]
    text = "\n".join(
        "".join(o.get("text", "")) + str(o.get("data", {}).get("text/markdown", "")) + str(o.get("data", {}).get("text/plain", ""))
        for o in outputs
    )
    problems = []
    n_fig = sum(1 for o in outputs if "image/png" in o.get("data", {}))
    if partial:
        if any(c.execution_count is None for c in code):
            problems.append("des cellules de code n'ont pas été exécutées")
        if any(o.output_type == "error" for o in outputs):
            problems.append("une cellule contient une erreur")
        print(f"{path}: {len(code)} cellules de code, {n_fig} figures")
        return problems
    for heading in REQUIRED_HEADINGS:
        if heading not in markdown:
            problems.append(f"section manquante : « {heading} »")
    if any(c.execution_count is None for c in code):
        problems.append("des cellules de code n'ont pas été exécutées")
    if any(o.output_type == "error" for o in outputs):
        problems.append("une cellule contient une erreur")
    if n_fig < MIN_FIGURES:
        problems.append(f"seulement {n_fig} figures (minimum {MIN_FIGURES})")
    if "Total params" not in text:
        problems.append("model.summary() absent (E1)")
    if "binary_crossentropy" not in text:
        problems.append("code de build_cnn absent des sorties (E1)")
    if text.count("Lecture automatique") < MIN_AUTO_READINGS:
        problems.append(f"moins de {MIN_AUTO_READINGS} cellules « Lecture automatique »")
    if "disjoints" not in text:
        problems.append("contrôle anti-fuite absent")
    print(f"{path}: {len(code)} cellules de code, {n_fig} figures, {text.count('Lecture automatique')} lectures automatiques")
    return problems


if __name__ == "__main__":
    issues = check(sys.argv[1], partial="--partial" in sys.argv)
    for issue in issues:
        print("  ✗", issue)
    print("OK" if not issues else f"{len(issues)} problème(s)")
    sys.exit(1 if issues else 0)
