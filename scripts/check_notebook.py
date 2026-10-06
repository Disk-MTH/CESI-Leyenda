"""Check that an executed deliverable notebook is complete: run without errors, sections and figures present.

Usage: python scripts/check_notebook.py notebooks/deliverable1_photo_classification.ipynb [--partial]
--partial: notebook still being built (or quick run) -> only checks that it ran without errors and prints counts.
Exit code 0 if everything is fine, 1 otherwise.
"""
import sys
from pathlib import Path

import nbformat

# Per-deliverable requirements: (number of numbered sections, minimum figures, strings expected in the outputs)
REQUIREMENTS = {
    "deliverable1": (12, 14, ["Total params", "binary_crossentropy", "pairwise disjoint", "photo_classifier.keras"]),
    "deliverable2": (11, 12, ["Total params", "PSNR", "SSIM", "denoiser.keras"]),
    "deliverable3": (11, 10, ["Total params", "BLEU-4", "captioner.keras"]),
}
MIN_RESULTS = 4                # computed "Results." cells (real numbers, no hand-written figures)
FORBIDDEN = ["—"]         # em dash: kept out of the deliverables on purpose


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
    n_results = text.count("**Results")
    if any(c.execution_count is None for c in code):
        problems.append("some code cells were not executed")
    if any(o.output_type == "error" for o in outputs):
        problems.append("a cell raised an error")
    print(f"{path}: {len(code)} code cells, {n_fig} figures, {n_results} computed result cells")
    if partial:
        return problems
    key = Path(path).name.split("_")[0]
    n_sections, min_fig, expected = REQUIREMENTS[key]
    for i in range(n_sections):
        if f"## {i}." not in markdown:
            problems.append(f"missing section: '## {i}.'")
    if n_fig < min_fig:
        problems.append(f"only {n_fig} figures (minimum {min_fig})")
    for s in expected:
        if s not in text:
            problems.append(f"expected output not found: '{s}'")
    if n_results < MIN_RESULTS:
        problems.append(f"fewer than {MIN_RESULTS} computed result cells")
    for s in FORBIDDEN:
        if s in markdown or s in text:
            problems.append(f"forbidden character {s!r} found")
    if "QUICK mode: False" not in text:
        problems.append("the notebook was not run in full mode")
    return problems


if __name__ == "__main__":
    issues = check(sys.argv[1], partial="--partial" in sys.argv)
    for issue in issues:
        print("  x", issue)
    print("OK" if not issues else f"{len(issues)} problem(s)")
    sys.exit(1 if issues else 0)
