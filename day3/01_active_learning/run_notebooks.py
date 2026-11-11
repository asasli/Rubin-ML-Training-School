#!/usr/bin/env python3
"""Execute the two standalone active-learning school notebooks on CPU.

Run from any working directory after installing al-pipeline[teaching].
Outputs are saved separately; the distributed source notebooks are never edited.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
SCHOOL_ROOT = HERE.parents[1]
NOTEBOOKS = {
    "digits": ("01_active_learning_digits.ipynb", 5, "Selected from VALIDATION:"),
    "astronomy": ("02_applecider_astronomy_active_learning.ipynb", 6, "Contract checks passed"),
}


def execute(name: str, *, kernel: str, output_dir: Path) -> Path:
    source_name, minimum_figures, required_marker = NOTEBOOKS[name]
    source = HERE / source_name
    output = output_dir / source_name.replace(".ipynb", "_executed.ipynb")
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(
            f"{output} already exists. Remove it explicitly if you want to rerun."
        )
    notebook = nbformat.read(source, as_version=4)
    nbformat.validate(notebook)
    try:
        finished = NotebookClient(
            notebook, timeout=600, kernel_name=kernel, allow_errors=False,
            resources={"metadata": {"path": str(HERE)}},
        ).execute()
        figures = sum(
            bool(item.get("data", {}).get("image/png"))
            for cell in finished.cells if cell.cell_type == "code"
            for item in cell.get("outputs", [])
        )
        streams = "\n".join(
            str(item.get("text", ""))
            for cell in finished.cells
            for item in cell.get("outputs", [])
            if item.get("output_type") == "stream"
        )
        if figures < minimum_figures or required_marker not in streams:
            raise RuntimeError(
                f"{name}: {figures} figures, required marker present: "
                f"{required_marker in streams}"
            )
    except Exception:
        nbformat.write(notebook, output)
        print(f"FAIL: {name}. Diagnostic notebook: {output}", flush=True)
        raise
    nbformat.write(finished, output)
    print(f"PASS: {name}; {figures} embedded figures; {len(finished.cells)} cells")
    print(f"Saved: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("all", "digits", "astronomy"), default="all")
    parser.add_argument("--kernel", default="python3", help="Installed Jupyter kernel name")
    parser.add_argument(
        "--output-dir", type=Path, default=SCHOOL_ROOT / "results" / "active_learning",
        help="Output directory; the school's results/ directory is Git-ignored",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    names = NOTEBOOKS if args.only == "all" else (args.only,)
    for name in names:
        execute(name, kernel=args.kernel, output_dir=output_dir)
    print("Both notebooks execute offline after installation; the astronomy notebook "
          "replots archived validation summaries rather than retraining AstroMiNN.")


if __name__ == "__main__":
    main()
