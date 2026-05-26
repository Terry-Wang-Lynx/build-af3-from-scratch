"""
Generate the student-facing `tutorials/` tree from the reference `solutions/`.

Each `TODO ... END OF YOUR CODE` block in solutions/ has its body replaced
by `pass`; the TODO pseudocode stays in place so students can fill in the
implementation step by step. Non-stubbed files are copied verbatim.

Run from the project root:

    python prepare_tutorials.py
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
from pathlib import Path

import numpy as np


def _lazy_nbformat():
    """Import nbformat lazily — we only need it if any .ipynb is found."""
    try:
        import nbformat  # type: ignore
        return nbformat
    except ImportError:
        print(
            "[prepare_tutorials] nbformat is not installed in the current env. "
            "Install it (`pip install nbformat`) if you have notebooks to process. "
            "Skipping notebook output-clearing for now."
        )
        return None


HERE = Path(__file__).resolve().parent
SOL = HERE / "solutions"
TUT = HERE / "tutorials"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
PYTHON_PATHS: list[str] = [
    # Each .py with TODO blocks. The .ipynb files in solutions/<chapter>/
    # are auto-picked up by glob below.
    "attention/mha.py",
    "attention/attention_pair_bias.py",
    "attention/transition.py",
    "attention/linear.py",
    "attention/layer_norm.py",
    "feature_embedding/input_embedder.py",
    "feature_embedding/relative_position_encoding.py",
    "feature_embedding/atom_attention.py",
    "feature_embedding/local_attention.py",
    "feature_embedding/constraint_embedder.py",
    "pairformer/pair_stack.py",
    "pairformer/msa_stack.py",
    "pairformer/template_embedder.py",
    "pairformer/triangle.py",
    "pairformer/triangle_ops.py",
    "pairformer/dropout.py",
    "diffusion/diffusion_module.py",
    "diffusion/diffusion_transformer.py",
    "diffusion/frames.py",
    "diffusion/sampler.py",
    "confidence/confidence_head.py",
    "confidence/distogram_head.py",
    "model/model.py",
    "model/utils.py",
    "model/inference.py",
]

FILE_COPY_PATHS: list[str] = [
    # Static support files copied verbatim.
    "model/__init__.py",
]

FOLDER_COPY_PATHS: list[str] = [
    # Whole subdirectories copied verbatim (data, configs, runtime,
    # control_values, etc).
    "configs",
    "runtime",
    "confidence",  # the __init__.py + helpers students don't reimplement
    "feature_extraction",
    "examples",
    # Per-chapter control values + test helpers — students need both the
    # .pt artifacts and the *_checks.py / _generate.py drivers.
    # 各章 control_values + 测试驱动 (.pt 参考 + checks.py + _generate.py)。
    "attention/control_values",
    "pairformer/control_values",
    "feature_embedding/control_values",
    "diffusion/control_values",
    "confidence/control_values",
    "model/control_values",
]


# ---------------------------------------------------------------------------
# TODO-block excision
# ---------------------------------------------------------------------------
def parse_file(src: Path, dst: Path) -> None:
    """Read src, replace every TODO block with `pass`, write dst."""
    is_ipynb = src.suffix == ".ipynb"
    lines = src.read_text().splitlines(keepends=True)

    hashtag_inds = [i for i, l in enumerate(lines) if "#####" in l and "##########" in l]
    if len(hashtag_inds) % 4 != 0:
        # No / malformed TODO blocks; just copy verbatim
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return

    groups = [hashtag_inds[i:i+4] for i in range(0, len(hashtag_inds), 4)]
    to_replace_inds = []
    for g in groups:
        todo_lines = "\n".join(lines[g[0]:g[1]])
        end_lines = "\n".join(lines[g[2]:g[3]])
        if "TODO" not in todo_lines or "END OF YOUR CODE" not in end_lines:
            continue
        to_replace_inds.append((g[1] + 1, g[2]))

    def calc_lead(line: str) -> int:
        stripped = line.replace("\\n", "\n")
        if not any(c.isalnum() for c in stripped):
            return -1
        return len(line) - len(line.lstrip())

    def gen_replacement(indent: int) -> list[str]:
        if is_ipynb:
            return [
                "\n",
                " " * indent + '# Replace "pass" statement with your code\n',
                " " * indent + "pass\n",
                "\n",
            ]
        return [
            "\n",
            " " * indent + '# Replace "pass" statement with your code\n',
            " " * indent + "pass\n",
            "\n",
        ]

    for start, stop in reversed(to_replace_inds):
        leads = np.array([calc_lead(a) for a in lines[start:stop]])
        leads = leads[leads != -1]
        indent = int(np.min(leads)) if leads.size else 0
        lines = lines[:start] + gen_replacement(indent) + lines[stop:]

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("".join(lines))


def clear_notebook_outputs(p: Path) -> None:
    nbformat = _lazy_nbformat()
    if nbformat is None:
        return
    nb = nbformat.read(p, as_version=4)
    nb.metadata.clear()
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
    nbformat.write(nb, p)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true", help="rm -rf tutorials/ first")
    args = ap.parse_args()

    if args.clean and TUT.exists():
        shutil.rmtree(TUT)
        print(f"  removed {TUT}")

    # 1. .py files with TODO blocks
    for rel in PYTHON_PATHS:
        src = SOL / rel
        dst = TUT / rel
        assert src.is_file(), f"missing: {src}"
        parse_file(src, dst)

    # 2. notebooks: copy + clear outputs
    for nb_path in SOL.glob("**/*.ipynb"):
        rel = nb_path.relative_to(SOL)
        dst = TUT / rel
        parse_file(nb_path, dst)
        clear_notebook_outputs(dst)

    # 3. static files
    for rel in FILE_COPY_PATHS:
        src = SOL / rel
        if not src.exists():
            continue
        dst = TUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    # 4. whole folders. Skip:
    #    - junk (__pycache__, ipynb_checkpoints, etc.)
    #    - any file already processed by step 1 (PYTHON_PATHS) — otherwise
    #      copytree would clobber the stubbed version with the solution.
    # 拷整个目录。要跳过:
    #   - 临时垃圾文件
    #   - step 1 已经处理过的 .py (PYTHON_PATHS) —— 否则会被原版覆盖。
    _junk_patterns = ("__pycache__", "*.py[cod]", ".DS_Store", ".ipynb_checkpoints")
    _already_stubbed = {str(SOL / p) for p in PYTHON_PATHS}

    def _ignore(directory: str, names: list[str]) -> set[str]:
        ignored = set(shutil.ignore_patterns(*_junk_patterns)(directory, names))
        for name in names:
            full = os.path.join(directory, name)
            if full in _already_stubbed:
                ignored.add(name)
        return ignored

    for rel in FOLDER_COPY_PATHS:
        src = SOL / rel
        if not src.is_dir():
            continue
        dst = TUT / rel
        # Don't rmtree — step 1 may have already populated dst with stubbed
        # files. Just merge over the top.
        # 不再 rmtree —— step 1 可能已经把 stub 写进 dst 了，
        # 这里仅在 stub 之上补齐。
        shutil.copytree(src, dst, ignore=_ignore, dirs_exist_ok=True)

    # 5. per-chapter __init__.py.
    #    Copy verbatim from solutions/ (preserves module-level docstrings),
    #    falling back to an empty file only when none exists upstream.
    #    Skip ``examples/`` — it's a data folder, not a Python package, so
    #    don't fabricate an __init__.py for it.
    # 复制每章 __init__.py (保留模块级 docstring)；上游不存在则建空文件。
    # ``examples/`` 是数据目录，不是 Python 包，不要给它造 __init__.py。
    _skip_init = {"__pycache__", "examples"}
    for ch in (SOL.iterdir()):
        if not ch.is_dir() or ch.name in _skip_init:
            continue
        (TUT / ch.name).mkdir(parents=True, exist_ok=True)
        sol_init = ch / "__init__.py"
        tut_init = TUT / ch.name / "__init__.py"
        if sol_init.is_file():
            shutil.copy2(sol_init, tut_init)
        elif not tut_init.exists():
            tut_init.write_text("")

    print(f"\nDone. Tutorials written under {TUT}")


if __name__ == "__main__":
    main()
