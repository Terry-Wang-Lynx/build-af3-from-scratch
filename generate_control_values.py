"""Regenerate / verify every per-chapter ``control_values/*.pt`` artifact.

重新生成 / 校验每章的 ``control_values/*.pt``。

Usage / 用法 (run from the repository root)::

    # Regenerate every chapter's reference .pt files from solutions/.
    # 由 solutions/ 重新生成全部参考 .pt 文件:
    python generate_control_values.py

    # Verify your tutorials/ implementation matches the saved references.
    # 校验 tutorials/ 中你的实现是否与参考一致:
    python generate_control_values.py --verify --src tutorials

The script forces ``LAYERNORM_TYPE=torch`` so it runs everywhere (no
CUDA kernel needed).
脚本会强制 ``LAYERNORM_TYPE=torch``，无需 CUDA 算子，处处可跑。
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path


CHAPTERS = (
    "attention",
    "pairformer",
    "feature_embedding",
    "diffusion",
    "confidence",
    "model",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--verify", action="store_true",
        help="Verify that the saved .pt artifacts match the current solution. "
             "校验当前实现与保存的 .pt 一致 (不重写文件).")
    ap.add_argument(
        "--src", default="solutions",
        choices=("solutions", "tutorials"),
        help="Which source tree to test. 'solutions' (default) tests the "
             "reference impl; 'tutorials' tests the student version. "
             "测试的代码树：solutions (参考实现) 或 tutorials (学生版)。")
    ap.add_argument(
        "--chapters", nargs="+", default=list(CHAPTERS),
        choices=list(CHAPTERS),
        help="Subset of chapters to run. 仅测试列表中的章节。")
    args = ap.parse_args()

    os.environ["LAYERNORM_TYPE"] = "torch"
    repo_root = Path(__file__).resolve().parent
    src_dir = repo_root / args.src
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    overwrite = not args.verify
    for ch in args.chapters:
        mod = importlib.import_module(f"{ch}.control_values._generate")
        mod.main(overwrite=overwrite)


if __name__ == "__main__":
    main()
