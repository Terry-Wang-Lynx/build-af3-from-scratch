"""Execute every chapter notebook end-to-end and report pass / fail.

依次执行每章 notebook 并汇报通过 / 失败。

Usage / 用法::

    python check_solutions.py                  # check solutions/
    python check_solutions.py --src tutorials  # check the student tree
    python check_solutions.py --chapters attention pairformer

By default runs every chapter and reports a summary at the end (useful
for students who want the full picture of what they still have to fill
in). Add ``--fail-fast`` to stop at the first failure (useful for CI).

默认跑完所有章节再汇总 (适合学生看自己还差哪些)；加 ``--fail-fast`` 改为
第一个失败就停 (适合 CI)。

This is the AF3 educational project's analogue of the AF2 reference's
``check_solutions.py``.

对标 AF2 教学项目的 ``check_solutions.py``。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

# Notebook execution deps are optional at the system level (they ship in the
# conda env files and the README pip block, but a stale/minimal venv may lack
# them). Import them with a concise, actionable message instead of letting a
# raw ModuleNotFoundError traceback escape before we can explain ourselves.
# notebook 执行依赖是可选的系统级依赖（conda env / README pip 都已声明，但
# 老旧或精简的 venv 里可能没装）。这里给出简明可操作的提示，避免直接抛
# ModuleNotFoundError 让用户看不懂。
try:
    import ipykernel  # noqa: F401
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
    sys.exit(
        f"check_solutions.py: missing notebook checker dependency '{exc.name}'.\n"
        "Install the notebook toolchain, then re-run:\n"
        "    pip install nbformat nbconvert ipykernel\n"
        "(these are also in environment_cpu.yml / environment_mac.yml.)\n"
        "缺少 notebook 检查依赖。请先安装上面的包再重试。"
    )


# Ordered the same way as the README's "学习路径" table.
# 与 README 学习路径表同序。
#
# ``CHAPTER_NOTEBOOKS`` are the per-chapter labs — every one is
# self-contained and runs in ~10s on CPU without any downloaded weights.
#
# ``OVERVIEW_NOTEBOOK`` is the end-to-end demo which **requires** the
# Protenix checkpoint under ``checkpoints/``. It is opt-in via
# ``--with-overview`` so a CI / fresh-clone run can be fast and offline.
#
# ``CHAPTER_NOTEBOOKS`` 是每章独立的实验，CPU 上各约 10 秒，不需要下载权重。
# ``OVERVIEW_NOTEBOOK`` 是端到端 demo，**需要** ``checkpoints/`` 下的 Protenix
# 权重；默认不跑，加 ``--with-overview`` 才执行。
CHAPTER_NOTEBOOKS = [
    "feature_extraction/feature_extraction.ipynb",  # read-only tour
    "attention/attention.ipynb",
    "pairformer/pairformer.ipynb",
    "feature_embedding/feature_embedding.ipynb",
    "diffusion/diffusion.ipynb",
    "confidence/confidence.ipynb",
]
OVERVIEW_NOTEBOOK = "model/overview.ipynb"


@contextmanager
def current_python_kernel():
    """Expose ``sys.executable`` as a temporary Jupyter kernel.

    ``python3`` in a user's global kernelspec can point at a different venv,
    which makes notebook checks appear to pass while testing the wrong
    environment. This temporary kernelspec keeps the checker tied to the
    interpreter that launched it.
    """
    kernel_name = "af3-current-python"
    with tempfile.TemporaryDirectory(prefix="af3-kernel-") as tmp:
        kernel_dir = Path(tmp) / "kernels" / kernel_name
        kernel_dir.mkdir(parents=True)
        (kernel_dir / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [
                        sys.executable,
                        "-m",
                        "ipykernel_launcher",
                        "-f",
                        "{connection_file}",
                    ],
                    "display_name": f"Python ({Path(sys.executable).parent})",
                    "language": "python",
                },
                indent=2,
            )
        )

        old_jupyter_path = os.environ.get("JUPYTER_PATH")
        os.environ["JUPYTER_PATH"] = (
            tmp if not old_jupyter_path else tmp + os.pathsep + old_jupyter_path
        )
        try:
            yield kernel_name
        finally:
            if old_jupyter_path is None:
                os.environ.pop("JUPYTER_PATH", None)
            else:
                os.environ["JUPYTER_PATH"] = old_jupyter_path


def execute(
    nb_path: Path,
    cwd: Path,
    timeout: int,
    kernel_name: str,
) -> tuple[bool, str]:
    """Run ``nb_path`` in working dir ``cwd``. Returns (ok, message)."""
    try:
        with open(nb_path) as f:
            nb = nbformat.read(f, as_version=4)
    except FileNotFoundError as e:
        return False, f"missing notebook: {e}"

    ep = ExecutePreprocessor(timeout=timeout, kernel_name=kernel_name)
    try:
        ep.preprocess(nb, {"metadata": {"path": str(cwd)}})
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    return True, "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src", default="solutions", choices=("solutions", "tutorials"),
        help="Which source tree to test. 测试哪一棵代码树。",
    )
    ap.add_argument(
        "--chapters", nargs="+", default=None,
        help="Subset of chapters (notebook stems) to run. 仅运行指定章节。",
    )
    ap.add_argument(
        "--with-overview", action="store_true",
        help="Also run model/overview.ipynb (requires Protenix checkpoint under "
             "checkpoints/). 一并跑 overview (需要权重)。",
    )
    ap.add_argument(
        "--only-overview", action="store_true",
        help="Run only model/overview.ipynb. 只跑 overview。",
    )
    ap.add_argument(
        "--timeout", type=int, default=600,
        help="Per-cell timeout in seconds. 单 cell 超时时间。",
    )
    ap.add_argument(
        "--fail-fast", action="store_true",
        help="Stop at the first failure. Default is to run every notebook "
             "and summarize at the end (more useful for students wanting the "
             "full picture). 默认跑完所有 notebook 再汇总；加这个开关在第一次"
             "失败处即停 (适合 CI)。",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    src_dir = repo_root / args.src
    if not src_dir.is_dir():
        print(f"error: {src_dir} does not exist", file=sys.stderr)
        return 2

    os.environ["LAYERNORM_TYPE"] = "torch"

    if args.only_overview:
        notebooks = [OVERVIEW_NOTEBOOK]
    else:
        notebooks = list(CHAPTER_NOTEBOOKS)
        if args.with_overview:
            notebooks.append(OVERVIEW_NOTEBOOK)
    if args.chapters:
        notebooks = [
            nb for nb in notebooks
            if any(ch in nb for ch in args.chapters)
        ]
        if not notebooks:
            print(f"error: no notebooks match {args.chapters}", file=sys.stderr)
            return 2

    print(f"checking {len(notebooks)} notebook(s) under {src_dir}/")
    print(f"using notebook kernel from: {sys.executable}")
    failures: list[tuple[str, str]] = []
    with current_python_kernel() as kernel_name:
        for rel in notebooks:
            nb_path = src_dir / rel
            t0 = time.time()
            print(
                f"  [{time.strftime('%H:%M:%S')}] running {rel} … ",
                end="",
                flush=True,
            )
            ok, msg = execute(
                nb_path,
                cwd=src_dir,
                timeout=args.timeout,
                kernel_name=kernel_name,
            )
            elapsed = time.time() - t0
            if ok:
                print(f"PASS  ({elapsed:.1f}s)")
            else:
                print(f"FAIL  ({elapsed:.1f}s)\n         {msg}")
                failures.append((rel, msg))
                if args.fail_fast:
                    break

    print()
    if failures:
        print(f"{len(failures)} notebook(s) failed:")
        for rel, msg in failures:
            print(f"  · {rel}")
        return 1
    print(f"all {len(notebooks)} notebook(s) passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
