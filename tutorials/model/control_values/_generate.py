"""Regenerate ``model/control_values/*.pt``.

Run from ``solutions/``::

    python -m model.control_values._generate

This checks that the assembled ``Protenix(cfg)`` exposes the same set of
parameters (name + shape) as the reference. Anything missed, mis-named,
or wrong-shaped surfaces here.

校验装配好的 ``Protenix(cfg)`` 参数集 (name + shape) 与参考一致。漏建
模块、命名错、维度错都会在这里被发现。
"""
from __future__ import annotations

import os

os.environ.setdefault("LAYERNORM_TYPE", "torch")

from model.control_values.model_checks import (  # noqa: E402
    CONTROL_FOLDER,
    MODEL_NAME,
    build_tiny_protenix,
    test_module_shape,
)


def main(overwrite: bool = True) -> None:
    try:
        model = build_tiny_protenix()
    except (AttributeError, TypeError) as exc:
        # Most often: a child module's __init__ body is still ``pass`` so
        # ``super().__init__()`` never ran. nn.Module then trips over
        # missing ``_modules`` / ``_parameters`` when the parent tries to
        # register the child.
        # 通常出现在子模块 __init__ 还停在 ``pass``，
        # nn.Module 注册子模块时找不到 ``_modules`` / ``_parameters``。
        raise AssertionError(
            f"protenix_tiny construction failed ({exc.__class__.__name__}: "
            f"{exc}). This almost always means at least one chapter's "
            f"``__init__`` is still a ``pass`` stub. Fill in the per-chapter "
            f"blanks first, then re-run this check."
        ) from None

    test_module_shape(
        model,
        f"protenix_tiny_state_dict",
        CONTROL_FOLDER,
        overwrite_results=overwrite,
    )
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    if overwrite:
        print(f"Wrote control values under {CONTROL_FOLDER}")
        print(f"  preset = {MODEL_NAME}")
        print(f"  params = {n_params:.2f} M")
    else:
        print(f"All model/ control checks passed against {CONTROL_FOLDER}")
        print(f"  preset = {MODEL_NAME}  params = {n_params:.2f} M")


if __name__ == "__main__":
    main(overwrite=True)
