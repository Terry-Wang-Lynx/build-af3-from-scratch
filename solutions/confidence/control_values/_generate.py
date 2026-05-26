"""Regenerate ``confidence/control_values/*.pt``.

Run from ``solutions/``:

    python -m confidence.control_values._generate
"""
from __future__ import annotations

import os

os.environ.setdefault("LAYERNORM_TYPE", "torch")

import torch  # noqa: E402

from confidence.confidence_head import ConfidenceHead  # noqa: E402
from confidence.distogram_head import DistogramHead  # noqa: E402

from confidence.control_values.confidence_checks import (  # noqa: E402
    CONTROL_FOLDER,
    c_z,
    no_bins,
    test_inputs,
    test_module_forward,
    test_module_shape,
)


def main(overwrite: bool = True) -> None:
    torch.manual_seed(0)

    # ----- DistogramHead (Algorithm 1 line 17) -----------------------------
    dh = DistogramHead(c_z=c_z, no_bins=no_bins)
    test_module_shape(dh, "distogram_head", CONTROL_FOLDER, overwrite_results=overwrite)
    test_module_forward(
        dh, "distogram_head",
        inputs=(test_inputs["z"],),
        output_names="out",
        control_folder=CONTROL_FOLDER,
        overwrite_results=overwrite,
    )

    # ----- ConfidenceHead full module — shape check only -------------------
    # The full forward and memory_efficient_forward need a non-trivial
    # input_feature_dict (with atom-level indexing maps). A parameter
    # shape check is cheaper but still catches every dim / naming / module
    # wiring mistake in __init__.
    # 全前向需要 atom 级索引映射，构造测试输入复杂；shape 检查就足以暴露            #
    # __init__ 里的维度 / 命名 / 模块拼装错误。
    # ConfidenceHead's internal PairformerStack uses n_heads=16, so c_s must
    # be a multiple of 16. Use 32 for the smallest valid test size.
    # ConfidenceHead 内部 PairformerStack 用 n_heads=16，c_s 必须是 16 的倍数。
    ch = ConfidenceHead(
        n_blocks=1,
        c_s=32, c_z=c_z,
        c_s_inputs=32,
        b_pae=8, b_pde=8, b_plddt=10, b_resolved=2,
        max_atoms_per_token=5,
        pairformer_dropout=0.0,
        distance_bin_start=3.25, distance_bin_end=8.25, distance_bin_step=1.25,
    )
    test_module_shape(ch, "confidence_head_init", CONTROL_FOLDER, overwrite_results=overwrite)

    if overwrite:
        print(f"Wrote control values under {CONTROL_FOLDER}")
    else:
        print(f"All confidence/ control checks passed against {CONTROL_FOLDER}")


if __name__ == "__main__":
    main(overwrite=True)
