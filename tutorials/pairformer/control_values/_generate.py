"""Regenerate every ``*.pt`` reference in ``pairformer/control_values/``.

重新生成 ``pairformer/control_values/`` 下的所有参考 ``*.pt``。

Run from ``solutions/`` with::

    python -m pairformer.control_values._generate
"""
from __future__ import annotations

import os

os.environ.setdefault("LAYERNORM_TYPE", "torch")

import torch  # noqa: E402

from pairformer.pair_stack import PairformerBlock  # noqa: E402
from pairformer.triangle import (  # noqa: E402
    TriangleAttention,
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from pairformer.triangle_ops import OuterProductMean  # noqa: E402
from pairformer.msa_stack import MSAPairWeightedAveraging  # noqa: E402

from pairformer.control_values.pairformer_checks import (  # noqa: E402
    CONTROL_FOLDER,
    N_token,
    c_hidden,
    c_m,
    c_z,
    no_heads_pair,
    test_inputs,
    test_module_forward,
    test_module_method,
    test_module_shape,
)


def main(overwrite: bool = True) -> None:
    torch.manual_seed(0)

    # ----- OuterProductMean (Algorithm 10) ---------------------------------
    opm = OuterProductMean(c_m=c_m, c_z=c_z, c_hidden=c_hidden)
    test_module_shape(opm, "outer_product_mean", CONTROL_FOLDER, overwrite_results=overwrite)
    test_module_method(
        opm, "outer_product_mean",
        inputs=(test_inputs["m"], test_inputs["msa_mask"]),
        output_names="out",
        control_folder=CONTROL_FOLDER,
        method=lambda m, mask: opm(m, mask=mask),
        overwrite_results=overwrite,
    )

    # ----- TriangleMultiplication (outgoing + incoming) --------------------
    for variant, cls in [
        ("triangle_mul_out", TriangleMultiplicationOutgoing),
        ("triangle_mul_in",  TriangleMultiplicationIncoming),
    ]:
        mod = cls(c_z=c_z, c_hidden=c_hidden)
        test_module_shape(mod, variant, CONTROL_FOLDER, overwrite_results=overwrite)
        test_module_method(
            mod, variant,
            inputs=(test_inputs["z"], test_inputs["pair_mask"]),
            output_names="out",
            control_folder=CONTROL_FOLDER,
            method=lambda z, pm: mod(z, mask=pm),
            overwrite_results=overwrite,
        )

    # ----- TriangleAttention (starting node) -------------------------------
    tri_att = TriangleAttention(
        c_in=c_z, c_hidden=c_hidden, no_heads=no_heads_pair, starting=True,
    )
    test_module_shape(tri_att, "triangle_attention_start", CONTROL_FOLDER, overwrite_results=overwrite)
    test_module_method(
        tri_att, "triangle_attention_start",
        inputs=(test_inputs["z"], test_inputs["pair_mask"]),
        output_names="out",
        control_folder=CONTROL_FOLDER,
        method=lambda z, pm: tri_att(z, mask=pm),
        overwrite_results=overwrite,
    )

    # ----- MSAPairWeightedAveraging (Algorithm 10 inside MSAModule) -------
    mpwa = MSAPairWeightedAveraging(c_m=c_m, c=c_hidden, c_z=c_z, n_heads=no_heads_pair)
    test_module_shape(mpwa, "msa_pair_weighted_avg", CONTROL_FOLDER, overwrite_results=overwrite)
    test_module_forward(
        mpwa, "msa_pair_weighted_avg",
        inputs=(test_inputs["m"], test_inputs["z"]),
        output_names="out",
        control_folder=CONTROL_FOLDER,
        overwrite_results=overwrite,
    )

    # ----- PairformerBlock (full Algorithm 17 single step) -----------------
    # Use a smaller pair-attention head count for the small test shapes.
    # 测试形状小，pair attention 头数也调小。
    block = PairformerBlock(
        n_heads=no_heads_pair,
        c_z=c_z, c_s=0,                       # disable single-track to keep test small
        c_hidden_mul=c_hidden,
        c_hidden_pair_att=c_hidden,
        no_heads_pair=no_heads_pair,
        num_intermediate_factor=2,
        dropout=0.0,
    )
    # Like the AttentionPairBias test in attention/, route MHA via the
    # dtype-tolerant explicit path.
    # 同 attention 那边，走显式 attention 路径以兼容 double 输入。
    for sub in (block.tri_att_start.mha, block.tri_att_end.mha):
        if hasattr(sub, "use_efficient_implementation"):
            sub.use_efficient_implementation = False
    test_module_shape(block, "pairformer_block_no_single", CONTROL_FOLDER, overwrite_results=overwrite)
    # Forward returns (s, z); with c_s=0 it returns (None, z).
    test_module_method(
        block, "pairformer_block_no_single",
        inputs=(None, test_inputs["z"], test_inputs["pair_mask"]),
        output_names="z_out",
        control_folder=CONTROL_FOLDER,
        method=lambda s, z, pm: block(s, z, pair_mask=pm)[1],
        overwrite_results=overwrite,
    )

    if overwrite:
        print(f"Wrote control values under {CONTROL_FOLDER}")
    else:
        print(f"All pairformer/ control checks passed against {CONTROL_FOLDER}")


if __name__ == "__main__":
    main(overwrite=True)
