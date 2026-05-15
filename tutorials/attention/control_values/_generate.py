"""Regenerate every ``*.pt`` reference in ``attention/control_values/``.

重新生成 ``attention/control_values/`` 下的所有参考 ``*.pt``。

Run from the repository root with::

    python -m attention.control_values._generate

Set the environment variable ``LAYERNORM_TYPE=torch`` so the pure-Python
LayerNorm is used (the only path available without a CUDA kernel).

从仓库根目录运行::

    python -m attention.control_values._generate

需要先设置 ``LAYERNORM_TYPE=torch`` 让我们走纯 PyTorch 路径 (CUDA 算子
不可用时也能跑)。
"""
from __future__ import annotations

import os

os.environ.setdefault("LAYERNORM_TYPE", "torch")

import torch  # noqa: E402

from attention.attention_pair_bias import AttentionPairBias  # noqa: E402
from attention.linear import BiasInitLinear, Linear, LinearNoBias  # noqa: E402
from attention.mha import Attention, _attention  # noqa: E402
from attention.transition import AdaptiveLayerNorm, Transition  # noqa: E402
from pairformer.triangle_ops import LayerNorm  # noqa: E402

from attention.control_values.attention_checks import (  # noqa: E402
    CONTROL_FOLDER,
    N_head,
    c_a,
    c_hidden,
    c_s,
    c_z,
    n_factor,
    test_inputs,
    test_module_forward,
    test_module_method,
    test_module_shape,
)


def main(overwrite: bool = True) -> None:
    torch.manual_seed(0)

    # ----- Linear (initializer="default") ----------------------------------
    lin = Linear(in_features=c_a, out_features=c_z).double()
    test_module_shape(lin, "linear_default", CONTROL_FOLDER, overwrite_results=overwrite)
    test_module_forward(
        lin, "linear_default",
        inputs=(test_inputs["x_a"],),
        output_names="out",
        control_folder=CONTROL_FOLDER,
        overwrite_results=overwrite,
    )

    # ----- LinearNoBias ----------------------------------------------------
    lin_nb = LinearNoBias(in_features=c_a, out_features=c_z).double()
    test_module_shape(lin_nb, "linear_nobias", CONTROL_FOLDER, overwrite_results=overwrite)
    test_module_forward(
        lin_nb, "linear_nobias",
        inputs=(test_inputs["x_a"],),
        output_names="out",
        control_folder=CONTROL_FOLDER,
        overwrite_results=overwrite,
    )

    # ----- BiasInitLinear (biasinit=-2.0) ----------------------------------
    bil = BiasInitLinear(in_features=c_s, out_features=c_a, bias=True, biasinit=-2.0).double()
    test_module_shape(bil, "bias_init_linear", CONTROL_FOLDER, overwrite_results=overwrite)
    test_module_forward(
        bil, "bias_init_linear",
        inputs=(test_inputs["x_s"],),
        output_names="out",
        control_folder=CONTROL_FOLDER,
        overwrite_results=overwrite,
    )

    # ----- LayerNorm (default scale + offset) ------------------------------
    ln = LayerNorm(c_a).double()
    test_module_shape(ln, "layer_norm", CONTROL_FOLDER, overwrite_results=overwrite)
    test_module_forward(
        ln, "layer_norm",
        inputs=(test_inputs["x_a"],),
        output_names="out",
        control_folder=CONTROL_FOLDER,
        overwrite_results=overwrite,
    )

    # ----- _attention standalone function ----------------------------------
    # The function has no learnable parameters, so just compute the expected
    # output and save it.
    # 这个函数没有可学参数，直接算期望输出存起来。
    with torch.no_grad():
        out = _attention(
            test_inputs["q_raw"].double(),
            test_inputs["k_raw"].double(),
            test_inputs["v_raw"].double(),
            attn_bias=None,
            use_efficient_implementation=False,
        )
    path = os.path.join(CONTROL_FOLDER, "attention_function_out.pt")
    if overwrite:
        torch.save(out, path)

    # ----- Attention module (gating=True) ----------------------------------
    attn = Attention(
        c_q=c_a, c_k=c_a, c_v=c_a,
        c_hidden=c_hidden, num_heads=N_head,
        gating=True, q_linear_bias=True,
        use_efficient_implementation=False,
        zero_init=False,
    ).double()
    test_module_shape(attn, "mha_gated", CONTROL_FOLDER, overwrite_results=overwrite)
    test_module_method(
        attn, "mha_gated",
        inputs=(test_inputs["q_x"], test_inputs["kv_x"], test_inputs["attn_bias"]),
        output_names="out",
        control_folder=CONTROL_FOLDER,
        method=lambda q_x, kv_x, b: attn(q_x=q_x, kv_x=kv_x, attn_bias=b),
        overwrite_results=overwrite,
    )

    # ----- AdaptiveLayerNorm ----------------------------------------------
    adaln = AdaptiveLayerNorm(c_a=c_a, c_s=c_s).double()
    test_module_shape(adaln, "adaptive_layer_norm", CONTROL_FOLDER, overwrite_results=overwrite)
    test_module_forward(
        adaln, "adaptive_layer_norm",
        inputs=(test_inputs["x_a"], test_inputs["x_s"]),
        output_names="out",
        control_folder=CONTROL_FOLDER,
        overwrite_results=overwrite,
    )

    # ----- Transition ------------------------------------------------------
    tr = Transition(c_in=c_a, n=n_factor).double()
    test_module_shape(tr, "transition", CONTROL_FOLDER, overwrite_results=overwrite)
    test_module_forward(
        tr, "transition",
        inputs=(test_inputs["x_a"],),
        output_names="out",
        control_folder=CONTROL_FOLDER,
        overwrite_results=overwrite,
    )

    # ----- AttentionPairBias (has_s=True) ----------------------------------
    apb = AttentionPairBias(
        has_s=True, create_offset_ln_z=False,
        n_heads=N_head, c_a=c_a, c_s=c_s, c_z=c_z,
        biasinit=-2.0, cross_attention_mode=False,
    ).double()
    # F.scaled_dot_product_attention rejects mixed dtypes; the explicit path
    # is dtype-tolerant. We turn the efficient path off only for testing.
    # F.scaled_dot_product_attention 不容忍 dtype 不一致，
    # 这里切到 dtype-friendly 的显式路径，仅用于测试。
    apb.attention.use_efficient_implementation = False
    test_module_shape(apb, "attention_pair_bias", CONTROL_FOLDER, overwrite_results=overwrite)
    test_module_method(
        apb, "attention_pair_bias",
        inputs=(test_inputs["x_a"], test_inputs["x_s"], test_inputs["x_z"]),
        output_names="out",
        control_folder=CONTROL_FOLDER,
        method=lambda a, s, z: apb(a=a, s=s, z=z),
        overwrite_results=overwrite,
    )

    if overwrite:
        print(f"Wrote control values under {CONTROL_FOLDER}")
    else:
        print(f"All attention/ control checks passed against {CONTROL_FOLDER}")


if __name__ == "__main__":
    main(overwrite=True)
