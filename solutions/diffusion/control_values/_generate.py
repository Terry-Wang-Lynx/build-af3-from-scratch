"""Regenerate ``diffusion/control_values/*.pt``.

Run from ``solutions/``:

    python -m diffusion.control_values._generate
"""
from __future__ import annotations

import os

os.environ.setdefault("LAYERNORM_TYPE", "torch")

import torch  # noqa: E402

from diffusion.diffusion_transformer import (  # noqa: E402
    ConditionedTransitionBlock,
    DiffusionTransformer,
    DiffusionTransformerBlock,
)

from diffusion.control_values.diffusion_checks import (  # noqa: E402
    CONTROL_FOLDER,
    c_a,
    c_s,
    c_z,
    n_blocks,
    n_heads,
    test_inputs,
    test_module_method,
    test_module_shape,
)


def _disable_efficient_attn(mod: torch.nn.Module) -> None:
    """Force the dtype-tolerant explicit attention path on every submodule
    that exposes ``use_efficient_implementation``.

    递归地把所有 ``use_efficient_implementation`` 切到 False，避免 SDP
    在 double 输入下抛 dtype 错误。
    """
    for m in mod.modules():
        if hasattr(m, "use_efficient_implementation"):
            m.use_efficient_implementation = False


def main(overwrite: bool = True) -> None:
    torch.manual_seed(0)

    # ----- ConditionedTransitionBlock (Algorithm 25) ----------------------
    ctb = ConditionedTransitionBlock(c_a=c_a, c_s=c_s, n=2, biasinit=-2.0).double()
    test_module_shape(ctb, "conditioned_transition_block", CONTROL_FOLDER, overwrite_results=overwrite)
    test_module_method(
        ctb, "conditioned_transition_block",
        inputs=(test_inputs["a"], test_inputs["s"]),
        output_names="out",
        control_folder=CONTROL_FOLDER,
        method=lambda a, s: ctb(a=a, s=s),
        overwrite_results=overwrite,
    )

    # ----- DiffusionTransformerBlock (Algorithm 23 lines 2-3) -------------
    dtb = DiffusionTransformerBlock(
        c_a=c_a, c_s=c_s, c_z=c_z, n_heads=n_heads,
    ).double()
    _disable_efficient_attn(dtb)
    test_module_shape(dtb, "diffusion_transformer_block", CONTROL_FOLDER, overwrite_results=overwrite)
    test_module_method(
        dtb, "diffusion_transformer_block",
        inputs=(test_inputs["a"], test_inputs["s"], test_inputs["z"]),
        output_names="a_out",
        control_folder=CONTROL_FOLDER,
        method=lambda a, s, z: dtb(a=a, s=s, z=z)[0],
        overwrite_results=overwrite,
    )

    # ----- DiffusionTransformer (Algorithm 23 full stack) -----------------
    dt = DiffusionTransformer(
        c_a=c_a, c_s=c_s, c_z=c_z,
        n_blocks=n_blocks, n_heads=n_heads,
    ).double()
    _disable_efficient_attn(dt)
    test_module_shape(dt, "diffusion_transformer", CONTROL_FOLDER, overwrite_results=overwrite)
    test_module_method(
        dt, "diffusion_transformer",
        inputs=(test_inputs["a"], test_inputs["s"], test_inputs["z"]),
        output_names="out",
        control_folder=CONTROL_FOLDER,
        method=lambda a, s, z: dt(a=a, s=s, z=z),
        overwrite_results=overwrite,
    )

    if overwrite:
        print(f"Wrote control values under {CONTROL_FOLDER}")
    else:
        print(f"All diffusion/ control checks passed against {CONTROL_FOLDER}")


if __name__ == "__main__":
    main(overwrite=True)
