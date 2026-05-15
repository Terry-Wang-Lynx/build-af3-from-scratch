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
from diffusion.frames import expressCoordinatesInFrame  # noqa: E402
from model.utils import centre_random_augmentation  # noqa: E402

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

    # ----- expressCoordinatesInFrame (Algorithm 29) ------------------------
    # Pure function (no learnable params) — save the output directly and
    # compare bit-wise.
    # 纯函数（无参数）—— 直接保存输出做位级比较。
    with torch.no_grad():
        ecf_out = expressCoordinatesInFrame(
            test_inputs["coords"].double(),
            test_inputs["frame_atoms"].double(),
        )
    ecf_path = os.path.join(CONTROL_FOLDER, "express_coordinates_in_frame_out.pt")
    if overwrite:
        torch.save(ecf_out, ecf_path)
    else:
        expected = torch.load(ecf_path)
        assert torch.allclose(ecf_out, expected), (
            "expressCoordinatesInFrame output mismatch"
        )

    # ----- centre_random_augmentation (Algorithm 19, centre_only path) ----
    # The full path samples a random rotation and translation, which is hard
    # to reproduce bit-wise. We only check the deterministic centre_only=True
    # branch.
    # 完整路径用随机旋转/平移，难以位级复现；这里只测确定性的 centre_only 分支。
    with torch.no_grad():
        cra_out = centre_random_augmentation(
            test_inputs["coords"].double(),
            N_sample=2,
            centre_only=True,
        )
    cra_path = os.path.join(CONTROL_FOLDER, "centre_random_augmentation_centre_only_out.pt")
    if overwrite:
        torch.save(cra_out, cra_path)
    else:
        expected = torch.load(cra_path)
        assert torch.allclose(cra_out, expected), (
            "centre_random_augmentation(centre_only=True) output mismatch"
        )

    if overwrite:
        print(f"Wrote control values under {CONTROL_FOLDER}")
    else:
        print(f"All diffusion/ control checks passed against {CONTROL_FOLDER}")


if __name__ == "__main__":
    main(overwrite=True)
