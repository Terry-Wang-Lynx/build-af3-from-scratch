"""Regenerate ``confidence/control_values/*.pt``.

Run from ``solutions/``:

    python -m confidence.control_values._generate
"""
from __future__ import annotations

import os

os.environ.setdefault("LAYERNORM_TYPE", "torch")

import torch  # noqa: E402

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
    dh = DistogramHead(c_z=c_z, no_bins=no_bins).double()
    test_module_shape(dh, "distogram_head", CONTROL_FOLDER, overwrite_results=overwrite)
    test_module_forward(
        dh, "distogram_head",
        inputs=(test_inputs["z"],),
        output_names="out",
        control_folder=CONTROL_FOLDER,
        overwrite_results=overwrite,
    )

    if overwrite:
        print(f"Wrote control values under {CONTROL_FOLDER}")
    else:
        print(f"All confidence/ control checks passed against {CONTROL_FOLDER}")


if __name__ == "__main__":
    main(overwrite=True)
