"""Regenerate ``feature_embedding/control_values/*.pt``.

Run from ``solutions/`` with::

    python -m feature_embedding.control_values._generate
"""
from __future__ import annotations

import os

os.environ.setdefault("LAYERNORM_TYPE", "torch")

import torch  # noqa: E402

from feature_embedding.relative_position_encoding import (  # noqa: E402
    FourierEmbedding,
    RelativePositionEncoding,
)

from feature_embedding.control_values.feature_embedding_checks import (  # noqa: E402
    CONTROL_FOLDER,
    c_noise,
    c_z,
    r_max,
    s_max,
    test_inputs,
    test_module_forward,
    test_module_shape,
)


def main(overwrite: bool = True) -> None:
    torch.manual_seed(0)

    # ----- RelativePositionEncoding.forward (the linear at the tail) --------
    relpe = RelativePositionEncoding(r_max=r_max, s_max=s_max, c_z=c_z).double()
    test_module_shape(relpe, "relative_position_encoding", CONTROL_FOLDER, overwrite_results=overwrite)
    test_module_forward(
        relpe, "relative_position_encoding",
        inputs=(test_inputs["relp_feature"],),
        output_names="out",
        control_folder=CONTROL_FOLDER,
        overwrite_results=overwrite,
    )

    # ----- FourierEmbedding -------------------------------------------------
    fe = FourierEmbedding(c=c_noise).double()
    test_module_shape(fe, "fourier_embedding", CONTROL_FOLDER, overwrite_results=overwrite)
    test_module_forward(
        fe, "fourier_embedding",
        inputs=(test_inputs["noise_level"],),
        output_names="out",
        control_folder=CONTROL_FOLDER,
        overwrite_results=overwrite,
    )

    if overwrite:
        print(f"Wrote control values under {CONTROL_FOLDER}")
    else:
        print(f"All feature_embedding/ control checks passed against {CONTROL_FOLDER}")


if __name__ == "__main__":
    main(overwrite=True)
