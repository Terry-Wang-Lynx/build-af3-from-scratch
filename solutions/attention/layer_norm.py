"""
LayerNorm — pure PyTorch (OpenFoldLayerNorm only).

Protenix ships a CUDA-fused LayerNorm and a PyTorch fallback. We keep only
the fallback. The bf16 branch upcasts to fp32 inside an autocast(False)
context so normalization matches Protenix's reference numerics.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class OpenFoldLayerNorm(nn.Module):
    """LayerNorm with optional scale/offset. Parameter shapes match the
    fused-kernel version so checkpoints load unchanged."""

    def __init__(
        self,
        c_in: int,
        create_scale: bool = True,
        create_offset: bool = True,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.c_in: Tuple[int, ...] = (c_in,)
        self.eps = eps
        if create_scale:
            self.weight = nn.Parameter(torch.ones(c_in))
        else:
            self.register_parameter("weight", None)
        if create_offset:
            self.bias = nn.Parameter(torch.zeros(c_in))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.dtype
        if d is torch.bfloat16:
            with torch.amp.autocast("cuda", enabled=False):
                return nn.functional.layer_norm(
                    x,
                    self.c_in,
                    self.weight.to(d) if self.weight is not None else None,
                    self.bias.to(d) if self.bias is not None else None,
                    self.eps,
                )
        return nn.functional.layer_norm(
            x, self.c_in, self.weight, self.bias, self.eps
        )


def LayerNorm(
    c_in: int,
    create_scale: bool = True,
    create_offset: bool = True,
    eps: float = 1e-5,
) -> nn.Module:
    """Factory matching upstream API. Always returns OpenFoldLayerNorm."""
    return OpenFoldLayerNorm(c_in, create_scale, create_offset, eps)
