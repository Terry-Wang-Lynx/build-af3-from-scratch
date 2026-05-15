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
        ##########################################################################
        # TODO: LayerNorm with optional learnable scale and offset. Parameter    #
        #   names must match the fused-kernel version so Protenix checkpoints    #
        #   load with strict=True.                                               #
        #                                                                        #
        #   Step 1 — Record the normalized-shape tuple and ``eps``:              #
        #       self.c_in: Tuple[int, ...] = (c_in,)                             #
        #       self.eps = eps                                                   #
        #                                                                        #
        #   Step 2 — Optional learnable scale (``weight``). When                  #
        #     ``create_scale`` is False the module behaves like a pure mean/var  #
        #     normalizer with no learnable scale — register a ``None``           #
        #     placeholder so the state_dict path still resolves:                 #
        #       if create_scale:                                                 #
        #           self.weight = nn.Parameter(torch.ones(c_in))                 #
        #       else:                                                            #
        #           self.register_parameter("weight", None)                      #
        #                                                                        #
        #   Step 3 — Optional learnable offset (``bias``), same pattern:         #
        #       if create_offset:                                                #
        #           self.bias = nn.Parameter(torch.zeros(c_in))                  #
        #       else:                                                            #
        #           self.register_parameter("bias", None)                        #
        #                                                                        #
        # TODO: 带可选可学 scale / offset 的 LayerNorm。参数名必须与融合算子      #
        #   版本一致，使 Protenix checkpoint 可严格加载。                          #
        #                                                                        #
        #   步骤 1 — 记下归一化的形状元组和 ``eps``:                                #
        #       self.c_in: Tuple[int, ...] = (c_in,)                             #
        #       self.eps = eps                                                   #
        #                                                                        #
        #   步骤 2 — 可选可学 scale (``weight``)。``create_scale`` 为 False 时    #
        #     注册一个 None 占位符，state_dict 路径仍然成立:                       #
        #       if create_scale:                                                 #
        #           self.weight = nn.Parameter(torch.ones(c_in))                 #
        #       else:                                                            #
        #           self.register_parameter("weight", None)                      #
        #                                                                        #
        #   步骤 3 — 可选可学 offset (``bias``)，同上:                             #
        #       if create_offset:                                                #
        #           self.bias = nn.Parameter(torch.zeros(c_in))                  #
        #       else:                                                            #
        #           self.register_parameter("bias", None)                        #
        ##########################################################################

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

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ##########################################################################
        # TODO: LayerNorm forward. The bf16 input branch is special-cased to    #
        #   match Protenix reference numerics — autocast is disabled and the    #
        #   weight / bias are cast to bf16 inside the call so the result is bit-#
        #   compatible with the upstream fused kernel.                           #
        #                                                                        #
        #   Step 1 — Remember the input dtype:                                   #
        #       d = x.dtype                                                      #
        #                                                                        #
        #   Step 2 — bf16 special case (disable autocast, downcast params):     #
        #       if d is torch.bfloat16:                                          #
        #           with torch.amp.autocast("cuda", enabled=False):              #
        #               return nn.functional.layer_norm(                         #
        #                   x,                                                   #
        #                   self.c_in,                                           #
        #                   self.weight.to(d) if self.weight is not None         #
        #                       else None,                                       #
        #                   self.bias.to(d) if self.bias is not None             #
        #                       else None,                                       #
        #                   self.eps,                                            #
        #               )                                                        #
        #                                                                        #
        #   Step 3 — Default path: plain ``F.layer_norm`` with the stored        #
        #     weight / bias / eps:                                               #
        #       return nn.functional.layer_norm(                                 #
        #           x, self.c_in, self.weight, self.bias, self.eps,              #
        #       )                                                                #
        #                                                                        #
        # TODO: LayerNorm 前向。bf16 输入做特别处理以与 Protenix 融合算子的       #
        #   数值对齐 —— 关 autocast 并把 weight / bias 临时降到 bf16。           #
        #                                                                        #
        #   步骤 1 — 记下输入 dtype:                                                #
        #       d = x.dtype                                                      #
        #                                                                        #
        #   步骤 2 — bf16 特殊路径 (关 autocast，降参数 dtype):                    #
        #       if d is torch.bfloat16:                                          #
        #           with torch.amp.autocast("cuda", enabled=False):              #
        #               return nn.functional.layer_norm(                         #
        #                   x,                                                   #
        #                   self.c_in,                                           #
        #                   self.weight.to(d) if self.weight is not None         #
        #                       else None,                                       #
        #                   self.bias.to(d) if self.bias is not None             #
        #                       else None,                                       #
        #                   self.eps,                                            #
        #               )                                                        #
        #                                                                        #
        #   步骤 3 — 默认路径: 直接 ``F.layer_norm``:                              #
        #       return nn.functional.layer_norm(                                 #
        #           x, self.c_in, self.weight, self.bias, self.eps,              #
        #       )                                                                #
        ##########################################################################

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

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################


def LayerNorm(
    c_in: int,
    create_scale: bool = True,
    create_offset: bool = True,
    eps: float = 1e-5,
) -> nn.Module:
    """Factory matching upstream API. Always returns OpenFoldLayerNorm."""
    return OpenFoldLayerNorm(c_in, create_scale, create_offset, eps)
