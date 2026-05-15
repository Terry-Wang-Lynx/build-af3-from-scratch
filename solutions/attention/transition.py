# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from functools import partial
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention.linear import Linear, LinearNoBias
from pairformer.triangle_ops import LayerNorm, trunc_normal_init_
from model.utils import (
    chunk_layer,
    flatten_final_dims,
    move_final_dim_to_dim,
    pad_at_dim,
    reshape_at_dim,
)



class AdaptiveLayerNorm(nn.Module):
    """
    Implements Algorithm 26 in AF3

    Args:
        c_a (int, optional): the embedding dim of a(single feature aggregated atom info). Defaults to 768.
        c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
    """

    def __init__(self, c_a: int = 768, c_s: int = 384) -> None:
        super(AdaptiveLayerNorm, self).__init__()
        self.layernorm_a = LayerNorm(c_a, create_scale=False, create_offset=False)
        self.layernorm_s = LayerNorm(c_s, create_offset=False)
        self.linear_s = Linear(in_features=c_s, out_features=c_a, initializer="zeros")
        self.linear_nobias_s = LinearNoBias(
            in_features=c_s, out_features=c_a, initializer="zeros"
        )

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            a: [..., N_token, c_a] aggregated per-atom representation.
            s: [..., N_token, c_s] single embedding.

        Returns:
            [..., N_token, c_a] — AdaLN-modulated ``a``.
        """
        ##########################################################################
        # TODO: Algorithm 26 — Adaptive LayerNorm (AdaLN), as used inside the    #
        #   AF3 DiffusionTransformer to inject the single embedding ``s`` into   #
        #   the token/atom stream ``a``.                                         #
        #                                                                        #
        #   Step 1 — Normalize both inputs independently:                        #
        #       a = self.layernorm_a(a)   # no scale, no offset                  #
        #       s = self.layernorm_s(s)   # learnable scale, no offset           #
        #     (See __init__: ``layernorm_a`` has create_scale=False,             #
        #      create_offset=False, while ``layernorm_s`` keeps its scale.)      #
        #                                                                        #
        #   Step 2 — Compute the conditioning scale and bias from ``s`` and      #
        #     apply them to ``a`` (FiLM / AdaLN-Zero modulation):                #
        #       a = sigmoid(self.linear_s(s)) * a + self.linear_nobias_s(s)      #
        #     -  ``linear_s``        is Linear      (c_s -> c_a, zero-init)      #
        #     -  ``linear_nobias_s`` is LinearNoBias(c_s -> c_a, zero-init)      #
        #     Zero-initialization makes the layer start as identity:             #
        #       sigmoid(0)=0.5 -> a := 0.5 * a + 0   (handled by the downstream  #
        #       block that consumes this output).                                #
        #                                                                        #
        # TODO: 算法 26 — Adaptive LayerNorm (AdaLN)，                            #
        #   AF3 DiffusionTransformer 用它把单序列嵌入 ``s`` 注入                  #
        #   token / atom 流 ``a``。                                              #
        #                                                                        #
        #   步骤 1 — 分别对两路输入做 LayerNorm:                                  #
        #       a = self.layernorm_a(a)   # 无 scale、无 offset                  #
        #       s = self.layernorm_s(s)   # 有可学 scale、无 offset              #
        #     (见 __init__: ``layernorm_a`` 的 create_scale=False、              #
        #      create_offset=False；``layernorm_s`` 保留 scale。)                #
        #                                                                        #
        #   步骤 2 — 由 ``s`` 算出 scale / bias，按 FiLM / AdaLN-Zero 调制 ``a``: #
        #       a = sigmoid(self.linear_s(s)) * a + self.linear_nobias_s(s)      #
        #     -  ``linear_s``        是 Linear      (c_s -> c_a，zero-init)      #
        #     -  ``linear_nobias_s`` 是 LinearNoBias(c_s -> c_a，zero-init)      #
        #     零初始化保证该层一开始近似恒等映射:                                 #
        #       sigmoid(0)=0.5 -> a := 0.5 * a + 0 (由下游 block 进一步处理)。   #
        ##########################################################################

        a = self.layernorm_a(a)
        s = self.layernorm_s(s)
        a = torch.sigmoid(self.linear_s(s)) * a + self.linear_nobias_s(s)

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################
        return a


class Transition(nn.Module):
    """
    Implements Algorithm 11 in AF3

    Args:
        c_in (int): the input dimension.
        n (int): factor by which c_in is multiplied to obtain hidden dimension.
    """

    def __init__(self, c_in: int, n: int) -> None:
        super(Transition, self).__init__()
        self.n = n
        self.c_in = c_in
        self.layernorm1 = LayerNorm(c_in)
        self.linear_no_bias_a = LinearNoBias(
            in_features=c_in, out_features=n * c_in, initializer="relu"
        )
        self.linear_no_bias_b = LinearNoBias(
            in_features=c_in, out_features=n * c_in, initializer="relu"
        )
        self.linear_no_bias = LinearNoBias(
            in_features=n * c_in, out_features=c_in, initializer="zeros"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., c_in] input.

        Returns:
            [..., c_in] — SwiGLU-style gated FFN output.
        """
        ##########################################################################
        # TODO: Algorithm 11 — SwiGLU-style Transition block.                    #
        #   This is AF3's per-token / per-pair MLP. It widens by a factor ``n``  #
        #   (typically 2 or 4), then projects back.                              #
        #                                                                        #
        #   Step 1 — Pre-LayerNorm:                                              #
        #       x = self.layernorm1(x)                  # [..., c_in]            #
        #                                                                        #
        #   Step 2 — Two parallel linear branches widen to ``n * c_in``:         #
        #       a = self.linear_no_bias_a(x)            # [..., n * c_in]       #
        #       b = self.linear_no_bias_b(x)            # [..., n * c_in]       #
        #     (Both initialized with ``relu`` fan-in scaling; both no-bias.)     #
        #                                                                        #
        #   Step 3 — SwiGLU gate ``silu(a) * b`` and project back:               #
        #       return self.linear_no_bias(F.silu(a) * b)   # [..., c_in]       #
        #     (``linear_no_bias`` is zero-init so the residual branch starts    #
        #      as a no-op.)                                                      #
        #                                                                        #
        # TODO: 算法 11 — SwiGLU 风格的 Transition 块。                           #
        #   这是 AF3 中每 token / 每 pair 的 MLP，先按倍数 ``n`` 扩宽 (常用 2/4) #
        #   再投回原维度。                                                       #
        #                                                                        #
        #   步骤 1 — Pre-LayerNorm:                                              #
        #       x = self.layernorm1(x)                  # [..., c_in]            #
        #                                                                        #
        #   步骤 2 — 两条并行线性分支，均扩宽到 ``n * c_in``:                     #
        #       a = self.linear_no_bias_a(x)            # [..., n * c_in]       #
        #       b = self.linear_no_bias_b(x)            # [..., n * c_in]       #
        #     (两者均用 relu fan-in 初始化、不带 bias。)                          #
        #                                                                        #
        #   步骤 3 — SwiGLU 门控 ``silu(a) * b`` 再投回原维度:                    #
        #       return self.linear_no_bias(F.silu(a) * b)   # [..., c_in]       #
        #     (``linear_no_bias`` 零初始化，使残差分支初始为恒等。)              #
        ##########################################################################

        x = self.layernorm1(x)
        a = self.linear_no_bias_a(x)
        b = self.linear_no_bias_b(x)
        return self.linear_no_bias(F.silu(a) * b)

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################