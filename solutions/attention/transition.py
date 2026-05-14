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
        # TODO: Apply LayerNorm to ``a`` and ``s``, then modulate ``a`` with     #
        #   ``sigmoid(linear_s(s)) * a + linear_nobias_s(s)``.                   #
        # TODO: 对 ``a`` 和 ``s`` 分别 LayerNorm，然后用                        #
        #   ``sigmoid(linear_s(s)) * a + linear_nobias_s(s)`` 调制 ``a``。      #
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
        # TODO: Algorithm 11. Apply LayerNorm, project via two Linear branches    #
        #   (a, b), gate as ``silu(a) * b``, then project back via linear_no_bias.#
        # TODO: Algorithm 11。先 LayerNorm，分别经 linear_no_bias_a / b 投影得到 #
        #   a 和 b，按 ``silu(a) * b`` 门控，最后用 linear_no_bias 投回原维度。 #
        ##########################################################################

        x = self.layernorm1(x)
        a = self.linear_no_bias_a(x)
        b = self.linear_no_bias_b(x)
        return self.linear_no_bias(F.silu(a) * b)

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################