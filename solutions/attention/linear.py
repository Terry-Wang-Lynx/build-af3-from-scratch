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

from pairformer.triangle_ops import LayerNorm, trunc_normal_init_
from model.utils import (
    chunk_layer,
    flatten_final_dims,
    move_final_dim_to_dim,
    pad_at_dim,
    reshape_at_dim,
)



class Linear(nn.Linear):
    """Linear module with customized initialization.

    Args:
        in_features (int): Input dimension.
        out_features (int): Output dimension.
        bias (bool, optional): Whether to use bias. Defaults to True.
        device (torch.device, optional): Device. Defaults to None.
        dtype (torch.dtype, optional): Data type. Defaults to None.
        precision (torch.dtype, optional): Precision for calculation. Defaults to None.
        initializer (str, optional): initializer: choose one from ['default', 'relu', 'zeros']. Defaults to "default".
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        precision: Optional[torch.dtype] = None,
        initializer: str = "default",
    ) -> None:
        self.use_bias = bias
        self.precision = precision
        self.initializer = initializer
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )

        self._init_params()

    @torch.no_grad()
    def _init_params(self):
        if self.use_bias:
            nn.init.zeros_(self.bias)  # zero-init bias

        if self.initializer == "default":
            trunc_normal_init_(self.weight, scale=1.0)
        elif self.initializer == "relu":
            trunc_normal_init_(self.weight, scale=2.0)
        elif self.initializer == "zeros":
            nn.init.zeros_(self.weight)
        else:
            raise ValueError(f"Invalid initializer: {self.initializer}.")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.precision is not None:
            input_dtype = input.dtype
            with torch.amp.autocast("cuda", enabled=False):
                bias = (
                    self.bias.to(dtype=self.precision)
                    if self.bias is not None
                    else None
                )
                return F.linear(
                    input.to(dtype=self.precision),
                    self.weight.to(dtype=self.precision),
                    bias,
                ).to(dtype=input_dtype)
        else:
            return F.linear(input, self.weight, self.bias)


LinearNoBias = partial(Linear, bias=False)


class BiasInitLinear(Linear):
    """Support biasinit for nn.Linear Called just like torch.nn.Linear.

    Args:
        in_features (int): Input dimension.
        out_features (int): Output dimension.
        bias (bool, optional): whether add bias. Defaults to True.
        biasinit (float, optional): the initial bias value. Defaults to 0.0.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        biasinit: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super(BiasInitLinear, self).__init__(
            in_features=in_features, out_features=out_features, bias=bias, **kwargs
        )
        nn.init.zeros_(tensor=self.weight)
        if bias:
            nn.init.constant_(tensor=self.bias, val=biasinit)