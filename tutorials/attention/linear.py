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
        ##########################################################################
        # TODO: Custom parameter initialization driven by self.initializer.      #
        #   AF3 expects per-layer initialization knobs so different roles        #
        #   (gates, residual outputs, ReLU pre-activations) can start in the     #
        #   right scale.                                                          #
        #                                                                        #
        #   Step 1 — Zero out the bias if the layer has one. AF3 always zero-   #
        #     inits the bias regardless of the weight initializer:               #
        #       if self.use_bias:                                                #
        #           nn.init.zeros_(self.bias)                                    #
        #                                                                        #
        #   Step 2 — Initialize the weight according to ``self.initializer``:    #
        #     - "default": truncated normal with fan-in variance scale=1.0       #
        #         trunc_normal_init_(self.weight, scale=1.0)                     #
        #     - "relu":    same but scale=2.0 (He / Kaiming for ReLU activations)#
        #         trunc_normal_init_(self.weight, scale=2.0)                     #
        #     - "zeros":   zero-init the weight (used for residual / output       #
        #                   projections that should start as a no-op)            #
        #         nn.init.zeros_(self.weight)                                    #
        #     - any other value: raise ValueError with the offending string.     #
        #                                                                        #
        # TODO: 由 self.initializer 决定权重初始化方式。                          #
        #   AF3 不同位置的线性层对初始化尺度要求不同 (门控 / 残差输出 /          #
        #   ReLU 前激活 等)。                                                     #
        #                                                                        #
        #   步骤 1 — 若有 bias，先置零 (AF3 不论何种 initializer 都零初始化       #
        #     bias):                                                              #
        #       if self.use_bias:                                                #
        #           nn.init.zeros_(self.bias)                                    #
        #                                                                        #
        #   步骤 2 — 根据 ``self.initializer`` 初始化权重:                        #
        #     - "default": fan-in 截断正态，scale=1.0                              #
        #         trunc_normal_init_(self.weight, scale=1.0)                     #
        #     - "relu":    同上但 scale=2.0 (适配 ReLU 后激活的方差)              #
        #         trunc_normal_init_(self.weight, scale=2.0)                     #
        #     - "zeros":   零初始化 (用于残差 / 输出投影，起手为恒等)              #
        #         nn.init.zeros_(self.weight)                                    #
        #     - 其它字符串: 抛 ValueError(f"Invalid initializer: ...")            #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        ##########################################################################
        # TODO: Linear forward with an optional high-precision compute path.     #
        #   Some AF3 layers (coordinate projections, noise-level conditioning)   #
        #   must run in fp32 even under mixed precision to avoid catastrophic    #
        #   cancellation. ``self.precision`` (e.g. torch.float32) opts in.       #
        #                                                                        #
        #   Branch A — ``self.precision is not None``:                           #
        #     · remember the original input dtype                                #
        #     · disable CUDA autocast inside the block                           #
        #     · cast input, weight, (and bias if present) to ``self.precision``  #
        #     · run F.linear, then cast back to the original dtype:              #
        #         input_dtype = input.dtype                                      #
        #         with torch.amp.autocast("cuda", enabled=False):                #
        #             bias = (self.bias.to(dtype=self.precision)                 #
        #                     if self.bias is not None else None)                #
        #             return F.linear(                                           #
        #                 input.to(dtype=self.precision),                        #
        #                 self.weight.to(dtype=self.precision),                  #
        #                 bias,                                                  #
        #             ).to(dtype=input_dtype)                                    #
        #                                                                        #
        #   Branch B — ``self.precision is None`` (default): plain linear.       #
        #     return F.linear(input, self.weight, self.bias)                     #
        #                                                                        #
        # TODO: Linear 前向 + 可选的高精度计算路径。                              #
        #   AF3 的部分层 (坐标投影、噪声水平条件) 需要在 fp32 下计算，避免        #
        #   混合精度时的数值灾难。``self.precision`` (如 torch.float32) 即开关。 #
        #                                                                        #
        #   分支 A — ``self.precision is not None``:                              #
        #     · 记下输入 dtype                                                    #
        #     · 关闭 CUDA autocast                                                #
        #     · 把 input / weight / bias (若有) 转到 ``self.precision``           #
        #     · F.linear 之后再转回原 dtype:                                       #
        #         input_dtype = input.dtype                                      #
        #         with torch.amp.autocast("cuda", enabled=False):                #
        #             bias = (self.bias.to(dtype=self.precision)                 #
        #                     if self.bias is not None else None)                #
        #             return F.linear(                                           #
        #                 input.to(dtype=self.precision),                        #
        #                 self.weight.to(dtype=self.precision),                  #
        #                 bias,                                                  #
        #             ).to(dtype=input_dtype)                                    #
        #                                                                        #
        #   分支 B — ``self.precision is None`` (默认): 直接 F.linear。           #
        #     return F.linear(input, self.weight, self.bias)                     #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################


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
        ##########################################################################
        # TODO: BiasInitLinear — a Linear that starts as the constant function   #
        #   ``f(x) = biasinit``. AF3 uses this for sigmoid gates whose initial   #
        #   open state needs to be picked (e.g. AttentionPairBias's adaLN-Zero   #
        #   output gate uses biasinit=-2 so sigmoid(-2)≈0.12 at init).           #
        #                                                                        #
        #   Step 1 — Build the underlying Linear (parent class) with the         #
        #     forwarded kwargs:                                                  #
        #       super(BiasInitLinear, self).__init__(                            #
        #           in_features=in_features,                                     #
        #           out_features=out_features,                                   #
        #           bias=bias,                                                   #
        #           **kwargs,                                                    #
        #       )                                                                #
        #                                                                        #
        #   Step 2 — Overwrite the weight to all zeros (parent already zero-     #
        #     inited the bias):                                                  #
        #       nn.init.zeros_(tensor=self.weight)                               #
        #                                                                        #
        #   Step 3 — If bias is present, fill it with the constant ``biasinit``: #
        #       if bias:                                                         #
        #           nn.init.constant_(tensor=self.bias, val=biasinit)            #
        #                                                                        #
        # TODO: BiasInitLinear —— 起手就输出常数 ``biasinit`` 的 Linear。         #
        #   AF3 用它给 sigmoid 门设定初始开度 (例如 AttentionPairBias 的         #
        #   adaLN-Zero 输出门用 biasinit=-2，sigmoid(-2)≈0.12)。                  #
        #                                                                        #
        #   步骤 1 — 走父类初始化 Linear (转发 kwargs):                            #
        #       super(BiasInitLinear, self).__init__(                            #
        #           in_features=in_features,                                     #
        #           out_features=out_features,                                   #
        #           bias=bias,                                                   #
        #           **kwargs,                                                    #
        #       )                                                                #
        #                                                                        #
        #   步骤 2 — 权重直接置零 (bias 在父类已被零初始化):                       #
        #       nn.init.zeros_(tensor=self.weight)                               #
        #                                                                        #
        #   步骤 3 — 若有 bias，填上常数 ``biasinit``:                            #
        #       if bias:                                                         #
        #           nn.init.constant_(tensor=self.bias, val=biasinit)            #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################