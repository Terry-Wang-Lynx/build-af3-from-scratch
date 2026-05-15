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

import torch
import torch.nn as nn

from attention.linear import Linear


# Adapted From openfold.model.heads
class DistogramHead(nn.Module):
    """Implements Algorithm 1 [Line17] in AF3

    Computes a distogram probability distribution.
    For use in computation of distogram loss, subsection 1.9.8 (AF2)

    Args:
        c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        no_bins (int, optional): Number of distogram bins. Defaults to 64.
    """

    def __init__(self, c_z: int = 128, no_bins: int = 64) -> None:
        super(DistogramHead, self).__init__()

        self.c_z = c_z
        self.no_bins = no_bins

        self.linear = Linear(
            in_features=self.c_z, out_features=self.no_bins, initializer="zeros"
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # [*, N, N, C_z]
        """
        Args:
            z (torch.Tensor): pair embedding
                [*, N_token, N_token, C_z]

        Returns:
            torch.Tensor: distogram probability distribution
                [*, N_token, N_token, no_bins]
        """
        ##########################################################################
        # TODO: Algorithm 1 line 17 — distogram logits from the pair tensor.    #
        #   ``self.linear`` is zero-init (so the head outputs uniform at the    #
        #   start of training), mapping ``c_z -> no_bins``.                     #
        #                                                                        #
        #   Step 1 — Per-pair logits:                                            #
        #       logits = self.linear(z)            # [*, N, N, no_bins]          #
        #                                                                        #
        #   Step 2 — Symmetrize over (i, j) so the distogram is invariant to    #
        #     pair ordering. The Pairformer's pair tensor is not strictly       #
        #     symmetric; AF3 explicitly enforces it on the distogram readout:   #
        #       logits = logits + logits.transpose(-2, -3)                       #
        #                                                                        #
        #   Step 3 — Return raw logits — the caller does softmax / CE loss     #
        #     downstream:                                                        #
        #       return logits                                                    #
        #                                                                        #
        # TODO: 算法 1 第 17 行 —— 由 pair 张量产出 distogram logits。            #
        #   ``self.linear`` 是零初始化 (训练初期输出均匀分布)，c_z -> no_bins。   #
        #                                                                        #
        #   步骤 1 — 每 pair 的 logits:                                            #
        #       logits = self.linear(z)            # [*, N, N, no_bins]          #
        #                                                                        #
        #   步骤 2 — 沿 (i, j) 对称化，使 distogram 与 pair 顺序无关:               #
        #     (Pairformer 的 pair 张量并不严格对称，AF3 在这里显式强制)             #
        #       logits = logits + logits.transpose(-2, -3)                       #
        #                                                                        #
        #   步骤 3 — 返回原始 logits (softmax / CE 由调用方做):                    #
        #       return logits                                                    #
        ##########################################################################

        # [*, N, N, no_bins]
        logits = self.linear(z)
        logits = logits + logits.transpose(-2, -3)
        return logits

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################
