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

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention.linear import LinearNoBias
from feature_embedding.atom_attention import AtomAttentionEncoder
from runtime.logger import get_logger

logger = get_logger(__name__)



class RelativePositionEncoding(nn.Module):
    """
    Implements Algorithm 3 in AF3

    Args:
        r_max (int, optional): Relative position indices clip value. Defaults to 32.
        s_max (int, optional): Relative chain indices clip value. Defaults to 2.
        c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
    """

    def __init__(self, r_max: int = 32, s_max: int = 2, c_z: int = 128) -> None:
        super(RelativePositionEncoding, self).__init__()
        self.r_max = r_max
        self.s_max = s_max
        self.c_z = c_z
        self.linear_no_bias = LinearNoBias(
            in_features=(4 * self.r_max + 2 * self.s_max + 7), out_features=self.c_z
        )
        self.input_feature = {
            "asym_id": 1,
            "residue_index": 1,
            "entity_id": 1,
            "sym_id": 1,
            "token_index": 1,
        }

    def generate_relp(self, input_feature_dict: dict[str, Any]) -> dict[str, Any]:
        """asym_id, residue_index, entity_id, token_index, sym_id"""
        ##########################################################################
        # TODO: Algorithm 3 — build the relative-position feature ``relp``. It  #
        #   is an integer-valued, one-hot-encoded representation of position    #
        #   relationships between every pair of tokens. Three "boolean" channels#
        #   (same chain / same residue / same entity) gate three clipped        #
        #   pair-wise distances (residue offset, token offset, chain offset).  #
        #                                                                        #
        #   Step 1 — Wrap the whole computation in ``torch.no_grad()`` — relp   #
        #     is a fixed feature with no gradients, and we want to save memory:#
        #       with torch.no_grad():                                            #
        #         asym_id        = input_feature_dict["asym_id"]                #
        #         residue_index  = input_feature_dict["residue_index"]          #
        #         entity_id      = input_feature_dict["entity_id"]              #
        #         token_index    = input_feature_dict["token_index"]            #
        #         sym_id         = input_feature_dict["sym_id"]                 #
        #                                                                        #
        #   Step 2 — Pairwise booleans (1 inside the same chain / residue /     #
        #     entity, 0 otherwise). Use long() so they compose cleanly with    #
        #     integer arithmetic later:                                          #
        #         b_same_chain   = (asym_id[..., :, None]                        #
        #                           == asym_id[..., None, :]).long()             #
        #         b_same_residue = (residue_index[..., :, None]                  #
        #                           == residue_index[..., None, :]).long()      #
        #         b_same_entity  = (entity_id[..., :, None]                      #
        #                           == entity_id[..., None, :]).long()           #
        #                                                                        #
        #   Step 3 — Three clipped offsets, each mapped to a one-hot.            #
        #     The trick: clip into [0, 2*r_max], but for pairs that fail the    #
        #     gating condition assign a special "off-chain" code 2*r_max+1.    #
        #                                                                        #
        #     residue offset (gated by same-chain):                              #
        #         d_residue = (torch.clip(                                       #
        #             input=residue_index[..., :, None]                          #
        #                   - residue_index[..., None, :] + self.r_max,         #
        #             min=0, max=2 * self.r_max,                                 #
        #           ) * b_same_chain                                              #
        #           + (1 - b_same_chain) * (2 * self.r_max + 1))                #
        #         a_rel_pos = F.one_hot(d_residue, 2 * (self.r_max + 1))         #
        #                                                                        #
        #     token offset (gated by same-chain AND same-residue):              #
        #         d_token = (torch.clip(                                         #
        #             input=token_index[..., :, None]                            #
        #                   - token_index[..., None, :] + self.r_max,           #
        #             min=0, max=2 * self.r_max,                                 #
        #           ) * b_same_chain * b_same_residue                            #
        #           + (1 - b_same_chain * b_same_residue)                       #
        #             * (2 * self.r_max + 1))                                    #
        #         a_rel_token = F.one_hot(d_token, 2 * (self.r_max + 1))         #
        #                                                                        #
        #     chain (sym_id) offset (gated by same-entity, clipped at s_max):   #
        #         d_chain = (torch.clip(                                         #
        #             input=sym_id[..., :, None] - sym_id[..., None, :]         #
        #                   + self.s_max,                                        #
        #             min=0, max=2 * self.s_max,                                 #
        #           ) * b_same_entity                                             #
        #           + (1 - b_same_entity) * (2 * self.s_max + 1))                #
        #         a_rel_chain = F.one_hot(d_chain, 2 * (self.s_max + 1))         #
        #                                                                        #
        #   Step 4 — Concatenate along the feature axis. ``b_same_entity``      #
        #     is added in as a length-1 channel. Final width:                    #
        #     2*(r_max+1) + 2*(r_max+1) + 1 + 2*(s_max+1)                       #
        #     = 4*r_max + 2*s_max + 7. Cache and return the dict:               #
        #         relp = torch.cat(                                              #
        #             [a_rel_pos, a_rel_token,                                   #
        #              b_same_entity[..., None], a_rel_chain],                   #
        #             dim=-1,                                                    #
        #         ).float()                                                       #
        #         input_feature_dict["relp"] = relp                              #
        #       return input_feature_dict                                         #
        #                                                                        #
        # TODO: 算法 3 —— 构造相对位置特征 ``relp``。它是一个整数级、             #
        #   one-hot 编码的每对 token 之间的位置关系。三个"布尔"通道 (同链 /      #
        #   同残基 / 同 entity) 分别门控三个 clip 后的成对距离 (residue 偏移、   #
        #   token 偏移、chain 偏移)。                                            #
        #                                                                        #
        #   步骤 1 — 整段写在 ``torch.no_grad()`` 里 —— relp 是固定特征，无梯度，#
        #     也省显存:                                                          #
        #       with torch.no_grad():                                            #
        #         asym_id        = input_feature_dict["asym_id"]                #
        #         residue_index  = input_feature_dict["residue_index"]          #
        #         entity_id      = input_feature_dict["entity_id"]              #
        #         token_index    = input_feature_dict["token_index"]            #
        #         sym_id         = input_feature_dict["sym_id"]                 #
        #                                                                        #
        #   步骤 2 — pair 布尔门控 (同链 / 同残基 / 同 entity)。用 long() 与下面   #
        #     的整数算术干净组合:                                                  #
        #         b_same_chain   = (asym_id[..., :, None]                        #
        #                           == asym_id[..., None, :]).long()             #
        #         b_same_residue = (residue_index[..., :, None]                  #
        #                           == residue_index[..., None, :]).long()      #
        #         b_same_entity  = (entity_id[..., :, None]                      #
        #                           == entity_id[..., None, :]).long()           #
        #                                                                        #
        #   步骤 3 — 三组 clip 偏移，每一组映成 one-hot。技巧:                     #
        #     clip 到 [0, 2*r_max]，门控失败的 pair 用特殊编码 2*r_max+1。         #
        #                                                                        #
        #     residue 偏移 (同链门控):                                            #
        #         d_residue = (torch.clip(                                       #
        #             input=residue_index[..., :, None]                          #
        #                   - residue_index[..., None, :] + self.r_max,         #
        #             min=0, max=2 * self.r_max,                                 #
        #           ) * b_same_chain                                              #
        #           + (1 - b_same_chain) * (2 * self.r_max + 1))                #
        #         a_rel_pos = F.one_hot(d_residue, 2 * (self.r_max + 1))         #
        #                                                                        #
        #     token 偏移 (同链 AND 同残基门控):                                    #
        #         d_token = (torch.clip(                                         #
        #             input=token_index[..., :, None]                            #
        #                   - token_index[..., None, :] + self.r_max,           #
        #             min=0, max=2 * self.r_max,                                 #
        #           ) * b_same_chain * b_same_residue                            #
        #           + (1 - b_same_chain * b_same_residue)                       #
        #             * (2 * self.r_max + 1))                                    #
        #         a_rel_token = F.one_hot(d_token, 2 * (self.r_max + 1))         #
        #                                                                        #
        #     chain (sym_id) 偏移 (同 entity 门控，clip 到 [0, 2*s_max]):         #
        #         d_chain = (torch.clip(                                         #
        #             input=sym_id[..., :, None] - sym_id[..., None, :]         #
        #                   + self.s_max,                                        #
        #             min=0, max=2 * self.s_max,                                 #
        #           ) * b_same_entity                                             #
        #           + (1 - b_same_entity) * (2 * self.s_max + 1))                #
        #         a_rel_chain = F.one_hot(d_chain, 2 * (self.s_max + 1))         #
        #                                                                        #
        #   步骤 4 — 沿特征维拼接，``b_same_entity`` 作为长度 1 通道掺进去。       #
        #     最终宽度 4*r_max + 2*s_max + 7，缓存到 dict 后返回:                  #
        #         relp = torch.cat(                                              #
        #             [a_rel_pos, a_rel_token,                                   #
        #              b_same_entity[..., None], a_rel_chain],                   #
        #             dim=-1,                                                    #
        #         ).float()                                                       #
        #         input_feature_dict["relp"] = relp                              #
        #       return input_feature_dict                                         #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(self, relp_feature) -> torch.Tensor:
        """
        Args:
            asym_id / residue_index / entity_id / sym_id / token_index
                [..., N_tokens]
        Returns:
            torch.Tensor: relative position encoding
                [..., N_token, N_token, c_z]
        """
        return self.linear_no_bias(relp_feature)



class FourierEmbedding(nn.Module):
    """
    Implements Algorithm 22 in AF3

    Args:
        c (int): embedding dim.
        seed (int, optional): random seed. Defaults to 42.
    """

    def __init__(self, c: int, seed: int = 42) -> None:
        super(FourierEmbedding, self).__init__()
        self.c = c
        self.seed = seed
        generator = torch.Generator()
        generator.manual_seed(seed)
        w_value = torch.randn(size=(c,), generator=generator)
        self.w = nn.Parameter(w_value, requires_grad=False)
        b_value = torch.randn(size=(c,), generator=generator)
        self.b = nn.Parameter(b_value, requires_grad=False)

    def forward(self, t_hat_noise_level: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t_hat_noise_level (torch.Tensor): the noise level
                [..., N_sample]

        Returns:
            torch.Tensor: the output fourier embedding
                [..., N_sample, c]
        """
        ##########################################################################
        # TODO: Algorithm 22 — random-Fourier embedding of a scalar input.       #
        #   We turn a per-sample scalar (the diffusion noise level) into a      #
        #   dense c-vector by computing                                          #
        #       cos(2 * pi * (t * w + b))                                        #
        #   where ``w`` and ``b`` are FIXED random vectors of length c           #
        #   sampled at construction time (and registered as non-trainable       #
        #   parameters so they ride along with the checkpoint).                 #
        #                                                                        #
        #   The unsqueeze appends the channel axis: ``t`` is [..., N_sample],  #
        #   output is [..., N_sample, c]:                                        #
        #     return torch.cos(                                                  #
        #         input=2 * torch.pi * (                                         #
        #             t_hat_noise_level.unsqueeze(dim=-1) * self.w + self.b)    #
        #     )                                                                  #
        #                                                                        #
        # TODO: 算法 22 —— 标量输入的随机 Fourier 嵌入。把每个样本的标量           #
        #   (扩散噪声水平) 通过                                                   #
        #       cos(2 * pi * (t * w + b))                                        #
        #   变成长度 c 的稠密向量。``w`` 和 ``b`` 在构造时随机抽样、              #
        #   作为不可训练参数保存，所以会随 checkpoint 一起。                       #
        #                                                                        #
        #   ``t`` 形状 [..., N_sample]，unsqueeze 出通道轴后输出 [..., N_sample, c]:#
        #     return torch.cos(                                                  #
        #         input=2 * torch.pi * (                                         #
        #             t_hat_noise_level.unsqueeze(dim=-1) * self.w + self.b)    #
        #     )                                                                  #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

