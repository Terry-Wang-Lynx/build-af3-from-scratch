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

from functools import partial
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.diffusion_transformer import DiffusionTransformer
from attention.transition import AdaptiveLayerNorm
from attention.mha import Attention, DropPath
from attention.linear import BiasInitLinear, LinearNoBias
from feature_embedding.local_attention import broadcast_token_to_local_atom_pair, rearrange_qk_to_dense_trunk
from pairformer.triangle_ops import LayerNorm
from model.utils import (
    aggregate_atom_to_token,
    broadcast_token_to_atom,
    checkpoint_blocks,
    permute_final_dims,
)



class AtomTransformer(nn.Module):
    """
    Implements Algorithm 7 in AF3

    Performs local transformer among atom embeddings, with bias predicted from atom pair embeddings

    Args:
        c_atom (int, optional): embedding dim for atom feature. Defaults to 128.
        c_atompair (int, optional): embedding dim for atompair feature. Defaults to 16.
        n_blocks (int, optional): number of block in AtomTransformer. Defaults to 3.
        n_heads (int, optional): number of heads in attention. Defaults to 4.
        n_queries (int, optional): local window size of query tensor. If not None, will perform local attention. Defaults to 32.
        n_keys (int, optional): local window size of key tensor. Defaults to 128.
        blocks_per_ckpt (int, optional): number of AtomTransformer/DiffusionTransformer blocks in each activation checkpoint. Defaults to None.
    """

    def __init__(
        self,
        c_atom: int = 128,
        c_atompair: int = 16,
        n_blocks: int = 3,
        n_heads: int = 4,
        n_queries: int = 32,
        n_keys: int = 128,
        blocks_per_ckpt: Optional[int] = None,
    ) -> None:
        super(AtomTransformer, self).__init__()
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.n_queries = n_queries
        self.n_keys = n_keys
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.diffusion_transformer = DiffusionTransformer(
            n_blocks=n_blocks,
            n_heads=n_heads,
            c_a=c_atom,
            c_s=c_atom,
            c_z=c_atompair,
            cross_attention_mode=True,
            blocks_per_ckpt=blocks_per_ckpt,
        )

    def forward(
        self,
        q: torch.Tensor,
        c: torch.Tensor,
        p: torch.Tensor,
    ) -> torch.Tensor:
        """Algorithm 7 — local DiffusionTransformer over atoms.

        Args:
            q: [..., N_atom, c_atom]                                    atom single (query)
            c: [..., N_atom, c_atom]                                    atom single (conditioning)
            p: [..., n_blocks, n_queries, n_keys, c_atompair]           atom-pair (dense-trunk)
        Returns:
            [..., N_atom, c_atom]
        """
        _, n_queries, n_keys = p.shape[-4:-1]
        assert n_queries == self.n_queries
        assert n_keys == self.n_keys
        return self.diffusion_transformer(
            a=q, s=c, z=p, n_queries=self.n_queries, n_keys=self.n_keys
        )



class AtomAttentionEncoder(nn.Module):
    """
    Implements Algorithm 5 in AF3

    Args:
        has_coords (bool): whether the module input will contains coordinates (r_l).
        c_token (int): token embedding dim.
        c_atom (int, optional): atom embedding dim. Defaults to 128.
        c_atompair (int, optional): atompair embedding dim. Defaults to 16.
        c_s (int, optional):  single embedding dim. Defaults to 384.
        c_z (int, optional): pair embedding dim. Defaults to 128.
        n_blocks (int, optional): number of blocks in AtomTransformer. Defaults to 3.
        n_heads (int, optional): number of heads in AtomTransformer. Defaults to 4.
        n_queries (int, optional): local window size of query tensor. Defaults to 32.
        n_keys (int, optional): local window size of key tensor. Defaults to 128.
        blocks_per_ckpt (int, optional): number of AtomAttentionEncoder/AtomTransformer blocks in each activation checkpoint. Defaults to None.
    """

    def __init__(
        self,
        has_coords: bool,
        c_token: int,  # 384 or 768
        c_atom: int = 128,
        c_atompair: int = 16,
        c_s: int = 384,
        c_z: int = 128,
        n_blocks: int = 3,
        n_heads: int = 4,
        n_queries: int = 32,
        n_keys: int = 128,
        blocks_per_ckpt: Optional[int] = None,
    ) -> None:
        super(AtomAttentionEncoder, self).__init__()
        self.has_coords = has_coords
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_token = c_token
        self.c_s = c_s
        self.c_z = c_z
        self.n_queries = n_queries
        self.n_keys = n_keys
        self.local_attention_method = "local_cross_attention"

        self.input_feature = {
            # "ref_pos": 3,
            # "ref_charge": 1,
            "ref_mask": 1,
            "ref_element": 128,
            "ref_atom_name_chars": 4 * 64,
        }
        self.linear_no_bias_ref_pos = LinearNoBias(
            in_features=3, out_features=self.c_atom, precision=torch.float32
        )  # use high precision for ref_pos
        self.linear_no_bias_ref_charge = LinearNoBias(
            in_features=1, out_features=self.c_atom
        )
        self.linear_no_bias_f = LinearNoBias(
            in_features=sum(self.input_feature.values()), out_features=self.c_atom
        )
        self.linear_no_bias_d = LinearNoBias(
            in_features=3, out_features=self.c_atompair, precision=torch.float32
        )
        self.linear_no_bias_invd = LinearNoBias(
            in_features=1, out_features=self.c_atompair
        )
        self.linear_no_bias_v = LinearNoBias(
            in_features=1, out_features=self.c_atompair
        )

        if self.has_coords:
            # Line9
            self.layernorm_s = LayerNorm(self.c_s, create_offset=False)
            self.linear_no_bias_s = LinearNoBias(
                in_features=self.c_s,
                out_features=self.c_atom,
                initializer="zeros",
                precision=torch.float32,
            )
            # Line10
            self.layernorm_z = LayerNorm(
                self.c_z, create_offset=False
            )  # memory bottleneck
            self.linear_no_bias_z = LinearNoBias(
                in_features=self.c_z,
                out_features=self.c_atompair,
                initializer="zeros",
                precision=torch.float32,
            )
            # Line11
            self.linear_no_bias_r = LinearNoBias(
                in_features=3, out_features=self.c_atom, precision=torch.float32
            )
        self.linear_no_bias_cl = LinearNoBias(
            in_features=self.c_atom, out_features=self.c_atompair
        )
        self.linear_no_bias_cm = LinearNoBias(
            in_features=self.c_atom, out_features=self.c_atompair
        )
        self.small_mlp = nn.Sequential(
            nn.ReLU(),
            LinearNoBias(
                in_features=self.c_atompair,
                out_features=self.c_atompair,
                initializer="relu",
            ),
            nn.ReLU(),
            LinearNoBias(
                in_features=self.c_atompair,
                out_features=self.c_atompair,
                initializer="relu",
            ),
            nn.ReLU(),
            LinearNoBias(
                in_features=self.c_atompair,
                out_features=self.c_atompair,
                initializer="zeros",
            ),
        )
        self.atom_transformer = AtomTransformer(
            n_blocks=n_blocks,
            n_heads=n_heads,
            c_atom=c_atom,
            c_atompair=c_atompair,
            n_queries=n_queries,
            n_keys=n_keys,
            blocks_per_ckpt=blocks_per_ckpt,
        )
        self.linear_no_bias_q = LinearNoBias(
            in_features=self.c_atom, out_features=self.c_token
        )

    def prepare_cache(
        self,
        ref_pos: torch.Tensor,
        ref_charge: torch.Tensor,
        ref_mask: torch.Tensor,
        ref_element: torch.Tensor,
        ref_atom_name_chars: torch.Tensor,
        atom_to_token_idx: torch.Tensor,
        d_lm: torch.Tensor,
        v_lm: torch.Tensor,
        pad_info: dict,
        r_l: Union[torch.Tensor, bool, None] = None,
        z: Optional[torch.Tensor] = None,
        inplace_safe: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Algorithm 5 lines 1-6 — compute the time-invariant pair (p_lm) and
        single (c_l) atom-level conditioning. The result is cached once and
        reused across all diffusion timesteps.
        """
        del inplace_safe
        batch_shape = ref_pos.shape[:-2]
        N_atom = ref_pos.shape[-2]
        c_l = self.linear_no_bias_ref_pos(ref_pos) + self.linear_no_bias_ref_charge(
            torch.arcsinh(ref_charge).reshape(*batch_shape, N_atom, 1)
        )
        c_l = c_l + self.linear_no_bias_f(
            torch.cat(
                [
                    ref_mask.reshape(*batch_shape, N_atom, 1),
                    ref_element.reshape(*batch_shape, N_atom, 128),
                    ref_atom_name_chars.reshape(*batch_shape, N_atom, 4 * 64),
                ],
                dim=-1,
            ).to(c_l.dtype)
        )
        c_l = c_l * ref_mask.reshape(*batch_shape, N_atom, 1)

        # [..., n_blocks, n_queries, n_keys, C_atompair]
        p_lm = (self.linear_no_bias_d(d_lm) * v_lm) * pad_info["mask_trunked"].unsqueeze(-1)
        p_lm = p_lm + self.linear_no_bias_invd(
            1 / (1 + (d_lm**2).sum(dim=-1, keepdim=True))
        ) * v_lm
        p_lm = p_lm + self.linear_no_bias_v(v_lm.to(p_lm.dtype))

        # Add trunk pair embedding when called inside the diffusion loop.
        if r_l is not None:
            assert z is not None
            p_lm = p_lm.unsqueeze(-5) + broadcast_token_to_local_atom_pair(
                z_token=self.linear_no_bias_z(self.layernorm_z(z)),
                atom_to_token_idx=atom_to_token_idx,
                n_queries=self.n_queries,
                n_keys=self.n_keys,
                compute_mask=False,
            )[0]
        return p_lm, c_l

    def forward(
        self,
        atom_to_token_idx: torch.Tensor,
        ref_pos: torch.Tensor,
        ref_charge: torch.Tensor,
        ref_mask: torch.Tensor,
        ref_atom_name_chars: torch.Tensor,
        ref_element: torch.Tensor,
        d_lm: torch.Tensor,
        v_lm: torch.Tensor,
        pad_info: dict,
        r_l: Optional[torch.Tensor] = None,
        s: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        p_lm: Optional[torch.Tensor] = None,
        c_l: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Algorithm 5 — AtomAttentionEncoder.

        Returns (a, q_l, c_l, p_lm):
            a:    [..., (N_sample), N_token, c_token]
            q_l:  [..., (N_sample), N_atom, c_atom]
            c_l:  [..., (N_sample), N_atom, c_atom]
            p_lm: [..., (N_sample), n_blocks, n_queries, n_keys, c_atompair]
        """
        ##########################################################################
        # TODO: Algorithm 5. AtomAttentionEncoder.                               #
        #   1. Build time-invariant (c_l, p_lm) via ``prepare_cache`` if not     #
        #      already given.                                                    #
        #   2. If coords ``r_l`` are provided, add the trunk single broadcast     #
        #      and the noisy-position projection to get the query ``q_l``.       #
        #      Otherwise q_l = c_l.clone().                                      #
        #   3. Fuse the single conditioning into the pair representation         #
        #      ``p_lm`` via two ReLU-linear projections + a small MLP.           #
        #   4. Run AtomTransformer (local cross-attention) on (q_l, c_l, p_lm). #
        #   5. Aggregate atoms→tokens via mean-pool to produce ``a``.            #
        # TODO: Algorithm 5. AtomAttentionEncoder。                              #
        #   1. 通过 ``prepare_cache`` 算与时间无关的 (c_l, p_lm)。               #
        #   2. 若给了坐标 r_l，把主干 single 广播 + 噪声坐标投影加到 c_l，       #
        #      得到查询 q_l；否则 q_l = c_l.clone()。                            #
        #   3. 用两路 ReLU+Linear + 小 MLP 把 single 条件融入 p_lm。            #
        #   4. AtomTransformer 跑局部 cross-attention。                          #
        #   5. mean-pool 原子→token 得到 a。                                     #
        ##########################################################################

        if self.has_coords:
            assert r_l is not None and s is not None and z is not None

        if p_lm is None or c_l is None:
            p_lm, c_l = self.prepare_cache(
                ref_pos=ref_pos, ref_charge=ref_charge,
                ref_mask=ref_mask, ref_atom_name_chars=ref_atom_name_chars,
                ref_element=ref_element, atom_to_token_idx=atom_to_token_idx,
                d_lm=d_lm, v_lm=v_lm, pad_info=pad_info,
                r_l=r_l, z=z,
            )

        n_token = None
        if r_l is not None:
            assert s is not None
            n_token = s.size(-2)
            c_l = c_l.unsqueeze(-3) + broadcast_token_to_atom(
                x_token=self.linear_no_bias_s(self.layernorm_s(s)),
                atom_to_token_idx=atom_to_token_idx,
            )
            q_l = c_l + self.linear_no_bias_r(r_l)
        else:
            q_l = c_l.clone()

        c_l_q, c_l_k, _ = rearrange_qk_to_dense_trunk(
            q=c_l, k=c_l, dim_q=-2, dim_k=-2,
            n_queries=self.n_queries, n_keys=self.n_keys,
            compute_mask=False,
        )
        p_lm = (
            p_lm
            + self.linear_no_bias_cl(F.relu(c_l_q[..., None, :]))
            + self.linear_no_bias_cm(F.relu(c_l_k[..., None, :, :]))
        )
        p_lm = p_lm + self.small_mlp(p_lm)

        q_l = self.atom_transformer(q_l, c_l, p_lm)

        a = aggregate_atom_to_token(
            x_atom=F.relu(self.linear_no_bias_q(q_l)),
            atom_to_token_idx=atom_to_token_idx,
            n_token=n_token,
            reduce="mean",
        )

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################
        return a, q_l, c_l, p_lm



class AtomAttentionDecoder(nn.Module):
    """
    Implements Algorithm 6 in AF3

    Args:
        n_blocks (int, optional): number of blocks for AtomTransformer. Defaults to 3.
        n_heads (int, optional): number of heads for AtomTransformer. Defaults to 4.
        c_token (int, optional): feature channel of token (single a). Defaults to 384.
        c_atom (int, optional): embedding dim for atom embedding. Defaults to 128.
        c_atompair (int, optional): embedding dim for atom pair embedding. Defaults to 16.
        n_queries (int, optional): local window size of query tensor. Defaults to 32.
        n_keys (int, optional): local window size of key tensor. Defaults to 128.
        blocks_per_ckpt (int, optional): number of AtomAttentionDecoder/AtomTransformer blocks in each activation checkpoint. Defaults to None.
    """

    def __init__(
        self,
        n_blocks: int = 3,
        n_heads: int = 4,
        c_token: int = 384,
        c_atom: int = 128,
        c_atompair: int = 16,
        n_queries: int = 32,
        n_keys: int = 128,
        blocks_per_ckpt: Optional[int] = None,
    ) -> None:
        super(AtomAttentionDecoder, self).__init__()
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.c_token = c_token
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.n_queries = n_queries
        self.n_keys = n_keys
        self.linear_no_bias_a = LinearNoBias(in_features=c_token, out_features=c_atom)
        self.layernorm_q = LayerNorm(c_atom, create_offset=False)
        self.linear_no_bias_out = LinearNoBias(
            in_features=c_atom, out_features=3, precision=torch.float32
        )
        self.atom_transformer = AtomTransformer(
            n_blocks=n_blocks,
            n_heads=n_heads,
            c_atom=c_atom,
            c_atompair=c_atompair,
            n_queries=n_queries,
            n_keys=n_keys,
            blocks_per_ckpt=blocks_per_ckpt,
        )

    def forward(
        self,
        atom_to_token_idx: torch.Tensor,
        a: torch.Tensor,
        q_skip: torch.Tensor,
        c_skip: torch.Tensor,
        p_skip: torch.Tensor,
    ) -> torch.Tensor:
        """Algorithm 6 — AtomAttentionDecoder.

        Maps token-level activations + atom-level skip connections to a
        per-atom position update.

        Returns:
            r: [..., N_atom, 3] coordinate update.
        """
        ##########################################################################
        # TODO: Algorithm 6.                                                     #
        #   1. Broadcast linear_no_bias_a(a) from token to atom level and add    #
        #      the atom-level skip ``q_skip``.                                   #
        #   2. Run AtomTransformer with conditioning ``c_skip`` and pair         #
        #      ``p_skip``.                                                       #
        #   3. Project to a 3-vector update via LayerNorm + linear_no_bias_out.  #
        # TODO: Algorithm 6.                                                     #
        #   1. 把 linear_no_bias_a(a) 从 token 广播到原子级，再加 q_skip。       #
        #   2. AtomTransformer 用 c_skip + p_skip 作条件跑一次。                 #
        #   3. LayerNorm + linear_no_bias_out 投到 3 维坐标更新。                #
        ##########################################################################

        q = broadcast_token_to_atom(
            x_token=self.linear_no_bias_a(a),
            atom_to_token_idx=atom_to_token_idx,
        ) + q_skip
        q = self.atom_transformer(q, c_skip, p_skip)
        return self.linear_no_bias_out(self.layernorm_q(q))

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################
