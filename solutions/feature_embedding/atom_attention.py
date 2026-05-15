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
        ##########################################################################
        # TODO: AtomTransformer (Algorithm 7) is just a DiffusionTransformer    #
        #   restricted to the local windowed-attention path. The atom-pair      #
        #   tensor ``p`` arrives in dense-trunk form, so check the trunk width   #
        #   matches the configured window sizes, then delegate to the           #
        #   underlying transformer telling it to use local attention:           #
        #                                                                        #
        #       _, n_queries, n_keys = p.shape[-4:-1]                            #
        #       assert n_queries == self.n_queries                               #
        #       assert n_keys    == self.n_keys                                  #
        #       return self.diffusion_transformer(                               #
        #           a=q, s=c, z=p,                                               #
        #           n_queries=self.n_queries, n_keys=self.n_keys,                #
        #       )                                                                #
        #                                                                        #
        # TODO: AtomTransformer (算法 7) 就是把 DiffusionTransformer 限制到局部   #
        #   窗口注意力路径。pair 张量 ``p`` 已是 dense-trunk 形状，先校验 trunk     #
        #   宽度与窗口配置一致，再透传给底层 transformer 并指明启用局部注意力:    #
        #                                                                        #
        #       _, n_queries, n_keys = p.shape[-4:-1]                            #
        #       assert n_queries == self.n_queries                               #
        #       assert n_keys    == self.n_keys                                  #
        #       return self.diffusion_transformer(                               #
        #           a=q, s=c, z=p,                                               #
        #           n_queries=self.n_queries, n_keys=self.n_keys,                #
        #       )                                                                #
        ##########################################################################

        _, n_queries, n_keys = p.shape[-4:-1]
        assert n_queries == self.n_queries
        assert n_keys == self.n_keys
        return self.diffusion_transformer(
            a=q, s=c, z=p, n_queries=self.n_queries, n_keys=self.n_keys
        )

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################



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
        ##########################################################################
        # TODO: Algorithm 5 lines 1-6 — time-invariant atom-single (c_l) and    #
        #   atom-pair (p_lm) conditioning. These depend only on reference       #
        #   geometry, so we compute them once and cache across diffusion steps. #
        #                                                                        #
        #   Step 1 — Read shapes and start c_l from reference position +         #
        #     arcsinh-clamped formal charge (arcsinh is symmetric in sign and   #
        #     keeps the distribution tame):                                     #
        #       batch_shape = ref_pos.shape[:-2]                                 #
        #       N_atom      = ref_pos.shape[-2]                                  #
        #       c_l = (                                                          #
        #           self.linear_no_bias_ref_pos(ref_pos)                         #
        #           + self.linear_no_bias_ref_charge(                            #
        #               torch.arcsinh(ref_charge).reshape(                       #
        #                   *batch_shape, N_atom, 1))                            #
        #       )                                                                #
        #                                                                        #
        #   Step 2 — Fold the categorical features (ref_mask, ref_element       #
        #     one-hot 128-dim, ref_atom_name_chars 4*64-dim) into one wide      #
        #     concat, project to c_atom, and add. Cast the concat to c_l.dtype  #
        #     because element/atom_name_chars may be int64:                     #
        #       c_l = c_l + self.linear_no_bias_f(                               #
        #           torch.cat(                                                   #
        #               [                                                        #
        #                   ref_mask.reshape(*batch_shape, N_atom, 1),          #
        #                   ref_element.reshape(*batch_shape, N_atom, 128),     #
        #                   ref_atom_name_chars.reshape(                        #
        #                       *batch_shape, N_atom, 4 * 64),                  #
        #               ],                                                       #
        #               dim=-1,                                                  #
        #           ).to(c_l.dtype)                                              #
        #       )                                                                #
        #                                                                        #
        #   Step 3 — Mask invalid atoms to zero:                                  #
        #       c_l = c_l * ref_mask.reshape(*batch_shape, N_atom, 1)            #
        #                                                                        #
        #   Step 4 — Build the dense-trunk atom-pair tensor p_lm from the       #
        #     pre-computed per-pair distance vectors ``d_lm`` and validity      #
        #     ``v_lm``. The padding-trunk mask zeros pairs outside the local    #
        #     window:                                                            #
        #       # [..., n_blocks, n_queries, n_keys, C_atompair]                 #
        #       p_lm = (self.linear_no_bias_d(d_lm) * v_lm)                     #
        #              * pad_info["mask_trunked"].unsqueeze(-1)                  #
        #       p_lm = p_lm + self.linear_no_bias_invd(                         #
        #           1 / (1 + (d_lm**2).sum(dim=-1, keepdim=True))               #
        #       ) * v_lm                                                         #
        #       p_lm = p_lm + self.linear_no_bias_v(v_lm.to(p_lm.dtype))        #
        #                                                                        #
        #   Step 5 — When called inside the diffusion loop (``r_l is not None``)#
        #     also broadcast the trunk pair feature ``z`` into the dense-trunk  #
        #     layout and add. We add a leading length-1 axis to p_lm so it      #
        #     broadcasts with the sample axis carried by ``r_l``:                #
        #       if r_l is not None:                                              #
        #           assert z is not None                                         #
        #           p_lm = p_lm.unsqueeze(-5) + broadcast_token_to_local_atom_pair(#
        #               z_token=self.linear_no_bias_z(self.layernorm_z(z)),     #
        #               atom_to_token_idx=atom_to_token_idx,                    #
        #               n_queries=self.n_queries,                               #
        #               n_keys=self.n_keys,                                     #
        #               compute_mask=False,                                     #
        #           )[0]                                                         #
        #   Return ``(p_lm, c_l)``.                                              #
        #                                                                        #
        # TODO: 算法 5 第 1-6 行 —— 与时间无关的原子 single (c_l) 和原子 pair     #
        #   (p_lm) 条件。它们只依赖参考几何，可缓存复用。                          #
        #                                                                        #
        #   步骤 1 — 形状 + 由 ref_pos 与 arcsinh 后的 formal charge 起手 c_l:     #
        #       batch_shape = ref_pos.shape[:-2]                                 #
        #       N_atom      = ref_pos.shape[-2]                                  #
        #       c_l = (                                                          #
        #           self.linear_no_bias_ref_pos(ref_pos)                         #
        #           + self.linear_no_bias_ref_charge(                            #
        #               torch.arcsinh(ref_charge).reshape(                       #
        #                   *batch_shape, N_atom, 1))                            #
        #       )                                                                #
        #                                                                        #
        #   步骤 2 — 把 (ref_mask, ref_element one-hot 128 维,                    #
        #     ref_atom_name_chars 4*64 维) 拼宽，投到 c_atom 加到 c_l。           #
        #     concat 后转到 c_l.dtype (element 等可能是 int64):                    #
        #       c_l = c_l + self.linear_no_bias_f(                               #
        #           torch.cat([...], dim=-1).to(c_l.dtype)                      #
        #       )                                                                #
        #                                                                        #
        #   步骤 3 — 用 ref_mask 把无效原子置零:                                    #
        #       c_l = c_l * ref_mask.reshape(*batch_shape, N_atom, 1)            #
        #                                                                        #
        #   步骤 4 — 由预算好的 per-pair 距离向量 ``d_lm`` 和有效性 ``v_lm``      #
        #     构建 dense-trunk 原子 pair 张量 p_lm，``mask_trunked`` 把窗口外       #
        #     置零:                                                                #
        #       p_lm = (self.linear_no_bias_d(d_lm) * v_lm)                     #
        #              * pad_info["mask_trunked"].unsqueeze(-1)                  #
        #       p_lm = p_lm + self.linear_no_bias_invd(                         #
        #           1 / (1 + (d_lm**2).sum(dim=-1, keepdim=True))               #
        #       ) * v_lm                                                         #
        #       p_lm = p_lm + self.linear_no_bias_v(v_lm.to(p_lm.dtype))        #
        #                                                                        #
        #   步骤 5 — 在扩散循环内 (``r_l is not None``) 把主干 pair ``z`` 广播到   #
        #     dense-trunk 上叠加。在 p_lm 前插入一个长度 1 的轴，使其与 r_l        #
        #     携带的样本轴对齐:                                                    #
        #       if r_l is not None:                                              #
        #           assert z is not None                                         #
        #           p_lm = p_lm.unsqueeze(-5) + broadcast_token_to_local_atom_pair(#
        #               z_token=self.linear_no_bias_z(self.layernorm_z(z)),     #
        #               atom_to_token_idx=atom_to_token_idx,                    #
        #               n_queries=self.n_queries,                               #
        #               n_keys=self.n_keys,                                     #
        #               compute_mask=False,                                     #
        #           )[0]                                                         #
        #   返回 ``(p_lm, c_l)``。                                                #
        ##########################################################################

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

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

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
        # TODO: Algorithm 5 — AtomAttentionEncoder. Aggregates per-atom features #
        #   up to per-token features ``a`` while emitting the atom-level state   #
        #   (q, c, p) that the AtomAttentionDecoder later consumes as a skip.    #
        #                                                                        #
        #   Step 0 — Sanity-check inputs in the “with coords” branch (Diffusion  #
        #     loop): r_l / s / z must all be available:                          #
        #       if self.has_coords:                                               #
        #           assert r_l is not None and s is not None and z is not None   #
        #                                                                        #
        #   Step 1 — Build the time-invariant atom single (c_l) and atom-pair    #
        #     (p_lm) tensors once. They depend only on reference geometry, so    #
        #     they can be cached across diffusion timesteps; if the caller       #
        #     already supplied both we reuse them:                               #
        #       if p_lm is None or c_l is None:                                  #
        #           p_lm, c_l = self.prepare_cache(                              #
        #               ref_pos=ref_pos, ref_charge=ref_charge,                  #
        #               ref_mask=ref_mask,                                       #
        #               ref_atom_name_chars=ref_atom_name_chars,                 #
        #               ref_element=ref_element,                                 #
        #               atom_to_token_idx=atom_to_token_idx,                     #
        #               d_lm=d_lm, v_lm=v_lm, pad_info=pad_info,                 #
        #               r_l=r_l, z=z,                                            #
        #           )                                                            #
        #                                                                        #
        #   Step 2 — Build the query ``q_l``:                                    #
        #     With coords (diffusion path) we additively fold in the broadcast   #
        #     trunk-single ``s`` and the projected noisy coordinates ``r_l``;    #
        #     ``c_l`` is updated in-place so it carries the trunk conditioning   #
        #     for downstream pair fusion in Step 3.                              #
        #     Without coords (input embedder path) the query is just a copy of   #
        #     ``c_l``:                                                           #
        #       n_token = None                                                   #
        #       if r_l is not None:                                               #
        #           assert s is not None                                          #
        #           n_token = s.size(-2)                                          #
        #           c_l = c_l.unsqueeze(-3) + broadcast_token_to_atom(          #
        #               x_token=self.linear_no_bias_s(self.layernorm_s(s)),     #
        #               atom_to_token_idx=atom_to_token_idx,                    #
        #           )                                                            #
        #           q_l = c_l + self.linear_no_bias_r(r_l)                      #
        #       else:                                                            #
        #           q_l = c_l.clone()                                            #
        #                                                                        #
        #   Step 3 — Fuse atom-single information into the local atom-pair      #
        #     tensor. ``rearrange_qk_to_dense_trunk`` packs ``c_l`` into the     #
        #     dense-trunk layout [..., n_blocks, n_queries|n_keys, c_atom] used #
        #     by the local attention; the two ReLU-linear projections fan it    #
        #     in across query/key axes; the small zero-init MLP refines it:     #
        #       c_l_q, c_l_k, _ = rearrange_qk_to_dense_trunk(                  #
        #           q=c_l, k=c_l, dim_q=-2, dim_k=-2,                            #
        #           n_queries=self.n_queries, n_keys=self.n_keys,                #
        #           compute_mask=False,                                          #
        #       )                                                                #
        #       p_lm = (                                                         #
        #           p_lm                                                         #
        #           + self.linear_no_bias_cl(F.relu(c_l_q[..., None, :]))        #
        #           + self.linear_no_bias_cm(F.relu(c_l_k[..., None, :, :]))    #
        #       )                                                                #
        #       p_lm = p_lm + self.small_mlp(p_lm)                              #
        #                                                                        #
        #   Step 4 — Run the local atom-level transformer (Algorithm 7):         #
        #       q_l = self.atom_transformer(q_l, c_l, p_lm)                     #
        #                                                                        #
        #   Step 5 — Aggregate atoms back to tokens with mean-pooling, after a   #
        #     ReLU + projection from c_atom -> c_token:                          #
        #       a = aggregate_atom_to_token(                                    #
        #           x_atom=F.relu(self.linear_no_bias_q(q_l)),                  #
        #           atom_to_token_idx=atom_to_token_idx,                        #
        #           n_token=n_token,                                            #
        #           reduce="mean",                                              #
        #       )                                                                #
        #   Return ``(a, q_l, c_l, p_lm)`` — note ``q_l`` is returned post-      #
        #   transformer; ``c_l`` and ``p_lm`` are returned post-conditioning so  #
        #   the decoder can use them as skips later.                             #
        #                                                                        #
        # TODO: 算法 5 —— AtomAttentionEncoder。                                  #
        #   把每原子特征聚合成每 token 特征 ``a``，同时输出原子级状态 (q, c, p)，  #
        #   供后续 AtomAttentionDecoder 做 skip 用。                              #
        #                                                                        #
        #   步骤 0 — 带坐标分支 (扩散循环) 的输入校验:                              #
        #       if self.has_coords:                                               #
        #           assert r_l is not None and s is not None and z is not None   #
        #                                                                        #
        #   步骤 1 — 计算与时间无关的 c_l / p_lm (只依赖参考几何，可跨扩散步缓存)。  #
        #     调用方若已提供则复用:                                                #
        #       if p_lm is None or c_l is None:                                  #
        #           p_lm, c_l = self.prepare_cache(                              #
        #               ref_pos=ref_pos, ref_charge=ref_charge,                  #
        #               ref_mask=ref_mask,                                       #
        #               ref_atom_name_chars=ref_atom_name_chars,                 #
        #               ref_element=ref_element,                                 #
        #               atom_to_token_idx=atom_to_token_idx,                     #
        #               d_lm=d_lm, v_lm=v_lm, pad_info=pad_info,                 #
        #               r_l=r_l, z=z,                                            #
        #           )                                                            #
        #                                                                        #
        #   步骤 2 — 构造 query ``q_l``:                                          #
        #     带坐标 (扩散) 时把主干 single 广播 + 噪声坐标投影加到 c_l，          #
        #     这里 ``c_l`` 会被原地更新，从而带上 trunk 条件供步骤 3 使用；        #
        #     不带坐标 (输入嵌入路径) 直接复制 c_l:                                #
        #       n_token = None                                                   #
        #       if r_l is not None:                                               #
        #           assert s is not None                                          #
        #           n_token = s.size(-2)                                          #
        #           c_l = c_l.unsqueeze(-3) + broadcast_token_to_atom(          #
        #               x_token=self.linear_no_bias_s(self.layernorm_s(s)),     #
        #               atom_to_token_idx=atom_to_token_idx,                    #
        #           )                                                            #
        #           q_l = c_l + self.linear_no_bias_r(r_l)                      #
        #       else:                                                            #
        #           q_l = c_l.clone()                                            #
        #                                                                        #
        #   步骤 3 — 把原子 single 信息融入局部 atom-pair:                          #
        #     ``rearrange_qk_to_dense_trunk`` 把 ``c_l`` 打包成局部注意力期望的    #
        #     dense-trunk 形状；两个 ReLU+Linear 分别从 q / k 注入；               #
        #     再过一个零初始化的小 MLP:                                            #
        #       c_l_q, c_l_k, _ = rearrange_qk_to_dense_trunk(                  #
        #           q=c_l, k=c_l, dim_q=-2, dim_k=-2,                            #
        #           n_queries=self.n_queries, n_keys=self.n_keys,                #
        #           compute_mask=False,                                          #
        #       )                                                                #
        #       p_lm = (                                                         #
        #           p_lm                                                         #
        #           + self.linear_no_bias_cl(F.relu(c_l_q[..., None, :]))        #
        #           + self.linear_no_bias_cm(F.relu(c_l_k[..., None, :, :]))    #
        #       )                                                                #
        #       p_lm = p_lm + self.small_mlp(p_lm)                              #
        #                                                                        #
        #   步骤 4 — 跑局部 atom-level transformer (算法 7):                       #
        #       q_l = self.atom_transformer(q_l, c_l, p_lm)                     #
        #                                                                        #
        #   步骤 5 — 原子 -> token 的 mean-pool 聚合 (先 ReLU 再投到 c_token):     #
        #       a = aggregate_atom_to_token(                                    #
        #           x_atom=F.relu(self.linear_no_bias_q(q_l)),                  #
        #           atom_to_token_idx=atom_to_token_idx,                        #
        #           n_token=n_token,                                            #
        #           reduce="mean",                                              #
        #       )                                                                #
        #   返回 ``(a, q_l, c_l, p_lm)`` —— ``q_l`` 是过完 transformer 之后的；   #
        #   ``c_l`` / ``p_lm`` 是带上条件之后的，供解码器做 skip。                  #
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
        # TODO: Algorithm 6 — AtomAttentionDecoder.                              #
        #   Inverse direction of Algorithm 5: takes the token-level activation   #
        #   ``a`` and three atom-level skip tensors (``q_skip`` / ``c_skip`` /   #
        #   ``p_skip``) saved by the encoder, then predicts a per-atom 3-vector  #
        #   coordinate update.                                                   #
        #                                                                        #
        #   Step 1 — Re-broadcast the token activation back to atoms (with the   #
        #     atom→token index map) and fuse the encoder's atom-level skip       #
        #     ``q_skip`` via a simple sum residual:                              #
        #       q = broadcast_token_to_atom(                                    #
        #               x_token=self.linear_no_bias_a(a),                       #
        #               atom_to_token_idx=atom_to_token_idx,                    #
        #           ) + q_skip                          # [..., N_atom, c_atom] #
        #     ``linear_no_bias_a`` is LinearNoBias(c_token -> c_atom).           #
        #                                                                        #
        #   Step 2 — Run the local AtomTransformer (Algorithm 7), conditioning   #
        #     on the encoder's ``c_skip`` (atom-single) and ``p_skip`` (atom-    #
        #     pair, dense-trunk):                                                #
        #       q = self.atom_transformer(q, c_skip, p_skip)                    #
        #                                                                        #
        #   Step 3 — LayerNorm + project to a 3-vector coordinate update.        #
        #     ``layernorm_q`` runs without offset (LN with create_offset=False), #
        #     and ``linear_no_bias_out`` is set to fp32 precision so coordinate  #
        #     residuals stay numerically stable across diffusion samples:        #
        #       return self.linear_no_bias_out(self.layernorm_q(q))             #
        #                                                                        #
        # TODO: 算法 6 —— AtomAttentionDecoder。                                  #
        #   算法 5 的反向: 接收 token 级激活 ``a`` 以及编码器保留的三份             #
        #   原子级 skip (``q_skip`` / ``c_skip`` / ``p_skip``)，预测每个原子的     #
        #   3 维坐标更新。                                                        #
        #                                                                        #
        #   步骤 1 — 把 token 激活重新广播回原子，再叠加编码器的原子级 skip:        #
        #       q = broadcast_token_to_atom(                                    #
        #               x_token=self.linear_no_bias_a(a),                       #
        #               atom_to_token_idx=atom_to_token_idx,                    #
        #           ) + q_skip                          # [..., N_atom, c_atom] #
        #     ``linear_no_bias_a`` 是 LinearNoBias(c_token -> c_atom)。           #
        #                                                                        #
        #   步骤 2 — 跑局部 AtomTransformer (算法 7)，                              #
        #     用编码器的 ``c_skip`` (原子 single) 与 ``p_skip`` (原子 pair，       #
        #     dense-trunk) 作为条件:                                              #
        #       q = self.atom_transformer(q, c_skip, p_skip)                    #
        #                                                                        #
        #   步骤 3 — LayerNorm + 投到 3 维坐标增量。                                #
        #     ``layernorm_q`` 不带 offset (create_offset=False)，                  #
        #     ``linear_no_bias_out`` 用 fp32 精度保证坐标稳定:                     #
        #       return self.linear_no_bias_out(self.layernorm_q(q))             #
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
