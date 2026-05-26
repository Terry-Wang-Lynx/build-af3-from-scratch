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

from typing import Optional, Union

import torch
import torch.nn as nn

from pairformer.pair_stack import PairformerStack
from attention.linear import LinearNoBias
from pairformer.triangle_ops import LayerNorm
from model.utils import broadcast_token_to_atom, one_hot


class ConfidenceHead(nn.Module):
    """
    Implements Algorithm 31 in AF3

    Args:
        n_blocks (int, optional): number of blocks for ConfidenceHead. Defaults to 4.
        c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
        c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        c_s_inputs (int, optional): hidden dim [for single embedding from InputFeatureEmbedder]. Defaults to 449.
        b_pae (int, optional): the bin number for pae. Defaults to 64.
        b_pde (int, optional): the bin numer for pde. Defaults to 64.
        b_plddt (int, optional): the bin number for plddt. Defaults to 50.
        b_resolved (int, optional): the bin number for resolved. Defaults to 2.
        max_atoms_per_token (int, optional): max atoms in a token. Defaults to 20.
        pairformer_dropout (float, optional): dropout ratio for Pairformer. Defaults to 0.0.
        blocks_per_ckpt: number of Pairformer blocks in each activation checkpoint
        distance_bin_start (float, optional): Start of the distance bin range. Defaults to 3.25.
        distance_bin_end (float, optional): End of the distance bin range. Defaults to 52.0.
        distance_bin_step (float, optional): Step size for the distance bins. Defaults to 1.25.
        stop_gradient (bool, optional): Whether to stop gradient propagation. Defaults to True.
        hidden_scale_up (bool, optional): Whether to scale up hidden dimension. Defaults to False.
    """

    def __init__(
        self,
        n_blocks: int = 4,
        c_s: int = 384,
        c_z: int = 128,
        c_s_inputs: int = 449,
        b_pae: int = 64,
        b_pde: int = 64,
        b_plddt: int = 50,
        b_resolved: int = 2,
        max_atoms_per_token: int = 20,
        pairformer_dropout: float = 0.0,
        blocks_per_ckpt: Optional[int] = None,
        distance_bin_start: float = 3.25,
        distance_bin_end: float = 52.0,
        distance_bin_step: float = 1.25,
        stop_gradient: bool = True,
        hidden_scale_up: bool = False,
    ) -> None:
        super(ConfidenceHead, self).__init__()
        self.n_blocks = n_blocks
        self.c_s = c_s
        self.c_z = c_z
        self.c_s_inputs = c_s_inputs
        self.b_pae = b_pae
        self.b_pde = b_pde
        self.b_plddt = b_plddt
        self.b_resolved = b_resolved
        self.max_atoms_per_token = max_atoms_per_token
        self.stop_gradient = stop_gradient
        self.linear_no_bias_s1 = LinearNoBias(
            in_features=self.c_s_inputs, out_features=self.c_z
        )
        self.linear_no_bias_s2 = LinearNoBias(
            in_features=self.c_s_inputs, out_features=self.c_z
        )
        lower_bins = torch.arange(
            distance_bin_start, distance_bin_end, distance_bin_step
        )
        upper_bins = torch.cat([lower_bins[1:], lower_bins.new_tensor([1e6])], dim=-1)
        self.lower_bins = nn.Parameter(lower_bins, requires_grad=False)
        self.upper_bins = nn.Parameter(upper_bins, requires_grad=False)
        self.num_bins = len(lower_bins)  # + 1

        self.linear_no_bias_d = LinearNoBias(
            in_features=self.num_bins, out_features=self.c_z
        )
        self.linear_no_bias_d_wo_onehot = LinearNoBias(
            in_features=1, out_features=self.c_z
        )
        self.pairformer_stack = PairformerStack(
            c_z=self.c_z,
            c_s=self.c_s,
            n_blocks=n_blocks,
            dropout=pairformer_dropout,
            blocks_per_ckpt=blocks_per_ckpt,
            hidden_scale_up=hidden_scale_up,
        )
        self.linear_no_bias_pae = LinearNoBias(
            in_features=self.c_z, out_features=self.b_pae
        )
        self.linear_no_bias_pde = LinearNoBias(
            in_features=self.c_z, out_features=self.b_pde
        )
        self.plddt_weight = nn.Parameter(
            data=torch.empty(size=(self.max_atoms_per_token, self.c_s, self.b_plddt))
        )
        self.resolved_weight = nn.Parameter(
            data=torch.empty(size=(self.max_atoms_per_token, self.c_s, self.b_resolved))
        )

        self.input_strunk_ln = LayerNorm(self.c_s)
        self.pae_ln = LayerNorm(self.c_z)
        self.pde_ln = LayerNorm(self.c_z)
        self.plddt_ln = LayerNorm(self.c_s)
        self.resolved_ln = LayerNorm(self.c_s)

        with torch.no_grad():
            # Zero init for output layer (before softmax) to zero
            nn.init.zeros_(self.linear_no_bias_pae.weight)
            nn.init.zeros_(self.linear_no_bias_pde.weight)
            nn.init.zeros_(self.plddt_weight)
            nn.init.zeros_(self.resolved_weight)

    def forward(
        self,
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        pair_mask: Optional[torch.Tensor],
        x_pred_coords: torch.Tensor,
        use_embedding: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Algorithm 31 — ConfidenceHead.

        Args:
            input_feature_dict: input feature dict.
            s_inputs:  [..., N_tokens, c_s_inputs] single embedding from InputFeatureEmbedder.
            s_trunk:   [..., N_tokens, c_s]         single feature from PairformerStack.
            z_trunk:   [..., N_tokens, N_tokens, c_z] pair feature from PairformerStack.
            pair_mask: [..., N_token, N_token]      optional pair mask.
            x_pred_coords: [..., N_sample, N_atoms, 3] predicted coordinates.

        Returns:
            (plddt_preds, pae_preds, pde_preds, resolved_preds).
        """
        ##########################################################################
        # TODO: Algorithm 31 — ConfidenceHead. Produces four predictions for     #
        #   every diffusion sample:                                              #
        #     - pLDDT          (per-atom local-distance-difference test)         #
        #     - PAE            (predicted aligned error, per token-pair)         #
        #     - PDE            (predicted distance error, per token-pair)        #
        #     - resolved       (per-atom resolved-vs-unresolved classifier)      #
        #                                                                        #
        #   Step 1 — Optional stop-gradient: keep the trunk frozen so the        #
        #     confidence head trains without touching the structure model. The  #
        #     flag is configured in ``__init__`` (``self.stop_gradient``):       #
        #       if self.stop_gradient:                                           #
        #           s_inputs = s_inputs.detach()                                 #
        #           s_trunk  = s_trunk.detach()                                  #
        #           z_trunk  = z_trunk.detach()                                  #
        #                                                                        #
        #   Step 2 — Clamp + LayerNorm the trunk single track so very large      #
        #     activations (sometimes seen on long sequences) do not blow up the  #
        #     distance-bin softmax downstream:                                   #
        #       s_trunk = self.input_strunk_ln(                                  #
        #           torch.clamp(s_trunk, min=-512, max=512))                     #
        #                                                                        #
        #   Step 3 — Classifier-free path: zero the pair trunk if requested      #
        #     (corresponds to dropping the structure conditioning entirely):     #
        #       if not use_embedding:                                            #
        #           z_trunk *= 0                                                 #
        #                                                                        #
        #   Step 4 — Select the representative atom per token (Cα for amino     #
        #     acids, the per-residue centre for nucleic acids / ligands) and    #
        #     remember how many diffusion samples we got:                        #
        #       x_rep_atom_mask    = input_feature_dict[                         #
        #                              "distogram_rep_atom_mask"].bool()          #
        #       x_pred_rep_coords  = x_pred_coords[..., x_rep_atom_mask, :]       #
        #       N_sample           = x_pred_rep_coords.size(-3)                  #
        #                                                                        #
        #   Step 5 — Build the initial pair conditioning by broadcasting two    #
        #     LinearNoBias projections of ``s_inputs`` over the (token, token)   #
        #     grid (outer-sum) and adding it to ``z_trunk``:                     #
        #       z_init = (                                                       #
        #           self.linear_no_bias_s1(s_inputs)[..., None, :, :]           #
        #           + self.linear_no_bias_s2(s_inputs)[..., None, :]             #
        #       )                                                                #
        #       z_trunk = z_init + z_trunk                                       #
        #       if not self.training:                                            #
        #           del z_init                                                   #
        #           if torch.cuda.is_available():                                #
        #               torch.cuda.empty_cache()                                 #
        #                                                                        #
        #   Step 6 — Loop over diffusion samples. Each iteration calls           #
        #     ``memory_efficient_forward`` (which adds the per-sample distance   #
        #     bin features into a *clone* of the conditioning, then runs a       #
        #     small PairformerStack + four classifier heads) and collects the   #
        #     four output heads. For very long sequences move PAE/PDE to CPU    #
        #     after each sample to keep GPU memory bounded:                      #
        #       plddt_preds, pae_preds, pde_preds, resolved_preds = [],[],[],[] #
        #       for i in range(N_sample):                                       #
        #           (plddt_pred, pae_pred, pde_pred, resolved_pred              #
        #           ) = self.memory_efficient_forward(                          #
        #               input_feature_dict = input_feature_dict,                 #
        #               s_trunk            = s_trunk.clone(),                    #
        #               z_pair             = z_trunk.clone(),                    #
        #               pair_mask          = pair_mask,                          #
        #               x_pred_rep_coords  = x_pred_rep_coords[..., i, :, :],   #
        #           )                                                            #
        #           if z_trunk.shape[-2] > 2000 and not self.training:           #
        #               pae_pred = pae_pred.cpu()                                #
        #               pde_pred = pde_pred.cpu()                                #
        #               if torch.cuda.is_available():                            #
        #                   torch.cuda.empty_cache()                             #
        #           plddt_preds.append(plddt_pred)                               #
        #           pae_preds.append(pae_pred)                                   #
        #           pde_preds.append(pde_pred)                                   #
        #           resolved_preds.append(resolved_pred)                         #
        #                                                                        #
        #   Step 7 — Stack along the sample dimension. Mind the right axis for  #
        #     each head:                                                         #
        #       plddt_preds    = torch.stack(plddt_preds,    dim=-3)             #
        #       # -> [..., N_sample, N_atom, plddt_bins]                         #
        #       pae_preds      = torch.stack(pae_preds,      dim=-4)             #
        #       # -> [..., N_sample, N_token, N_token, pae_bins]                 #
        #       pde_preds      = torch.stack(pde_preds,      dim=-4)             #
        #       # -> [..., N_sample, N_token, N_token, pde_bins]                 #
        #       resolved_preds = torch.stack(resolved_preds, dim=-3)             #
        #       # -> [..., N_sample, N_atom, 2]                                  #
        #   Return ``(plddt_preds, pae_preds, pde_preds, resolved_preds)``.     #
        #                                                                        #
        # TODO: 算法 31 —— ConfidenceHead。对每个扩散样本输出四个置信预测:        #
        #     - pLDDT          (每原子的 local-distance-difference test)        #
        #     - PAE            (每 token pair 的 predicted aligned error)       #
        #     - PDE            (每 token pair 的 predicted distance error)      #
        #     - resolved       (每原子的可解析/未解析二分类)                      #
        #                                                                        #
        #   步骤 1 — 可选 stop-gradient: 冻结主干，让置信头独立训练。              #
        #     由 ``__init__`` 的 ``self.stop_gradient`` 控制:                     #
        #       if self.stop_gradient:                                           #
        #           s_inputs = s_inputs.detach()                                 #
        #           s_trunk  = s_trunk.detach()                                  #
        #           z_trunk  = z_trunk.detach()                                  #
        #                                                                        #
        #   步骤 2 — clamp + LayerNorm 主干 single，避免长序列大激活值在下游        #
        #     距离 bin softmax 中溢出:                                            #
        #       s_trunk = self.input_strunk_ln(                                  #
        #           torch.clamp(s_trunk, min=-512, max=512))                     #
        #                                                                        #
        #   步骤 3 — Classifier-free 路径: 若 use_embedding=False 则把 pair       #
        #     trunk 清零 (相当于丢掉结构条件):                                     #
        #       if not use_embedding:                                            #
        #           z_trunk *= 0                                                 #
        #                                                                        #
        #   步骤 4 — 取每个 token 的代表原子 (氨基酸的 Cα、核酸/配体的中心)，       #
        #     并记下样本数:                                                       #
        #       x_rep_atom_mask    = input_feature_dict[                         #
        #                              "distogram_rep_atom_mask"].bool()          #
        #       x_pred_rep_coords  = x_pred_coords[..., x_rep_atom_mask, :]       #
        #       N_sample           = x_pred_rep_coords.size(-3)                  #
        #                                                                        #
        #   步骤 5 — 由 ``s_inputs`` 的两次线性投影在 (token, token) 网格上做外加，#
        #     加到 ``z_trunk`` 作为初始 pair 条件:                                 #
        #       z_init = (                                                       #
        #           self.linear_no_bias_s1(s_inputs)[..., None, :, :]           #
        #           + self.linear_no_bias_s2(s_inputs)[..., None, :]             #
        #       )                                                                #
        #       z_trunk = z_init + z_trunk                                       #
        #       if not self.training:                                            #
        #           del z_init                                                   #
        #           if torch.cuda.is_available():                                #
        #               torch.cuda.empty_cache()                                 #
        #                                                                        #
        #   步骤 6 — 对每个扩散样本调 ``memory_efficient_forward`` (它会把当前样本 #
        #     的距离 bin 特征加到 ``z_pair`` 的克隆上，再跑小型 PairformerStack +  #
        #     四个分类头)；超长序列时把 PAE/PDE 转 CPU:                              #
        #       plddt_preds, pae_preds, pde_preds, resolved_preds = [],[],[],[] #
        #       for i in range(N_sample):                                       #
        #           (plddt_pred, pae_pred, pde_pred, resolved_pred              #
        #           ) = self.memory_efficient_forward(                          #
        #               input_feature_dict = input_feature_dict,                 #
        #               s_trunk            = s_trunk.clone(),                    #
        #               z_pair             = z_trunk.clone(),                    #
        #               pair_mask          = pair_mask,                          #
        #               x_pred_rep_coords  = x_pred_rep_coords[..., i, :, :],   #
        #           )                                                            #
        #           if z_trunk.shape[-2] > 2000 and not self.training:           #
        #               pae_pred = pae_pred.cpu()                                #
        #               pde_pred = pde_pred.cpu()                                #
        #               if torch.cuda.is_available():                            #
        #                   torch.cuda.empty_cache()                             #
        #           plddt_preds.append(plddt_pred)                               #
        #           pae_preds.append(pae_pred)                                   #
        #           pde_preds.append(pde_pred)                                   #
        #           resolved_preds.append(resolved_pred)                         #
        #                                                                        #
        #   步骤 7 — 按各自的正确轴沿样本维 stack:                                  #
        #       plddt_preds    = torch.stack(plddt_preds,    dim=-3)             #
        #       # -> [..., N_sample, N_atom, plddt_bins]                         #
        #       pae_preds      = torch.stack(pae_preds,      dim=-4)             #
        #       # -> [..., N_sample, N_token, N_token, pae_bins]                 #
        #       pde_preds      = torch.stack(pde_preds,      dim=-4)             #
        #       # -> [..., N_sample, N_token, N_token, pde_bins]                 #
        #       resolved_preds = torch.stack(resolved_preds, dim=-3)             #
        #       # -> [..., N_sample, N_atom, 2]                                  #
        #   返回 ``(plddt_preds, pae_preds, pde_preds, resolved_preds)``。       #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################
        return (
            plddt_preds,
            pae_preds,
            pde_preds,
            resolved_preds,
        )

    def memory_efficient_forward(
        self,
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
        s_trunk: torch.Tensor,
        z_pair: torch.Tensor,
        pair_mask: Optional[torch.Tensor],
        x_pred_rep_coords: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Confidence head for a single diffusion sample (avoids OOM)."""
        ##########################################################################
        # TODO: Per-sample confidence head — predicts pLDDT, PAE, PDE,           #
        #   resolved for ONE diffusion sample. Same idea repeated by the caller #
        #   for every sample.                                                    #
        #                                                                        #
        #   Step 1 — Pairwise distance between every pair of representative     #
        #     atoms, in fp32 (autocast off):                                     #
        #       with torch.amp.autocast("cuda", enabled=False):                  #
        #           x_pred_rep_coords = x_pred_rep_coords.to(torch.float32)      #
        #           distance_pred = torch.cdist(                                 #
        #               x_pred_rep_coords, x_pred_rep_coords)                    #
        #                                                                        #
        #   Step 2 — Bin the distances and inject into the pair conditioning.   #
        #     Two paths: one-hot binning (``one_hot`` returns ``num_bins``      #
        #     columns) and a raw distance projection — sum both in place:       #
        #       z_pair += self.linear_no_bias_d(                                 #
        #           one_hot(distance_pred, self.lower_bins, self.upper_bins)) #
        #       z_pair += self.linear_no_bias_d_wo_onehot(                       #
        #           distance_pred.unsqueeze(-1))                                 #
        #                                                                        #
        #   Step 3 — Run the small PairformerStack (uses both single and pair  #
        #     tracks):                                                           #
        #       s_single, z_pair = self.pairformer_stack(                        #
        #           s_trunk, z_pair, pair_mask)                                  #
        #                                                                        #
        #   Step 4 — Upcast both streams to fp32; pull index maps from the     #
        #     feature dict (atom -> token idx for broadcast; atom -> per-token #
        #     local atom idx for the slot-specific plddt / resolved weights):  #
        #       z_pair   = z_pair.to(torch.float32)                              #
        #       s_single = s_single.to(torch.float32)                            #
        #       atom_to_token_idx  = input_feature_dict["atom_to_token_idx"]    #
        #       atom_to_tokatom_idx = input_feature_dict["atom_to_tokatom_idx"]  #
        #                                                                        #
        #   Step 5 — Heads (run under autocast(False) for numerical stability): #
        #       with torch.amp.autocast("cuda", enabled=False):                  #
        #         # PAE: directional, no symmetrization                         #
        #         pae_pred = self.linear_no_bias_pae(self.pae_ln(z_pair))       #
        #         # PDE: symmetrized over (i, j) before LN -> output is        #
        #         #   pairwise distance error, which is symmetric              #
        #         pde_pred = self.linear_no_bias_pde(                           #
        #             self.pde_ln(z_pair + z_pair.transpose(-2, -3)))           #
        #         # Broadcast s_single back to atoms:                           #
        #         a = broadcast_token_to_atom(                                  #
        #             x_token=s_single, atom_to_token_idx=atom_to_token_idx)    #
        #         # pLDDT / resolved use per-(atom-slot) weight matrices       #
        #         # indexed by ``atom_to_tokatom_idx``:                         #
        #         plddt_pred = torch.einsum(                                    #
        #             "...nc,ncb->...nb",                                       #
        #             self.plddt_ln(a),                                          #
        #             self.plddt_weight[atom_to_tokatom_idx],                   #
        #         )                                                              #
        #         resolved_pred = torch.einsum(                                  #
        #             "...nc,ncb->...nb",                                       #
        #             self.resolved_ln(a),                                       #
        #             self.resolved_weight[atom_to_tokatom_idx],                #
        #         )                                                              #
        #                                                                        #
        #   Step 6 — Release GPU cache for very long sequences:                 #
        #       if (not self.training and z_pair.shape[-2] > 2000               #
        #           and torch.cuda.is_available()):                              #
        #           torch.cuda.empty_cache()                                     #
        #   Return ``(plddt_pred, pae_pred, pde_pred, resolved_pred)``.          #
        #                                                                        #
        # TODO: 单个扩散样本的置信度头 —— 预测 pLDDT、PAE、PDE、resolved。       #
        #   外层调用方对每个样本各跑一次这里。                                       #
        #                                                                        #
        #   步骤 1 — 在 fp32 (autocast off) 下算每对代表原子的距离矩阵:             #
        #       with torch.amp.autocast("cuda", enabled=False):                  #
        #           x_pred_rep_coords = x_pred_rep_coords.to(torch.float32)      #
        #           distance_pred = torch.cdist(                                 #
        #               x_pred_rep_coords, x_pred_rep_coords)                    #
        #                                                                        #
        #   步骤 2 — 把距离分 bin 后塞回 pair 条件。两路: one-hot bin 投影 + 原始    #
        #     距离投影，原地累加:                                                    #
        #       z_pair += self.linear_no_bias_d(                                 #
        #           one_hot(distance_pred, self.lower_bins, self.upper_bins)) #
        #       z_pair += self.linear_no_bias_d_wo_onehot(                       #
        #           distance_pred.unsqueeze(-1))                                 #
        #                                                                        #
        #   步骤 3 — 跑小型 PairformerStack (有 single 通道):                       #
        #       s_single, z_pair = self.pairformer_stack(                        #
        #           s_trunk, z_pair, pair_mask)                                  #
        #                                                                        #
        #   步骤 4 — 两路升 fp32；取出索引映射 (原子->token；原子-> 每 token 内的    #
        #     原子序号，用于 plddt / resolved 的 slot-specific 权重):                #
        #       z_pair   = z_pair.to(torch.float32)                              #
        #       s_single = s_single.to(torch.float32)                            #
        #       atom_to_token_idx  = input_feature_dict["atom_to_token_idx"]    #
        #       atom_to_tokatom_idx = input_feature_dict["atom_to_tokatom_idx"]  #
        #                                                                        #
        #   步骤 5 — 四个头 (autocast(False) 稳数值):                               #
        #       with torch.amp.autocast("cuda", enabled=False):                  #
        #         # PAE: 有向，不对称化                                            #
        #         pae_pred = self.linear_no_bias_pae(self.pae_ln(z_pair))       #
        #         # PDE: 距离误差 (对称量)，进 LN 前先对称化                       #
        #         pde_pred = self.linear_no_bias_pde(                           #
        #             self.pde_ln(z_pair + z_pair.transpose(-2, -3)))           #
        #         # s_single 广播回原子级                                          #
        #         a = broadcast_token_to_atom(                                  #
        #             x_token=s_single, atom_to_token_idx=atom_to_token_idx)    #
        #         # pLDDT / resolved 使用按 (atom-slot) 索引的权重矩阵            #
        #         plddt_pred = torch.einsum(                                    #
        #             "...nc,ncb->...nb",                                       #
        #             self.plddt_ln(a),                                          #
        #             self.plddt_weight[atom_to_tokatom_idx],                   #
        #         )                                                              #
        #         resolved_pred = torch.einsum(                                  #
        #             "...nc,ncb->...nb",                                       #
        #             self.resolved_ln(a),                                       #
        #             self.resolved_weight[atom_to_tokatom_idx],                #
        #         )                                                              #
        #                                                                        #
        #   步骤 6 — 超长序列释放显存:                                              #
        #       if (not self.training and z_pair.shape[-2] > 2000               #
        #           and torch.cuda.is_available()):                              #
        #           torch.cuda.empty_cache()                                     #
        #   返回 ``(plddt_pred, pae_pred, pde_pred, resolved_pred)``。            #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################
