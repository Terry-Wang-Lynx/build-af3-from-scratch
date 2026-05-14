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

from feature_embedding.relative_position_encoding import FourierEmbedding, RelativePositionEncoding
from attention.linear import LinearNoBias
from attention.transition import Transition
from feature_embedding.atom_attention import AtomAttentionDecoder, AtomAttentionEncoder
from diffusion.diffusion_transformer import DiffusionTransformer
from pairformer.triangle_ops import LayerNorm
from model.utils import expand_at_dim, get_checkpoint_fn, permute_final_dims


class DiffusionConditioning(nn.Module):
    """Diffusion conditioning — AF3 Algorithm 21.

    扩散过程的条件输入 —— AF3 Algorithm 21.

    Produces the single (s) and pair (z) representations that condition every
    diffusion timestep. Inputs are the trunk outputs from the Pairformer plus
    the noise-level ``t`` for the current step; everything that does *not*
    depend on ``t`` is precomputed once via :meth:`prepare_cache` and reused
    for all sampling steps.

    生成在每一步扩散去噪中作为条件的 single (s) 与 pair (z) 表示。输入是
    Pairformer 主干的输出加上当前步骤的噪声水平 t；所有与 t 无关的部分
    通过 :meth:`prepare_cache` 提前算一次，给所有采样步骤复用。

    Args / 参数:
        sigma_data:         std-dev of the data distribution (16.0 in AF3, 1.0 in EDM).
                            数据分布的标准差 (AF3 用 16.0，EDM 原论文用 1.0)。
        c_z, c_s:           pair / single hidden dims / pair 和 single 隐藏维度.
        c_s_inputs:         single-input dim from InputFeatureEmbedder.
                            来自 InputFeatureEmbedder 的 single-input 维度。
        c_noise_embedding:  Fourier-embedding dim of the noise level.
                            噪声水平的 Fourier embedding 维度。
    """

    def __init__(
        self,
        sigma_data: float = 16.0,
        c_z: int = 128,
        c_s: int = 384,
        c_s_inputs: int = 449,
        c_noise_embedding: int = 256,
    ) -> None:
        super(DiffusionConditioning, self).__init__()
        self.sigma_data = sigma_data
        self.c_z = c_z
        self.c_s = c_s
        self.c_s_inputs = c_s_inputs
        # Line1-Line3:
        self.relpe = RelativePositionEncoding(c_z=c_z)
        self.layernorm_z = LayerNorm(2 * self.c_z, create_offset=False)
        self.linear_no_bias_z = LinearNoBias(
            in_features=2 * self.c_z, out_features=self.c_z, precision=torch.float32
        )
        # Line3-Line5:
        self.transition_z1 = Transition(c_in=self.c_z, n=2)
        self.transition_z2 = Transition(c_in=self.c_z, n=2)

        # Line6-Line7
        self.layernorm_s = LayerNorm(self.c_s + self.c_s_inputs, create_offset=False)
        self.linear_no_bias_s = LinearNoBias(
            in_features=self.c_s + self.c_s_inputs,
            out_features=self.c_s,
            precision=torch.float32,
        )
        # Line8-Line9
        self.fourier_embedding = FourierEmbedding(c=c_noise_embedding)
        self.layernorm_n = LayerNorm(c_noise_embedding, create_offset=False)
        self.linear_no_bias_n = LinearNoBias(
            in_features=c_noise_embedding,
            out_features=self.c_s,
            precision=torch.float32,
        )
        # Line10-Line12
        self.transition_s1 = Transition(c_in=self.c_s, n=2)
        self.transition_s2 = Transition(c_in=self.c_s, n=2)

    def prepare_cache(
        self,
        relp_feature: torch.Tensor,
        z_trunk: torch.Tensor,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        # Pair conditioning
        pair_z = torch.cat(
            tensors=[
                z_trunk,
                self.relpe(relp_feature),
            ],
            dim=-1,
        )  # [..., N_tokens, N_tokens, 2*c_z]
        pair_z = self.linear_no_bias_z(self.layernorm_z(pair_z))
        if inplace_safe:
            pair_z += self.transition_z1(pair_z)
            pair_z += self.transition_z2(pair_z)
        else:
            pair_z = pair_z + self.transition_z1(pair_z)
            pair_z = pair_z + self.transition_z2(pair_z)
        return pair_z

    def forward(
        self,
        t_hat_noise_level: torch.Tensor,
        relp_feature: torch.Tensor,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        pair_z: torch.Tensor,
        inplace_safe: bool = False,
        use_conditioning: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            t_hat_noise_level (torch.Tensor): the noise level
                [..., N_sample]
            asym_id (torch.Tensor): asym_id
            residue_index (torch.Tensor): residue_index
            entity_id (torch.Tensor): entity_id
            token_index (torch.Tensor): token_index
            sym_id (torch.Tensor): sym_id
            s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
                [..., N_tokens, c_s_inputs]
            s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
                [..., N_tokens, c_s]
            z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
                [..., N_tokens, N_tokens, c_z]
            inplace_safe (bool): Whether it is safe to use inplace operations.
            use_conditioning (bool): Whether to drop the s/z embeddings.
        Returns:
            tuple[torch.Tensor, torch.Tensor]: embeddings s and z
                - s (torch.Tensor): [..., N_sample, N_tokens, c_s]
                - z (torch.Tensor): [..., N_tokens, N_tokens, c_z]
        """
        ##########################################################################
        # TODO: Algorithm 21. Build (s, z) for the current diffusion step.       #
        #   1. If ``pair_z`` cache is missing, compute it from relp + z_trunk via#
        #      ``prepare_cache``.                                                #
        #   2. Concatenate s_trunk + s_inputs along channel, project to c_s.    #
        #   3. Add a Fourier embedding of the noise level (log(t/sigma)/4),     #
        #      LayerNorm + Linear.                                              #
        #   4. Apply two Transition blocks (residual SwiGLU FFN) to single_s.   #
        # TODO: Algorithm 21. 为当前扩散步构造 (s, z)。                          #
        #   1. ``pair_z`` 缓存为空时，用 ``prepare_cache`` 从 relp + z_trunk 算。#
        #   2. 沿通道拼 s_trunk + s_inputs，再线性投影到 c_s。                   #
        #   3. 噪声水平做 Fourier embedding (log(t/sigma)/4)，LN+Linear 后加到 s。#
        #   4. 跑两次 Transition (残差 SwiGLU FFN)。                             #
        ##########################################################################

        if pair_z is None:
            if not use_conditioning:
                if inplace_safe:
                    s_trunk *= 0
                    z_trunk *= 0
                else:
                    s_trunk = 0 * s_trunk
                    z_trunk = 0 * z_trunk
            pair_z = self.prepare_cache(relp_feature, z_trunk, inplace_safe)
        else:
            # Pair conditioning
            if inplace_safe:
                pair_z_clone = pair_z.clone()
                pair_z = pair_z_clone
        # Single conditioning
        single_s = torch.cat(
            tensors=[s_trunk, s_inputs], dim=-1
        )  # [..., N_tokens, c_s + c_s_inputs]
        single_s = self.linear_no_bias_s(self.layernorm_s(single_s))
        noise_n = self.fourier_embedding(
            t_hat_noise_level=torch.log(input=t_hat_noise_level / self.sigma_data) / 4
        ).to(
            single_s.dtype
        )  # [..., N_sample, c_in]
        single_s = single_s.unsqueeze(dim=-3) + self.linear_no_bias_n(
            self.layernorm_n(noise_n)
        ).unsqueeze(
            dim=-2
        )  # [..., N_sample, N_tokens, c_s]
        if inplace_safe:
            single_s += self.transition_s1(single_s)
            single_s += self.transition_s2(single_s)
        else:
            single_s = single_s + self.transition_s1(single_s)
            single_s = single_s + self.transition_s2(single_s)

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################
        return single_s, pair_z


class DiffusionSchedule:
    """
    Diffusion schedule for training and inference.

    Args:
        sigma_data (float, optional): The standard deviation of the data. Defaults to 16.0.
        s_max (float, optional): The maximum noise level. Defaults to 160.0.
        s_min (float, optional): The minimum noise level. Defaults to 4e-4.
        p (float, optional): The exponent for the noise schedule. Defaults to 7.0.
        dt (float, optional): The time step size. Defaults to 1/200.
        p_mean (float, optional): The mean of the log-normal distribution for noise level sampling. Defaults to -1.2.
        p_std (float, optional): The standard deviation of the log-normal distribution for noise level sampling. Defaults to 1.5.
    """

    def __init__(
        self,
        sigma_data: float = 16.0,
        s_max: float = 160.0,
        s_min: float = 4e-4,
        p: float = 7.0,
        dt: float = 1 / 200,
        p_mean: float = -1.2,
        p_std: float = 1.5,
    ) -> None:
        self.sigma_data = sigma_data
        self.s_max = s_max
        self.s_min = s_min
        self.p = p
        self.dt = dt
        self.p_mean = p_mean
        self.p_std = p_std
        # self.T
        self.T = int(1 / dt) + 1  # 201

    def get_train_noise_schedule(self) -> torch.Tensor:
        return self.sigma_data * torch.exp(self.p_mean + self.p_std * torch.randn(1))

    def get_inference_noise_schedule(self) -> torch.Tensor:
        time_step_lists = torch.arange(start=0, end=1 + 1e-10, step=self.dt)
        inference_noise_schedule = (
            self.sigma_data
            * (
                self.s_max ** (1 / self.p)
                + time_step_lists
                * (self.s_min ** (1 / self.p) - self.s_max ** (1 / self.p))
            )
            ** self.p
        )
        return inference_noise_schedule


class DiffusionModule(nn.Module):
    """Single-step denoising network — AF3 Algorithm 20.

    单步去噪网络 —— AF3 Algorithm 20.

    Given noisy atom coordinates ``x_noisy`` and a noise level ``t_hat``,
    this module predicts the clean coordinates ``x_denoised``. The outer
    sampling loop (:func:`diffusion.sampler.sample_diffusion`) calls this
    once per Euler step.

    给定加噪原子坐标 ``x_noisy`` 和噪声水平 ``t_hat``，该模块预测去噪后的
    坐标 ``x_denoised``。外层采样循环 (sample_diffusion) 每一步 Euler 迭代
    调用一次。

    Following the EDM formulation:
        F(x_noisy, t) = denoised
        D = c_skip * x_noisy + c_out * F(c_in * x_noisy, c_noise(t))

    Internally we run:
        1. ``DiffusionConditioning``        — build (s, z) from trunk + noise level.
        2. ``AtomAttentionEncoder``         — lift atom-level features to tokens.
        3. ``DiffusionTransformer``         — token-level self-attention with AdaLN.
        4. ``AtomAttentionDecoder``         — map back to per-atom updates.

    Args / 参数:
        sigma_data:     std-dev of training data distribution / 训练分布标准差。
        c_atom, c_atompair, c_token, c_s, c_z, c_s_inputs:
                        hidden sizes for the various representations / 各种表示的隐藏维度。
        atom_encoder / transformer / atom_decoder:
                        sub-module configs (block count, head count, …) /
                        子模块的配置 (block 数量、注意力头数等)。
        drop_path_rate: training-only stochastic-depth rate / 仅训练用的 stochastic depth。
        blocks_per_ckpt / use_fine_grained_checkpoint:
                        gradient-checkpointing knobs (training only) /
                        梯度检查点开关 (仅训练用)。
    """

    def __init__(
        self,
        sigma_data: float = 16.0,
        c_atom: int = 128,
        c_atompair: int = 16,
        c_token: int = 768,
        c_s: int = 384,
        c_z: int = 128,
        c_s_inputs: int = 449,
        atom_encoder: dict[str, int] = {"n_blocks": 3, "n_heads": 4},
        transformer: dict[str, int] = {
            "n_blocks": 24,
            "n_heads": 16,
            "drop_path_rate": 0,
        },
        atom_decoder: dict[str, int] = {"n_blocks": 3, "n_heads": 4},
        drop_path_rate: float = 0.0,
        blocks_per_ckpt: Optional[int] = None,
        use_fine_grained_checkpoint: bool = False,
    ) -> None:
        super(DiffusionModule, self).__init__()
        self.sigma_data = sigma_data
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_token = c_token
        self.c_s_inputs = c_s_inputs
        self.c_s = c_s
        self.c_z = c_z

        # Grad checkpoint setting
        self.blocks_per_ckpt = blocks_per_ckpt
        self.use_fine_grained_checkpoint = use_fine_grained_checkpoint

        self.diffusion_conditioning = DiffusionConditioning(
            sigma_data=self.sigma_data, c_z=c_z, c_s=c_s, c_s_inputs=c_s_inputs
        )
        self.atom_attention_encoder = AtomAttentionEncoder(
            **atom_encoder,
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_token=c_token,
            has_coords=True,
            c_s=c_s,
            c_z=c_z,
            blocks_per_ckpt=blocks_per_ckpt,
        )
        # Alg20: line4
        self.layernorm_s = LayerNorm(c_s, create_offset=False)
        self.linear_no_bias_s = LinearNoBias(
            in_features=c_s,
            out_features=c_token,
            precision=torch.float32,
            initializer="zeros",
        )
        self.diffusion_transformer = DiffusionTransformer(
            **transformer,
            c_a=c_token,
            c_s=c_s,
            c_z=c_z,
            blocks_per_ckpt=blocks_per_ckpt,
        )
        self.layernorm_a = LayerNorm(c_token, create_offset=False)
        self.atom_attention_decoder = AtomAttentionDecoder(
            **atom_decoder,
            c_token=c_token,
            c_atom=c_atom,
            c_atompair=c_atompair,
            blocks_per_ckpt=blocks_per_ckpt,
        )
        self.normalize = LayerNorm(c_z, create_offset=False, create_scale=False)

    def f_forward(
        self,
        r_noisy: torch.Tensor,
        t_hat_noise_level: torch.Tensor,
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        pair_z: torch.Tensor,
        p_lm: torch.Tensor,
        c_l: torch.Tensor,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        use_conditioning: bool = True,
        enable_efficient_fusion: bool = False,
    ) -> torch.Tensor:
        """The raw denoising network F_theta — EDM equation (7).
        EDM 公式 (7) 中的去噪网络 F_theta。

        Computes F_theta(c_in * x, c_noise(sigma)); c_noise(sigma) is built
        inside :class:`DiffusionConditioning`.
        计算 F_theta(c_in * x, c_noise(sigma))；c_noise(sigma) 在
        :class:`DiffusionConditioning` 内部构造。

        Args:
            r_noisy (torch.Tensor): scaled x_noisy (i.e., c_in * x)
                [..., N_sample, N_atom, 3]
            t_hat_noise_level (torch.Tensor): the noise level, as well as the time step t
                [..., N_sample]
            input_feature_dict (dict[str, Union[torch.Tensor, int, float, dict]]): input feature
            s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
                [..., N_tokens, c_s_inputs]
            s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
                [..., N_tokens, c_s]
            z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
                [..., N_tokens, N_tokens, c_z]
            pair_z (torch.Tensor): diffusion pair embedding
                [..., N_tokens, N_tokens, c_z]
            p_lm (torch.Tensor): MSA embedding
                [..., N_tokens, c_p_lm]
            c_l (torch.Tensor): ligand embedding
                [..., N_tokens, c_c_l]
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.
            use_conditioning (bool): Whether to drop the s/z embeddings in DiffusionConditioning.
            enable_efficient_fusion (bool): Whether to enable efficient fusion. Defaults to False.

        Returns:
            torch.Tensor: coordinates update
                [..., N_sample, N_atom, 3]
        """
        N_sample = r_noisy.size(-3)
        assert t_hat_noise_level.size(-1) == N_sample

        # Conditioning is shared across diffusion samples.
        s_single, z_pair = self.diffusion_conditioning(
            t_hat_noise_level,
            input_feature_dict["relp"],
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            pair_z=pair_z,
            inplace_safe=inplace_safe,
            use_conditioning=use_conditioning,
        )

        s_trunk = expand_at_dim(s_trunk, dim=-3, n=1)
        z_pair = expand_at_dim(z_pair, dim=-4, n=1)

        # Atom-attention encoder: token aggregation over sequence-local atoms.
        a_token, q_skip, c_skip, p_skip = self.atom_attention_encoder(
            input_feature_dict["atom_to_token_idx"],
            input_feature_dict["ref_pos"],
            input_feature_dict["ref_charge"],
            input_feature_dict["ref_mask"],
            input_feature_dict["ref_atom_name_chars"],
            input_feature_dict["ref_element"],
            input_feature_dict["d_lm"],
            input_feature_dict["v_lm"],
            input_feature_dict["pad_info"],
            r_l=r_noisy,
            s=s_trunk,
            z=z_pair,
            p_lm=p_lm,
            c_l=c_l,
        )

        a_token = a_token.to(torch.float32)
        a_token = a_token + self.linear_no_bias_s(self.layernorm_s(s_single))

        # Token-level DiffusionTransformer.
        z = z_pair.to(torch.float32)
        a_token = self.diffusion_transformer(
            a=a_token.to(torch.float32),
            s=s_single.to(torch.float32),
            z=z,
        )
        a_token = self.layernorm_a(a_token)

        # Atom-attention decoder: produce per-atom coordinate update.
        r_update = self.atom_attention_decoder(
            atom_to_token_idx=input_feature_dict["atom_to_token_idx"],
            a=a_token,
            q_skip=q_skip,
            c_skip=c_skip,
            p_skip=p_skip,
        )
        return r_update

    def forward(
        self,
        x_noisy: torch.Tensor,
        t_hat_noise_level: torch.Tensor,
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        pair_z: torch.Tensor,
        p_lm: torch.Tensor,
        c_l: torch.Tensor,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        use_conditioning: bool = True,
        enable_efficient_fusion: bool = False,
    ) -> torch.Tensor:
        """One step denoise: x_noisy, noise_level -> x_denoised

        Args:
            x_noisy (torch.Tensor): the noisy version of the input atom coords
                [..., N_sample, N_atom,3]
            t_hat_noise_level (torch.Tensor): the noise level, as well as the time step t
                [..., N_sample]
            input_feature_dict (dict[str, Union[torch.Tensor, int, float, dict]]): input meta feature dict
            s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
                [..., N_tokens, c_s_inputs]
            s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
                [..., N_tokens, c_s]
            z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
                [..., N_tokens, N_tokens, c_z]
            pair_z (torch.Tensor): diffusion pair embedding
                [..., N_tokens, N_tokens, c_z]
            p_lm (torch.Tensor): MSA embedding
                [..., N_tokens, c_p_lm]
            c_l (torch.Tensor): ligand embedding
                [..., N_tokens, c_c_l]
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.
            use_conditioning (bool): Whether to drop the s/z embeddings in DiffusionConditioning.
            enable_efficient_fusion (bool): Whether to enable efficient fusion. Defaults to False.

        Returns:
            torch.Tensor: the denoised coordinates of x
                [..., N_sample, N_atom,3]
        """
        ##########################################################################
        # TODO: EDM single-step denoise wrapper (Algorithm 20).                  #
        #   1. Scale ``x_noisy`` by c_in = 1/sqrt(sigma_data^2 + sigma^2).        #
        #   2. Call ``self.f_forward`` to get the raw network update r_update.   #
        #   3. Combine: x_denoised = c_skip*x + c_out*r_update where             #
        #        s_ratio = sigma / sigma_data                                   #
        #        c_skip = 1 / (1 + s_ratio^2)                                   #
        #        c_out  = sigma / sqrt(1 + s_ratio^2)                            #
        # TODO: EDM 单步去噪封装 (Algorithm 20)。                                #
        #   1. 用 c_in = 1/sqrt(sigma_data^2 + sigma^2) 缩放 x_noisy。           #
        #   2. 调 ``self.f_forward`` 得到原始网络输出 r_update。                 #
        #   3. 按 EDM 公式合成 x_denoised = c_skip*x + c_out*r_update。          #
        ##########################################################################

        r_noisy = (
            x_noisy
            / torch.sqrt(self.sigma_data**2 + t_hat_noise_level**2)[..., None, None]
        )

        # Compute the update given r_noisy (the scaled x_noisy)
        # As in EDM:
        #     r_update = F(r_noisy, c_noise(sigma))
        r_update = self.f_forward(
            r_noisy=r_noisy,
            t_hat_noise_level=t_hat_noise_level,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            pair_z=pair_z,
            p_lm=p_lm,
            c_l=c_l,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
            use_conditioning=use_conditioning,
            enable_efficient_fusion=enable_efficient_fusion,
        )

        # Rescale updates to positions and combine with input positions
        # As in EDM:
        #     D = c_skip * x_noisy + c_out * r_update
        #     c_skip = sigma_data^2 / (sigma_data^2 + sigma^2)
        #     c_out = (sigma_data * sigma) / sqrt(sigma_data^2 + sigma^2)
        #     s_ratio = sigma / sigma_data
        #     c_skip = 1 / (1 + s_ratio^2)
        #     c_out = sigma / sqrt(1 + s_ratio^2)

        s_ratio = (t_hat_noise_level / self.sigma_data)[..., None, None].to(
            r_update.dtype
        )
        x_denoised = (
            1 / (1 + s_ratio**2) * x_noisy
            + t_hat_noise_level[..., None, None]
            / torch.sqrt(1 + s_ratio**2)
            * r_update
        ).to(r_update.dtype)

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################
        return x_denoised
