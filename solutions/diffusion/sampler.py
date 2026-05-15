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

from typing import Any, Callable, Optional

import torch

from model.utils import centre_random_augmentation
from runtime.logger import get_logger

logger = get_logger(__name__)


class TrainingNoiseSampler:
    """
    Sample the noise-level of training samples.

    Args:
        p_mean (float, optional): gaussian mean. Defaults to -1.2.
        p_std (float, optional): gaussian std. Defaults to 1.5.
        sigma_data (float, optional): scale. Defaults to 16.0, but this is 1.0 in EDM.
    """

    def __init__(
        self,
        p_mean: float = -1.2,
        p_std: float = 1.5,
        sigma_data: float = 16.0,  # NOTE: in EDM, this is 1.0
    ) -> None:
        self.sigma_data = sigma_data
        self.p_mean = p_mean
        self.p_std = p_std

    def __call__(
        self, size: torch.Size, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """Sampling

        Args:
            size (torch.Size): the target size
            device (torch.device, optional): target device. Defaults to torch.device("cpu").

        Returns:
            torch.Tensor: sampled noise-level
        """
        rnd_normal = torch.randn(size=size, device=device)
        noise_level = (rnd_normal * self.p_std + self.p_mean).exp() * self.sigma_data
        return noise_level


class InferenceNoiseScheduler:
    """
    Scheduler for noise-level (time steps).

    Args:
        s_max (float, optional): maximal noise level. Defaults to 160.0.
        s_min (float, optional): minimal noise level. Defaults to 4e-4.
        rho (float, optional): the exponent numerical part. Defaults to 7.
        sigma_data (float, optional): scale. Defaults to 16.0, but this is 1.0 in EDM.
    """

    def __init__(
        self,
        s_max: float = 160.0,
        s_min: float = 4e-4,
        rho: float = 7,
        sigma_data: float = 16.0,  # NOTE: in EDM, this is 1.0
    ) -> None:
        self.sigma_data = sigma_data
        self.s_max = s_max
        self.s_min = s_min
        self.rho = rho

    def __call__(
        self,
        N_step: int = 200,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Schedule the noise-level (time steps). No sampling is performed.

        Args:
            N_step (int, optional): number of time steps. Defaults to 200.
            device (torch.device, optional): target device. Defaults to torch.device("cpu").
            dtype (torch.dtype, optional): target dtype. Defaults to torch.float32.

        Returns:
            torch.Tensor: noise-level (time_steps)
                [N_step+1]
        """
        step_size = 1 / N_step
        step_indices = torch.arange(N_step + 1, device=device, dtype=dtype)
        t_step_list = (
            self.sigma_data
            * (
                self.s_max ** (1 / self.rho)
                + step_indices
                * step_size
                * (self.s_min ** (1 / self.rho) - self.s_max ** (1 / self.rho))
            )
            ** self.rho
        )
        # replace the last time step by 0
        t_step_list[..., -1] = 0  # t_N = 0

        return t_step_list


def sample_diffusion(
    denoise_net: Callable,
    input_feature_dict: dict[str, Any],
    s_inputs: torch.Tensor,
    s_trunk: torch.Tensor,
    z_trunk: torch.Tensor,
    pair_z: torch.Tensor,
    p_lm: torch.Tensor,
    c_l: torch.Tensor,
    noise_schedule: torch.Tensor,
    N_sample: int = 1,
    gamma0: float = 0.8,
    gamma_min: float = 1.0,
    noise_scale_lambda: float = 1.003,
    step_scale_eta: float = 1.5,
    diffusion_chunk_size: Optional[int] = None,
    inplace_safe: bool = False,
    attn_chunk_size: Optional[int] = None,
    enable_efficient_fusion: bool = False,
    guidance_configs: Optional[dict[str, Any]] = None,
) -> torch.Tensor:
    """AF3 Algorithm 18 — Euler-step coordinate diffusion sampler.

    AF3 Algorithm 18 —— 坐标扩散的 Euler 步采样器.

    Walks the noise schedule from t=T (pure Gaussian noise) down to t=0
    (clean coordinates), calling ``denoise_net`` once per step.

    沿噪声调度从 t=T (纯高斯噪声) 走到 t=0 (干净坐标)，每一步调用一次
    ``denoise_net``。

    Args:
        denoise_net (Callable): the network that performs the denoising step.
        input_feature_dict (dict[str, Any]): input meta feature dict
        s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
            [..., N_tokens, c_s_inputs]
        s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
            [..., N_tokens, c_s]
        z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
            [..., N_tokens, N_tokens, c_z]
        pair_z (torch.Tensor): pair feature embedding from InputFeatureEmbedder
            [..., N_tokens, N_tokens, c_z_inputs]
        p_lm (torch.Tensor): MSA embedding
            [..., N_tokens, c_p_lm]
        c_l (torch.Tensor): ligand embedding
            [..., N_tokens, c_c_l]
        noise_schedule (torch.Tensor): noise-level schedule (which is also the time steps) since sigma=t.
            [N_iterations]
        N_sample (int): number of generated samples
        gamma0 (float): params in Alg.18.
        gamma_min (float): params in Alg.18.
        noise_scale_lambda (float): params in Alg.18.
        step_scale_eta (float): params in Alg.18.
        diffusion_chunk_size (Optional[int]): Chunk size for diffusion operation. Defaults to None.
        inplace_safe (bool): Whether to use inplace operations safely. Defaults to False.
        attn_chunk_size (Optional[int]): Chunk size for attention operation. Defaults to None.
        enable_efficient_fusion (bool): Whether to enable efficient fusion. Defaults to False.
        guidance_configs (Optional[dict[str, Any]]): training free guidance configs. Defaults to None.

    Returns:
        torch.Tensor: the denoised coordinates of x in inference stage
            [..., N_sample, N_atom, 3]
    """
    del guidance_configs
    ##########################################################################
    # TODO: Algorithm 18 — Euler-step (predictor-corrector) coordinate        #
    #   diffusion sampler. Walks ``noise_schedule`` from t=T (pure Gaussian   #
    #   noise) down to t=0 (clean coordinates), calling ``denoise_net``       #
    #   once per step.                                                        #
    #                                                                        #
    #   Step 1 — Read the static shapes from the inputs:                     #
    #       N_atom      = input_feature_dict["atom_to_token_idx"].size(-1)   #
    #       batch_shape = s_inputs.shape[:-2]                                #
    #       device      = s_inputs.device                                    #
    #       dtype       = s_inputs.dtype                                     #
    #                                                                        #
    #   Step 2 — Inner closure that runs ONE chunk of N_sample (so very-     #
    #     large N_sample can be split into smaller passes):                  #
    #                                                                        #
    #     def _chunk_sample_diffusion(chunk_n_sample, inplace_safe):         #
    #         # init at t=T with pure Gaussian, scaled by noise_schedule[0]: #
    #         x_l = noise_schedule[0] * torch.randn(                          #
    #             size=(*batch_shape, chunk_n_sample, N_atom, 3),             #
    #             device=device, dtype=dtype,                                 #
    #         )                                                               #
    #                                                                        #
    #         for step_i, (c_tau_last, c_tau) in enumerate(                   #
    #             zip(noise_schedule[:-1], noise_schedule[1:])                #
    #         ):                                                              #
    #             # (a) re-centre + random rotate each sample at each step:  #
    #             x_l = (                                                     #
    #                 centre_random_augmentation(                             #
    #                     x_input_coords=x_l, N_sample=1)                     #
    #                 .squeeze(dim=-3)                                        #
    #                 .to(dtype)                                              #
    #             )                                                           #
    #                                                                        #
    #             # (b) Predictor: hop noise level c_tau_last -> t_hat by    #
    #             #     re-noising. ``gamma`` is non-zero only when current  #
    #             #     noise level is above ``gamma_min``:                  #
    #             gamma = float(gamma0) if c_tau > gamma_min else 0           #
    #             t_hat = c_tau_last * (gamma + 1)                            #
    #             delta_noise_level = torch.sqrt(t_hat**2 - c_tau_last**2)    #
    #             x_noisy = x_l + (                                           #
    #                 noise_scale_lambda * delta_noise_level                  #
    #                 * torch.randn(                                          #
    #                     size=x_l.shape, device=device, dtype=dtype)        #
    #             )                                                           #
    #                                                                        #
    #             # (c) Corrector: a single Euler denoise step from t_hat    #
    #             #     to c_tau. Broadcast t_hat across the batch dims for #
    #             #     ``denoise_net``:                                      #
    #             t_hat = (                                                   #
    #                 t_hat.reshape((1,) * (len(batch_shape) + 1))            #
    #                 .expand(*batch_shape, chunk_n_sample)                   #
    #                 .to(dtype)                                              #
    #             )                                                           #
    #             x_denoised = denoise_net(                                   #
    #                 x_noisy=x_noisy,                                        #
    #                 t_hat_noise_level=t_hat,                                #
    #                 input_feature_dict=input_feature_dict,                  #
    #                 s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=z_trunk,   #
    #                 pair_z=pair_z, p_lm=p_lm, c_l=c_l,                     #
    #                 chunk_size=attn_chunk_size,                             #
    #                 inplace_safe=inplace_safe,                              #
    #                 enable_efficient_fusion=enable_efficient_fusion,       #
    #             )                                                           #
    #                                                                        #
    #             # Euler integration step: delta = (x_noisy - x_denoised)/  #
    #             #     t_hat is the score-like direction; the step is       #
    #             #     scaled by (c_tau - t_hat) and the global eta:        #
    #             delta = (x_noisy - x_denoised) / t_hat[..., None, None]    #
    #             dt    = c_tau - t_hat                                       #
    #             x_l   = x_noisy + step_scale_eta * dt[..., None, None] * delta#
    #                                                                        #
    #         return x_l                                                      #
    #                                                                        #
    #   Step 3 — Run once, or chunk if requested:                            #
    #     if diffusion_chunk_size is None:                                   #
    #         x_l = _chunk_sample_diffusion(                                  #
    #             N_sample, inplace_safe=inplace_safe)                        #
    #     else:                                                              #
    #         x_l = []                                                       #
    #         no_chunks = N_sample // diffusion_chunk_size + (               #
    #             N_sample % diffusion_chunk_size != 0)                       #
    #         for i in range(no_chunks):                                     #
    #             chunk_n_sample = (                                          #
    #                 diffusion_chunk_size                                    #
    #                 if i < no_chunks - 1                                    #
    #                 else N_sample - i * diffusion_chunk_size)               #
    #             chunk_x_l = _chunk_sample_diffusion(                        #
    #                 chunk_n_sample, inplace_safe=inplace_safe)              #
    #             x_l.append(chunk_x_l)                                       #
    #         x_l = torch.cat(x_l, -3)                                        #
    #     return x_l                                                          #
    #                                                                        #
    # TODO: 算法 18 —— Euler 步 (预测-校正) 坐标扩散采样器。沿 ``noise_schedule``#
    #   从 t=T (纯高斯噪声) 走到 t=0 (干净坐标)，每步调用 ``denoise_net`` 一次。  #
    #                                                                        #
    #   步骤 1 — 读取静态形状:                                                  #
    #       N_atom      = input_feature_dict["atom_to_token_idx"].size(-1)   #
    #       batch_shape = s_inputs.shape[:-2]                                #
    #       device      = s_inputs.device                                    #
    #       dtype       = s_inputs.dtype                                     #
    #                                                                        #
    #   步骤 2 — 内部闭包：跑一个 N_sample 的子块 (大 N_sample 时可分块):       #
    #     def _chunk_sample_diffusion(chunk_n_sample, inplace_safe):         #
    #         # t=T 时纯高斯，幅度由 noise_schedule[0] 决定:                    #
    #         x_l = noise_schedule[0] * torch.randn(                          #
    #             size=(*batch_shape, chunk_n_sample, N_atom, 3),             #
    #             device=device, dtype=dtype,                                 #
    #         )                                                               #
    #         for step_i, (c_tau_last, c_tau) in enumerate(                   #
    #             zip(noise_schedule[:-1], noise_schedule[1:])                #
    #         ):                                                              #
    #             # (a) 每步对每个样本重新居中 + 随机旋转:                       #
    #             x_l = (                                                     #
    #                 centre_random_augmentation(                             #
    #                     x_input_coords=x_l, N_sample=1)                     #
    #                 .squeeze(dim=-3)                                        #
    #                 .to(dtype)                                              #
    #             )                                                           #
    #             # (b) Predictor: 从 c_tau_last 重新加噪到 t_hat。            #
    #             #     当前噪声水平超过 ``gamma_min`` 时 ``gamma`` 才非零:    #
    #             gamma = float(gamma0) if c_tau > gamma_min else 0           #
    #             t_hat = c_tau_last * (gamma + 1)                            #
    #             delta_noise_level = torch.sqrt(t_hat**2 - c_tau_last**2)    #
    #             x_noisy = x_l + (                                           #
    #                 noise_scale_lambda * delta_noise_level                  #
    #                 * torch.randn(                                          #
    #                     size=x_l.shape, device=device, dtype=dtype)        #
    #             )                                                           #
    #             # (c) Corrector: 一次 Euler 去噪步 (t_hat -> c_tau)。        #
    #             #     先把 t_hat 广播到 batch 维:                             #
    #             t_hat = (                                                   #
    #                 t_hat.reshape((1,) * (len(batch_shape) + 1))            #
    #                 .expand(*batch_shape, chunk_n_sample)                   #
    #                 .to(dtype)                                              #
    #             )                                                           #
    #             x_denoised = denoise_net(                                   #
    #                 x_noisy=x_noisy,                                        #
    #                 t_hat_noise_level=t_hat,                                #
    #                 input_feature_dict=input_feature_dict,                  #
    #                 s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=z_trunk,   #
    #                 pair_z=pair_z, p_lm=p_lm, c_l=c_l,                     #
    #                 chunk_size=attn_chunk_size,                             #
    #                 inplace_safe=inplace_safe,                              #
    #                 enable_efficient_fusion=enable_efficient_fusion,       #
    #             )                                                           #
    #             # Euler 积分:                                                #
    #             delta = (x_noisy - x_denoised) / t_hat[..., None, None]    #
    #             dt    = c_tau - t_hat                                       #
    #             x_l   = x_noisy + step_scale_eta * dt[..., None, None] * delta#
    #         return x_l                                                      #
    #                                                                        #
    #   步骤 3 — 直接跑或分块跑:                                                #
    #     if diffusion_chunk_size is None:                                   #
    #         x_l = _chunk_sample_diffusion(                                  #
    #             N_sample, inplace_safe=inplace_safe)                        #
    #     else:                                                              #
    #         x_l = []                                                       #
    #         no_chunks = N_sample // diffusion_chunk_size + (               #
    #             N_sample % diffusion_chunk_size != 0)                       #
    #         for i in range(no_chunks):                                     #
    #             chunk_n_sample = (                                          #
    #                 diffusion_chunk_size                                    #
    #                 if i < no_chunks - 1                                    #
    #                 else N_sample - i * diffusion_chunk_size)               #
    #             chunk_x_l = _chunk_sample_diffusion(                        #
    #                 chunk_n_sample, inplace_safe=inplace_safe)              #
    #             x_l.append(chunk_x_l)                                       #
    #         x_l = torch.cat(x_l, -3)                                        #
    #     return x_l                                                          #
    ##########################################################################

    N_atom = input_feature_dict["atom_to_token_idx"].size(-1)
    batch_shape = s_inputs.shape[:-2]
    device = s_inputs.device
    dtype = s_inputs.dtype

    def _chunk_sample_diffusion(chunk_n_sample, inplace_safe):
        # init noise
        # [..., N_sample, N_atom, 3]
        x_l = noise_schedule[0] * torch.randn(
            size=(*batch_shape, chunk_n_sample, N_atom, 3), device=device, dtype=dtype
        )  # NOTE: set seed in distributed training

        for step_i, (c_tau_last, c_tau) in enumerate(
            zip(noise_schedule[:-1], noise_schedule[1:])
        ):
            # [..., N_sample, N_atom, 3]
            x_l = (
                centre_random_augmentation(x_input_coords=x_l, N_sample=1)
                .squeeze(dim=-3)
                .to(dtype)
            )

            # Denoise with a predictor-corrector sampler
            # 1. Add noise to move x_{c_tau_last} to x_{t_hat}
            gamma = float(gamma0) if c_tau > gamma_min else 0
            t_hat = c_tau_last * (gamma + 1)

            delta_noise_level = torch.sqrt(t_hat**2 - c_tau_last**2)
            x_noisy = x_l + noise_scale_lambda * delta_noise_level * torch.randn(
                size=x_l.shape, device=device, dtype=dtype
            )

            # 2. Denoise from x_{t_hat} to x_{c_tau}
            # Euler step only
            t_hat = (
                t_hat.reshape((1,) * (len(batch_shape) + 1))
                .expand(*batch_shape, chunk_n_sample)
                .to(dtype)
            )

            x_denoised = denoise_net(
                x_noisy=x_noisy,
                t_hat_noise_level=t_hat,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                pair_z=pair_z,
                p_lm=p_lm,
                c_l=c_l,
                chunk_size=attn_chunk_size,
                inplace_safe=inplace_safe,
                enable_efficient_fusion=enable_efficient_fusion,
            )

            # Euler step. AF3 paper line 9 says 'x_l_hat'; we believe that's a typo.
            delta = (x_noisy - x_denoised) / t_hat[..., None, None]
            dt = c_tau - t_hat
            x_l = x_noisy + step_scale_eta * dt[..., None, None] * delta

        return x_l

    if diffusion_chunk_size is None:
        x_l = _chunk_sample_diffusion(N_sample, inplace_safe=inplace_safe)
    else:
        x_l = []
        no_chunks = N_sample // diffusion_chunk_size + (
            N_sample % diffusion_chunk_size != 0
        )
        for i in range(no_chunks):
            chunk_n_sample = (
                diffusion_chunk_size
                if i < no_chunks - 1
                else N_sample - i * diffusion_chunk_size
            )
            chunk_x_l = _chunk_sample_diffusion(
                chunk_n_sample, inplace_safe=inplace_safe
            )
            x_l.append(chunk_x_l)
        x_l = torch.cat(x_l, -3)  # [..., N_sample, N_atom, 3]
    return x_l

    ##########################################################################
    #               END OF YOUR CODE                                         #
    ##########################################################################


def sample_diffusion_training(
    noise_sampler: TrainingNoiseSampler,
    denoise_net: Callable,
    label_dict: dict[str, Any],
    input_feature_dict: dict[str, Any],
    s_inputs: torch.Tensor,
    s_trunk: torch.Tensor,
    z_trunk: torch.Tensor,
    pair_z: torch.Tensor,
    p_lm: torch.Tensor,
    c_l: torch.Tensor,
    N_sample: int = 1,
    diffusion_chunk_size: Optional[int] = None,
    use_conditioning: bool = True,
    enable_efficient_fusion: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Implements diffusion training as described in AF3 Appendix at page 23.
    It performances denoising steps from time 0 to time T.
    The time steps (=noise levels) are given by noise_schedule.

    Args:
        noise_sampler (TrainingNoiseSampler): sampler for training noise-level.
        denoise_net (Callable): the network that performs the denoising step.
        label_dict (dict[str, Any]) : a dictionary containing the followings.
            "coordinate": the ground-truth coordinates
                [..., N_atom, 3]
            "coordinate_mask": whether true coordinates exist.
                [..., N_atom]
        input_feature_dict (dict[str, Any]): input meta feature dict
        s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
            [..., N_tokens, c_s_inputs]
        s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
            [..., N_tokens, c_s]
        z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
            [..., N_tokens, N_tokens, c_z]
        pair_z (torch.Tensor): pair feature embedding from InputFeatureEmbedder
            [..., N_tokens, N_tokens, c_z_inputs]
        p_lm (torch.Tensor): MSA embedding
            [..., N_tokens, c_p_lm]
        c_l (torch.Tensor): ligand embedding
            [..., N_tokens, c_c_l]
        N_sample (int): number of training samples
        diffusion_chunk_size (Optional[int]): Chunk size for diffusion operation. Defaults to None.
        use_conditioning (bool): Whether to use conditioning. Defaults to True.
        enable_efficient_fusion (bool): Whether to enable efficient fusion. Defaults to False.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            x_gt_augment: the augmented ground-truth coordinates [..., N_sample, N_atom, 3]
            x_denoised: the denoised coordinates [..., N_sample, N_atom, 3]
            sigma: the sampled noise-level [..., N_sample]
    """
    batch_size_shape = label_dict["coordinate"].shape[:-2]
    device = label_dict["coordinate"].device
    dtype = label_dict["coordinate"].dtype
    # Areate N_sample versions of the input structure by randomly rotating and translating
    x_gt_augment = centre_random_augmentation(
        x_input_coords=label_dict["coordinate"],
        N_sample=N_sample,
        mask=label_dict["coordinate_mask"],
    ).to(
        dtype
    )  # [..., N_sample, N_atom, 3]

    # Add independent noise to each structure
    # sigma: independent noise-level [..., N_sample]
    sigma = noise_sampler(size=(*batch_size_shape, N_sample), device=device).to(dtype)
    # noise: [..., N_sample, N_atom, 3]
    noise = torch.randn_like(x_gt_augment, dtype=dtype) * sigma[..., None, None]

    # Get denoising outputs [..., N_sample, N_atom, 3]
    if diffusion_chunk_size is None:
        x_denoised = denoise_net(
            x_noisy=x_gt_augment + noise,
            t_hat_noise_level=sigma,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            pair_z=pair_z,
            p_lm=p_lm,
            c_l=c_l,
            use_conditioning=use_conditioning,
            enable_efficient_fusion=enable_efficient_fusion,
        )
    else:
        x_denoised = []
        no_chunks = N_sample // diffusion_chunk_size + (
            N_sample % diffusion_chunk_size != 0
        )
        for i in range(no_chunks):
            x_noisy_i = (x_gt_augment + noise)[
                ..., i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size, :, :
            ]
            t_hat_noise_level_i = sigma[
                ..., i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size
            ]
            x_denoised_i = denoise_net(
                x_noisy=x_noisy_i,
                t_hat_noise_level=t_hat_noise_level_i,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                pair_z=pair_z,
                p_lm=p_lm,
                c_l=c_l,
                use_conditioning=use_conditioning,
                enable_efficient_fusion=enable_efficient_fusion,
            )
            x_denoised.append(x_denoised_i)
        x_denoised = torch.cat(x_denoised, dim=-3)

    return x_gt_augment, x_denoised, sigma
