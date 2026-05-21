"""Coordinate diffusion — turn the trunk's (s, z) into atom coordinates.

坐标扩散模块 —— 把主干输出 (s, z) 变成原子坐标。

AF3 generates atom coordinates by iteratively denoising 3D points
conditioned on the trunk's (s, z).

AF3 通过对 3D 点反复去噪来生成原子坐标，去噪过程以主干输出的 (s, z) 作为
条件。

1. ``DiffusionModule`` — one denoising step F_theta(c_in · x_noisy, c_noise).
   单步去噪网络.
2. ``DiffusionTransformer`` — token-level transformer with AdaLN + pair bias.
   token 级 transformer，使用 AdaLN-Zero 与 pair bias.
3. ``sample_diffusion`` — outer Euler-step loop that turns Gaussian noise into
   the predicted structure (Algorithm 18).
   外层 Euler-step 采样循环.

Files / 文件:
    diffusion_module.py        DiffusionModule + DiffusionConditioning
    diffusion_transformer.py   DiffusionTransformer/Block + ConditionedTransitionBlock
    sampler.py                 InferenceNoiseScheduler + sample_diffusion  (Algorithm 18)
    frames.py                  Backbone-frame helpers
"""
