# 第 4 章 · Diffusion（坐标扩散）

> **章节定位**：AF3 与 AF2 在结构生成端**最大的不同** —— 抛弃 AF2 的等变
> Structure Module + IPA，换成 EDM 风格的扩散模型，在原子坐标空间逐步去噪。
>
> **配套 notebook**：`tutorials/diffusion/diffusion.ipynb`
>
> **本系列受 [The Illustrated AlphaFold](https://elanapearl.github.io/blog/2024/the-illustrated-alphafold/) 启发，内容为原创中文版。**

## 4.1 AF2 → AF3 的范式切换

AlphaFold 2 的结构头是一个**确定性的**等变 Transformer：Invariant Point
Attention (IPA) + Structure Module。给定 trunk 输出，它一次性回归出 backbone
frames，再用角度网络生成侧链。整套依赖**精心设计的等变层**保证刚体不变性。

AlphaFold 3 把它**全部丢掉**，换成扩散：

1. **训练**: 给真实坐标加 σ 量级高斯噪声 $x = y + \sigma \epsilon$，让网络
   $F_\theta$ 学会从带噪 x 预测 y（实际是预测带 EDM 预条件的 y）。
2. **推理**: 从纯噪声 $x \sim \mathcal{N}(0, \sigma_\text{max}^2 I)$ 出发，
   按 noise schedule 逐步降低 σ，每步调一次 $F_\theta$、用 Euler 步把 x
   推向去噪后的位置。

这一改给 AF3 三个 AF2 难做到的能力：

- **多构象采样**: 不同初始噪声 → 不同样本，自然支持「一个序列的多种 3D 解」
- **配体 / 核酸通用**: 扩散在坐标空间通用，不需要为每种分子设计专用 frame
  几何
- **训练更稳**: 去噪目标比一次性回归坐标光滑得多

代价是推理慢了 N_step 倍（每步都要跑一遍完整 trunk-下游网络）；
AF3 base 模型用 ~200 步，Tiny 模型用 5 步。

## 4.2 EDM 数学骨架

AF3 的扩散用 [Karras et al. 2022 (EDM)](https://arxiv.org/abs/2206.00364)
框架。核心是**预条件**（pre-conditioning）：把网络 $F_\theta$ 包成

$$D_\theta(x; \sigma) = c_\text{skip}(\sigma)\,x + c_\text{out}(\sigma) \, F_\theta\big(c_\text{in}(\sigma)\,x; \,c_\text{noise}(\sigma)\big)$$

其中三个系数：

$$c_\text{in}(\sigma) = \frac{1}{\sqrt{\sigma_\text{data}^2 + \sigma^2}}, \quad c_\text{out}(\sigma) = \frac{\sigma \cdot \sigma_\text{data}}{\sqrt{\sigma_\text{data}^2 + \sigma^2}}, \quad c_\text{skip}(\sigma) = \frac{\sigma_\text{data}^2}{\sigma_\text{data}^2 + \sigma^2}$$

### 这些系数从哪里来

假设真实数据分布方差 $\sigma_\text{data}^2$（AF3 用 16²=256 Å²），
加噪后 $\mathrm{Var}(x) = \sigma_\text{data}^2 + \sigma^2$。

- $c_\text{in}$ 把 x 缩到单位方差，让网络看到归一化输入
- $c_\text{out}$ 和 $c_\text{skip}$ 是 Karras 推导出来的最优组合，
  让训练损失在 σ 范围内**信号方差与噪声方差平衡**——
  详细推导见 EDM 论文 eq. (5)-(7)

直观行为：

- $\sigma \to 0$ 时 $c_\text{skip} \to 1, c_\text{out} \to 0$：
  模型几乎直接返回输入 x（接近真实，不需要大改）
- $\sigma \to \infty$ 时 $c_\text{skip} \to 0, c_\text{out} \to \sigma_\text{data}$：
  完全靠网络重建 y

### Noise schedule

推理时从一个**逆向**的 σ 序列采样：$\sigma_T \gg \cdots \gg \sigma_0 = 0$。
AF3 的 `InferenceNoiseScheduler` 用 sigmoid-based schedule，比 DDIM 的几何
序列在中间区域更密集——直观上是「最难去的噪声集中在中段，多花预算」。

仓库里默认 Tiny 模型 N_step=5（推理 ~10s on CPU），AF3 base 用 200 步。
更多步 = 更精细的去噪 = 更高 pLDDT，但也线性贵。

## 4.3 三层扩散网络

AF3 的 $F_\theta$ 是**三层结构**:

1. **`DiffusionConditioning` (Alg 21)**: 把 trunk 给的 single / pair 表示，
   加上当前噪声水平的 FourierEmbedding，生成 sample-aware 的 (s, z) 条件
2. **`DiffusionModule.f_forward` (Alg 20 的中间块)**: 核心网络
   - AtomAttentionEncoder（带坐标 has_coords=True）→ 原子 → token 聚合
   - DiffusionTransformer（token 级，Alg 23）
   - AtomAttentionDecoder → token → 原子，输出 r_update
3. **`DiffusionModule.forward` (Alg 20 完整 EDM 包装)**: 套上
   $c_\text{in}, c_\text{out}, c_\text{skip}$ 得到 x_denoised

最外层是 `sample_diffusion` (Alg 18)，实现完整的 Euler 步预测-校正采样循环。

### DiffusionTransformerBlock (Algorithm 23)

DiffusionTransformer 的最小单元，每个 block 由两个分支组成：

```
a_in ──── AttentionPairBias(a, s, z) ────► attn_out (+= a_in)
                                              │
                                              ▼
                                   ConditionedTransitionBlock(a, s)
                                              │
                                              ▼
                                            a_out (+= attn_out)
```

`AttentionPairBias`（第 1 章 Alg 24）跑 multi-head attention 并用 pair
张量 z 做 bias、AdaLN-Zero 调制 s；`ConditionedTransitionBlock`（Alg 25）
是 SwiGLU FFN + adaLN-Zero 输出门。两个分支都套 DropPath（stochastic depth），
推理时是 identity。

`s, z` 沿 block 间**原样透传**——为方便后续接 activation checkpoint。

### ConditionedTransitionBlock (Algorithm 25)

形式上是带 adaLN-Zero 输出门的 SwiGLU FFN：

```
a = AdaLN(a, s)
b = SiLU(linear_a1(a)) * linear_a2(a)
a = sigmoid(linear_s(s)) * linear_b(b)   ← adaLN-Zero output gate
```

`linear_s` 是 `BiasInitLinear`（biasinit=-2），起手输出 $\sigma(-2) \approx 0.12$，
深层 block 起手对残差 ≈ 0 贡献，训练才稳。

## 4.4 几何 helper

AF2 用一整章（Structure Module）处理刚体 / 帧 / quaternion；AF3 把这些
**散进扩散里**当 helper 函数用。两个最常用的：

### `expressCoordinatesInFrame` (Algorithm 29)

把每个原子坐标投影到 frame 的局部正交基。frame 由 3 个原子 (a, b, c) 定义，
b 是原点。**Confidence head 计算 PAE 用它**。

构造正交基的技巧不是 Gram-Schmidt，而是更**数值稳定**的版本：

$$\mathbf{w}_1 = \widehat{a - b},\quad \mathbf{w}_2 = \widehat{c - b}$$
$$\mathbf{e}_1 = \widehat{\mathbf{w}_1 + \mathbf{w}_2},\quad \mathbf{e}_2 = \widehat{\mathbf{w}_2 - \mathbf{w}_1},\quad \mathbf{e}_3 = \mathbf{e}_1 \times \mathbf{e}_2$$

为什么不用 Gram-Schmidt？当 $\mathbf{w}_1, \mathbf{w}_2$ 几乎共线（蛋白主链
相邻三原子经常如此），经典 Gram-Schmidt 投影 $\mathbf{w}_2 - (\mathbf{w}_2 \cdot \mathbf{e}_1)\mathbf{e}_1$
会出现 **catastrophic cancellation** —— 两个相近大数相减放大噪声，
normalize 后方向跳到完全不相关的位置。

sum/diff 形式等价于一个 45° 旋转 + scale，对共线退化是**对称退化**
（$\mathbf{e}_1 \to \widehat{\mathbf{w}_1}$，$\mathbf{e}_2 \to 0$），eps 保护下不会爆炸，
梯度连续。

### `centre_random_augmentation` (Algorithm 19)

扩散采样**每一步开头**都做的事：

1. 减去（masked）质心 —— 抵消坐标整体平移
2. 给每个 sample 抽一个**随机 SE(3) 变换**（3D 旋转 + 平移），应用
3. 走 mask 后处理

为什么？AF3 网络对**全局刚体变换不是天然等变**的（输入是绝对坐标）。
如果不每步增广，模型学到的就是某个固定参考系下的去噪，泛化差。
随机增广强迫网络在新参考系下也能去噪 → 等价于训练目标对刚体变换不敏感。

## 4.5 sample_diffusion (Algorithm 18)：完整采样循环

```python
x_l = sigma_max * noise                              # 起手纯高斯
for c_tau_last, c_tau in zip(schedule[:-1], schedule[1:]):
    x_l = centre_random_augmentation(x_l)            # 每步刚体增广
    # Predictor: 加噪到 t_hat (γ-scaled，仅 c_tau > gamma_min 时)
    gamma = gamma0 if c_tau > gamma_min else 0
    t_hat = c_tau_last * (gamma + 1)
    x_noisy = x_l + lambda * sqrt(t_hat² - c_tau_last²) * noise
    # Corrector: 一次 EDM 去噪 + Euler 步
    x_denoised = DiffusionModule(x_noisy, t_hat, ..., conditions...)
    delta = (x_noisy - x_denoised) / t_hat
    x_l = x_noisy + eta * (c_tau - t_hat) * delta
```

这是经典的 **predictor-corrector 采样器**：predictor 用 γ 系数往噪声方向
轻微回退（为了多样性），corrector 用 EDM 去噪 + Euler 步往真实方向推进。
γ_min / γ_0 / eta / lambda 都是 sampler 超参，sigmoid schedule 之外的所有
调节都集中在这里。

## 4.6 与本仓库代码对应

```
diffusion/
├── diffusion_transformer.py
│   ├── ConditionedTransitionBlock    ← __init__ + forward (Alg 25)
│   ├── DiffusionTransformerBlock     ← __init__ + forward (Alg 23 一块)
│   └── DiffusionTransformer          ← n_blocks 个 block stack
├── diffusion_module.py
│   ├── DiffusionConditioning         ← __init__ + prepare_cache + forward (Alg 21)
│   └── DiffusionModule               ← __init__ + f_forward + forward (Alg 20)
├── sampler.py
│   ├── InferenceNoiseScheduler       ← sigmoid-based noise schedule
│   ├── TrainingNoiseSampler          ← 训练时按 σ 分布抽样
│   └── sample_diffusion              ← Alg 18 完整 predictor-corrector 循环
└── frames.py
    ├── expressCoordinatesInFrame     ← Alg 29 局部正交基投影
    └── gather_frame_atom_by_indices  ← 索引工具

model/utils.py 里还有:
    centre_random_augmentation        ← Alg 19 recentre + SE(3) 增广
```

## 4.7 设计取舍

**为什么 AF3 用 EDM 而不是 DDPM / DDIM？**
EDM 把不同扩散框架（VE / VP / DDIM）统一成同一个参数化，
$c_\text{in}, c_\text{out}, c_\text{skip}$ 这套预条件让训练目标对各种 σ
都数值平衡。Karras 论文实验显示同算力下 EDM 在 CIFAR-10 / ImageNet 都
比 DDPM 好一档。AF3 直接拿来用。

**为什么是 token 级 transformer 而不是 atom 级？**
原子总数 ~5×token 数；atom-level transformer 在 1000 token 系统上是
$5000^2 / 1000^2 = 25$ 倍显存。AF3 折中：原子级用局部窗口注意力
（AtomAttentionEncoder/Decoder），token 级用全连接 DiffusionTransformer ——
每层的 atom-attention 计算量 ~常数，DiffusionTransformer 计算量 $O(N_\text{token}^2)$。

**为什么每步重新 recentre + 随机刚体增广？**
让网络看到的输入分布在 SE(3) 群下尽量均匀。如果只在初始一次增广，
扩散后期 σ 小时坐标已经向某个特定姿态收敛，模型会过拟合那个姿态的偏置。

## 4.8 延伸阅读

- AF3 主论文 Algorithm 18, 19, 20, 21, 23, 25, 29
- [Elucidating the Design Space of Diffusion-Based Generative Models (Karras et al. 2022, EDM)](https://arxiv.org/abs/2206.00364) —— 必读
- [DiT (Peebles & Xie 2023)](https://arxiv.org/abs/2212.09748) —— AdaLN-Zero 的来源
- [The Illustrated AlphaFold §3 Structure Prediction](https://elanapearl.github.io/blog/2024/the-illustrated-alphafold/)
