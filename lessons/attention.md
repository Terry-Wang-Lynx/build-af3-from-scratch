# 第 1 章 · Attention（注意力基础设施）

> **章节定位**：从最底层的线性层到 AF3 标志性的 AttentionPairBias，一路搭起来。
> 完成本章之后，Pairformer / Diffusion / Confidence 三章就只剩「按图接线」。
>
> **配套 notebook**：`tutorials/attention/attention.ipynb`
>
> **本系列受 [The Illustrated AlphaFold](https://elanapearl.github.io/blog/2024/the-illustrated-alphafold/) 启发，内容为原创中文版。**

## 1.1 为什么 AF3 整个模型都在跑同一种 attention

AF3 的主干（Pairformer）、扩散网络（DiffusionTransformer）、置信头里的
小型 Pairformer，**所有 attention 都基于同一段缩放点积数学**：

$$\mathrm{Attention}(Q, K, V, b) = \mathrm{softmax}\Big(\frac{Q K^\top}{\sqrt{d}} + b\Big) V$$

差别只是怎么算 $Q, K, V, b$、按哪些轴展开、要不要加 sigmoid 门控。
本章把这套基础设施搭好，下面三章就是反复组合它们。

具体来说，本章交付 7 个组件：

| 组件 | 文件 | AF3 Algorithm |
|---|---|---|
| `Linear` / `LinearNoBias` / `BiasInitLinear` | `attention/linear.py` | — (实现细节) |
| `OpenFoldLayerNorm` | `attention/layer_norm.py` | — |
| `_attention` + `Attention` | `attention/mha.py` | — |
| `AdaptiveLayerNorm` | `attention/transition.py` | Alg 26 |
| `Transition` (SwiGLU FFN) | `attention/transition.py` | Alg 11 |
| `AttentionPairBias` | `attention/attention_pair_bias.py` | Alg 24 |

## 1.2 自定义 Linear：为什么 PyTorch 自带的不够用

`torch.nn.Linear` 全部用默认 Kaiming 初始化。AF3 在不同位置用**不同初始化策略**：

- **ReLU/SiLU 前的层**（如 Transition 内部的扩宽分支）：fan-in 截断正态，
  $\mathrm{Var}(W) = 2/n_\text{in}$。这是 He 初始化的形式——
  补偿 SiLU 砍掉一半激活方差。
- **残差块的输出投影**：**zero-init**。让每个 block 起手对残差是零贡献，
  深层堆叠（48 + 24 + 4 = 76 层）才不会方差爆炸。
- **门控分支**（sigmoid 之前）：gating 初始化（OpenFold 风格的截断正态），
  让 sigmoid 起手在 0.5 附近。
- **其它**：默认 fan-in 截断正态 $\mathrm{Var}(W) = 1/n_\text{in}$，
  保前向方差。

我们通过 `Linear(initializer="default" | "relu" | "zeros")` 字符串选策略。

### `BiasInitLinear`：adaLN-Zero 门的专用 Linear

`AttentionPairBias` 等位置需要 sigmoid 输出门**起手几乎关闭**——
让每个 block 残差贡献 ≈ 0，深层稳定。具体做法：weight 全 0，
bias 设成一个负常数（AF3 用 `biasinit = -2`）。

为什么是 -2？$\sigma(-2) = 1/(1+e^2) \approx 0.119$。L=48 层后总放大
$\sim 48 \cdot 0.119^2 \cdot \mathrm{Var}(\text{Block}) \approx 0.68$，
稳定。如果 bias 取 0（默认 sigmoid 半开），同样深度下方差会指数级爆炸。

### 精度路径

`Linear.forward` 还有一个**高精度计算路径**：当 `precision` 不为 None
（典型 `torch.float32`），即使主干跑在 bf16，这一层也会临时升 fp32。
用于扩散里的坐标投影 / 噪声水平条件 —— bf16 mantissa 只有 7 bit，
小数值容易归零。

## 1.3 LayerNorm：可选 scale / offset

PyTorch 的 `nn.LayerNorm` 标配 $\gamma$（scale）和 $\beta$（offset）。
AF3 有两个地方需要关闭其中之一：

- **AdaLN 内部对 `a` 做 LN**：scale / offset 都关（由 `s` 经线性层生成，
  见 1.5 节）。如果再叠 $\gamma$/$\beta$，参数化冗余、训练歧义。
- **DiffusionConditioning 里对 z 做 LN**：只关 offset，保留 scale。

所以仓库自己写一个 `OpenFoldLayerNorm`，构造时通过 `create_scale` /
`create_offset` 决定要不要建参数。**注意**: 关掉的参数仍要用
`register_parameter(name, None)` 注册——这样 state_dict 里的 key 才齐全，
Protenix 权重才能严格加载。

## 1.4 Scaled dot-product attention 的数学

整个 AF3 attention 体系最底层是无参数函数 `_attention(q, k, v, attn_bias)`。
公式上就是

$$\mathrm{out} = \mathrm{softmax}\Big(\frac{Q K^\top}{\sqrt{d}} + b\Big) \, V$$

但有两个**容易踩坑的细节**：

### 为什么 $1/\sqrt{d}$ 而不是 $1/d$

设 $Q, K$ 各分量独立同方差 1。点积 $Q_i \cdot K_j$ 是 $d$ 项之和，方差是 $d$。
$d$ 较大（典型 64）时，softmax 输入有相当部分 $|z| > 8$——softmax 输出近似
one-hot，梯度几乎为零（**饱和的 softmax**）。

$1/\sqrt{d}$ 把方差缩回 1，softmax 工作在线性区，梯度健康。
这是 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) §3.2.1
的原始观察。

### 为什么提前缩放 Q

数学上 $(\tilde Q) K^\top = QK^\top / \sqrt{d}$ 等价。但我们在
`_prep_qkv` 里就把 Q 除以 $\sqrt{d}$，然后传 `scale=1.0` 给底层 attention——
这样 CUDA 上 `F.scaled_dot_product_attention` 调用接口更简单。

## 1.5 AdaptiveLayerNorm (Algorithm 26)：FiLM 的 sigmoid 变体

AF3 的 DiffusionTransformer 每个 block 都要把**当前噪声水平 + 单序列条件**
注入到主流。`AdaptiveLayerNorm` 就是注入通道：

$$\mathrm{AdaLN}(a, s) = \sigma(W_g \, \mathrm{LN}_s(s)) \cdot \mathrm{LN}_a(a) + W_b \, \mathrm{LN}_s(s)$$

这是 [FiLM](https://arxiv.org/abs/1709.07871)（Feature-wise Linear Modulation）
的演化版本——一路 sigmoid 门当 scale，一路线性当 shift。

**关键设计**：

- $W_g, W_b$ 都 **zero-init**。起手 $\sigma(0) = 0.5$，shift 为 0；
  整个 AdaLN 起手相当于 $\mathrm{LN}(a) \cdot 0.5$，是一个稳定的恒等近似。
- $\mathrm{LN}_a$ 关 scale + offset，调制完全由 s 控制。
- $\mathrm{LN}_s$ 保留 scale，让 s 进入门控前先稳定归一。

这是 [DiT](https://arxiv.org/abs/2212.09748)（Peebles & Xie 2023）的
adaLN-Zero 设计，被 AF3 / Stable Diffusion 3 / Sora 等大量借鉴。

## 1.6 Transition (Algorithm 11)：SwiGLU FFN

Vanilla Transformer 的 FFN 是 $W_2 \,\text{GELU}(W_1 x)$。AF3 / Llama /
Gemini 普遍换成 SwiGLU（[Shazeer 2020](https://arxiv.org/abs/2002.05202)）：

$$\mathrm{Transition}(x) = W_o \, \big(\mathrm{SiLU}(W_a \, \mathrm{LN}(x)) \odot W_b \, \mathrm{LN}(x)\big)$$

中间多一路 value 分支 $W_b x$，与 SiLU 门做 element-wise 乘。同算力下
比 GELU MLP 在 LM 任务低约 0.4 PPL。代价是参数量从 $2 c \cdot nc$ 涨到
$3 c \cdot nc$ —— AF3 把 expansion 系数 $n$ 调到 2 或 4 来抵消。

- 两路扩宽线性 (`linear_no_bias_a/b`) 用 `"relu"` 初始化
- 输出投影 `linear_no_bias` 用 `"zeros"` 初始化（起手恒等）

## 1.7 AttentionPairBias (Algorithm 24)：本章的顶点

这是 AF3 最高频出现的**复合块**，地位等同于 Vanilla Transformer 的
`SelfAttention + FFN`。出现在：

- PairformerBlock 的单序列分支
- DiffusionTransformerBlock 的 attention 分支
- AtomTransformer 的局部窗口分支
- ConfidenceHead 内部小 Pairformer 的 attention 分支

### Forward 流程

```
a, s, z  →  AdaLN(s) or plain LN  →  q = kv  →  attention(q, kv, bias=Linear(LN(z)))
                                                              │
                                                              ↓
                                                       × sigmoid(BiasInitLinear(s))
                                                              ↓
                                                       a_update  ← 调用方做残差加
```

三个关键点：

1. **bias 来自 pair 张量 z**: `Linear(LN(z))` 投到 `n_heads` 维，
   作为每头一个标量加到 QK^T 上。让 token i 关注 token j 时，
   pair 表示 $z_{ij}$ 决定多看多少。
2. **AdaLN 仅在 has_s=True 时启用**（DiffusionTransformer 路径）。
   Pairformer 路径用普通 LN，因为那里没有噪声水平要注入。
3. **adaLN-Zero 输出门**: `linear_a_last` 是 `BiasInitLinear`(biasinit=-2)，
   起手输出 $\sigma(-2) \approx 0.12$ —— block 残差初始 ≈ 0，深层稳。

### 局部 vs 全连接两条路径

- `standard_multihead_attention(q, kv, z)`: `z` 是正方形 `[N, N, c_z]`，
  PairformerBlock 用。
- `local_multihead_attention(q, kv, z, n_queries, n_keys)`: `z` 已经
  rearrange 成 `[n_blocks, n_q, n_k, c_z]` 的 dense-trunk 形式，
  AtomTransformer 用以避免 $O(N^2)$ 显存。

## 1.8 与本仓库代码对应

```
attention/
├── linear.py
│   ├── Linear                        ← _init_params + forward
│   ├── LinearNoBias = partial(Linear, bias=False)
│   └── BiasInitLinear                ← __init__
├── layer_norm.py
│   └── OpenFoldLayerNorm             ← __init__ + forward
├── mha.py
│   ├── _attention (函数)              ← 缩放点积数学
│   └── Attention                     ← __init__ / _prep_qkv / _wrap_up / forward
├── transition.py
│   ├── AdaptiveLayerNorm             ← __init__ + forward (Alg 26)
│   └── Transition                    ← __init__ + forward (Alg 11)
└── attention_pair_bias.py
    └── AttentionPairBias             ← __init__ / local_* / standard_* / forward (Alg 24)
```

## 1.9 延伸阅读

- AF3 主论文 Algorithm 11, 24, 26
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) §3.2 —— $1/\sqrt{d}$ 的来源
- [DiT (Peebles & Xie 2023)](https://arxiv.org/abs/2212.09748) —— adaLN-Zero 设计
- [SwiGLU (Shazeer 2020)](https://arxiv.org/abs/2002.05202) —— Transition 的 FFN 选择
- [FiLM (Perez et al. 2017)](https://arxiv.org/abs/1709.07871) —— AdaLN 的祖先
