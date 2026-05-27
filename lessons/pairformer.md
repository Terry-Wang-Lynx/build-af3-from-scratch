# 第 2 章 · Pairformer（pair 表示主干）

> **章节定位**：AF3 的核心主干。它把蛋白序列 + MSA + 模板提炼成一个
> $N_\text{token} \times N_\text{token} \times c_z$ 的 **pair 表示** $z$，
> 让下游的扩散模块有「token (i, j) 之间应该有多近 / 角度如何」的几何先验。
>
> **配套 notebook**：`tutorials/pairformer/pairformer.ipynb`
>
> **本系列受 [The Illustrated AlphaFold](https://elanapearl.github.io/blog/2024/the-illustrated-alphafold/) 启发，内容为原创中文版。**

## 2.1 Pair 表示 z 是什么

AlphaFold 系列的核心抽象：**pair 表示** $z \in \mathbb{R}^{N_\text{token} \times N_\text{token} \times c_z}$，
其中 $z_{ij}$ 是一个 $c_z$ 维向量，编码 token i 和 token j 之间的关系。

它不直接是一个距离值，但训练目标里包含「让 $z_{ij}$ 能解码出真实距离」
（distogram loss，见第 5 章）。可以把它想成一份**抽象的接触图**：

- 既有距离信息（distogram 监督）
- 也有方向 / 角度 / 化学键 / 链关系等所有 pair 级信号
- 是扩散模块的主要条件输入

Pair 表示的好处是**显式表达 pair 关系**，让模型在结构空间「自然」推理。
缺点是显存随 $N_\text{token}^2$ 增长——AF3 base 用 c_z=128，2000 token
时 z 就要约 1 GB（fp16）。

## 2.2 「三角」更新的几何动机

AF2 / AF3 主干的特点是反复跑 **triangle update** 算子。直观地说：

> $z_{ij}$ 表示 i, j 之间的关系；如果模型见到第三个 token k，
> 那么 $z_{ij}$ 应当显式地依赖 $z_{ik}$ 和 $z_{kj}$。

这是把**三角不等式**注入模型的方式：

$$d(i, j) \le d(i, k) + d(k, j) \quad \forall k$$

如果 z 学到的距离不显式约束这个不等式，预测 $d_{12} = 100$Å 但
$d_{13} = d_{32} = 1$Å 这种**物理不可能**的组合就不会被惩罚。
AF2 ablation 显示移掉所有三角操作 GDT-TS 下降约 5 分。

AF3 主干一个 PairformerBlock 里有四个三角算子，加一个 transition：

| 算子 | 算法 | 角色 |
|---|---|---|
| TriangleMultiplicationOutgoing | Alg 11 | 沿 i 方向的外积式更新 |
| TriangleMultiplicationIncoming | Alg 12 | 沿 j 方向的外积式更新 |
| TriangleAttentionStartingNode | Alg 13 | 沿行的注意力，bias 来自列 |
| TriangleAttentionEndingNode | Alg 14 | 沿列的注意力，bias 来自行 |
| Transition (Alg 11) | | pair 上的 SwiGLU FFN |

加上从 MSA 注入信号的 `OuterProductMean` 和反向的 `MSAPairWeightedAveraging`，
本章一共要交付 6 个组件。

## 2.3 OuterProductMean (Algorithm 10)：MSA → pair

**共进化**（coevolution）是结构预测最古老的信号——蛋白演化中，有的位置
**必须协同突变**才能维持物理接触（如果残基 i 改成大体积，附近的 j 也
要协同改小）。这条线索从 90 年代的 Direct Coupling Analysis 开始，
一直是 AlphaFold 1 / 2 / 3 的核心信号源。

OuterProductMean 是把这种共进化统计抽象成可学张量代数的版本：

$$z_{ij} \;+\!\!= \;W_o \, \mathrm{flat}\Big(\frac{1}{N_\text{eff}(i,j)} \sum_s m_{si} m_{sj} \cdot a_{si} \otimes b_{sj}\Big)$$

其中:

- $a, b$ 是 MSA 张量经 LayerNorm + 两路线性投影到 `c_hidden` 后的产物
- $m_{si}$ 是 MSA mask（位置 i 在序列 s 中是否非 gap）
- $a_{si} \otimes b_{sj}$ 是 $c_\text{hidden} \times c_\text{hidden}$ 的外积矩阵
- 沿 MSA 维 s 加权平均，flatten 成 $c_\text{hidden}^2$ 维，最后 `linear_out` 投回 $c_z$

代码上用一行 einsum 就完成: `torch.einsum('bac, dae -> bdce', a, b)` —
通道维放最后让 PyTorch 自动选最优 contract。

`linear_out` 用 **`init="final"`** 即零初始化——训练初期 OPM 对 pair 没贡献，
让模型从「pair 只有 RelativePositionEncoding 提供的先验」这种安全状态出发。

## 2.4 TriangleMultiplication (Algorithms 11 & 12)：乘法式三角更新

最便宜的三角算子。不跑注意力，直接做带门控的外积：

- **Outgoing**: $z_{ij} \,+\!\!=\, \sum_k a_{ik} \odot b_{jk}$
- **Incoming**: $z_{ij} \,+\!\!=\, \sum_k a_{ki} \odot b_{kj}$

为什么用**乘法**而不是加法？乘法实现 AND 逻辑——
只有「存在某个 k 使得 (i, k) 和 (j, k) **同时**强」时 $z_{ij}$ 才被显著放大。
这正好是「i 通过 k 间接连接 j」的拓扑信号。加法（OR-like）会让任一边强就触发，
没法传递三角约束。

实现上有一个很漂亮的 **matmul trick**：数学上沿 k 求和的是
$\sum_k a_{ikc} b_{jkc}$。**把通道维 c 提到最前面**（`permute_final_dims`），
就能让 `torch.matmul` 沿 (i, k) @ (k, j) 一次性算完所有 c × N² 元素，
比手写循环或 einsum 更利用 BLAS：

```
a permute: [..., c, N, N]      # outgoing 的 a，通道在前
b permute: [..., c, N, N]      # outgoing 的 b 转置后
torch.matmul → [..., c, N, N]
permute 回 [..., N, N, c]
```

Outgoing 与 Incoming 共用一个类 `TriangleMultiplicativeUpdate`，
只是 `_outgoing` 标志影响 a/b 的 permute 路径。子类用 `partialmethod`
固定这个标志。

## 2.5 TriangleAttention (Algorithms 13 & 14)：注意力式三角更新

比乘法版更强——让模型**主动选择**通过哪个第三个 token k 来连接 (i, j)。

公式上，对 starting-node 变种（Alg 13），在每一行（固定 i）内部跑多头
注意力，**注意力 bias 来自另一行 z**：

$$z'_{ij} \;+\!\!=\; \sum_{j'} \mathrm{softmax}_{j'}\!\Big(\frac{Q_{ij} K_{ij'}^\top}{\sqrt{d}} + b_{jj'}\Big) V_{ij'}$$

注意 bias 是 $b_{jj'}$ 而**不是** $b_{ij'}$——它来自 token j 与 j' 之间的
pair 表示 $z_{jj'}$ 过 Linear 投到 `no_heads` 维。"第三个 token" 通过这个
bias 隐式参与。

「Around starting/ending node」命名来自 AF2 论文里把 pair $z_{ij}$ 画成
有向边 i → j：「starting」= 起点 i，「ending」= 终点 j。沿一行扫共起点的边，
沿一列扫共终点的边——绕第三个 token 的两种方式。

ending-node 实现上**复用 starting-node 类**：在 forward 里先把 z 物理转置
（`z.transpose(-2, -3)`），跑完再转回来。这样模型只学一份 attention 权重。

底层 attention 是 OpenFold 风格的 `Attention`（不是 AF3 的那个！），
接受 `biases=[mask_bias, triangle_bias]` 列表。

## 2.6 MSAPairWeightedAveraging：pair → MSA 反向通道

OuterProductMean 把 MSA 信息送入 pair，**信号只能单向流**。
MSAPairWeightedAveraging 让 pair 学到的关系反过来精修 MSA：

$$m'_{si} = \sum_h g_{si}^h \sum_j \mathrm{softmax}_j(b_{ij}^h) \cdot v_{sj}^h$$

- $b_{ij}^h$ 由 pair 张量经 LN + Linear 投出，每头一个标量
- softmax 沿 j 归一化，得到「位置 i 应该多看位置 j」的权重
- $v_{sj}^h$ 是 MSA 在位置 j 的 value（跨 MSA 序列 s 共享 b 权重）
- 沿 j 加权平均，再用 sigmoid 门 `g`（zero-init，起手关闭）收尾

这块在 MSAModule 内部使用，不直接出现在 PairformerBlock。

## 2.7 PairformerBlock (Algorithm 17)：把所有零件拼起来

一个 block 的伪代码：

```
z += TriangleMultiplicationOutgoing(z)        # Alg 11
z += TriangleMultiplicationIncoming(z)        # Alg 12
z += TriangleAttentionStartingNode(z)         # Alg 13
z += TriangleAttentionEndingNode(z)           # Alg 14 (内部转置)
z += pair_transition(z)                       # Alg 11 (FFN over z)

if c_s > 0:                                   # 单序列分支 (PairformerStack)
    s += AttentionPairBias(a=s, s=None, z=z)  # 第 1 章 Alg 24
    s += single_transition(s)
```

两个工程细节：

- 前两个三角乘走 `inplace_safe=True, _add_with_inplace=True` 路径，
  把 `z += op(z)` 做成融合操作（省一份 z 的 buffer）。
- 后两个 attention 用普通 `z = z + op(z)`。EndingNode 通过物理转置
  实现 Alg 14。

AF3 base 模型把 48 个 PairformerBlock 串起来；Tiny 配置只用 8 个。
单序列分支可以关掉（`c_s=0`）——例如 TemplateEmbedder 内部跑一个小型
PairformerStack 时只更新 pair。

## 2.8 与本仓库代码对应

```
pairformer/
├── triangle_ops.py
│   ├── OuterProductMean              ← __init__ + _opm + _forward (Alg 10)
│   ├── LayerNorm                     ← (re-export from attention/layer_norm.py)
│   └── (Attention, OpenfoldLinear)   ← OpenFold 风格 attention 的工具
├── triangle.py
│   ├── BaseTriangleMultiplicativeUpdate  ← __init__
│   ├── TriangleMultiplicativeUpdate      ← __init__ + _combine_projections + forward
│   ├── TriangleMultiplicationOutgoing    ← partialmethod 固定 _outgoing=True
│   ├── TriangleMultiplicationIncoming    ← partialmethod 固定 _outgoing=False
│   ├── TriangleAttention                 ← __init__ + forward (Alg 13/14)
│   └── TriangleAttentionEndingNode       ← partialmethod 固定 starting=False
├── msa_stack.py
│   ├── MSAPairWeightedAveraging      ← __init__ + forward
│   ├── MSAStack                      ← Pair-weighted avg + Transition
│   └── MSAModule / MSABlock          ← Alg 8 完整 MSA 模块
├── pair_stack.py
│   ├── PairformerBlock               ← __init__ + forward (Alg 17 一块)
│   └── PairformerStack               ← n_blocks 个 block 顺序应用
└── template_embedder.py
    └── TemplateEmbedder              ← Alg 16，内部跑小型 PairformerStack
```

## 2.9 设计取舍

**为什么 AF3 把 AF2 的 Evoformer 简化为 Pairformer？**
AF2 同时维护 MSA 表示和 pair 表示，两者反复交换信息。AF3 发现可以让
MSA 表示**只在 MSAModule 里短暂出现**，把信号汇入 pair 后就丢掉——
省了大量显存，且实验显示精度不降。

**为什么有四个三角算子，不能合一**？理论上一个广义版能覆盖，但实践显示
**乘法版 + 注意力版分工不同**：乘法快但表达受限（只能学外积形式的关系），
注意力慢但能学任意 query/key 相似度。每个 PairformerBlock 各跑两次（in / out
方向）让 z 在每一轮都收到来自所有四个角度的更新。

**为什么 MSA 维要 sub-sample**？典型 MSA 几万行，全跑显存爆。
MSAModule 在 forward 时随机抽 512 / 2048 / 16384 行（训练 / 推理 cutoff
在 `msa_configs` 配）。

## 2.10 延伸阅读

- AF3 主论文 Algorithm 8（MSAModule）, 10, 11, 12, 13, 14, 17
- AF3 Supplementary Section 3.6（Pairformer 完整伪代码）
- [The Illustrated AlphaFold §2 Representation Learning](https://elanapearl.github.io/blog/2024/the-illustrated-alphafold/)
