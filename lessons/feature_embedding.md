# 第 3 章 · Feature Embedding（特征嵌入）

> **章节定位**：把第 0 章 feature_extraction 输出的稀疏 / 异质特征，
> 翻译成 trunk 可直接消费的稠密张量。两块小但关键的基础设施：
> `RelativePositionEncoding`（pair 通道的初始先验）和
> `FourierEmbedding`（扩散里把标量噪声水平变向量）。
>
> **配套 notebook**：`tutorials/feature_embedding/feature_embedding.ipynb`
>
> **本系列受 [The Illustrated AlphaFold](https://elanapearl.github.io/blog/2024/the-illustrated-alphafold/) 启发，内容为原创中文版。**

## 3.1 为什么有这一章

第 0 章的 featurizer 输出张量字典，但有的字段需要再加工才能变成 trunk 能吃的形式：

- 每对 token 的**相对位置**: 不能直接喂离散的「相差 5 个残基」，要 one-hot
  后再投到 pair 通道
- **扩散噪声水平**: 一个标量 σ，要展成 $c_\text{noise}$ 维向量才能加到
  AdaLN 条件流上
- 序列 + MSA 派生的**单序列特征**: 需要拼接 + 线性投影
- 原子级特征 → token 级聚合: 由 `AtomAttentionEncoder` 处理

后两块（`InputFeatureEmbedder` / `AtomAttentionEncoder`）的工程量大、
依赖太多上下文，单元测试难写——我们在端到端 `overview.ipynb` 整体验证。
本章主要单测**两块通用的小零件**。

## 3.2 RelativePositionEncoding (Algorithm 3)：pair 通道的初始先验

Pairformer 第一次见到 pair 张量 z 之前，z 是怎么初始化的？

- 单序列特征 $s_\text{init}$ 经两路线性外加和得到一个 $[N, N, c_z]$ 的 base
- 加上 RelativePositionEncoding 的输出（编码相对位置）
- 加上 token_bonds 的投影（共价键邻接）

**RelativePositionEncoding 的角色是「相对位置先验」**。绝对位置在 AF3 里
没意义（链是无序的、多链复合物里 chain A 和 chain B 可以任意调换），
所以编码相对关系才符合等变性。

### 三个 token 偏移特征

对每对 token (i, j) 算三类整数偏移，每类 clip 到固定范围，再用一个特殊
编码 (`2*r_max+1` 或 `2*s_max+1`) 表示「超出范围或不同 chain」:

1. **residue 偏移** (gated by 同链):
   ```
   clip(residue_index[i] - residue_index[j] + r_max, 0, 2*r_max)
   ```
   只有 i, j 在同一条链时给出真实偏移，否则给出特殊编码 `2*r_max+1`。

2. **token 偏移** (gated by 同链 ∧ 同残基):
   只在「同一个残基里的不同 token」（修饰残基 / 配体 / 多原子 token）
   有非平凡值。同链不同残基的 pair 给出特殊编码。

3. **chain 偏移** (gated by 同 entity, `s_max`-clip):
   寡聚体（同一 entity 多个 chain）里，sym_id 给出第几个对称拷贝。

三类各自 one-hot，加上 1 维 `same_entity` 布尔，**总宽度 4·r_max + 2·s_max + 7**。
AF3 用 r_max=32, s_max=2 → 总宽度 139。

### forward 是一个线性层

`generate_relp` 计算 one-hot 张量并塞回 `input_feature_dict`；
`forward` 就是把 one-hot 经一个 `LinearNoBias(139 → c_z)` 投到 pair 通道。

### 为什么把 relp 算一次缓存起来

trunk 的 N_cycle recycling 每轮都需要 relp，但 relp 是**和 cycle 无关**的常量。
所以 `generate_relp` 在 `Protenix.forward` 一开始算一次，结果塞进
`input_feature_dict["relp"]`，后续 cycle 直接读，避免每轮重算。

整段写在 `torch.no_grad()` 里——relp 是离散索引产物，无梯度需求。

## 3.3 FourierEmbedding (Algorithm 22)：标量噪声水平的向量化

扩散模型在每一步去噪都需要让网络知道**当前噪声水平 σ**。但 σ 是一个标量，
怎么喂给一个吃 `(B, N, c)` 张量的 Transformer？

标准做法：**random Fourier embedding**。把 σ 映成一个 c 维向量：

$$\mathrm{FourierEmbed}(\tau)_k = \cos\big(2\pi \,(\tau \cdot w_k + b_k)\big), \quad k = 1, \ldots, c$$

其中 $\tau$ 是 σ 经过取对数 + 归一化的版本（见下面），$w_k, b_k$ 是
**固定**的随机数（构造时一次性抽样，作为不可训练的 `nn.Parameter` 存进
state_dict）。

### 数学背景：random feature map

Rahimi & Recht 2007 的论文证明：**平移不变核** $k(t, t') = k(t - t')$ 可通过
随机三角函数特征近似:

$$k(t, t') \approx \phi(t)^\top \phi(t'), \quad \phi(t) = \sqrt{2/c}\,[\cos(w_1 t + b_1), \ldots, \cos(w_c t + b_c)]$$

c 越大近似越准。AF3 借用这个想法（不再严格要求近似某个核），用它**给 σ 生成
稠密 fingerprint**——不同 σ 在 c 维 cos 空间上得到独特位置，让网络容易区分。

### 为什么用 log(σ/σ_data) 而不是 σ 本身

扩散里 σ 跨越多个数量级（典型 $\sigma_\text{min} \approx 0.002$, $\sigma_\text{max} \approx 80$，
AF3 范围更广）。如果直接用 σ:

- σ ≈ 0 时几乎所有 $w_k \sigma + b_k \approx b_k$，cos 值全相同——
  **不同 σ 撞码**
- σ 很大时 cos 振荡极快，相邻 σ 的 embedding 完全无关——**学不到平滑结构**

**取 log 后**: σ 跨 4 个数量级时 log 只跨 9 倍，cos 在合理频段振荡——
embedding 既能区分远端 σ、又对相邻 σ 平滑。AF3 具体用
$\tau = \log(\sigma / \sigma_\text{data}) / 4$，做了一个额外的 1/4 缩放
让频段更窄。

### 它在哪里被用到

唯一调用方在第 4 章：`DiffusionConditioning.forward` 里把 t_hat 经
FourierEmbedding 得到 c_noise 维向量，LN + Linear 后加到单序列条件
`single_s` 上。这是把噪声水平注入 AdaLN-Zero 的通道。

## 3.4 InputFeatureEmbedder (Algorithm 2)（不挖空，但要懂）

它把第 0 章给的特征字典聚成主干能吃的**单序列张量** `s_inputs`：

1. 跑 `AtomAttentionEncoder`（有 coords=False 模式），从原子级特征聚合出
   token 级激活 a
2. 把 a 与三个 token 级特征（restype 32 维 + profile 32 维 + deletion_mean 1 维）
   沿通道拼接，得到 `[N_token, c_token + 65]`
3. 若启用了 ESM（Mini-ESM 变体），把 ESM token embedding 经
   `linear_esm`（zero-init）加到 s_inputs 上

输出 `s_inputs` 之后会进入 trunk 的 recycling 起点。

## 3.5 AtomAttentionEncoder (Algorithm 5)（不挖空，但要懂）

把每原子特征（参考几何、电荷、原子名）聚合成 per-token 激活的核心组件。
内部跑一个**局部窗口 atom transformer**：每个原子只与窗口内的邻居做
attention，避免 $O(N_\text{atom}^2)$ 显存。

两种模式：

- `has_coords=False`：仅用参考几何（CCD lookup 出的标准构象），给
  InputFeatureEmbedder 用
- `has_coords=True`：还把当前噪声坐标 $r_l$ 折进去，给扩散模块用

详细 forward 由 5 步组成（见 TODO 伪代码），核心是：

1. 由参考坐标 + 电荷 + mask + element + atom_name_chars 算 `c_l`（原子 single）
2. 由 d_lm / v_lm / pad_info 算 dense-trunk `p_lm`（原子 pair）
3. 在 trunk 流上加上 single 投影 + 噪声坐标 r_l → q_l（query）
4. 跑 AtomTransformer 做局部窗口注意力
5. mean-pool 原子 → token，输出 `a`

## 3.6 与本仓库代码对应

```
feature_embedding/
├── relative_position_encoding.py
│   ├── RelativePositionEncoding      ← generate_relp + forward (Alg 3)
│   └── FourierEmbedding              ← __init__ + forward (Alg 22)
├── input_embedder.py
│   └── InputFeatureEmbedder          ← __init__ + forward (Alg 2)
├── atom_attention.py
│   ├── AtomTransformer               ← forward = local DiffusionTransformer
│   ├── AtomAttentionEncoder          ← __init__ + forward (Alg 5)
│   └── AtomAttentionDecoder          ← __init__ + forward (Alg 6)
├── local_attention.py                ← 局部窗口注意力的工具函数
└── constraint_embedder.py            ← 用户约束嵌入 (contact / pocket / bond)
```

## 3.7 设计取舍

**为什么 RelativePositionEncoding 把 relp 缓存在 input_feature_dict？**
N_cycle = 4 时每轮主干都要读 relp，缓存避免重算。但代价是 feature dict
要传一份大约 `[N, N, 139]` 的张量；对 1000 token 蛋白 ~550 MB（fp32）。
所以训练 + 推理时都要保证主干输入是按引用传递，不深拷。

**为什么 FourierEmbedding 的 w 和 b 不可训练？**
固定随机权重已经能区分不同 σ；如果可训练，模型可能收敛到「只看几个特定
噪声水平」，反而损失泛化。Karras 2022 的实验也是固定 random Fourier。

**为什么 ESM 用 zero-init 线性层加进 s_inputs？**
Mini-ESM 是「兼容版」—— 同一份代码既要能加载 PLM 变体的权重，也要能加载
非 PLM 变体（如 Tiny）。后者 `linear_esm` 也存在但权重是 0，所以即使加到
s_inputs 上也无效。两份代码完全统一，靠权重决定行为。

## 3.8 延伸阅读

- AF3 主论文 Algorithm 2, 3, 5, 6, 22
- [Random Features for Large-Scale Kernel Machines (Rahimi & Recht 2007)](https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf) —— Fourier embedding 的理论根
- [DDIM (Song et al. 2020)](https://arxiv.org/abs/2010.02502) —— σ 取 log 在扩散里的标准做法
- [The Illustrated AlphaFold §1 Input Preparation](https://elanapearl.github.io/blog/2024/the-illustrated-alphafold/)
