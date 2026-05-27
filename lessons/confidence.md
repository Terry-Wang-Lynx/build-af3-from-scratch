# 第 5 章 · Confidence（置信度系统）

> **章节定位**：AF3 不只输出坐标，更**附带每个原子 / 每对 token 的置信度**。
> 这是它能在湿实验里被采纳的关键产品差异：用户能挑出预测里高置信的部分
> 直接用，低置信的部分先做实验验证。
>
> **配套 notebook**：`tutorials/confidence/confidence.ipynb`
>
> **本系列受 [The Illustrated AlphaFold](https://elanapearl.github.io/blog/2024/the-illustrated-alphafold/) 启发，内容为原创中文版。**

## 5.1 AlphaFold 系列的产品哲学

「AI 预测的结构能用吗」—— 是结构生物学家面对深度学习产物的第一个问题。
AlphaFold 给出的答案是：**不告诉你「能不能用」，告诉你「哪部分能用、
哪部分不能用」**。每个原子带 pLDDT，每对 token 带 PAE / PDE，让用户
**逐位置**做判断。这套机制比单一的「整体精度评分」实用得多。

AF3 输出 5 类置信信号：

| 量 | 全名 | 维度 | 含义 |
|---|---|---|---|
| **pLDDT** | predicted local-distance-difference test | per atom | 局部精度 (0-100，越高越好) |
| **PAE** | predicted aligned error | per token pair | 给定一对 token，二者相对位姿误差分布 |
| **PDE** | predicted distance error | per token pair | 二者距离误差分布（对称量） |
| **resolved** | 是否在实验中可见 | per atom | 二分类：原子是否能被实验观察到 |
| **distogram** | predicted pair distances | per token pair | 真实距离的分布（64 个 bin） |

Distogram 严格说是训练目标，不是置信量；但同样由置信度路径产出，所以放一起。

## 5.2 LDDT：经典的局部精度度量

[**LDDT**](https://academic.oup.com/bioinformatics/article/29/21/2722/195896)
(Mariani et al. 2013) 不依赖整体对齐，对每个原子算「我和邻居的距离误差
是否在阈值内」：

$$\mathrm{LDDT}(\ell) = \frac{1}{|\mathcal{N}(\ell)|} \sum_{m \in \mathcal{N}(\ell)} \frac{1}{4} \sum_{t \in \{0.5, 1, 2, 4\}} \mathbb{1}\big[|d_{\ell m}^\text{pred} - d_{\ell m}^\text{gt}| < t\big]$$

其中 $\mathcal{N}(\ell)$ 是真实结构里距 $\ell$ 在 15 Å 内的原子集合。
本质：把「预测距离与真实距离的误差是否在 0.5 / 1 / 2 / 4 Å 之内」做 4 个阈值的均值。

LDDT 的优点：

- **局部敏感**：不受整体对齐误差影响。蛋白的某个 domain 完全错位，但 domain
  内部相对位置对的话，那部分 LDDT 仍很高
- **稀疏可解释**：直接告诉你「哪个残基附近预测可靠」
- **方便采样**: 即使预测是配体 / 修饰残基也能算

## 5.3 pLDDT：模型自己估计 LDDT

推理时模型不知道 ground truth，于是预测**LDDT 的分布**。AF3 把 [0, 100]
切成 50 个 bin，per-atom 输出 50 维 logits：

$$\mathrm{pLDDT}_\ell = \mathrm{Linear}_{50}(\mathrm{LN}(s_\ell))$$

softmax 后做 bin 中心加权得到标量预测值。训练时把真实 LDDT 离散化到
50 bin 做 cross-entropy loss。

实现上 `plddt_weight` 是个 `[max_atoms_per_token, c_s, b_plddt]` 的张量 ——
**按原子在 token 内的 slot 编号查权重**。这样不同类型的原子（CA / CB / N /
配体的某个特定原子）有专门的预测器。slot 编号由
`input_feature_dict["atom_to_tokatom_idx"]` 给出。

```python
plddt_pred = einsum("nc, ncb -> nb",
                    plddt_ln(broadcast_token_to_atom(s_single)),
                    plddt_weight[atom_to_tokatom_idx])
```

## 5.4 PAE：相对位姿误差

PAE 衡量「如果**对齐到 token i 的局部坐标系**，token j 的预测位置离真实位置
多远」。形式化：

$$\mathrm{AE}(i, j) = \big\|T_i^\text{pred,-1} \cdot x_j^\text{pred} - T_i^\text{gt,-1} \cdot x_j^\text{gt}\big\|$$

$T_i$ 是 token i 的局部 frame（用第 4 章的 `expressCoordinatesInFrame`
构造）。PAE 是个 **N×N 矩阵**，**有方向**：从 i 看 j 的误差 ≠ 从 j 看 i 的误差。
所以 PAE head 内部对 z 不做对称化:

```python
pae_pred = linear_no_bias_pae(pae_ln(z_pair))   # [N, N, b_pae]
```

PAE 最常见的用途：**判断蛋白结构域之间的相对姿态是否可信**。蛋白经常有两个
domain 内部预测都好（pLDDT 高），但两个 domain 之间的相对方向却不确定 ——
PAE 矩阵在两个 domain 边界附近会有大的预测值。

复合物 / 多链系统里，PAE 在 chain 边界附近的统计是 **iPTM** (interface pTM)
的来源 —— 判断两条链界面接触是否可信。

## 5.5 PDE：距离误差（对称版）

PDE 比 PAE 简单：不涉及 frame 对齐，只看**距离差的绝对值**：

$$\mathrm{DE}(i, j) = \big|\|x_i^\text{pred} - x_j^\text{pred}\| - \|x_i^\text{gt} - x_j^\text{gt}\|\big|$$

天然对称：$\mathrm{DE}(i, j) = \mathrm{DE}(j, i)$。所以 PDE head 内部要先做
`z + z.transpose(-2, -3)` 对称化再 LN + Linear：

```python
pde_pred = linear_no_bias_pde(pde_ln(z_pair + z_pair.transpose(-2, -3)))
```

PDE 不像 PAE 那样能告诉你「方向不对」，但能告诉你「距离不可信」——
两个互补的视角。

## 5.6 resolved：实验可见性

PDB 文件里常有**missing residues / atoms** —— 实验决定不了它们的位置
（disordered region, low electron density, etc.）。AF3 预测这些位置存在与否
本身就是有用的信号。

训练时把 PDB 里有坐标的原子标 1、没有的标 0；推理时给出 **per-atom 二分类 logits**：

```python
resolved_pred = einsum("nc, ncb -> nb",
                       resolved_ln(broadcast_token_to_atom(s_single)),
                       resolved_weight[atom_to_tokatom_idx])
```

`b_resolved = 2`（两个类）。同样按原子 slot 查权重。

## 5.7 Distogram (Algorithm 1 line 17)

训练目标之一：给每对 token (i, j) 预测**真实距离**落在 64 个 bin 的哪一个。
由 pair 张量 z 直接投出 logits，对称化保证物理对称：

```python
logits = self.linear(z)                          # [N, N, 64]
logits = logits + logits.transpose(-2, -3)       # symmetrize
return logits
```

**`self.linear` 用 `initializer="zeros"`**：训练初期所有 logits 是 0，
softmax 是均匀分布，让模型从「不知道距离」状态出发，慢慢学习。

distogram 本身不是「置信度」，但它是 AF2 / AF3 损失里很重要的一项——
模型必须先学会预测距离分布，才能在扩散里把坐标推到对的地方。

## 5.8 ConfidenceHead：把一切拼起来

完整 forward 流程（`confidence_head.py::ConfidenceHead.forward`）：

1. **可选 stop-gradient**：训练时让 confidence head 不影响 trunk 梯度
2. **clamp + LayerNorm s_trunk**：长序列里 trunk 输出方差大，clamp 到 ±512 防爆
3. **构造 pair 条件**: $z_\text{init} = \text{linear}_1(s) \oplus \text{linear}_2(s) + z_\text{trunk}$
4. **每个 diffusion sample 调一次 memory_efficient_forward**:
   - 算预测坐标的距离矩阵，分 bin 一路 / 不分 bin 一路加到 z_pair 上
   - 跑一个**小型 PairformerStack**（默认 4 个 block，可独立配置）
   - 投出 4 个 head: pae, pde, plddt, resolved
5. 沿 sample 维 stack 输出

为什么 confidence head 内部要再跑一遍 PairformerStack？
**预测坐标本身是 confidence 的重要信号**——只有看到预测距离才能判断
「这对 token 是否预测准确」。所以 confidence 不能直接读 trunk 的 z，
要先把预测距离 fold 进去再过一轮 pairformer。

## 5.9 与本仓库代码对应

```
confidence/
├── distogram_head.py
│   └── DistogramHead                ← forward (Alg 1 line 17)
├── confidence_head.py
│   └── ConfidenceHead                ← __init__ + forward + memory_efficient_forward (Alg 31)
├── bins.py                          ← 距离 bin / pLDDT bin 工具
├── scores.py                        ← pTM / iPTM / pLDDT 标量计算
├── summary.py                       ← 推理输出 summary dict
├── clash.py / external_clash.py     ← 物理性 clash 检测 (后处理)
```

注意 `bins.py` / `scores.py` / `summary.py` / `clash.py` 是**后处理**：
ConfidenceHead 给出 per-atom / per-pair 的分布，这些工具再算汇总分数
（如 ranking_score）。

## 5.10 设计取舍

**为什么 pLDDT / resolved 按 atom-slot 查权重，而 PAE / PDE 不需要？**
pLDDT 和 resolved 是 per-atom 量；同一个 token 里不同原子（CA / CB / 侧链等）
精度可能差很大——CA 通常预测最准，侧链末端最不准。所以用 slot-aware 权重
让模型学到 per-slot 的预测器。PAE / PDE 是 per-token-pair 量，token 已经聚合了
所有原子，不需要再分 slot。

**为什么 PAE 不对称、PDE 对称？**
PAE 涉及 frame 对齐 —— frame 选 i 还是 j 是不一样的，方向不对称。
PDE 只看距离差，是几何对称量。强制 head 对称化保证物理量合法。

**为什么 confidence head 要 stop-gradient？**
让 trunk 不被 confidence 损失干扰 —— trunk 主要学预测距离 / 接触，
confidence 学的是「模型自己预测的不确定性」。两个任务梯度方向不一致时，
stop-gradient 让 confidence 不去推 trunk 的权重，专心学自己。

## 5.11 延伸阅读

- AF3 主论文 Algorithm 31（ConfidenceHead 完整伪代码）
- [LDDT (Mariani et al. 2013)](https://academic.oup.com/bioinformatics/article/29/21/2722/195896) —— pLDDT 的真实定义
- AF2 Supplementary §1.9.5-1.9.8 —— PAE / pTM 的原始定义
- [The Illustrated AlphaFold §4 Confidence](https://elanapearl.github.io/blog/2024/the-illustrated-alphafold/)
