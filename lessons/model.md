# 第 6 章 · Model（Protenix 顶层装配）

> **章节定位**：把前 5 章造好的所有零件，按 AlphaFold 3 Algorithm 1 的
> 顺序拼成完整模型，端到端跑一次推理。
>
> **配套 notebook**：`tutorials/model/overview.ipynb`
>
> **本系列受 [The Illustrated AlphaFold](https://elanapearl.github.io/blog/2024/the-illustrated-alphafold/) 启发，内容为原创中文版。**

## 6.1 全图概览

AF3 完整推理（论文 Algorithm 1）大致是：

```
input_feature_dict
       │
       ├─ InputFeatureEmbedder ─────────► s_inputs                  (第 3 章)
       │
       └─ RelativePositionEncoding ────► relp                       (第 3 章)
              │
              ▼
       ┌───────────────────────────────────────────────────────┐
       │  Trunk (recycled N_cycle times)                       │
       │  ┌─────────────────────────────────────────────────┐  │
       │  │ TemplateEmbedder (optional)                      │  │
       │  │ MSAModule                                        │  │
       │  │ PairformerStack                  (第 2 章)        │  │
       │  └─────────────────────────────────────────────────┘  │
       │                                                       │
       │   ↓ 输出: s_trunk (single)、z_trunk (pair)            │
       └───────────────────────────────────────────────────────┘
              │
              ├─ sample_diffusion (Alg 18)  ◄── DiffusionModule (第 4 章)
              │       ▼
              │     x_pred_coords (扩散采样的 N_sample 份坐标)
              │
              └─ ConfidenceHead + DistogramHead              (第 5 章)
                      ▼
                   pLDDT / PAE / PDE / resolved / distogram
```

主干（Trunk）跑 N_cycle 轮 recycling，每轮把上一轮的 (s, z) 折回输入。
跑完拿 (s_trunk, z_trunk) 喂给扩散和置信头。

这一章本身**没有新算法**——只需要把前 5 章的零件按上面这张图接好。
本仓库的 `Protenix` 类就是这件事。

## 6.2 Protenix.\_\_init\_\_

构造一个 Protenix 实例要按顺序建以下子模块（顺序无关，但命名必须和 Protenix
checkpoint 一致才能加载权重）：

```python
self.input_embedder          = InputFeatureEmbedder(...)
self.constraint_embedder     = ConstraintEmbedder(...)         # 可选
self.relative_position_encoding = RelativePositionEncoding(...)
self.template_embedder       = TemplateEmbedder(...)           # 可选
self.msa_module              = MSAModule(...)
self.pairformer_stack        = PairformerStack(...)
self.diffusion_module        = DiffusionModule(...)
self.confidence_head         = ConfidenceHead(...)
self.distogram_head          = DistogramHead(...)

# trunk 单序列的 recycle 路径用的两个线性层
self.linear_no_bias_sinit    = LinearNoBias(c_s_inputs, c_s)
self.linear_no_bias_zinit1   = LinearNoBias(c_s, c_z)
self.linear_no_bias_zinit2   = LinearNoBias(c_s, c_z)
self.linear_no_bias_token_bond = LinearNoBias(1, c_z)
self.linear_no_bias_z_cycle  = LinearNoBias(c_z, c_z)
self.layernorm_z_cycle       = LayerNorm(c_z)
self.linear_no_bias_s        = LinearNoBias(c_s, c_s)
self.layernorm_s             = LayerNorm(c_s)
```

每个子模块的具体维度从 `configs/configs_model_type.py` 读，
对应 Protenix 不同变体（base / mini / tiny，MSA / ESM / ISM）。

## 6.3 Trunk forward (Algorithm 1 lines 1-13)

`get_pairformer_output(input_feature_dict, N_cycle)` 实现 trunk 部分：

```python
# 1. 输入嵌入 → s_inputs
s_inputs = self.input_embedder(input_feature_dict)
z_constraint = self.constraint_embedder(...) if 'constraint_feature' in ifd else None

# 2. 单序列投到 c_s，作为 recycling 起点
s_init = self.linear_no_bias_sinit(s_inputs)

# 3. pair 初始: 由 s_init 两路投影外加得到 base，再加 RelativePositionEncoding 和共价键
z_init = linear_no_bias_zinit1(s_init)[..., None, :]     # [N, 1, c_z]
       + linear_no_bias_zinit2(s_init)[..., None, :, :]  # [1, N, c_z]
z_init += relative_position_encoding(ifd["relp"])
z_init += linear_no_bias_token_bond(ifd["token_bonds"].unsqueeze(-1))
if z_constraint is not None:
    z_init += z_constraint

# 4. recycle N_cycle 轮
z = torch.zeros_like(z_init)
s = torch.zeros_like(s_init)
for _ in range(N_cycle):
    # z 的 recycle: 上一轮 z 经 LN+Linear 折回，加回 z_init
    z = z_init + linear_no_bias_z_cycle(layernorm_z_cycle(z))
    # 可选模板嵌入
    if self.template_embedder.n_blocks > 0:
        z = z + self.template_embedder(ifd, z)
    # MSAModule 改 z
    z = self.msa_module(ifd, z, s_inputs, pair_mask=None)
    # s 的 recycle: 上一轮 s 经 LN+Linear 折回，加回 s_init
    s = s_init + linear_no_bias_s(layernorm_s(s))
    # PairformerStack 共同更新 s, z
    s, z = self.pairformer_stack(s, z, pair_mask=None)

return s_inputs, s, z
```

注意几点：

- **z 的 recycle 加的是 z_init 而不是 z**：每轮都从 z_init 重新开始，
  上一轮的 z 只通过一个轻量 LN+Linear 影响 z_init 的偏移。
  这是 AF2 的设计，AF3 沿用。
- **N_cycle = 4** 在 AF3 base 默认；Tiny 模型常用 1 或 4。每多一轮约 25%
  推理时间，精度回报递减。
- **模板可选**: tiny 模型默认 n_blocks=0 跳过整个 TemplateEmbedder。

## 6.4 Diffusion + Confidence forward (Algorithm 1 lines 14-17)

trunk 跑完后:

```python
s_inputs, s_trunk, z_trunk = self.get_pairformer_output(ifd, N_cycle)

# Algorithm 18: 跑扩散采样得到 N_sample 份坐标
x_pred_coords = sample_diffusion(
    denoise_net=self.diffusion_module,
    input_feature_dict=ifd,
    s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=z_trunk,
    pair_z=None, p_lm=None, c_l=None,                   # 首次调用时由 prepare_cache 算
    noise_schedule=InferenceNoiseScheduler(N_step).schedule,
    N_sample=N_sample,
    gamma0=..., gamma_min=..., step_scale_eta=..., ...
)

# Algorithm 31: 置信度头
plddt, pae, pde, resolved = self.confidence_head(
    ifd, s_inputs, s_trunk, z_trunk, pair_mask=None,
    x_pred_coords=x_pred_coords,
)

# Algorithm 1 line 17: distogram
distogram_logits = self.distogram_head(z_trunk)
```

## 6.5 Protenix.forward 与输出格式

顶层 `forward(input_feature_dict, mode='inference')` 把上面全套串起来:

1. 校验 mode（仅 inference）
2. 生成 / 缓存 relp，更新 input_feature_dict（一次性）
3. 调 `main_inference_loop`
4. 返回 `(pred_dict, None, log_dict)` —— 中间那个 `None` 是为训练 mode 的
   label-aligned 输出留位

`pred_dict` 里关键字段：

| key | 形状 | 含义 |
|---|---|---|
| `coordinate` | `[N_sample, N_atom, 3]` | 扩散采样的坐标 |
| `summary_confidence` | list of dict | 每个 sample 一份汇总，含 plddt / ptm / iptm / ranking_score / has_clash |
| `plddt` | `[N_sample, N_atom, 50]` logits | per-atom pLDDT 分布 |
| `pae` / `pde` | `[N_sample, N_token, N_token, 64]` | pair 误差分布 |
| `resolved` | `[N_sample, N_atom, 2]` | per-atom 二分类 |
| `distogram_logits` | `[N_token, N_token, 64]` | distogram |

最常用的 `summary_confidence[0]["plddt"]` 是 sample 0 的标量 pLDDT 总分，
经常作为「这次预测好不好」的快速指标。

## 6.6 Recycling 的设计哲学

AF2 第一次提出 recycling：让模型**自己迭代精炼自己的输出**。AF3 沿用：

- 第 1 轮：从零开始，trunk 看到 z=0、s=0；纯靠 RelPE / 共价键 / MSA / 模板
  把 pair / single 推到合理位置
- 第 2 轮：trunk 看到上一轮的 z, s；可以「在上一轮基础上微调」
- ... 直到 N_cycle 轮

为什么 work？trunk 跑一轮就是 48 个 PairformerBlock = ~24×4 个三角更新。
**Recycling 等价于「把 trunk 跑 4 倍长」**，但参数量不变 —— 因为每轮共享同一
份 PairformerStack 权重。这是经典的 deep-equilibrium-style 设计:
推理时**加深度而不加参数**。

代价: 推理时间线性增长。N_cycle=4 比 N_cycle=1 慢 4 倍。

## 6.7 与本仓库代码对应

```
model/
├── model.py
│   └── Protenix                       ← __init__ + get_pairformer_output (Alg 1 lines 1-13)
│                                        + main_inference_loop + forward
├── utils.py                            ← centre_random_augmentation (Alg 19), broadcast / aggregate 工具
├── inference.py                        ← 命令行入口：加载权重 + 跑 forward + 输出 CIF
└── __init__.py
```

`model/inference.py` 是 CLI 入口，命令行调用:

```bash
python -m model.inference \
    --input_json examples/example.json \
    --ckpt_dir ../checkpoints \
    --device cpu \
    --model_name protenix_tiny_default_v0.5.0
```

它的工作：解析参数 → `Protenix(cfg)` → 加载 ckpt → 跑 `forward` → 把
预测坐标写成 mmCIF 文件、把 summary 写成 JSON。

`tutorials/model/overview.ipynb` 是 notebook 形式的同一件事 ——
学完前 5 章后跑一次 overview，验证整套你填的代码组合起来能加载 Protenix
官方权重、给出 pLDDT ≈ 30+（Tiny 模型 5 步采样的合理水平）。

## 6.8 设计取舍

**为什么 trunk / diffusion / confidence 分成三阶段而不是端到端？**
三阶段各自的输入输出意义不同 —— trunk 输出抽象 pair / single 表示，
diffusion 输出坐标，confidence 输出置信。如果端到端一个 Transformer 同时
处理所有 N_token² + N_atom + N_pair 信号，显存 / 算法上几乎不可能。
分阶段允许每阶段用适合的 / 最便宜的 网络。

**为什么 confidence 在 diffusion 后？**
预测坐标本身是 confidence 的重要输入（距离矩阵 → 分 bin → 加到 z_pair）。
没有坐标，confidence head 给不出 PAE / PDE。所以顺序必须 diffusion → confidence。

**为什么 RelPE 在 forward 一开始算？**
推理时 N_cycle 轮每轮都要 relp，且 relp 与 cycle 无关。一次性算并塞回
`input_feature_dict` 后续直接读，省 (N_cycle-1) 倍重复计算。

## 6.9 端到端测试

`overview.ipynb` 跑完会输出类似：

```
Protenix built — 109.50 M parameters
Loaded protenix_tiny_default_v0.5.0.pt — missing=0 unexpected=1   ← linear_esm.weight (PLM 变体留的)
forward time: 10.26s
  pLDDT         = 33.82      ← Tiny 模型 5 步采样
  pTM           = 0.209
  ranking score = 0.124
  has_clash     = True
```

`pLDDT = 33` 不高，但这是 Tiny 模型 + 5 步采样在 CPU 上的合理基线。
AF3 base 模型 200 步会给出 pLDDT > 80 的高质量预测——但单次推理要 30+
分钟，做教学不合适。

## 6.10 延伸阅读

- AF3 主论文 Algorithm 1（顶层伪代码）+ Methods §2.2「Inference setup」
- [Protenix 官方文档](https://github.com/bytedance/Protenix) —— 包括如何下载更大的 base 模型权重
- [The Illustrated AlphaFold §1-4 全篇](https://elanapearl.github.io/blog/2024/the-illustrated-alphafold/)

## 全系列小结

读完这 7 个 lesson，你应该:

- ✅ 理解 AF3 为什么把 AF2 的 Evoformer 简化为 Pairformer + MSAModule，
  把 IPA Structure Module 换成 EDM 扩散
- ✅ 能在白板上画出 Trunk → Diffusion → Confidence 的数据流
- ✅ 能对每一节论文 Algorithm 编号说出 1-2 个工程实现要点
- ✅ 跑通了一次 7r6r 蛋白的端到端推理

下一步推荐：

- 把 base 模型权重下下来跑一次，看 pLDDT 能不能上到 80+
- 自己拿一个感兴趣的蛋白做预测，对比 AlphaFold Server / ColabFold
- 读 AF3 Supplementary 没完全展开的算法 (e.g. Atom transformer 内部的局部窗口设计)
