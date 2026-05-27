# 第 0 章 · Feature Extraction（数据流水线）

> **章节定位**：只读 walkthrough，不挖空。学完后你知道 AF3 推理时
> 「一个 JSON 进去，到底变成了哪些张量」。
>
> **配套 notebook**：`tutorials/feature_extraction/feature_extraction.ipynb`
>
> **本系列受 [The Illustrated AlphaFold](https://elanapearl.github.io/blog/2024/the-illustrated-alphafold/) 启发，内容为原创中文版。论文图请直接参考 Abramson et al. 2024 *Nature*。**

## 0.1 为什么需要 feature extraction

AlphaFold 系列的输入是一段**生物意义上的描述**——蛋白序列、可能的同源序列
对齐 (MSA)、模板结构、配体的化学组分等。但模型本体是一个吃张量、吐张量的
`nn.Module`，它不认识 FASTA / CIF / SMILES。中间这一层「**异质数据 → 张量字典**」
就是 feature extraction 干的事。

AF3 比 AF2 显著复杂的地方就在这里：

- AF2 主要处理蛋白单体，输入基本是「序列 + MSA + 模板」。
- AF3 要处理**蛋白 + 核酸 + 配体 + 修饰残基 + 实验约束**任意组合，
  feature extraction 要把所有这些归一到同一份 token / atom 表示。

这一章我们**只读不写**——`feature_extraction/` 下的代码本质是数据工程，
没有 AF3 算法核心，不挖 TODO。学生只需要知道「我喂给 trunk 的张量是什么样的」。

## 0.2 输入 JSON 长什么样

仓库自带的示例 `solutions/examples/example.json`：

```json
[
  {
    "name": "7r6r_protein_only",
    "sequences": [
      {
        "proteinChain": {
          "sequence": "MGSSHHH...EELLSEP",
          "count": 1,
          "id": ["A"],
          "msa": {
            "precomputed_msa_dir": "examples/7r6r/msa/1",
            "pairing_db": "uniref100"
          }
        }
      }
    ]
  }
]
```

一个用户提交是一个 sample，sample 里可以放任意多个 `sequences` 项，
每项可以是 `proteinChain` / `dnaSequence` / `rnaSequence` / `ligand` /
`ion`。`count` 大于 1 表示寡聚体。MSA 可以预先算好放在
`precomputed_msa_dir`，也可以由 AF3 内部 BLAST/HHblits 实时搜（成本极高）。

## 0.3 流水线主线

`feature_extraction/` 顶层入口是 `inference/infer_dataloader.py::get_inference_dataloader`，
它在内部串起三个 featurizer：

1. **`SampleDictToFeatures`** (`inference/json_to_feature.py`)
   把 JSON 解析成 token + atom 数组。处理:
   - 序列 → token 序列；每个氨基酸 / 核苷酸 = 1 个 token
   - 配体 / 修饰残基 → 每个原子 = 1 个 token（atom-level tokenization）
   - 各 token 对应的 `ref_pos` / `ref_charge` / `ref_element` / `ref_atom_name_chars`
     从 CCD（Chemical Component Dictionary）查询得到
2. **`InferenceMSAFeaturizer`** (`msa/msa_featurizer.py`)
   读取 `precomputed_msa_dir` 或在线搜索；做 deduplication、pairing、sub-sampling
3. **`InferenceTemplateFeaturizer`** (`template/template_featurizer.py`)
   找同源结构作为初始模板。Tiny 模型一般跳过。

## 0.4 输出：`input_feature_dict`

featurizer 跑完吐出一份字典 `data["input_feature_dict"]`，里面是几十个张量。
关键几条（在 notebook 4.3 节有完整列表）：

| 字段 | 形状 | 含义 |
|---|---|---|
| `restype` | `[N_token, 32]` | 残基 / 核苷酸 one-hot |
| `profile` | `[N_token, 32]` | MSA-derived 残基类型分布 |
| `deletion_mean` | `[N_token, 1]` | MSA 删除均值 |
| `msa` | `[N_msa, N_token]` | 子采样后的 MSA 序列 |
| `has_deletion`, `deletion_value` | `[N_msa, N_token]` | MSA 行级删除指示 |
| `ref_pos` | `[N_atom, 3]` | 参考几何（来自 CCD） |
| `ref_charge` / `ref_mask` / `ref_element` / `ref_atom_name_chars` | per-atom | 各项原子属性 |
| `d_lm`, `v_lm`, `pad_info` | dense-trunk | 原子对之间的距离 + 有效性 |
| `atom_to_token_idx` | `[N_atom]` | 原子 → token 的映射 |
| `atom_to_tokatom_idx` | `[N_atom]` | 原子 → token 内部 slot 编号 |
| `asym_id` / `entity_id` / `sym_id` / `residue_index` / `token_index` | per-token | 用于 `RelativePositionEncoding`（第 3 章） |
| `token_bonds` | `[N_token, N_token]` | 共价键邻接（用于 InputFeatureEmbedder） |
| `template_*` | per-template | 模板特征 |
| `relp` | `[N_token, N_token, 4·r_max+2·s_max+7]` | 由 `RelativePositionEncoding.generate_relp` 在 trunk forward 起始生成 |

## 0.5 Token vs Atom：AF3 的两级抽象

这是 AF3 设计里最容易让新读者迷糊的概念。简单说：

- **Token** = 一个抽象的「单元」。蛋白 / 核酸残基本身是 1 个 token；配体 / 修饰残基的
  每个重原子都各算 1 个 token。整个系统大约几百到几千个 token。
- **Atom** = 物理原子。重原子总数远多于 token（典型蛋白每残基 ~8 重原子）。

Trunk（Pairformer + MSAModule + Template）跑在 **token 级**，因为 pair 张量
`[N_token, N_token, c_z]` 在 token 数为千的量级仍能装下。**扩散模块**则在
**atom 级**输出 `[N_atom, 3]` 坐标——蛋白侧链每个原子都要预测准。两个级别
通过 `atom_to_token_idx` 等索引张量来回切换。

`feature_extraction` 这一层就负责把所有信息编排到这两个抽象上：每个
token 知道自己包含哪几个 atom、每个 atom 知道自己属于哪个 token 的第几个
slot（用于 `plddt_weight[atom_to_tokatom_idx]` 之类的 lookup）。

## 0.6 与本仓库代码对应

我们的目录布局：

```
feature_extraction/
├── inference/
│   ├── infer_dataloader.py       ← get_inference_dataloader 入口
│   └── json_to_feature.py        ← SampleDictToFeatures
├── msa/msa_featurizer.py         ← InferenceMSAFeaturizer
├── template/template_featurizer.py ← InferenceTemplateFeaturizer
├── constraint/constraint_featurizer.py ← 用户约束 (contact / pocket / bond)
├── core/                          ← mmCIF / PDB 解析、CCD lookup
├── esm/                           ← PLM (ESM) 特征 (Mini-ESM 变体用)
├── tokenizer.py                   ← 残基类型 / 元素 / 原子名编码表
└── constants.py                   ← `STD_RESIDUES_WITH_GAP` 等常量
```

代码量大但**没有可学参数**——全是规则映射、字典 lookup、shape 排布。
学生只要知道入口和输出格式就够。

## 0.7 设计取舍

**为什么 AF3 选择 atom-level token 而不是统一 atom-level？**
统一 atom-level 会让 token 数膨胀 ~8 倍，pair 张量显存膨胀 ~64 倍——
直接超出 GPU。所以 AF3 折中：蛋白 / 核酸保留 residue-level token 节省算力；
配体 / 修饰残基用 atom-level token 提供必要的化学细节。

**为什么 MSA 维要 sub-sample**？典型 MSA 几万行，全跑显存爆。AF3 在
`MSAModule.forward`（见第 2 章 lessons）里随机抽 512 / 2048 / 16384 行。

**为什么把 `relp` 留到 trunk forward 才生成**？relp 是 `[N_token, N_token, ~23]`
的张量，dataloader 阶段就生成会让 batch 在 CPU 端变重；trunk 在 GPU 上
按 token 关系实时算更便宜。

## 0.8 延伸阅读

- AF3 主论文 Methods §2.1「Tokenization and featurization」
- AF3 Supplementary 第 2 节（Input feature embedding 之前的所有处理）
- 仓库 `solutions/feature_extraction/feature_extraction.ipynb` —— 跑一次看实际数值
