# build-your-af3

[English version](README.en.md)

一个面向教学、对 Mac 友好的 **AlphaFold 3** 复现项目。沿用
[alphafold-decoded](https://github.com/kilianmandon/alphafold-decoded)（AF2 版）
按章节拆分的写法，但目标换成 AF3 架构和权重。可直接加载字节跳动 **Protenix**
官方 checkpoint，CPU / Apple Silicon MPS / CUDA 都能跑。

## 项目结构

照搬参考项目的三部分结构：

```
build-af3-from-scratch/
├── lessons/      # 章节讲解（markdown）
├── tutorials/    # 学生填空版（自动生成）
└── solutions/    # 完整参考实现
    ├── attention/                # MHA + LayerNorm + AttentionPairBias
    ├── feature_extraction/       # JSON / MSA / template → 张量
    ├── feature_embedding/        # 输入嵌入 + Atom Attention Encoder
    ├── pairformer/               # Pairformer + MSAModule + Triangle ops
    ├── diffusion/                # DiffusionModule + Transformer + sampler
    ├── confidence/               # ConfidenceHead + 置信度计算
    ├── model/                    # 顶层 Protenix 装配 + 推理入口
    ├── configs/                  # 配置字典 + ConfigDict 解析（公共）
    └── runtime/                  # seed / logger / torch utils（公共）
```

每一章都是一个 *flat Python package*，章内文件之间用
`from <chapter>.<file> import <Class>` 互相 import —— 与 AF2 参考完全一致。

## 三件套：solutions / tutorials / lessons

| 目录 | 内容 | 用法 |
|---|---|---|
| `solutions/` | 完整参考实现 | 写完 tutorials 后对照检查；学习时尽量不要先看。 |
| `tutorials/` | `prepare_tutorials.py` 从 `solutions/` 自动剥出的填空版：所有被包裹的 `forward` 内容替换成 `pass`，TODO 上方保留详细伪代码 | 学生按 TODO 一步步把空填上。 |
| `lessons/` | 每章的教学 markdown | 想了解算法 / 数学背景时阅读。 |

重新生成填空版：

```bash
python prepare_tutorials.py        # solutions/* → tutorials/*
python prepare_tutorials.py --clean  # 先清空 tutorials/ 再生成
```

## 为什么要做 AF3 版？

AF3 在架构上有几个关键变化，没法直接复用 AF2 那一套章节：

| AF2 章节              | AF3 对应                                  |
|-----------------------|-------------------------------------------|
| `evoformer`           | `pairformer`（+ `msa_stack`）              |
| `structure_module`    | `diffusion`                               |
| `feature_embedding`   | + AtomAttentionEncoder                    |
| `geometry`            | 大部分融进 diffusion module               |

中间几章因此和 AF2 不同，但底层 primitives（attention / residual / axial ops）
还是同一套。

## 快速开始

### 1. 安装

Mac / Apple Silicon：

```bash
conda env create -f environment_mac.yml
conda activate af3
```

CPU only（Linux / Mac）：

```bash
conda env create -f environment_cpu.yml
conda activate af3
```

或者 venv + pip：

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision \
            rdkit biopython biotite modelcif gemmi pdbeccdutils \
            ml-collections scipy pandas scikit-learn scikit-learn-extra \
            matplotlib ipykernel ipywidgets py3dmol icecream fair-esm
```

### 2. 下载权重

用官方 Protenix-Tiny（约 110 M 参数，MSA 版）：

```bash
mkdir -p checkpoints
curl -L -o checkpoints/protenix_tiny_default_v0.5.0.pt \
    https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix_tiny_default_v0.5.0.pt
```

辅助缓存（CCD 化学组件等）会在首次推理时自动下载到 `~/common/`。

### 3. 运行推理

```bash
cd solutions
LAYERNORM_TYPE=torch python -m model.inference \
    --input_json examples/example.json \
    --dump_dir   ./out \
    --device     mps            # 或 cpu / cuda
```

每个样本会产出一个 `*.cif` 和一个 `*_summary_confidence_*.json`。
完整示例 JSON 见 `solutions/examples/example.json`（自带 7r6r 蛋白 + MSA）。

### 4. 端到端 demo

```bash
jupyter notebook solutions/overview.ipynb
```

notebook 走完一整圈：构建模型 → 加载权重 → 特征化 → 推理 → 写 CIF → 可视化。

### 5. 验证

约 200 残基的蛋白，单次前向在 CPU 上约 4–10 秒，pLDDT 大致落在 30–75 区间
（依赖 PyTorch 版本和后端）；切到 MPS 大约快 1.6 倍。

## 章节如何拼起来

```
                       ┌─ feature_extraction ─┐
                       │  JSON → atom array   │
                       │  MSA / templates     │
                       └──────┬───────────────┘
                              ▼
       ┌────────────────────────────────────────────┐
       │  feature_embedding                         │
       │  · InputFeatureEmbedder (Alg 2)            │
       │  · AtomAttentionEncoder (Alg 5)            │
       │  · RelativePositionEncoding                │
       └──────┬─────────────────────────────────────┘
              ▼
       ┌────────────────────────────────────────────┐
       │  pairformer  (Alg 8/16/17)                 │
       │  · MSAModule · TemplateEmbedder            │
       │  · PairformerStack with Triangle ops       │
       └──────┬─────────────────────────────────────┘
              ▼
       ┌────────────────────────────────────────────┐
       │  diffusion  (Alg 18/23/25)                 │
       │  · DiffusionModule + Transformer           │
       │  · Sampler (InferenceNoiseScheduler)       │
       └──────┬─────────────────────────────────────┘
              ▼
       ┌────────────────────────────────────────────┐
       │  confidence  (Alg 26–31)                   │
       │  · ConfidenceHead + DistogramHead          │
       │  · pTM / iPTM / pLDDT / clash              │
       └────────────────────────────────────────────┘
```

`model/model.py` 把这些模块按 Algorithm 1 串起来。

## 特性

- 按章节拆分的 Python 包结构 —— 每个模块住在以它命名的文件里。
- 纯 PyTorch 实现，CPU / MPS / CUDA 都能跑。
- 开箱即用加载字节跳动 Protenix Tiny / Mini 官方权重。
- 公开 API 全部带中英双语 docstring。
- 每个被包裹的 `forward` / 关键 `__init__` 上方都写有详细伪代码 TODO，
  学生照着填出来的就是与 Protenix 权重完全兼容的实现。

## 致谢

- **字节跳动 Protenix** —— AF3 架构实现 + 开源权重。
- **`alphafold-decoded`**（Kilian Mandon）—— 按章节拆分的教学结构。
- **DeepMind** —— AlphaFold 3 论文。

## License

Apache 2.0，见 [LICENSE](LICENSE)。
