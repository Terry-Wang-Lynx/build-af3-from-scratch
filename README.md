# build-your-af3

[English version](README.en.md)

一个面向教学、对 Mac 友好的 **AlphaFold 3** 复现项目。按章节拆分的 Python 包结构，
可直接加载字节跳动 **Protenix** 官方 checkpoint，CPU / Apple Silicon MPS / CUDA
都能跑。

## 项目结构

项目按三个目录组织：

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
`from <chapter>.<file> import <Class>` 互相 import。

## 三件套：solutions / tutorials / lessons

| 目录 | 内容 | 用法 |
|---|---|---|
| `solutions/` | 完整参考实现 | 写完 tutorials 后对照检查；学习时尽量不要先看。 |
| `tutorials/` | `prepare_tutorials.py` 从 `solutions/` 自动剥出的填空版：所有被包裹的 `forward` 内容替换成 `pass`，TODO 上方保留详细伪代码 | 学生按 TODO 一步步把空填上。 |
| `lessons/` | 每章的教学 markdown（7 章已成稿，持续打磨）| 想了解算法 / 数学背景时阅读，配合 notebook 内的 TODO 伪代码使用。 |

重新生成填空版：

```bash
python prepare_tutorials.py          # solutions/* → tutorials/*
python prepare_tutorials.py --clean  # 先清空 tutorials/ 再生成
```

## 学习路径 · How to use this project

推荐按下面 6 步走，章节本身共 7 个 (chapter 0 是只读 tour)；
每一章大约 0.5 – 2 小时不等：

1. **拉取仓库 + 生成填空版**

   ```bash
   git clone https://github.com/Terry-Wang-Lynx/build-af3-from-scratch.git
   cd build-af3-from-scratch
   python prepare_tutorials.py --clean
   ```

   生成出来的 `tutorials/` 就是你的工作目录。`solutions/` 是参考答案，建议
   不要先翻。

2. **配置环境** （二选一）

   ```bash
   conda env create -f environment_mac.yml      # Mac / Apple Silicon
   # 或
   conda env create -f environment_cpu.yml      # Linux / Mac CPU
   conda activate af3
   ```

3. **逐章学习**。每章 `tutorials/<chapter>/<chapter>.ipynb` 都是一份引导
   笔记本：每个小节先指明要打开哪个 `.py`、要填哪几个 TODO，紧跟着一格
   测试代码。填好就跑测试 cell：

   ```python
   test_module_shape(mha, 'mha_gated', control_folder)
   test_module_forward(mha, 'mha_gated', inputs=(...), ...)
   ```

   - 通过 → 章节继续往下走。
   - 不通过 → 测试函数会告诉你哪个输出对不上，回 `.py` 文件改实现，
     `autoreload` 直接生效，不用重启 kernel。

   推荐顺序：

   | 顺序 | 章节 | notebook | 主要内容 |
   |---|---|---|---|
   | 0 | `feature_extraction/` | `feature_extraction.ipynb` | **只读** walkthrough：JSON → 张量字典 |
   | 1 | `attention/` | `attention.ipynb` | Linear / LayerNorm / MHA / AdaLN / Transition / AttentionPairBias |
   | 2 | `pairformer/` | `pairformer.ipynb` | OuterProductMean / TriangleMul / TriangleAttention / MSAPairWeightedAveraging / PairformerBlock |
   | 3 | `feature_embedding/` | `feature_embedding.ipynb` | RelativePositionEncoding / FourierEmbedding |
   | 4 | `diffusion/` | `diffusion.ipynb` | ConditionedTransitionBlock / DiffusionTransformerBlock / DiffusionTransformer / expressCoordinatesInFrame / centre_random_augmentation |
   | 5 | `confidence/` | `confidence.ipynb` | DistogramHead |
   | 6 | `model/` | `overview.ipynb` | 端到端：加载 Protenix 权重 → 推理 → 出 CIF |

4. **章末自查**：每章末尾对应 `python generate_control_values.py
   --verify --src tutorials --chapters <chapter>` 一行回归检查。
   也可以用顶层 `python check_solutions.py --src tutorials` 一次性跑完六个
   章节 notebook (默认不跑 overview，避免还没下载权重的同学卡住)；
   想连同端到端推理一起测就加 `--with-overview`。

5. **跑端到端**：所有空填完以后下载 Protenix 权重（见 "快速开始"），
   打开 `tutorials/model/overview.ipynb` 跑通一次推理，会出 `7r6r_pred.cif`
   并报告 pLDDT / pTM。

6. **对答案**：把你写完的实现与 `solutions/` 对比，看看哪里写得不够干净。
   两边推理输出应一致（state_dict 基本兼容；tiny / default checkpoint 会
   留一个未启用的 `input_embedder.linear_esm.weight` 键被安全忽略，详见
   `solutions/model/inference.py::load_checkpoint`）。

## 单元测试 (control values)

每章 `control_values/` 下保存了一套小尺寸的参考张量 (`*.pt`)，配合
`runtime/checks.py` 里的 `test_module_shape` / `test_module_forward`，
学生可以按章 / 按函数验证自己的实现：

- 测试时把模块参数临时替换成 `torch.linspace(-1, 1, numel)`，再用固定
  的 `test_inputs` 跑前向，输出与 `<name>_out.pt` 对照；参数 shape 也
  会与 `<name>_param_shapes.pt` 对照。
- `solutions/<chapter>/control_values/_generate.py` 既能生成参考值
  (`overwrite=True`)，也能回归校验 (`overwrite=False`)。
- 仓库根目录有顶层入口 `generate_control_values.py`：

```bash
# 用 solutions/ 重新生成全部参考 .pt
python generate_control_values.py

# 用 tutorials/ 验证自己的实现 (未填空时会失败)
python generate_control_values.py --verify --src tutorials
```

每个 `.pt` 都很小 (KB 级)，已和源代码一起提交进 GitHub。`prepare_tutorials.py`
会把 `control_values/` 整目录拷到 `tutorials/`，因此学生 fork 后可以
立刻开始函数级测试。

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
            matplotlib ipykernel ipywidgets py3dmol icecream fair-esm \
            nbformat nbconvert
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
    --ckpt_dir   ../checkpoints \
    --device     mps            # 或 cpu / cuda
```

每个样本会产出一个 `*.cif` 和一个 `*_summary_confidence_*.json`。
完整示例 JSON 见 `solutions/examples/example.json`（自带 7r6r 蛋白 + MSA）。

### 4. 端到端 demo

```bash
jupyter notebook solutions/model/overview.ipynb
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
  学生照着填出来的实现可加载 Protenix 权重；tiny / default checkpoint 中
  未启用的 ESM 投影键会被安全忽略。

## 致谢

- **字节跳动 Protenix** —— AF3 架构实现 + 开源权重。
- **`alphafold-decoded`**（Kilian Mandon，AF2 教学项目）—— 教学结构灵感来源。
- **DeepMind** —— AlphaFold 3 论文。

## License

本仓库以 Apache 2.0 发布，见 [LICENSE](LICENSE)（含完整 Apache 2.0 全文）。
第三方组件（ByteDance Protenix、AlQuraishi/OpenFold 衍生工具、MIT 许可的
`alphafold-decoded` 教学脚手架）的版权声明与许可全文见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
