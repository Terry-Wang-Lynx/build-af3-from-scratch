# build-your-af3

<table><tr><td>

**English** — An educational, Mac-friendly **AlphaFold 3** reimplementation,
built in the same chapter-by-chapter format as
[alphafold-decoded](https://github.com/kilianmandon/alphafold-decoded) (AF2)
but targeting AF3 architecture and weights. Loads the official **ByteDance
Protenix** checkpoints unchanged; runs on CPU, Apple-Silicon MPS, or CUDA.

</td><td>

**中文** — 一个面向教学、对 Mac 友好的 **AlphaFold 3** 复现项目。沿用
[alphafold-decoded](https://github.com/kilianmandon/alphafold-decoded)（AF2 版）
按章节拆分的写法，但目标是 AF3 架构和权重。可直接加载字节跳动 **Protenix**
官方 checkpoint，CPU / Apple Silicon MPS / CUDA 都能跑。

</td></tr></table>

## 项目结构 · Project layout

照搬参考项目的三部分结构 · Mirrors the reference's three-fold structure:

```
build-af3-from-scratch/
├── lessons/      # 章节讲解 (markdown)            ·  per-chapter writeups
├── tutorials/    # 学生填空版 (auto-generated)     ·  fill-the-blank notebooks
└── solutions/    # 完整参考实现                    ·  complete reference impl
    ├── attention/                # MHA + LayerNorm + AttentionPairBias
    ├── feature_extraction/       # JSON / MSA / template → tensors  数据特征化
    ├── feature_embedding/        # 输入嵌入 + Atom Attention Encoder
    ├── pairformer/               # Pairformer + MSAModule + Triangle ops
    ├── diffusion/                # DiffusionModule + Transformer + sampler
    ├── confidence/               # ConfidenceHead + 置信度计算
    ├── model/                    # 顶层 Protenix 装配 + inference driver
    ├── configs/                  # 配置字典 + ConfigDict 解析  (shared)
    └── runtime/                  # seed / logger / torch utils  (shared)
```

每一章都是一个 *flat Python package*，内部文件之间通过
`from <chapter>.<file> import <Class>` 互相 import——和 AF2 参考完全一致。

Each chapter is a flat Python package; intra-chapter imports use the
fully-qualified form `from <chapter>.<file> import <Class>`, identical to
the AF2 reference.

## 为什么要做 AF3 版？ · Why an AF3 version?

AF3 在架构上有几个关键变化，没法直接复用 AF2 那一套章节：

AF3 introduces architectural changes that don't fit the AF2 mold:

| AF2 chapter           | AF3 equivalent                            |
|-----------------------|-------------------------------------------|
| `evoformer`           | `pairformer` (+ `msa_stack`)              |
| `structure_module`    | `diffusion`                               |
| `feature_embedding`   | + AtomAttentionEncoder                    |
| `geometry`            | 大部分融进 diffusion module · absorbed into diffusion |

中间几章因此和 AF2 不同，但底层 primitives（attention / residual /
axial ops）还是同一套。

## 快速开始 · Quickstart

### 1. 安装 · Install

Mac / Apple Silicon:

```bash
conda env create -f environment_mac.yml
conda activate af3
```

CPU only (Linux/Mac):

```bash
conda env create -f environment_cpu.yml
conda activate af3
```

或用 venv + pip:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision \
            rdkit biopython biotite modelcif gemmi pdbeccdutils \
            ml-collections scipy pandas scikit-learn scikit-learn-extra \
            matplotlib ipykernel ipywidgets py3dmol icecream fair-esm
```

### 2. 下载权重 · Download a checkpoint

We use the official Protenix-Tiny (~110 M parameters, MSA-based):

```bash
mkdir -p checkpoints
curl -L -o checkpoints/protenix_tiny_default_v0.5.0.pt \
    https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix_tiny_default_v0.5.0.pt
```

辅助缓存（CCD 化学组件等）会在首次推理时自动下到 `~/common/`。
Auxiliary caches (CCD chemistry, etc.) auto-download to `~/common/` on first run.

### 3. 运行推理 · Run inference

```bash
cd solutions
LAYERNORM_TYPE=torch python -m model.inference \
    --input_json /path/to/example.json \
    --dump_dir   ./out \
    --device     mps            # 或 cpu / cuda
```

每个样本会产出 `*.cif` 和 `*_summary_confidence_*.json`。
For each sample the runner produces a `*.cif` and a `*_summary_confidence_*.json`.

完整示例 JSON 见 `solutions/examples/example.json`（自带 7r6r 蛋白 + MSA）。
A complete example JSON is at `solutions/examples/example.json` (bundled 7r6r protein + MSA).

### 4. 验证 · Verify

200 残基左右的蛋白，在 M2 Max（CPU）上：pLDDT ≈ 74，单次前向 ≈ 4.3 s。
切到 MPS 后约快 1.6×。

## 章节如何拼起来 · How the chapters compose

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
       │  confidence  (Alg 26-31)                   │
       │  · ConfidenceHead + DistogramHead          │
       │  · pTM / iPTM / pLDDT / clash              │
       └────────────────────────────────────────────┘
```

`model/model.py` 把这些模块按 Algorithm 1 串起来。

## 特性 · Features

- Chapter-by-chapter Python package layout — every block lives in the file
  named after it.
  按章节拆分的 Python 包结构 —— 每个模块住在以它命名的文件里。
- Pure-PyTorch model: runs on CPU, Apple Silicon MPS, and CUDA.
  纯 PyTorch 实现，CPU / MPS / CUDA 都能跑。
- Loads the official ByteDance Protenix Tiny / Mini checkpoints out of the box.
  开箱即用加载字节跳动 Protenix Tiny / Mini 官方权重。
- Bilingual (English + 中文) docstrings on the public API.
  公开 API 全部带中英双语 docstring。

## 致谢 · Acknowledgements

- **ByteDance Protenix** — AF3 architecture implementation and open weights.
  字节跳动 Protenix —— AF3 架构实现 + 开源权重。
- **`alphafold-decoded`** by Kilian Mandon — the chapter-by-chapter
  pedagogical format.
  alphafold-decoded（Kilian Mandon）—— 按章节拆分的教学结构。
- **DeepMind** — AlphaFold 3 paper.

## License

Apache 2.0. See [LICENSE](LICENSE).
