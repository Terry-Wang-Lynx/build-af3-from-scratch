"""Generate per-chapter learning notebooks under ``solutions/<chapter>/``.

为每章生成学习引导 notebook，结构与 AF2 教学项目对齐:

    1. 基本环境 (cwd / sys.path / autoreload)
    2. 章节简介
    3. 每个模块一对 (指引 markdown + 调用 control_values 测试的 code cell)

Run from the repository root::

    python scripts/build_chapter_notebooks.py

This script is not used at runtime — it's only here so the notebooks are
reproducible. The committed ``.ipynb`` files are what students open.

脚本只负责生成 notebook，正式使用时学生打开 ``.ipynb`` 即可。
"""
from __future__ import annotations

import os
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOL = os.path.join(HERE, "solutions")


# ---------------------------------------------------------------------------
# Shared cell templates
# ---------------------------------------------------------------------------

def setup_cells(chapter: str) -> list:
    """Setup cells common to every chapter."""
    return [
        new_markdown_cell(
            "## 基本环境 · Basic setup\n\n"
            "首次打开运行下面 3 个 cell。它们做的事:\n"
            "1. 把工作目录切到所在的代码树根 (`solutions/` 或 `tutorials/`)，\n"
            "   这样 `from attention.mha import ...` 这种导入能直接生效。\n"
            "2. 启用 `autoreload`，编辑 .py 文件保存后 notebook 里立刻可用，不用重启 kernel。\n"
            "3. 设 `LAYERNORM_TYPE=torch`，避免 CUDA-only 算子的 import 失败。\n\n"
            "First time you open the notebook, run the 3 cells below: cd to the tree "
            "root (whichever of `solutions/` or `tutorials/` this notebook lives in), "
            "turn on autoreload, force the pure-PyTorch LayerNorm path."
        ),
        new_code_cell(
            "import os, sys\n"
            "\n"
            "# Walk up from the notebook's CWD until we find a directory named\n"
            "# `solutions` or `tutorials`. Works no matter which tree the student\n"
            "# opened. 不论 notebook 位于 solutions/ 还是 tutorials/ 都能正确定位。\n"
            "ROOTS = {'solutions', 'tutorials'}\n"
            "if os.path.basename(os.getcwd()) not in ROOTS:\n"
            "    while os.path.basename(os.getcwd()) not in ROOTS and os.getcwd() != '/':\n"
            "        os.chdir('..')\n"
            "    if os.path.basename(os.getcwd()) not in ROOTS:\n"
            "        # Fallback: maybe we were started at the repo root.\n"
            "        if os.path.isdir('tutorials'):\n"
            "            os.chdir('tutorials')\n"
            "        elif os.path.isdir('solutions'):\n"
            "            os.chdir('solutions')\n"
            "\n"
            "assert os.path.basename(os.getcwd()) in ROOTS, (\n"
            "    f'could not locate solutions/ or tutorials/ from {os.getcwd()}')\n"
            "if os.getcwd() not in sys.path:\n"
            "    sys.path.insert(0, os.getcwd())\n"
            "os.environ.setdefault('LAYERNORM_TYPE', 'torch')\n"
            "print('cwd =', os.getcwd())"
        ),
        new_code_cell(
            "%load_ext autoreload\n"
            "%autoreload 2"
        ),
        new_code_cell(
            "import torch\n"
            f"# Folder where the {chapter} chapter's reference .pt files live\n"
            f"control_folder = '{chapter}/control_values'\n"
            "assert os.path.isdir(control_folder), f'missing {control_folder}'"
        ),
    ]


def section_md(text: str) -> nbformat.NotebookNode:
    return new_markdown_cell(text)


def section_code(text: str) -> nbformat.NotebookNode:
    return new_code_cell(text)


# ---------------------------------------------------------------------------
# attention/attention.ipynb
# ---------------------------------------------------------------------------

def build_attention() -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.cells = setup_cells("attention")

    nb.cells += [
        section_md(
            "# 第 1 章 · Attention\n\n"
            "## 这一章你要学什么\n\n"
            "AlphaFold 3 整个模型可以看作"
            "**一堆 attention 块以不同方式排列组合**:\n\n"
            "- **Pairformer** 里有 *triangle attention* (沿一对残基的两条 starting/ending"
            "  轴) 与 *standard attention with pair bias*\n"
            "- **DiffusionTransformer** 里有 *AttentionPairBias* (用 pair 张量作为偏置)，"
            "  并按 token 级 / atom 级两种粒度运行\n"
            "- **ConfidenceHead** 内部又跑了一份小型 Pairformer + 四个分类头\n\n"
            "这些 attention 长得花样多，但**它们底下都在跑同一段 scaled dot-product 数学**: \n\n"
            "$$\\mathrm{Attention}(Q, K, V, b) = \\mathrm{softmax}\\Big(\\frac{QK^\\top}{\\sqrt{d}} + b\\Big)\\, V$$\n\n"
            "差别只是怎么算 $Q, K, V, b$、按哪些轴展开、要不要加 sigmoid 门控。本章把这套基础设施"
            "**从最底层的线性层一路搭到 AF3 标志性的 AttentionPairBias**，下面的"
            "Pairformer / Diffusion / Confidence 章节就只剩 \"用这些零件按论文 Algorithm 接线\"。\n\n"
            "## 本章模块速览\n\n"
            "| 文件 | 类 / 函数 | 角色 |\n"
            "|---|---|---|\n"
            "| `linear.py` | `Linear`, `LinearNoBias`, `BiasInitLinear` | 可选不同初始化策略的线性层 |\n"
            "| `layer_norm.py` | `OpenFoldLayerNorm` | 参数名与 Protenix 融合算子对齐的 LN |\n"
            "| `mha.py` | `_attention`, `Attention` | 缩放点积数学 + 多头封装 |\n"
            "| `transition.py` | `AdaptiveLayerNorm`, `Transition` | FiLM 风格调制 (Alg 26) + SwiGLU FFN (Alg 11) |\n"
            "| `attention_pair_bias.py` | `AttentionPairBias` | AF3 算法 24，主干最高频的复合块 |\n\n"
            "## 测试是怎么工作的\n\n"
            "每个 cell 调用 `test_module_shape` / `test_module_forward` 之前，"
            "测试 harness 会把你的模块参数**暂时**替换成 `torch.linspace(-1, 1, numel)`。这样:\n\n"
            "- **数值是确定的**：参考解和你的实现见到同样的权重 → 任何对的实现都得到同样的输出。\n"
            "- **不依赖训练好的权重**：每章测试都能离线、几毫秒内跑完。\n"
            "- **形状对了不代表实现对了**：测试会先比 shape (`*_param_shapes.pt`)，再比"
            "  forward 输出 (`torch.allclose` 到 `atol=1e-6`)。\n\n"
            "如果你跑测试得到 `output 'out' is None` 这种友好错误，说明 forward 还停留在 `pass` 没填；"
            "如果是 `appears to be uninitialized`，说明 `__init__` 没调 `super().__init__()`。\n\n"
            "## 论文 + 实现的对照\n\n"
            "本章每个组件都和 AF3 Supplementary 的某条 Algorithm 一一对应:\n\n"
            "| 组件 | 论文 Alg | 论文页 (Abramson et al. 2024) |\n"
            "|---|---|---|\n"
            "| Linear / LinearNoBias / BiasInitLinear | (实现细节，非算法) | — |\n"
            "| LayerNorm | (实现细节) | — |\n"
            "| `_attention` 数学 | 见 attention is all you need | 经典 |\n"
            "| `Attention` 模块 | (类于 OpenFold/AF2) | 经典 |\n"
            "| `AdaptiveLayerNorm` | **Algorithm 26** | Suppl. p27 |\n"
            "| `Transition` (SwiGLU) | **Algorithm 11** | Suppl. p21 |\n"
            "| `AttentionPairBias` | **Algorithm 24** | Suppl. p27 |\n\n"
            "Algorithm 编号在每节标题里都写明了。建议把 AF3 Supplementary PDF 打开放在一边对照看。"
        ),

        # --- Linear ---
        section_md(
            "## 1.1 Linear (自定义初始化的线性层)\n\n"
            "`torch.nn.Linear` 把一切都用默认 Kaiming 初始化。AF3 不接受这个 ——"
            "它对**不同角色的线性层用不同初始化**。\n\n"
            "### 为什么初始化选择重要 — 一点信号传播分析\n\n"
            "考虑一层线性变换 $y = W x$，输入 $x \\in \\mathbb{R}^{n_\\text{in}}$ 服从均值 0 方差 1 的分布、"
            "$W_{ij}$ 独立同分布且均值 0。则:\n\n"
            "$$\\mathrm{Var}(y_j) = \\sum_i W_{ij}^2 \\, \\mathrm{Var}(x_i) = n_\\text{in} \\cdot \\mathrm{Var}(W) \\cdot 1$$\n\n"
            "要让前向方差**保持为 1**，需要 $\\mathrm{Var}(W) = 1 / n_\\text{in}$ —— 这就是 **fan-in 初始化**。\n\n"
            "如果 W 之后紧跟 ReLU，**激活会砍掉一半**，方差减半，于是要把 $\\mathrm{Var}(W)$ 加倍 → "
            "He 初始化 (scale=2)。SiLU/GELU 砍得没有 ReLU 那么硬，但实践中也用 He 风格。\n\n"
            "### AF3 的四种策略\n\n"
            "| `initializer` | 用于哪里 | 数学 | 直观 |\n"
            "|---|---|---|---|\n"
            "| `\"default\"` | 普通线性层 | $\\mathrm{Var}(W) = 1/n_\\text{in}$, 截断正态 | 保前向方差 |\n"
            "| `\"relu\"` | ReLU/SiLU 前的层 (Transition 的 a/b 分支) | $\\mathrm{Var}(W) = 2/n_\\text{in}$ | 补偿激活方差损失 |\n"
            "| `\"zeros\"` | 残差块的最末投影 | $W = 0$ | 起手贡献 0，残差恒等 |\n"
            "| (隐含) | sigmoid 门控源 | 截断正态 (用于 OpenFold-style \"gating\" init) | sigmoid(z) 起手 ≈ 0.5 |\n\n"
            "`bias` 一律零初始化 (即使 weight 非零)。\n\n"
            "### Linear.forward 的两条路径\n\n"
            "AF3 主干一般跑 bf16 节省显存，但有些位置 (坐标投影、噪声水平条件) **必须 fp32**:\n\n"
            "- bf16 mantissa 只有 7 bit，坐标差只有几埃米时无法分辨 0.01 Å 的细节 → 配位错位\n"
            "- 扩散噪声水平 `t / sigma_data` 在 σ 接近 0 时是个很小的数，bf16 直接归零 → 网络永远看不到清晰信号\n\n"
            "所以 `Linear` 提供 `precision` 参数: 给定时就 disable autocast + 临时把 input/weight/bias 上提到该 dtype，"
            "算完再回到原 dtype。这样混合精度训练时只有这几条危险路径走 fp32，其余照样 bf16，速度 / 精度兼得。\n\n"
            "**任务**: 打开 `attention/linear.py` 把 `Linear._init_params` 和 "
            "`Linear.forward` 两个 TODO 块填好。每个 TODO 上方都有详细伪代码 + "
            "中英双语说明。这一格只测 `default` 策略 + 普通 forward 路径，"
            "1.3 节会单独检查 `BiasInitLinear` 的常数偏置初始化。"
        ),
        section_code(
            "from attention.linear import Linear\n"
            "from attention.control_values.attention_checks import (\n"
            "    c_a, c_z, test_module_shape, test_module_forward, test_inputs,\n"
            ")\n\n"
            "lin = Linear(in_features=c_a, out_features=c_z)\n"
            "test_module_shape(lin, 'linear_default', control_folder)\n"
            "test_module_forward(lin, 'linear_default',\n"
            "                    inputs=(test_inputs['x_a'],),\n"
            "                    output_names='out',\n"
            "                    control_folder=control_folder)\n"
            "print('Linear (default init) ✓')"
        ),

        # --- LinearNoBias ---
        section_md(
            "## 1.2 LinearNoBias (无 bias 的 Linear)\n\n"
            "Pairformer / Diffusion 主干里**大部分线性层都不带 bias** —— LayerNorm 已经"
            "提供了平移，再叠 bias 多余而且会让某些初始化策略 (如零初始化) 出现"
            "对称性问题。所以仓库里你会看到很多 `LinearNoBias` 而不是 `Linear`。\n\n"
            "这里没有新代码: `LinearNoBias = partial(Linear, bias=False)` —— 它就是"
            "`Linear` 固定 `bias=False`。**只要 1.1 你的 `Linear` 写对了，这格自动通过**。"
            "我们单独跑测试既是冗余检查，也提醒你「无 bias 的 Linear」是 AF3 默认形态。"
        ),
        section_code(
            "from attention.linear import LinearNoBias\n\n"
            "lin_nb = LinearNoBias(in_features=c_a, out_features=c_z)\n"
            "test_module_shape(lin_nb, 'linear_nobias', control_folder)\n"
            "test_module_forward(lin_nb, 'linear_nobias',\n"
            "                    inputs=(test_inputs['x_a'],),\n"
            "                    output_names='out',\n"
            "                    control_folder=control_folder)\n"
            "print('LinearNoBias ✓')"
        ),

        # --- BiasInitLinear ---
        section_md(
            "## 1.3 BiasInitLinear (起手输出常数的 Linear)\n\n"
            "AdaLN-Zero ([Peebles & Xie 2023, DiT](https://arxiv.org/abs/2212.09748)) 是 AF3"
            "**深层 Transformer 能稳训练**的关键技巧之一。它要求 attention / FFN 的"
            "输出门是一个 sigmoid，并且**初始几乎关闭**，让每个 block 起手对残差是近零贡献:\n\n"
            "$$x \\leftarrow x + \\sigma(W_s s + b_s) \\cdot \\mathrm{Block}(x)$$\n\n"
            "### 为什么不能让门初始在 0.5\n\n"
            "如果让 $W_s$ / $b_s$ 都走默认初始化，sigmoid 在 0.5 附近 —— 每个 block 都给残差"
            "贡献约 0.5·Block(x)，于是连续 L 层后 $x$ 的方差 ≈ $L \\cdot 0.25 \\cdot \\mathrm{Var}(\\text{Block})$。"
            "AF3 主干 48 个 PairformerBlock + 24 个 DiffusionTransformerBlock，"
            "如果每个都贡献 0.5 倍的 noise，**激活方差会指数级爆炸**，需要极小的 learning rate 才能稳。\n\n"
            "### 为什么是 biasinit = -2\n\n"
            "$$\\sigma(-2) = \\frac{1}{1 + e^2} \\approx 0.119$$\n\n"
            "也就是说 block 起手只贡献 12% 而非 50% 的 update。L=48 层后总方差只放大到约"
            "$48 \\cdot 0.119^2 \\cdot \\mathrm{Var}(\\text{Block}) \\approx 0.68$，远远稳定 ——"
            "网络能从这个状态出发，慢慢学习\"该让哪些 block 开门\"。\n\n"
            "AF3 用 -2，DiT 原论文用 -3 (更保守)。在 AF3 这个量级足够稳定，也保留了\"门能向上学到 0.5+\"的余量。\n\n"
            "### 为什么 weight 还要零初始化\n\n"
            "光让 bias = -2 让 sigmoid 起手在 0.12 是不够的 —— 如果 $W_s$ 非零，那么 sigmoid 的输入"
            "就成了 $W_s s - 2$，**取决于 $s$ 的实际值**: $s$ 大时门可能仍开得很大。零初始化让"
            "**门完全只由 bias 控制**: $\\sigma(b) = \\sigma(-2)$ 与 $s$ 无关。训练才学到怎么"
            "让 $W_s$ 起作用。\n\n"
            "### 代码 3 步\n\n"
            "`BiasInitLinear` 就是承担这个角色的 Linear 变体: 继承 `Linear`，覆盖 `__init__`:\n\n"
            "1. `super().__init__(...)` — 走父类的 Kaiming 初始化\n"
            "2. `nn.init.zeros_(self.weight)` — 覆盖 weight 为 0\n"
            "3. `nn.init.constant_(self.bias, biasinit)` — bias 设为 -2 (默认) 或调用方传入\n\n"
            "**任务**: 在 `attention/linear.py` 填 `BiasInitLinear.__init__` 的 TODO 块。"
        ),
        section_code(
            "from attention.linear import BiasInitLinear\n"
            "from attention.control_values.attention_checks import c_s\n\n"
            "bil = BiasInitLinear(in_features=c_s, out_features=c_a,\n"
            "                     bias=True, biasinit=-2.0)\n"
            "test_module_shape(bil, 'bias_init_linear', control_folder)\n"
            "test_module_forward(bil, 'bias_init_linear',\n"
            "                    inputs=(test_inputs['x_s'],),\n"
            "                    output_names='out',\n"
            "                    control_folder=control_folder)\n"
            "print('BiasInitLinear ✓')"
        ),

        # --- LayerNorm ---
        section_md(
            "## 1.4 LayerNorm (可选 scale / offset)\n\n"
            "PyTorch 自带的 `nn.LayerNorm` 给定 `c_in` 后默认带 scale ($\\gamma$) 和 offset"
            "($\\beta$):\n\n"
            "$$\\mathrm{LN}(x)_i = \\gamma_i \\cdot \\frac{x_i - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} + \\beta_i$$\n\n"
            "但 AF3 有两个地方需要**关闭** scale 或 offset:\n\n"
            "1. **AdaLN-Zero (Algorithm 26)** 内部对 `a` 做 LN 时**两个都关**: scale / offset 改由"
            "   `s` 经过线性层生成 —— 标准的 FiLM 调制。如果再带原生 $\\gamma$ / $\\beta$，"
            "   两套调制混在一起容易学到 trivial 解。\n"
            "2. **DiffusionConditioning 里的 z / s 归一化**只关 offset，保留 scale。\n\n"
            "所以我们自己写一个 `OpenFoldLayerNorm`，构造时通过 `create_scale` / `create_offset`"
            "决定要不要建可学参数。**关键**: 关掉某个参数时，仍要在 `state_dict` 里给它留"
            "一个 `None` 位 (`register_parameter(name, None)`)，否则 Protenix 权重加载会失败。\n\n"
            "另外 forward 里给 **bf16** 输入做了个特殊路径: autocast off + 临时把 weight / bias"
            "降到 bf16，这样得到的输出与上游融合算子位级一致。\n\n"
            "**任务**: 打开 `attention/layer_norm.py` 填两个 TODO 块。"
        ),
        section_code(
            "from pairformer.triangle_ops import LayerNorm\n\n"
            "ln = LayerNorm(c_a)\n"
            "test_module_shape(ln, 'layer_norm', control_folder)\n"
            "test_module_forward(ln, 'layer_norm',\n"
            "                    inputs=(test_inputs['x_a'],),\n"
            "                    output_names='out',\n"
            "                    control_folder=control_folder)\n"
            "print('LayerNorm ✓')"
        ),

        # --- _attention ---
        section_md(
            "## 1.5 `_attention` (核心点积数学)\n\n"
            "整个 AF3 attention 体系都在调一个无可学参数的纯函数:\n\n"
            "$$\\mathrm{out} = \\mathrm{softmax}\\Big(\\frac{Q K^\\top}{\\sqrt{d}} + b\\Big) \\, V$$\n\n"
            "### 为什么是 $1/\\sqrt{d}$ 而不是 $1/d$ 或 $1$\n\n"
            "假设 $Q, K$ 各分量独立同分布、均值 0 方差 1。点积 $Q_i \\cdot K_j = \\sum_k Q_{ik} K_{jk}$ 是"
            "$d$ 个独立项之和，因此:\n\n"
            "$$\\mathbb{E}[Q_i \\cdot K_j] = 0, \\quad \\mathrm{Var}(Q_i \\cdot K_j) = d$$\n\n"
            "注意标准差是 $\\sqrt{d}$。如果不缩放、d 大 (例如 64)，attention logits 的方差就是 64，"
            "softmax 输入有相当部分 $|z| > 8$；这导致 softmax 输出近似 one-hot，**几乎所有梯度都死掉**:\n\n"
            "$$\\frac{\\partial \\, \\mathrm{softmax}_i}{\\partial z_j} = \\mathrm{softmax}_i \\cdot (\\delta_{ij} - \\mathrm{softmax}_j)$$\n\n"
            "当 $\\mathrm{softmax}_i \\to 1$ 或 $\\to 0$，导数趋近 0 ——「饱和的 softmax」。\n\n"
            "缩放 $1/\\sqrt{d}$ 让 logits 方差回到 1，softmax 工作在线性区，梯度健康。"
            "这是 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) 第 3.2.1 节的核心观察。\n\n"
            "### 为什么我们提前缩放 Q 而不是在 `_attention` 内部缩放\n\n"
            "理论上等价: $\\frac{Q K^\\top}{\\sqrt{d}} = (\\frac{Q}{\\sqrt{d}}) K^\\top$。但实现上有差:\n\n"
            "- **提前缩放 Q (我们的做法)**: matmul 算 $\\tilde{Q} K^\\top$，传给 `F.scaled_dot_product_attention` 时 `scale=1.0`。\n"
            "- **运行时缩放**: matmul 完再除一次，多一次 elementwise op。CUDA 上不是瓶颈但 CPU 上能省 5-10%。\n\n"
            "在 `_attention(q, k, v, attn_bias, use_efficient_implementation, inplace_safe)` 中:\n\n"
            "- `q, k, v` 都已经被外面拆好头维 + 缩放，shape 是 `[..., H, T_q, d]` 和 "
            "  `[..., H, T_kv, d]`。**调用前 Q 已乘 1/sqrt(d)**，所以函数内部传 `scale=1.0`。\n"
            "- `attn_bias` 是 broadcastable 的 `[..., H, T_q, T_kv]`，包含 mask 和"
            "  pair / triangle bias 的总和。\n\n"
            "**两条路径**:\n\n"
            "1. `use_efficient_implementation=True`: 调 `F.scaled_dot_product_attention`，"
            "   PyTorch 在 CUDA 上能融合 + 走 FlashAttention，速度最快；缺点是要求 Q/K/V"
            "   dtype 一致 (我们测试时显式 disable 走显式路径)。\n"
            "2. 显式数学路径: `Q @ K^T + bias → softmax → · V`。在 `autocast(\"cuda\", enabled=False)`"
            "   下跑、Q/K 升 fp32 算 attention，再把 weight 强转回原 dtype 与 V 相乘 ——"
            "   避免 bf16 / fp16 下 softmax 精度灾难。\n\n"
            "**任务**: 在 `attention/mha.py` 文件顶部 (类 `Attention` 之前) 找到 "
            "`_attention` 函数，把整个函数体 (上面 TODO 块) 实现出来。"
            "这是后续所有 attention 模块共同调用的底座。"
        ),
        section_code(
            "from attention.mha import _attention\n\n"
            "out = _attention(\n"
            "    test_inputs['q_raw'].double(),\n"
            "    test_inputs['k_raw'].double(),\n"
            "    test_inputs['v_raw'].double(),\n"
            "    attn_bias=None,\n"
            "    use_efficient_implementation=False,\n"
            ")\n"
            "expected = torch.load(f'{control_folder}/attention_function_out.pt')\n"
            "assert torch.allclose(out, expected), '_attention output mismatch'\n"
            "print('_attention ✓')"
        ),

        # --- Attention (full MHA) ---
        section_md(
            "## 1.6 Attention (多头注意力模块)\n\n"
            "现在把 `_attention` 包成一个**带参数、带门控、能搬到 GPU 上跑批量数据**的 `nn.Module`。"
            "拆解成 3 个小方法 + 1 个 forward，更易读也好测:\n\n"
            "**`__init__`** 创建 5 个线性层 (命名要与 Protenix state_dict 严格一致):\n\n"
            "- `linear_q`, `linear_k`, `linear_v`: 把输入 `c_q` / `c_k` / `c_v` 各自投到"
            "  `H * c_hidden`。Q 是否带 bias 由 `q_linear_bias` 控制 (AF3 默认带)；K/V 不带。\n"
            "- `linear_o`: 输出投影 `H * c_hidden → c_q`，无 bias，**可选 zero-init** —— 当外层"
            "  没有 adaLN-Zero 门时 (`zero_init=True`) 让 attention block 起手为 0。\n"
            "- `linear_g` (可选): sigmoid 门控源，**zero-init** + `sigmoid` 一起作用让门起手在 0.5。\n\n"
            "**`_prep_qkv`** 做 4 件事 (按这个顺序最稳):\n\n"
            "1. 三个线性投影 → `[*, T, H * c_hidden]`\n"
            "2. `view(... + (H, c_hidden))` 拆出头维 → `[*, T, H, c_hidden]`\n"
            "3. `transpose(-2, -3)` 把头维提前 → `[*, H, T, c_hidden]` (这样 `_attention` 沿 T 求和)\n"
            "4. **预先**对 Q 乘 `1/sqrt(c_hidden)` —— 这样 `_attention` 内部 `scale=1.0`\n\n"
            "**`_wrap_up`**: 三步 —— 可选门控、展平头维、`linear_o` 投回 `c_q`。\n\n"
            "**`forward`** 是调度逻辑: 给了 `n_queries`/`n_keys` 就走局部窗口路径"
            "(`global_attention_with_bias` 或 `local_cross_attention`)，否则走全连接路径。"
            "局部 attention 是 AtomTransformer 用的 —— 让原子级注意力 O(N) 而非 O(N²)。\n\n"
            "**任务**: 把 `Attention.__init__` / `_prep_qkv` / `_wrap_up` / `forward` 四个 TODO 全部填好。"
        ),
        section_code(
            "from attention.mha import Attention\n"
            "from attention.control_values.attention_checks import N_head, c_hidden\n\n"
            "attn = Attention(\n"
            "    c_q=c_a, c_k=c_a, c_v=c_a,\n"
            "    c_hidden=c_hidden, num_heads=N_head,\n"
            "    gating=True, q_linear_bias=True,\n"
            "    use_efficient_implementation=False,\n"
            "    zero_init=False,\n"
            ")\n"
            "test_module_shape(attn, 'mha_gated', control_folder)\n\n"
            "from attention.control_values.attention_checks import test_module_method\n"
            "test_module_method(\n"
            "    attn, 'mha_gated',\n"
            "    inputs=(test_inputs['q_x'], test_inputs['kv_x'], test_inputs['attn_bias']),\n"
            "    output_names='out',\n"
            "    control_folder=control_folder,\n"
            "    method=lambda q_x, kv_x, b: attn(q_x=q_x, kv_x=kv_x, attn_bias=b),\n"
            ")\n"
            "print('Attention ✓')"
        ),

        # --- AdaptiveLayerNorm + Transition ---
        section_md(
            "## 1.7 AdaptiveLayerNorm + Transition\n\n"
            "两个看上去无关、但都在 `attention/transition.py` 的高频组件:\n\n"
            "### `AdaptiveLayerNorm` (Algorithm 26) — FiLM 的演化版\n\n"
            "FiLM (Feature-wise Linear Modulation, [Perez et al. 2017](https://arxiv.org/abs/1709.07871))"
            "提出了一个简单想法: **给一个条件 c，用它生成 affine 变换 (γ, β)，按 channel 改主流 x**:\n\n"
            "$$\\mathrm{FiLM}(x, c) = \\gamma(c) \\odot x + \\beta(c)$$\n\n"
            "原版 FiLM 直接乘 γ。AF3 / DiT 把 γ 换成 sigmoid 之后再乘 —— 这是 AdaLN-Zero 的关键变体:\n\n"
            "$$\\mathrm{AdaLN}(a, s) = \\sigma(W_g \\, \\mathrm{LN}_s(s)) \\cdot \\mathrm{LN}_a(a) + W_b \\, \\mathrm{LN}_s(s)$$\n\n"
            "为什么用 sigmoid 而不是直接的 γ?\n\n"
            "- γ 没有上下界，初始一旦偏大会瞬间放大主流方差。sigmoid 自然 clamp 到 (0, 1)，方差不会爆炸。\n"
            "- 配合 zero-init 让 sigmoid 起手在 0.5：主流被「半衰」缩放、shift 完全 0 —— 起手等价于"
            "  $a \\leftarrow 0.5 \\cdot \\mathrm{LN}(a)$，深层堆叠下不爆炸 (参 1.3 节的方差分析)。\n\n"
            "**关键设计**:\n\n"
            "- `LN_a` **关掉 scale + offset**: 调制完全由 s 控制。如果再叠原生 γ/β，参数化冗余、训练歧义。\n"
            "- `LN_s` **保留 scale**: 让 s 进入门控前先做一次稳定归一化。\n"
            "- `linear_s` (sigmoid 门源) + `linear_nobias_s` (shift 源) 都 **zero-init**: 起手只有"
            "  sigmoid 的 0.5 倍主流，shift 严格为 0。\n\n"
            "### `Transition` (Algorithm 11) — 为什么 SwiGLU 而不是普通 MLP\n\n"
            "Vanilla Transformer 的 FFN 是 $W_2 \\,\\text{GELU}(W_1 x)$。AF3 / Gemini / Llama 普遍换成 SwiGLU"
            "([Shazeer 2020](https://arxiv.org/abs/2002.05202)):\n\n"
            "$$\\mathrm{Transition}(x) = W_o \\, (\\mathrm{SiLU}(W_a \\, \\mathrm{LN}(x)) \\odot W_b \\, \\mathrm{LN}(x))$$\n\n"
            "区别在中间多一路 \"value\" 分支 $W_b x$，与 SiLU 门做 element-wise 乘:\n\n"
            "- **更强表达**: GLU 风格门控让 FFN 学到「按通道选择性放大 / 关闭」，普通 MLP 做不到。\n"
            "- **同算力下性能更好**: 论文实测同 FLOP 比 GELU MLP 在 LM 任务低 ~0.4 PPL。\n"
            "- **代价**: 参数量从 $2 \\cdot c \\cdot n c$ 变成 $3 \\cdot c \\cdot n c$。AF3 把 n 从 4 调到 2/4，"
            "  保持总参数量接近的同时享受 SwiGLU 的能力。\n\n"
            "**实现要点**:\n\n"
            "- 两路扩宽线性层 `linear_no_bias_a` / `linear_no_bias_b` 都用 `\"relu\"` 初始化"
            "  (fan-in 截断正态 scale=2)，因为后面跟 SiLU/逐元素乘。\n"
            "- 输出投影 `linear_no_bias` 用 `\"zeros\"` 初始化，残差起手为 0。\n"
            "- `n` 通常 2 或 4: AF3 主干 `c_a=384` 时 transition 隐藏维到 1536 (n=4)；\n"
            "  diffusion 的 `ConditionedTransitionBlock` 用 n=2 节省 FLOPs。\n\n"
            "**任务**: 在 `attention/transition.py` 填:\n\n"
            "- `AdaptiveLayerNorm.__init__` (4 个子模块) + `forward` (LN + 调制公式)\n"
            "- `Transition.__init__` (LN + 3 linears) + `forward` (LN + SwiGLU + out)"
        ),
        section_code(
            "from attention.transition import AdaptiveLayerNorm, Transition\n"
            "from attention.control_values.attention_checks import n_factor\n\n"
            "adaln = AdaptiveLayerNorm(c_a=c_a, c_s=c_s)\n"
            "test_module_shape(adaln, 'adaptive_layer_norm', control_folder)\n"
            "test_module_forward(adaln, 'adaptive_layer_norm',\n"
            "                    inputs=(test_inputs['x_a'], test_inputs['x_s']),\n"
            "                    output_names='out',\n"
            "                    control_folder=control_folder)\n\n"
            "tr = Transition(c_in=c_a, n=n_factor)\n"
            "test_module_shape(tr, 'transition', control_folder)\n"
            "test_module_forward(tr, 'transition',\n"
            "                    inputs=(test_inputs['x_a'],),\n"
            "                    output_names='out',\n"
            "                    control_folder=control_folder)\n"
            "print('AdaLN + Transition ✓')"
        ),

        # --- AttentionPairBias ---
        section_md(
            "## 1.8 AttentionPairBias (AF3 算法 24)\n\n"
            "本章的**顶点**。这一个组件出现在:\n\n"
            "- **PairformerBlock** 的单序列 update (用 pair `z` 作为偏置 bias 调整每对 token 的注意力)\n"
            "- **DiffusionTransformerBlock** 的核心 attention 分支 (`has_s=True` 启用 AdaLN-Zero 门)\n"
            "- **AtomTransformer** 的局部窗口版 (走 `local_multihead_attention` 路径)\n"
            "- **ConfidenceHead** 内部的小 Pairformer\n\n"
            "把这块写对 = AF3 主干基本就接通了。\n\n"
            "### Forward 流程\n\n"
            "```text\n"
            "    a ─── AdaLN(s) or LN ──┐\n"
            "                            ├── attention(q=a, kv=a, bias = Linear(LN(z)))\n"
            "                            │       │\n"
            "                            │       └── 局部窗口路径 (n_queries/n_keys 给了时)\n"
            "                            │           或全连接路径\n"
            "                            │\n"
            "                            └── × sigmoid(linear_a_last(s))    ← adaLN-Zero output gate\n"
            "                                       (起手 ≈ sigmoid(-2) = 0.12，初闭)\n"
            "    最后返回累计 update，由调用方做残差加。\n"
            "```\n\n"
            "### Pair bias 怎么来\n\n"
            "$$b_{ij}^h = (\\mathrm{Linear}_{c_z \\to H} \\, \\mathrm{LN}(z))_{ij}^h$$\n\n"
            "投出的 H 通道恰好作为多头注意力的每头 bias，加到 $Q K^\\top$ 上。"
            "几何意义: 让 token i 在关注 token j 时，看 pair 表示 $z_{ij}$ 决定多 / 少关注。\n\n"
            "### `local_*` vs `standard_*`\n\n"
            "- `standard_multihead_attention(q, kv, z)`: pair `z` 是 `[..., N, N, c_z]`，"
            "  PairformerBlock 用这条。\n"
            "- `local_multihead_attention(q, kv, z, n_queries, n_keys)`: pair `z` 已经被"
            "  rearrange 成 `[..., n_blocks, n_queries, n_keys, c_z]` (dense-trunk 形)，"
            "  AtomTransformer 用这条避免 O(N²) 显存。\n\n"
            "**任务**: 在 `attention/attention_pair_bias.py` 填四个 TODO ——"
            "`__init__`、两个 multihead 辅助、和顶层 `forward`。"
        ),
        section_code(
            "from attention.attention_pair_bias import AttentionPairBias\n\n"
            "apb = AttentionPairBias(\n"
            "    has_s=True, create_offset_ln_z=False,\n"
            "    n_heads=N_head, c_a=c_a, c_s=c_s, c_z=c_z,\n"
            "    biasinit=-2.0, cross_attention_mode=False,\n"
            ")\n"
            "# Disable the SDP fast path so the test runs in double precision.\n"
            "apb.attention.use_efficient_implementation = False\n\n"
            "test_module_shape(apb, 'attention_pair_bias', control_folder)\n"
            "test_module_method(\n"
            "    apb, 'attention_pair_bias',\n"
            "    inputs=(test_inputs['x_a'], test_inputs['x_s'], test_inputs['x_z']),\n"
            "    output_names='out',\n"
            "    control_folder=control_folder,\n"
            "    method=lambda a, s, z: apb(a=a, s=s, z=z),\n"
            ")\n"
            "print('AttentionPairBias ✓')"
        ),

        section_md(
            "## 章节小结\n\n"
            "完成本章后，你已经手写了:\n\n"
            "1. **`Linear` / `LinearNoBias` / `BiasInitLinear`** —— 带初始化策略选择的线性层 + adaLN-Zero 门的特殊变体；\n"
            "2. **`OpenFoldLayerNorm`** —— 可选 scale / offset 的 LN，bf16 走融合算子兼容路径；\n"
            "3. **`_attention`** —— 缩放点积 + softmax + 加权和的纯函数，两条 dtype 策略；\n"
            "4. **`Attention`** —— 包成模块的多头注意力，含 5 个线性层、可选 sigmoid 门、局部 / 全连接两条 forward 路径；\n"
            "5. **`AdaptiveLayerNorm`** —— FiLM 风格的条件归一化 (Algorithm 26)；\n"
            "6. **`Transition`** —— SwiGLU FFN (Algorithm 11)；\n"
            "7. **`AttentionPairBias`** —— AF3 主干最高频的复合块 (Algorithm 24)。\n\n"
            "**下一站**: 这些零件马上会在第 2 章 Pairformer 里被反复组合 ——"
            "你会发现 TriangleAttention 内部用的 mha、PairformerBlock 用的 AttentionPairBias、"
            "OuterProductMean 用的 LayerNorm，全是本章交付的成品。"
        ),
    ]
    return nb


# ---------------------------------------------------------------------------
# pairformer/pairformer.ipynb
# ---------------------------------------------------------------------------

def build_pairformer() -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.cells = setup_cells("pairformer")

    nb.cells += [
        section_md(
            "# 第 2 章 · Pairformer\n\n"
            "## 为什么需要 Pairformer\n\n"
            "结构预测最关键的问题: **判断哪两个残基会在 3D 中接近**。如果模型只对单序列做"
            "Transformer，永远拿不到 pair (i, j) 的几何先验。AF2 用 Evoformer 同时跑"
            "MSA + pair 两个表示，反复交换信息；**AF3 简化为 Pairformer**: pair 表示是主角，"
            "MSA 表示只在 MSAModule 里短暂出现，把信息汇入 pair 后就消失。\n\n"
            "Pairformer 让 $z_{ij}$ 满足结构上的"
            "[三角不等式约束](https://en.wikipedia.org/wiki/Triangle_inequality):\n\n"
            "$$\\forall i, j, k: \\quad |z_{ij}| \\le |z_{ik}| + |z_{kj}|$$\n\n"
            "实现这点的关键是**让 (i, j) 的更新明确依赖第三个 token k** —— 这就是"
            "「三角」一词的来源。具体有 4 种三角算子，本章逐个手写。\n\n"
            "## 本章模块\n\n"
            "| 文件 | 类 | 算法 | 简述 |\n"
            "|---|---|---|---|\n"
            "| `triangle_ops.py` | `OuterProductMean` | Alg 10 | MSA → pair |\n"
            "| `triangle.py` | `TriangleMultiplication{Outgoing,Incoming}` | Alg 11 / 12 | pair 上的外积式更新 |\n"
            "| `triangle.py` | `TriangleAttention` | Alg 13 / 14 | pair 上沿一条 residue 轴的注意力 |\n"
            "| `msa_stack.py` | `MSAPairWeightedAveraging` | (MSAModule 内) | pair → MSA 反向通道 |\n"
            "| `pair_stack.py` | `PairformerBlock` | Alg 17 | 把以上 4 个 + pair_transition + 可选 single update 串起来 |\n\n"
            "## 前置\n\n"
            "本章会反复用到第 1 章实现的 `Linear` / `LinearNoBias` / `LayerNorm` / `Transition` /"
            "`AttentionPairBias`。如果第 1 章测试还没全绿，回去补完。\n\n"
            "## 三角不等式的形式定义\n\n"
            "本章常常说\"满足三角不等式\"，准确的定义:\n\n"
            "对任意三点 i, j, k 在度量空间内 (real distance):\n\n"
            "$$d(i, j) \\le d(i, k) + d(k, j)$$\n\n"
            "如果 Pairformer 预测的 pair 张量学到的 $z_{ij}$ 完全独立于其他 token 的存在，**没人保证"
            "这个不等式被遵守** —— 例如可能预测出 $z_{12} = 100\\text{Å}$ 但 $z_{13} = z_{32} = 1\\text{Å}$，"
            "在物理上不可能。\n\n"
            "AF2/AF3 的三角更新算子让 $z_{ij}$ 的更新**显式依赖每个第三个 token k**"
            "(\"绕一圈\"路径)，把这条软约束注入到模型。论文实验显示移掉所有 triangle update"
            "在 CASP14 上 GDT-TS 下降 ~5 分 (AF2 ablation)。"
        ),

        section_md(
            "## 2.1 OuterProductMean (Algorithm 10) — MSA → pair\n\n"
            "### 共进化:经典的结构预测信号\n\n"
            "蛋白序列在演化中很多突变发生 —— 但**有的位置不能独立突变**，因为它们要"
            "维持物理接触: 如果残基 i 是疏水的、突变成亲水，残基 j 也必须协同突变才能维持折叠。\n\n"
            "如果对一个蛋白找几千条同源序列做对齐 (MSA)，**统计哪些位置 (i, j) 突变模式相关**，"
            "就能找出物理接触对。这条线索 (direct coupling analysis, DCA / Potts model)"
            "早于深度学习几十年，是 AlphaFold 1/2 的核心信号。\n\n"
            "### OuterProductMean 用张量代数实现共进化统计\n\n"
            "经典 DCA: 对每对 (i, j)，统计 MSA 中残基类型 (a, b) 共现频率矩阵 $f_{ij}(a, b)$ —— "
            "这是个 21×21 矩阵 (20 种氨基酸 + gap)。AF 把它推广到任意学习到的特征:\n\n"
            "$$z_{ij} \\;+\\!\\!= \\;W_o \\, \\mathrm{flat}\\Big(\\frac{1}{N_{\\text{eff}}(i,j)} \\sum_{s=1}^{N_{\\text{msa}}} \\mathrm{mask}_{si}\\mathrm{mask}_{sj}\\;a_{si} \\otimes b_{sj}\\Big)$$\n\n"
            "其中 $a, b$ 是 MSA 张量经 LayerNorm + 两路线性 (`linear_1` / `linear_2`) 投到"
            "`c_hidden` 后的产物。**外积**把两个 (c_hidden,) 向量变成 (c_hidden × c_hidden) 矩阵 ——"
            "捕获每对位置 (i, j) 在通道维上的所有交互组合。然后沿 MSA 维 (s) 加权平均，flatten"
            "再投回 `c_z`。\n\n"
            "**这是经典 21×21 共现矩阵的稠密学习版本**:\n\n"
            "- 经典版本: $f_{ij}(a, b)$ —— 离散类别，组合维度 21×21 = 441\n"
            "- AF3 版本: $\\frac{1}{N_\\text{eff}} \\sum_s a_{si} \\otimes b_{sj}$ —— 稠密 `c_hidden × c_hidden`，"
            "  让模型自主学到「应该提取哪些维度」\n\n"
            "### einsum 实现技巧\n\n"
            "三维外积外加 MSA 维求和的爱因斯坦符号写法:\n\n"
            "```python\n"
            "outer = torch.einsum('bac, dae -> bdce', a_transposed, b_transposed)\n"
            "```\n\n"
            "(`a` 字母代表 MSA 维，会被自动 contract; `b/d` 是 i/j，`c/e` 是 c_hidden 双轴)。\n"
            "PyTorch 自动选择最优 contract 顺序，单条 einsum 完成什么本来要 3 步矩阵操作。\n\n"
            "### Mask 归一化\n\n"
            "$N_\\text{eff}(i, j) = \\sum_s \\mathrm{mask}_{si} \\mathrm{mask}_{sj}$ 是\"位置 (i, j) 上"
            "都有有效残基的序列数\"。MSA 里如果某条序列对应位置是 gap，那一行被 mask 屏蔽，"
            "不参与统计。加 ε 避免 N_eff = 0 (例如 i / j 至少一处全 gap)。\n\n"
            "### 输出投影零初始化\n\n"
            "`linear_out` 用 `init=\"final\"` 即零初始化 → 训练初期 OPM 对 pair 不贡献，"
            "让模型从「pair 张量只来自 RelativePositionEncoding」这种安全状态出发，"
            "慢慢学到\"什么时候从 MSA 拉信号\"。\n\n"
            "**任务**: 打开 `pairformer/triangle_ops.py` 填三处 TODO ——"
            "`OuterProductMean.__init__` (4 个子模块) / `_opm` (einsum + flatten + 投影) / `_forward` (整个流程)。"
        ),
        section_code(
            "from pairformer.triangle_ops import OuterProductMean\n"
            "from pairformer.control_values.pairformer_checks import (\n"
            "    c_m, c_z, c_hidden, no_heads_pair, test_inputs,\n"
            "    test_module_shape, test_module_method, test_module_forward,\n"
            ")\n\n"
            "opm = OuterProductMean(c_m=c_m, c_z=c_z, c_hidden=c_hidden)\n"
            "test_module_shape(opm, 'outer_product_mean', control_folder)\n"
            "test_module_method(\n"
            "    opm, 'outer_product_mean',\n"
            "    inputs=(test_inputs['m'], test_inputs['msa_mask']),\n"
            "    output_names='out',\n"
            "    control_folder=control_folder,\n"
            "    method=lambda m, mask: opm(m, mask=mask),\n"
            ")\n"
            "print('OuterProductMean ✓')"
        ),

        section_md(
            "## 2.2 TriangleMultiplication (Algorithms 11 & 12)\n\n"
            "Pair 表示 $z_{ij}$ 的「三角」更新里，最便宜的一种: 不跑注意力，直接做**带门控的外积**。\n\n"
            "### 几何直观\n\n"
            "想象 3 个 token i, j, k 之间存在一个三角形。$z_{ik}$ 与 $z_{jk}$ 都包含"
            "关于 k 的信息 —— 那么 $z_{ij}$ 也应该能通过\"绕一圈\"得到约束:\n\n"
            "- **Outgoing** (Alg 11): $z_{ij} \\,+\\!\\!=\\, \\sum_k a_{ik} \\odot b_{jk}$  \n"
            "  从 i 出发、k 是「中介」，每条三角形边 (i, k) 与 (j, k) 提供一对乘子。\n"
            "- **Incoming** (Alg 12): $z_{ij} \\,+\\!\\!=\\, \\sum_k a_{ki} \\odot b_{kj}$  \n"
            "  反过来，从 k 流入 i 和 j。\n\n"
            "Pairformer 一个 block 同时跑这两条，让 $z_{ij}$ 在 i / j 两个角色上都收到信号。\n\n"
            "### 为什么\"乘\"能模拟三角不等式约束\n\n"
            "考虑一个特例: 让 $a_{ik}, b_{jk}$ 都把 \"接触\" 编码成大正值，否则接近 0。则 outgoing 更新:\n\n"
            "$$z_{ij}^{\\text{new}} = z_{ij}^{\\text{old}} + \\sum_k a_{ik} \\, b_{jk}$$\n\n"
            "$z_{ij}^{\\text{new}}$ 只在「**存在某个 k 使得 (i, k) 和 (j, k) 都强**」时被显著放大。"
            "这正是\"i 通过 k 间接连接 j\"的拓扑信号。乘法 (而非加) 让 k 必须**同时**对 i 和 j 起作用 ——"
            "天然实现 AND 逻辑，比加法 (OR-like) 在传递约束上更精确。\n\n"
            "### 计算流程\n\n"
            "1. LN(z)  \n"
            "2. 每路两个线性 + sigmoid 门: $a = \\sigma(W_{ag} z) \\odot W_{ap} z$，同样得 $b$\n"
            "3. mask 一下 (`mask * a`, `mask * b`)\n"
            "4. **核心 trick** —— 把通道维提到最前 (`permute_final_dims`)，再 `torch.matmul`，"
            "   让 BLAS 帮你算 $\\sum_k$。Outgoing / Incoming 的差别在于 a / b 的 permute 方式不同。\n"
            "5. LN(combined) + 投回 c_z + sigmoid 门 (`linear_g`)\n\n"
            "### 矩阵乘 trick 的细节\n\n"
            "数学上 outgoing 是 $z'_{ij} = \\sum_k a_{ik} b_{jk}$。如果把通道维 (c_hidden) 也写出来:\n\n"
            "$$z'_{ijc} = \\sum_k a_{ikc} \\, b_{jkc}$$\n\n"
            "**关键观察**: 沿 k 求和的是 `a[i, k, c] * b[j, k, c]`，即两个张量在 (k, c) 位置都对齐"
            "再乘并 contract over k。如果把 c 维 permute 到最前面 (轴 0):\n\n"
            "```\n"
            "a_perm: [c, i, k]      # outgoing 的 a 通道在前、i 行在中、k 列在后\n"
            "b_perm: [c, j, k]      # 等价地，b 转置后 [c, k, j]\n"
            "```\n\n"
            "对每个 c 通道**独立**做一次 (i, k) @ (k, j) 矩阵乘，正好得到 (i, j) ——\n"
            "`torch.matmul([c, i, k], [c, k, j])` 一次性算完所有 c × N² 元素。\n\n"
            "**对 incoming**，求和是 $\\sum_k a_{ki} b_{kj}$，permute 把 k 提到中间维，"
            "matmul 沿 (i, k) @ (k, j) 即可。两个变种共享同一个 `_combine_projections` 函数，"
            "只是 a / b 的 permute 路径不同。\n\n"
            "**任务**: 在 `pairformer/triangle.py` 填:\n\n"
            "- `BaseTriangleMultiplicativeUpdate.__init__` — 7 个共享子模块 (4 个 LN/Linear + 2 LN + sigmoid)\n"
            "- `TriangleMultiplicativeUpdate.__init__` — 追加 4 个 a/b projection\n"
            "- `_combine_projections` — permute + matmul + permute 回来 (附带可选 inplace_chunk)\n"
            "- `forward` — 串起 LN → 两路门控投影 → combine → out projection → 输出门\n\n"
            "子类 `TriangleMultiplicationOutgoing` / `Incoming` 不需要写 —— 它们用 `partialmethod`"
            "固定 `_outgoing=True/False`。"
        ),
        section_code(
            "from pairformer.triangle import (\n"
            "    TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming,\n"
            ")\n\n"
            "for variant, cls in [\n"
            "    ('triangle_mul_out', TriangleMultiplicationOutgoing),\n"
            "    ('triangle_mul_in',  TriangleMultiplicationIncoming),\n"
            "]:\n"
            "    mod = cls(c_z=c_z, c_hidden=c_hidden)\n"
            "    test_module_shape(mod, variant, control_folder)\n"
            "    test_module_method(\n"
            "        mod, variant,\n"
            "        inputs=(test_inputs['z'], test_inputs['pair_mask']),\n"
            "        output_names='out',\n"
            "        control_folder=control_folder,\n"
            "        method=lambda z, pm, mod=mod: mod(z, mask=pm),\n"
            "    )\n"
            "print('TriangleMultiplication (outgoing + incoming) ✓')"
        ),

        section_md(
            "## 2.3 TriangleAttention (Algorithms 13 & 14)\n\n"
            "刚才的 TriangleMultiplication 只能学到\"哪两条三角形边相乘起来强\"。**TriangleAttention**"
            "更进一步: 让模型**主动选择**通过哪个第三个节点 k 来连接 (i, j)。\n\n"
            "把 pair 表示 $z_{ij}$ 想象成一张 NxN 的图。沿一行 (固定 i, 遍历 j) 跑多头注意力，"
            "**注意力分数本身又来自另一行的 pair 张量** —— 这就是「triangle」的味道。\n\n"
            "### 数学公式\n\n"
            "对 starting-node 变种 (Algorithm 13)，在每一行 (固定 i) 内部:\n\n"
            "$$z'_{ij} \\;+\\!\\!=\\; \\sum_{j'} \\mathrm{softmax}_{j'}\\Big(\\frac{Q_{ij} K_{ij'}^\\top}{\\sqrt{d}} + b_{j j'}\\Big) V_{ij'}$$\n\n"
            "**注意 bias** $b_{j j'}$ 而不是 $b_{ij j'}$ —— 这个 bias 来自另一行的 pair 张量,\n"
            "**编码 j 和 j' 之间的关系** (作为「triangle」第三边)。Linear 把 $z_{j j'}$ 投到 `no_heads` 维就行。\n\n"
            "Algorithm 14 (ending-node) 是同样的事，但 attention 沿**列** (固定 j) 跑，bias 用 $z_{ii'}$。\n\n"
            "### \"Around starting/ending node\" 命名的由来\n\n"
            "AF2 论文的图: 把 pair 张量 $z_{ij}$ 画成 N×N 网格，每个 cell 看成一条**有向边**"
            "i → j。\"Starting node\" = 边的起点 i；\"ending node\" = 终点 j。\n\n"
            "- 起点-中心注意力 (starting): 沿一行扫描所有从 i 出发的边，互相比较 / 重组。\n"
            "- 终点-中心注意力 (ending): 沿一列扫描所有进入 j 的边。\n\n"
            "两种变种合起来，每对 (i, j) 的边都收到来自共起点 OR 共终点的 (N-1) 条邻居信号 ——"
            "这正是「绕第三个 token 走一圈」的注意力版本。\n\n"
            "### 两种变种\n\n"
            "- **Around starting node** (Alg 13, `starting=True`): 在 z[i, :, :] 这一**行**里跑 attention，"
            "  每行内部 i 固定、j 是 query / key/value 序列。\n"
            "- **Around ending node** (Alg 14, `starting=False`): 在 z[:, j, :] 这一**列**里跑 attention。\n\n"
            "代码实现共用同一个类，**ending 版用 `transpose(-2, -3)` 把列翻成行就行**。\n\n"
            "### 实现细节\n\n"
            "- 通道维 LayerNorm。\n"
            "- 一个 `linear: c_in → no_heads`，把 pair 投成每头一个 bias 标量，再 `permute_final_dims`"
            "  把头维提前 + `unsqueeze(-4)` 给一个假 row 维，让它能广播加到 attention 分数上。\n"
            "- mask 走 `mask_bias = inf * (mask - 1)`，softmax 自然把无效位置压到 0。\n"
            "- `self.mha` 是 OpenFold 风格 `Attention` (与第 1 章 AF3 那个不一样!)，接受 `biases=[...]` 列表。\n\n"
            "**任务**: 填 `TriangleAttention.__init__` + `forward`。"
        ),
        section_code(
            "from pairformer.triangle import TriangleAttention\n\n"
            "tri_att = TriangleAttention(\n"
            "    c_in=c_z, c_hidden=c_hidden, no_heads=no_heads_pair, starting=True,\n"
            ")\n"
            "test_module_shape(tri_att, 'triangle_attention_start', control_folder)\n"
            "test_module_method(\n"
            "    tri_att, 'triangle_attention_start',\n"
            "    inputs=(test_inputs['z'], test_inputs['pair_mask']),\n"
            "    output_names='out',\n"
            "    control_folder=control_folder,\n"
            "    method=lambda z, pm: tri_att(z, mask=pm),\n"
            ")\n"
            "print('TriangleAttention ✓')"
        ),

        section_md(
            "## 2.4 MSAPairWeightedAveraging\n\n"
            "OuterProductMean 是 **MSA → pair** 通道；它的兄弟 **pair → MSA** 是这里的"
            "`MSAPairWeightedAveraging`。它出现在 MSAModule 内部，每个 block 跑一次:\n\n"
            "$$m'_{si} = \\sum_h g_{si}^h \\;\\sum_j \\mathrm{softmax}_j(b_{ij}^h) \\cdot v_{sj}^h$$\n\n"
            "解读:\n\n"
            "- $b_{ij}^h$ 来自 pair 张量经 LN + linear 投到每头一个标量；它告诉每个 MSA 序列"
            "  「在更新位置 i 时，应该多看位置 j」。\n"
            "- $v_{sj}^h$ 是 MSA 在位置 j 经线性投影得到的 value (跨 MSA 序列 s 共享 b 权重)。\n"
            "- 沿 j 加权平均，再用 sigmoid 门 $g_{si}^h$ 收尾 (zero-init，起手关闭)。\n\n"
            "**为什么重要**: 单跑 OuterProductMean MSA 信息只能流向 pair，再不回头。"
            "加上 MSAPairWeightedAveraging，pair 学到的关系能反过来精修 MSA，让 OPM 下一轮的输入更好。\n\n"
            "**任务**: 在 `pairformer/msa_stack.py` 填 `MSAPairWeightedAveraging.__init__` 和 `.forward`。"
        ),
        section_code(
            "from pairformer.msa_stack import MSAPairWeightedAveraging\n\n"
            "mpwa = MSAPairWeightedAveraging(c_m=c_m, c=c_hidden, c_z=c_z, n_heads=no_heads_pair)\n"
            "test_module_shape(mpwa, 'msa_pair_weighted_avg', control_folder)\n"
            "test_module_forward(\n"
            "    mpwa, 'msa_pair_weighted_avg',\n"
            "    inputs=(test_inputs['m'], test_inputs['z']),\n"
            "    output_names='out',\n"
            "    control_folder=control_folder,\n"
            ")\n"
            "print('MSAPairWeightedAveraging ✓')"
        ),

        section_md(
            "## 2.5 PairformerBlock (Algorithm 17 — 一整块)\n\n"
            "这一节把前面 4 个三角操作 + Transition + 单序列更新拼成 AF3 主干的最小重复单元。"
            "Algorithm 17 一个 block 的伪代码:\n\n"
            "```text\n"
            "  z += TriangleMultiplicationOutgoing(z)\n"
            "  z += TriangleMultiplicationIncoming(z)\n"
            "  z += TriangleAttention starting_node(z)\n"
            "  z += TriangleAttention ending_node(z)         ← 通过物理转置实现\n"
            "  z += pair_transition(z)\n"
            "  if c_s > 0:                                    ← 单序列分支\n"
            "      s += AttentionPairBias(a=s, s=None, z=z)\n"
            "      s += single_transition(s)\n"
            "```\n\n"
            "### 几个工程细节\n\n"
            "- 前两个三角乘走 `inplace_safe=True, _add_with_inplace=True` 路径，"
            "  把 `z += op(z)` 做成融合操作 (省一份 z 的 buffer)。\n"
            "- 后两个 attention 用普通 `z = z + op(z)`。ending_node 通过先转置再调"
            "  `tri_att_end(z.T)` 再转回来实现 Algorithm 14 —— 模型只学一份 starting 权重。\n"
            "- 单序列分支用第 1 章的 `AttentionPairBias` 但 `has_s=False` (LN 而非 AdaLN)，"
            "  并把 LN over z 的 offset 打开 (`create_offset_ln_z=True`) —— Pairformer 习惯。\n\n"
            "### 测试简化\n\n"
            "本格用 `c_s=0` 跳过单序列分支，只验证 pair 通道的更新链路是对的。"
            "完整 `c_s > 0` 的 pairformer 会在端到端推理时由 Protenix 装配出来。\n\n"
            "**任务**: 填 `PairformerBlock.__init__` 和 `.forward` 两处 TODO。"
        ),
        section_code(
            "from pairformer.pair_stack import PairformerBlock\n\n"
            "block = PairformerBlock(\n"
            "    n_heads=no_heads_pair,\n"
            "    c_z=c_z, c_s=0,\n"
            "    c_hidden_mul=c_hidden,\n"
            "    c_hidden_pair_att=c_hidden,\n"
            "    no_heads_pair=no_heads_pair,\n"
            "    num_intermediate_factor=2,\n"
            "    dropout=0.0,\n"
            ")\n"
            "for sub in (block.tri_att_start.mha, block.tri_att_end.mha):\n"
            "    if hasattr(sub, 'use_efficient_implementation'):\n"
            "        sub.use_efficient_implementation = False\n\n"
            "test_module_shape(block, 'pairformer_block_no_single', control_folder)\n"
            "test_module_method(\n"
            "    block, 'pairformer_block_no_single',\n"
            "    inputs=(None, test_inputs['z'], test_inputs['pair_mask']),\n"
            "    output_names='z_out',\n"
            "    control_folder=control_folder,\n"
            "    method=lambda s, z, pm: block(s, z, pair_mask=pm)[1],\n"
            ")\n"
            "print('PairformerBlock ✓')"
        ),

        section_md(
            "## 章节小结\n\n"
            "完成本章后你掌握了 AF3 主干的 5 个核心三角算子:\n\n"
            "| 方向 | 算子 | 文件 |\n"
            "|---|---|---|\n"
            "| MSA → pair | OuterProductMean | `pairformer/triangle_ops.py` |\n"
            "| pair → MSA | MSAPairWeightedAveraging | `pairformer/msa_stack.py` |\n"
            "| pair → pair (乘法) | TriangleMul Out + In | `pairformer/triangle.py` |\n"
            "| pair → pair (注意力) | TriangleAttention Start + End | `pairformer/triangle.py` |\n"
            "| 集成 | PairformerBlock | `pairformer/pair_stack.py` |\n\n"
            "把 ~48 个 PairformerBlock 堆叠起来就是 AF3 主干 (我们的 tiny 配置只用 8 个)。每个 block"
            "都会让 pair 张量更接近\"满足三角形不等式的实际几何距离矩阵\"。"
            "**下一站**: Feature embedding 把原子 / 残基特征翻译成 trunk 能吃的张量；"
            "之后 Diffusion 用 trunk 学到的几何信号反向去噪出真坐标。"
        ),
    ]
    return nb


# ---------------------------------------------------------------------------
# feature_embedding/feature_embedding.ipynb
# ---------------------------------------------------------------------------

def build_feature_embedding() -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.cells = setup_cells("feature_embedding")
    nb.cells += [
        section_md(
            "# 第 3 章 · Feature embedding\n\n"
            "## 这一章在干啥\n\n"
            "前面两章你写的全是\"如何把张量变成另一张量\"的几何 / 注意力运算。"
            "但 AF3 真实输入根本不是张量 —— 是**异质数据**:\n\n"
            "- 蛋白序列 (一串字符)\n"
            "- 多重序列对齐 (MSA, 不定行不定列)\n"
            "- 同源模板结构 (3D 坐标 + atom mask)\n"
            "- 化学组分 (CCD codes、原子电荷、原子名)\n"
            "- 实验约束 (用户给的 contact / pocket / bond)\n"
            "- 扩散里的噪声水平 (一个标量 σ)\n\n"
            "**Feature embedding 这一章负责把异质 → 张量**。但工程量太大 (有 InputFeatureEmbedder /"
            "AtomAttentionEncoder / ConstraintEmbedder 等好几个 wrapper)，单元测试也难写"
            "(它们的输入是一整个 feature dict 而非简单张量)。\n\n"
            "我们在这一章只单测**两块通用基础设施**：每对 token 的相对位置编码、和标量到向量的"
            "Fourier 嵌入。它们小而独立，是其它 embedder 内部反复调用的零件。其余更重的"
            "embedder 在端到端 notebook (`model/overview.ipynb`) 里整体验证。\n\n"
            "## 本章模块\n\n"
            "| 文件 | 类 / 函数 | 算法 |\n"
            "|---|---|---|\n"
            "| `relative_position_encoding.py` | `RelativePositionEncoding.generate_relp` + `.forward` | Algorithm 3 |\n"
            "| `relative_position_encoding.py` | `FourierEmbedding` | Algorithm 22 |"
        ),

        section_md(
            "## 3.1 RelativePositionEncoding (Algorithm 3)\n\n"
            "Pairformer 和 DiffusionConditioning 都需要在初始 pair 张量上**叠加一个相对位置先验**:\n\n"
            "「两个 token 在同一条链且只差 1 个残基」与「两个 token 来自完全不同的实体」"
            "这两种 pair 应该在初始时就有不同表示。绝对位置 (positional encoding) 在多链系统里"
            "没意义 (链是无序的) —— **相对位置才有意义**。\n\n"
            "### Algorithm 3 的拼图\n\n"
            "对每对 token (i, j) 算 3 类整数偏移，每类 clip 到一个固定范围、用一个特殊编码"
            "(`2*r_max+1` 或 `2*s_max+1`) 表示「超出范围或不同 chain」:\n\n"
            "1. **residue 偏移** (gated by 同链): `clip(residue_index[i] - residue_index[j] + r_max, 0, 2r_max)`\n"
            "2. **token 偏移** (gated by 同链 ∧ 同残基): 同样 clip，但 gating 更严格 ——"
            "   只在「同一个残基里的不同 token」(比如修饰残基/糖基/多原子 token) 才有非平凡值\n"
            "3. **chain 偏移** (gated by 同 entity，`s_max`-clip): 给出蛋白寡聚体里第几条"
            "   对称拷贝\n\n"
            "三类各自 one-hot，再拼上 1 维 `same_entity` 布尔，**总宽度 4·r_max + 2·s_max + 7**。\n\n"
            "### forward 部分\n\n"
            "用一个 `LinearNoBias((4*r_max + 2*s_max + 7), c_z)` 把上面的 one-hot 投到 pair 通道。"
            "这一步是可学的，作为整个 trunk 的「相对位置感知」起点。\n\n"
            "### 关于 `generate_relp`\n\n"
            "为什么是\"generate\"而不是\"compute on the fly\"? AF3 推理在每个 N_cycle 里都需要 relp，"
            "但 relp 是和 t 无关的常量。所以我们一次性算好塞回 `input_feature_dict`，"
            "后续 cycle 直接读。注意整段写在 `torch.no_grad()` 里 —— relp 无梯度。\n\n"
            "**任务**: 在 `feature_embedding/relative_position_encoding.py` 填 `generate_relp` 的"
            "TODO 块。`forward` 仅是 `self.linear_no_bias(relp_feature)` 一行，不需要你新写。"
        ),
        section_code(
            "from feature_embedding.relative_position_encoding import RelativePositionEncoding\n"
            "from feature_embedding.control_values.feature_embedding_checks import (\n"
            "    r_max, s_max, c_z, test_inputs, test_module_shape, test_module_forward,\n"
            ")\n\n"
            "relpe = RelativePositionEncoding(r_max=r_max, s_max=s_max, c_z=c_z)\n"
            "test_module_shape(relpe, 'relative_position_encoding', control_folder)\n"
            "test_module_forward(\n"
            "    relpe, 'relative_position_encoding',\n"
            "    inputs=(test_inputs['relp_feature'],),\n"
            "    output_names='out',\n"
            "    control_folder=control_folder,\n"
            ")\n"
            "print('RelativePositionEncoding ✓')"
        ),

        section_md(
            "## 3.2 FourierEmbedding (Algorithm 22)\n\n"
            "扩散模型每一步去噪都需要让网络**知道当前噪声水平 σ**。但 σ 是一个标量，"
            "怎么把它喂给一个吃 (B, N, c) 张量的 Transformer? 标准做法是 sinusoidal /"
            "random Fourier embedding，把 σ 映成一个 c 维向量。\n\n"
            "### 数学背景: random feature map\n\n"
            "Random Fourier Features ([Rahimi & Recht 2007](https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf))"
            "证明了一个关键事实: **平移不变核** $k(t, t') = k(t - t')$ 可以通过随机三角函数特征近似:\n\n"
            "$$k(t, t') \\approx \\phi(t)^\\top \\phi(t'), \\quad \\phi(t) = \\sqrt{2/c} \\,\\big[\\cos(w_1 t + b_1), \\dots, \\cos(w_c t + b_c)\\big]$$\n\n"
            "其中 $w_k \\sim p(w)$ (Fourier 谱)、$b_k \\sim U(0, 2\\pi)$。**c 越大近似越准确**，"
            "且 $\\phi(t)$ 之间的内积就是核函数。\n\n"
            "AF3 借用这个想法 (不再要求核近似严格)，把噪声水平 $\\tau = \\log(\\sigma / \\sigma_\\text{data})$"
            "(注意是 log) 映成 $c_\\text{noise}$ 维稠密向量:\n\n"
            "$$\\mathrm{FourierEmbed}(\\tau)_k = \\cos\\big(2 \\pi \\,(\\tau \\cdot w_k + b_k)\\big), \\quad k = 1 \\ldots c$$\n\n"
            "$w_k, b_k$ 是构造时一次性抽样的高斯/均匀随机数，作为**不可训练的 nn.Parameter** 存进 state_dict。\n\n"
            "### 为什么用 log(σ/σ_data) 而不是 σ 本身\n\n"
            "扩散里 σ 跨越多个数量级 (典型 $\\sigma_\\text{min} \\approx 0.002, \\sigma_\\text{max} \\approx 80$，"
            "AF3 范围更广)。如果直接用 σ:\n\n"
            "- σ ≈ 0 时几乎所有 $w_k \\cdot \\sigma + b_k \\approx b_k$，cos 值全相同 —— 不同 σ 撞码\n"
            "- σ 很大时 cos 振荡极快、相邻 σ 的 embedding 完全无关 —— 学不到平滑结构\n\n"
            "**取 log**: σ 跨越 ~4 个数量级时 log 只跨 9 倍，cos 在合理频段振荡 ——\n"
            "embedding 既能区分远端 σ、又对相邻 σ 平滑。\n\n"
            "### 为什么是 cos 而不是 sin + cos / position encoding?\n\n"
            "AF3 直接用 cos —— 因为 $\\cos(\\phi - \\pi/2) = \\sin(\\phi)$，加上随机相位 $b_k$"
            "已经隐含覆盖了 sin/cos 两个相位。c 维 \"随机方向\" 的 cos 值合起来就是一个稠密"
            "可分辨的 fingerprint，让网络容易区分不同 σ。\n\n"
            "另一个选择是 Transformer 经典的 sinusoidal positional encoding (对数频率序列)。"
            "AF3 用 random feature 而非固定频率 —— 让模型不依赖人工设计的频率层次，靠数据决定哪个 $w_k$ 重要。\n\n"
            "### 它在哪里被用到\n\n"
            "唯一调用方: `DiffusionConditioning` (第 4 章) 里把 `t_hat / sigma_data` 取对数"
            "再过 FourierEmbedding，得到一个 c 维向量、LN + Linear 后加到单序列条件 `single_s` 上。"
            "这是把噪声水平注入 AdaLN-Zero 的路径。\n\n"
            "**任务**: 在同一个文件里填 `FourierEmbedding.__init__` 和 `.forward`。"
            "注意 `__init__` 用一个 manual_seed 的 generator 才能让权重每次构造一致 ——"
            "测试 harness 会再把它们覆盖成 linspace。"
        ),
        section_code(
            "from feature_embedding.relative_position_encoding import FourierEmbedding\n"
            "from feature_embedding.control_values.feature_embedding_checks import c_noise\n\n"
            "fe = FourierEmbedding(c=c_noise)\n"
            "test_module_shape(fe, 'fourier_embedding', control_folder)\n"
            "test_module_forward(\n"
            "    fe, 'fourier_embedding',\n"
            "    inputs=(test_inputs['noise_level'],),\n"
            "    output_names='out',\n"
            "    control_folder=control_folder,\n"
            ")\n"
            "print('FourierEmbedding ✓')"
        ),

        section_md(
            "## 章节小结\n\n"
            "本章你交付了两个小而关键的零件:\n\n"
            "1. **`RelativePositionEncoding`** —— 算法 3。给 Pairformer 和 DiffusionConditioning"
            "   提供\"两个 token 在结构空间的相对偏移\"先验。\n"
            "2. **`FourierEmbedding`** —— 算法 22。给 DiffusionConditioning 提供\"当前噪声水平的"
            "   稠密表示\"，是 AdaLN-Zero 让 transformer 感知 σ 的唯一通道。\n\n"
            "本章对其它 embedder (InputFeatureEmbedder / AtomAttentionEncoder /"
            "ConstraintEmbedder) 留了 TODO 但没单测 —— 它们依赖整个 `input_feature_dict`，"
            "在端到端 `overview.ipynb` 里一并验证。\n\n"
            "**下一站**: 第 4 章 Diffusion 会大量用到本章的 FourierEmbedding (在"
            "DiffusionConditioning 里) 和 RelativePositionEncoding (作为 pair 条件起点)。"
        ),
    ]
    return nb


# ---------------------------------------------------------------------------
# diffusion/diffusion.ipynb
# ---------------------------------------------------------------------------

def build_diffusion() -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.cells = setup_cells("diffusion")
    nb.cells += [
        section_md(
            "# 第 4 章 · Diffusion\n\n"
            "## 这是 AF3 与 AF2 最大的不同\n\n"
            "AlphaFold 2 的结构头是一个**确定性的**等变 Transformer (Invariant Point Attention "
            "+ Structure Module)，一次性回归出 3D 坐标。AlphaFold 3 把它**全部丢掉**，换成一个"
            "[EDM](https://arxiv.org/abs/2206.00364) 风格的**扩散模型**:\n\n"
            "1. **训练**: 给真实坐标加上 σ-量级的 Gaussian 噪声 → 让网络 $F_\\theta$ 学会去噪。\n"
            "2. **推理**: 从纯 Gaussian 噪声 $x \\sim \\mathcal{N}(0, \\sigma_{\\max}^2 I)$ 出发，"
            "   按 noise schedule 逐步降低 σ、每步调一次 $F_\\theta$、用 Euler 步把 $x$ 推向去噪后的位置。\n\n"
            "这给 AF3 三个能力 AF2 做不到的:\n\n"
            "- **采样多构象**: 从不同初始噪声出发就是不同样本，自然支持\"一个序列的多个 3D 解\"。\n"
            "- **配体 / 核酸通用**: 扩散模型在坐标空间通用，不需要为每种分子设计专用 frame。\n"
            "- **训练更稳**: 去噪目标比一次性回归坐标更易学。\n\n"
            "## EDM 数学骨架\n\n"
            "EDM 的核心公式 (来自 [Karras et al. 2022](https://arxiv.org/abs/2206.00364)) 把网络 $F_\\theta$ "
            "包成 **pre-conditioning**:\n\n"
            "$$D_\\theta(x; \\sigma) = c_\\text{skip}(\\sigma)\\, x + c_\\text{out}(\\sigma) \\, F_\\theta\\big(c_\\text{in}(\\sigma)\\, x;\\, c_\\text{noise}(\\sigma)\\big)$$\n\n"
            "其中 $c_\\text{skip}$, $c_\\text{out}$, $c_\\text{in}$ 都是 σ 的简单函数 ——"
            "保证不论 σ 多大网络都看到大致单位方差的输入、输出也保持合理量级。\n\n"
            "### $c_\\text{skip}, c_\\text{out}, c_\\text{in}$ 的来源 (Karras 2022 eq. 7)\n\n"
            "考虑数据分布方差为 $\\sigma_\\text{data}^2$ (AF3 用 16.0² ≈ 256 Å²)。加噪后的 $x = y + n$，"
            "$y \\sim p_\\text{data}, n \\sim \\mathcal{N}(0, \\sigma^2 I)$。则:\n\n"
            "$$\\mathrm{Var}(x) = \\sigma_\\text{data}^2 + \\sigma^2$$\n\n"
            "为了让网络 $F_\\theta$ 看到**单位方差输入**:\n\n"
            "$$c_\\text{in}(\\sigma) = \\frac{1}{\\sqrt{\\sigma_\\text{data}^2 + \\sigma^2}}$$\n\n"
            "Karras 推导了让训练损失 (在 $\\sigma$ 上的期望) 满足\"信号方差与噪声方差平衡\"的 $c_\\text{out}, c_\\text{skip}$:\n\n"
            "$$c_\\text{out}(\\sigma) = \\frac{\\sigma \\, \\sigma_\\text{data}}{\\sqrt{\\sigma_\\text{data}^2 + \\sigma^2}}, \\qquad "
            "c_\\text{skip}(\\sigma) = \\frac{\\sigma_\\text{data}^2}{\\sigma_\\text{data}^2 + \\sigma^2}$$\n\n"
            "**直觉**: σ → 0 (噪声小)，$c_\\text{skip} → 1, c_\\text{out} → 0$，模型几乎直接复用噪声坐标。"
            "σ → ∞ (纯噪声)，$c_\\text{skip} → 0, c_\\text{out} → \\sigma_\\text{data}$，"
            "模型主导输出，残差通道无所谓。\n\n"
            "AF3 的具体实现 (`DiffusionModule.forward`):\n\n"
            "```python\n"
            "ratio = sigma / sigma_data            # σ/σ_data\n"
            "c_skip = 1 / (1 + ratio**2)            # = σ_data² / (σ_data² + σ²)\n"
            "c_out  = sigma / sqrt(1 + ratio**2)    # 把 sqrt 提出来\n"
            "c_in   = 1 / sqrt(sigma_data**2 + sigma**2)\n"
            "x_denoised = c_skip * x_noisy + c_out * F_theta(c_in * x_noisy, sigma)\n"
            "```\n\n"
            "(代码里 $c_\\text{noise}$ 通过 `FourierEmbedding(log(σ/σ_data) / 4)` 注入 —— 见第 3 章 §3.2。)\n\n"
            "### Noise schedule (推理时的 σ 序列)\n\n"
            "推理时从一个**逆向**的 σ 序列采样: $\\sigma_T \\gg \\cdots \\gg \\sigma_0 = 0$。"
            "AF3 用 sigmoid-based schedule，比 DDIM 的几何序列更密集地花预算在中间区域。"
            "看 `solutions/diffusion/sampler.py` 里的 `InferenceNoiseScheduler`。\n\n"
            "AF3 tiny 模型默认 `N_step=5` (推理便宜)，base 模型 200 步。每多一步都让网络多调一次，\n"
            "用计算换精度。\n\n"
            "## 本章模块\n\n"
            "| 文件 | 类 / 函数 | 算法 | 简述 |\n"
            "|---|---|---|---|\n"
            "| `diffusion_transformer.py` | `ConditionedTransitionBlock` | 25 | AdaLN-Zero SwiGLU FFN |\n"
            "| `diffusion_transformer.py` | `DiffusionTransformerBlock` | 23 (单块) | AttentionPairBias + ConditionedTransitionBlock |\n"
            "| `diffusion_transformer.py` | `DiffusionTransformer` | 23 (整 stack) | 堆叠 n_blocks 个 block |\n"
            "| `diffusion_module.py` | `DiffusionConditioning` | 21 | 把 trunk + 噪声水平 → (s, z) |\n"
            "| `diffusion_module.py` | `DiffusionModule.f_forward / forward` | 20 | EDM 包装 + atom→token→atom 主路径 |\n"
            "| `sampler.py` | `sample_diffusion` | 18 | 完整的 Euler 步采样循环 |\n"
            "| `frames.py` | `expressCoordinatesInFrame` | 29 | 坐标投影到局部正交基 |\n"
            "| `model/utils.py` | `centre_random_augmentation` | 19 | recentre + 随机刚体增广 |\n\n"
            "本章测试覆盖 4 个 `nn.Module` (CTB / DT block / DT 整 stack / 几何 helper)。"
            "顶层 DiffusionModule + sample_diffusion 在 `overview.ipynb` 里端到端跑。"
        ),

        section_md(
            "## 4.1 ConditionedTransitionBlock (Algorithm 25)\n\n"
            "DiffusionTransformer 每个 block 的 **FFN 分支**。看上去就是个 SwiGLU + AdaLN，"
            "但有一个微妙的工程要点 —— 输出门**用的不是 attention 那种 zero-init `linear_o`**，"
            "而是再叠一个 sigmoid 门 (`linear_s` 是 BiasInitLinear with biasinit=-2)。\n\n"
            "### 完整流程\n\n"
            "1. `a = AdaLN(a, s)`  — AdaLN 用 single 条件调制 a\n"
            "2. `b = SiLU(linear_a1(a)) * linear_a2(a)`  — SwiGLU 在隐藏维 `n*c_a` 上的门控\n"
            "3. `a = sigmoid(linear_s(s)) * linear_b(b)`  — **adaLN-Zero 输出门** + 投回 c_a\n\n"
            "**关键**: 步骤 3 的 sigmoid 门让这个 transition block 起手对残差是 ≈0 贡献"
            "(sigmoid(-2)≈0.12)，配合 DropPath 是深层 stack 不爆炸的根本原因。\n\n"
            "**任务**: 打开 `diffusion/diffusion_transformer.py`，"
            "把 `ConditionedTransitionBlock.forward` 的 TODO 填好。"
        ),
        section_code(
            "from diffusion.diffusion_transformer import ConditionedTransitionBlock\n"
            "from diffusion.control_values.diffusion_checks import (\n"
            "    c_a, c_s, c_z, n_heads, n_blocks, test_inputs,\n"
            "    test_module_shape, test_module_method,\n"
            ")\n\n"
            "ctb = ConditionedTransitionBlock(c_a=c_a, c_s=c_s, n=2, biasinit=-2.0)\n"
            "test_module_shape(ctb, 'conditioned_transition_block', control_folder)\n"
            "test_module_method(\n"
            "    ctb, 'conditioned_transition_block',\n"
            "    inputs=(test_inputs['a'], test_inputs['s']),\n"
            "    output_names='out',\n"
            "    control_folder=control_folder,\n"
            "    method=lambda a, s: ctb(a=a, s=s),\n"
            ")\n"
            "print('ConditionedTransitionBlock ✓')"
        ),

        section_md(
            "## 4.2 DiffusionTransformerBlock (Algorithm 23 — 一块)\n\n"
            "把 attention 与 FFN 两个分支拼起来:\n\n"
            "```text\n"
            "    a_in    s     z\n"
            "      \\    |    /\n"
            "       AttentionPairBias(a, s, z)        ← 第 1 章已实现\n"
            "       │\n"
            "       DropPath (stochastic depth)\n"
            "       │\n"
            "       a_in + ─────► a_mid\n"
            "                       \\\n"
            "                        ConditionedTransitionBlock(a_mid, s)\n"
            "                        │\n"
            "                        DropPath\n"
            "                        │\n"
            "                        a_mid + ─────► a_out\n"
            "```\n\n"
            "**两个细节**:\n\n"
            "- 返回 `(a_out, s, z)` 而不是只返回 `a_out`。这样 `DiffusionTransformer.forward`"
            "  能在 block 之间透传 s/z，而不必每次重新构造 (有助于激活检查点)。\n"
            "- DropPath 在推理 (eval) 模式下是 nn.Identity()，所以本测试与训练时的输出"
            "  会有微小不同 (训练时随机)，但 control values 是在 eval 状态生成的。\n\n"
            "**任务**: 在同一个文件里把 `DiffusionTransformerBlock.forward` 的 TODO 填好。"
        ),
        section_code(
            "from diffusion.diffusion_transformer import DiffusionTransformerBlock\n\n"
            "def _disable_efficient_attn(mod):\n"
            "    for m in mod.modules():\n"
            "        if hasattr(m, 'use_efficient_implementation'):\n"
            "            m.use_efficient_implementation = False\n\n"
            "dtb = DiffusionTransformerBlock(c_a=c_a, c_s=c_s, c_z=c_z, n_heads=n_heads)\n"
            "_disable_efficient_attn(dtb)\n"
            "test_module_shape(dtb, 'diffusion_transformer_block', control_folder)\n"
            "test_module_method(\n"
            "    dtb, 'diffusion_transformer_block',\n"
            "    inputs=(test_inputs['a'], test_inputs['s'], test_inputs['z']),\n"
            "    output_names='a_out',\n"
            "    control_folder=control_folder,\n"
            "    method=lambda a, s, z: dtb(a=a, s=s, z=z)[0],\n"
            ")\n"
            "print('DiffusionTransformerBlock ✓')"
        ),

        section_md(
            "## 4.3 DiffusionTransformer (Algorithm 23 — 整 stack)\n\n"
            "把 `n_blocks` 个 DiffusionTransformerBlock 顺序叠起来。看似 1 行 for 循环，"
            "但**注意**: 不是单纯 `a = block(a, s, z)`，而是 `a, s, z = block(a, s, z)` ——\n\n"
            "三个张量都从 block 出来再喂下一个 block，这样如果未来要加 activation checkpointing"
            "(把 s/z 也一并 checkpoint)，只需把这个 for 循环换成 `checkpoint_blocks(...)` 即可。\n\n"
            "AtomTransformer (第 1 章 attention/ 末尾) 内部用的也是这个类 ——"
            "只是配置 `cross_attention_mode=True` 让 attention 走 cross 路径而非 self。\n\n"
            "**任务**: 在同一个文件里填 `DiffusionTransformer.forward`。"
        ),
        section_code(
            "from diffusion.diffusion_transformer import DiffusionTransformer\n\n"
            "dt = DiffusionTransformer(\n"
            "    c_a=c_a, c_s=c_s, c_z=c_z,\n"
            "    n_blocks=n_blocks, n_heads=n_heads,\n"
            ")\n"
            "_disable_efficient_attn(dt)\n"
            "test_module_shape(dt, 'diffusion_transformer', control_folder)\n"
            "test_module_method(\n"
            "    dt, 'diffusion_transformer',\n"
            "    inputs=(test_inputs['a'], test_inputs['s'], test_inputs['z']),\n"
            "    output_names='out',\n"
            "    control_folder=control_folder,\n"
            "    method=lambda a, s, z: dt(a=a, s=s, z=z),\n"
            ")\n"
            "print('DiffusionTransformer ✓')"
        ),

        # --- Geometry helpers ---
        section_md(
            "## 4.4 几何 helper · 局部坐标系 + 刚体增广\n\n"
            "AF2 用一整章 (Structure Module) 处理刚体 / 帧 / quaternion，AF3 把这些**塞进扩散**:"
            "需要刚体不变性时就在原子坐标空间用几何运算实现。本节实现两个最常用的:\n\n"
            "### `expressCoordinatesInFrame` (Algorithm 29)\n\n"
            "**Confidence head 计算 PAE (predicted aligned error) 的核心**。给定一组"
            "「frame」(每个 frame 由 3 个原子定义) 和一组目标原子，要把每个原子投影到每个 frame "
            "的**局部正交基**上，得到形状 `[..., N_frame, N_atom, 3]` 的相对坐标。\n\n"
            "构造正交基的技巧不是 Gram-Schmidt 而是更稳定的版本: 设 a, b, c 是 frame 的三个原子，\n\n"
            "$$\\mathbf{w}_1 = \\widehat{a - b}, \\quad \\mathbf{w}_2 = \\widehat{c - b}$$\n"
            "$$\\mathbf{e}_1 = \\widehat{\\mathbf{w}_1 + \\mathbf{w}_2}, \\quad \\mathbf{e}_2 = \\widehat{\\mathbf{w}_2 - \\mathbf{w}_1}, \\quad \\mathbf{e}_3 = \\mathbf{e}_1 \\times \\mathbf{e}_2$$\n\n"
            "### 为什么 sum/diff 比 Gram-Schmidt 数值更稳\n\n"
            "经典 Gram-Schmidt 写法是: $\\mathbf{e}_1 = \\hat{\\mathbf{w}}_1$, "
            "$\\mathbf{e}_2 = \\widehat{\\mathbf{w}_2 - (\\mathbf{w}_2 \\cdot \\mathbf{e}_1) \\mathbf{e}_1}$。"
            "当 $\\mathbf{w}_1$ 和 $\\mathbf{w}_2$ **几乎共线** (例如蛋白主链上相邻三个原子接近一条直线)，"
            "投影 $\\mathbf{w}_2 - (\\mathbf{w}_2 \\cdot \\mathbf{e}_1)\\mathbf{e}_1 \\to 0$，再 normalize 就放大噪声 ——\n"
            "**catastrophic cancellation** 经典案例，输出可能跳到完全不相关的方向。\n\n"
            "AF3 用的版本 $\\mathbf{e}_1 \\propto \\mathbf{w}_1 + \\mathbf{w}_2$, $\\mathbf{e}_2 \\propto \\mathbf{w}_2 - \\mathbf{w}_1$:\n\n"
            "- 几何上等价于一个 45° 旋转，但用加 / 减替代了 dot + project\n"
            "- 当 $\\mathbf{w}_1 \\approx \\mathbf{w}_2$，$\\mathbf{e}_1$ 仍然有定义"
            "  (≈ 2$\\mathbf{w}_1$ → normalize 给出 $\\hat{\\mathbf{w}}_1$)；$\\mathbf{e}_2 \\to 0$ ——\n"
            "  确实退化但是**对称**退化，eps 保护下不会爆炸。\n"
            "- 对 $\\mathbf{w}_1, \\mathbf{w}_2$ 的小扰动是 Lipschitz 连续的，梯度稳定。\n\n"
            "投影就是相对位移 $d = x - b$ 与三个基向量取内积。\n\n"
            "### `centre_random_augmentation` (Algorithm 19)\n\n"
            "**扩散采样每一步**都先做的事:\n\n"
            "1. 减去 (masked) 质心 —— 抵消坐标的整体平移\n"
            "2. 给每个 sample 各抽一个**随机 SE(3) 变换** (3D 旋转 + 平移)，应用\n"
            "3. mask 后处理\n\n"
            "为什么? AF3 的网络对**全局刚体变换不是天然等变**的 (输入是绝对坐标)。"
            "如果不每步增广，模型学到的就是某个固定参考系下的去噪，泛化差。"
            "随机增广强迫每次去噪都用一个新的参考系 → 等价于训练目标对刚体变换不敏感。\n\n"
            "### 测试只覆盖确定性分支\n\n"
            "`centre_random_augmentation` 完整路径要抽 SO(3) 随机数，无法跨平台位级复现；"
            "我们测试 `centre_only=True` 分支 (只做减质心)，确定性、可测。\n\n"
            "**任务**: 填两个 TODO ——`diffusion/frames.py::expressCoordinatesInFrame` (完整) 和"
            "`model/utils.py::centre_random_augmentation` (包括 centre_only + 完整路径)。"
        ),
        section_code(
            "from diffusion.frames import expressCoordinatesInFrame\n"
            "from model.utils import centre_random_augmentation\n"
            "from diffusion.control_values.diffusion_checks import test_inputs\n\n"
            "# expressCoordinatesInFrame (Algorithm 29)\n"
            "out = expressCoordinatesInFrame(\n"
            "    test_inputs['coords'].double(),\n"
            "    test_inputs['frame_atoms'].double(),\n"
            ")\n"
            "expected = torch.load(f'{control_folder}/express_coordinates_in_frame_out.pt')\n"
            "assert torch.allclose(out, expected), 'expressCoordinatesInFrame output mismatch'\n"
            "print('expressCoordinatesInFrame ✓')\n\n"
            "# centre_random_augmentation, deterministic centre_only=True branch\n"
            "out = centre_random_augmentation(\n"
            "    test_inputs['coords'].double(), N_sample=2, centre_only=True,\n"
            ")\n"
            "expected = torch.load(f'{control_folder}/centre_random_augmentation_centre_only_out.pt')\n"
            "assert torch.allclose(out, expected), 'centre_random_augmentation(centre_only=True) output mismatch'\n"
            "print('centre_random_augmentation (centre_only) ✓')"
        ),

        section_md(
            "## 章节小结\n\n"
            "本章你实现了 AF3 扩散主干的 5 个核心组件:\n\n"
            "| 类 / 函数 | 角色 |\n"
            "|---|---|\n"
            "| `ConditionedTransitionBlock` | DiffusionTransformer 的 FFN 分支 (Alg 25) |\n"
            "| `DiffusionTransformerBlock` | attention + FFN 一块 (Alg 23) |\n"
            "| `DiffusionTransformer` | n_blocks 块堆叠 |\n"
            "| `expressCoordinatesInFrame` | PAE 内核 (Alg 29) |\n"
            "| `centre_random_augmentation` | 采样每步必备的刚体增广 (Alg 19) |\n\n"
            "顶层的 `DiffusionConditioning` (Alg 21)、`DiffusionModule.f_forward` / `forward`"
            "(Alg 20 EDM scaling)、和 `sample_diffusion` (Alg 18 完整采样循环) **TODO 已写好**"
            "但没单测 (它们要一整份特征 dict)。完成本章后这些 TODO 全部能填，"
            "在端到端 `overview.ipynb` 跑一次推理就会一次跑通。\n\n"
            "**下一站**: 第 5 章 Confidence 把扩散输出的坐标转成 pLDDT / PAE / PDE 等置信度。"
        ),
    ]
    return nb


# ---------------------------------------------------------------------------
# confidence/confidence.ipynb
# ---------------------------------------------------------------------------

def build_confidence() -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.cells = setup_cells("confidence")
    nb.cells += [
        section_md(
            "# 第 5 章 · Confidence\n\n"
            "## AF3 输出的不只是坐标\n\n"
            "AF2 / AF3 的最大产品差异之一: **每个预测都附带置信度**。给 100 个结构告诉用户"
            "「这 100 个里哪些可信哪些瞎猜」，是这类模型在湿实验里能用起来的前提。\n\n"
            "AF3 置信度系统输出 4 类:\n\n"
            "| 量 | 全名 | 维度 | 含义 |\n"
            "|---|---|---|---|\n"
            "| **pLDDT** | predicted local-distance-difference test | per atom | 局部精度 (0-100，越高越好) |\n"
            "| **PAE** | predicted aligned error | per token pair | 给定一对 token，二者相对位姿误差的分布 |\n"
            "| **PDE** | predicted distance error | per token pair | 二者距离的误差分布 (对称量) |\n"
            "| **resolved** | 是否在实验中可见 | per atom | 二分类: 这个原子在最终结构里有没有坐标 |\n\n"
            "再加上**distogram** —— 训练目标之一，预测每对 token 之间的距离分布 (64 个 bin)。"
            "Distogram 不直接是置信度但在推理时也由置信度路径产出。\n\n"
            "### 这四个量的数学定义\n\n"
            "**LDDT** ([Mariani et al. 2013](https://academic.oup.com/bioinformatics/article/29/21/2722/195896)): "
            "对原子 $\\ell$，定义它与一组距离接触原子的相对距离误差:\n\n"
            "$$\\mathrm{LDDT}(\\ell) = \\frac{1}{|\\mathcal{N}(\\ell)|} \\sum_{m \\in \\mathcal{N}(\\ell)} \\frac{1}{4} \\sum_{t \\in \\{0.5, 1, 2, 4\\}} \\mathbb{1}\\big[|d_{\\ell m}^\\text{pred} - d_{\\ell m}^\\text{gt}| < t\\big]$$\n\n"
            "其中 $\\mathcal{N}(\\ell)$ 是真实结构里离 $\\ell$ 在 15Å 内的原子。本质: 把"
            "「预测距离与真实距离误差是否在 0.5/1/2/4Å 之内」做 4 个阈值平均。LDDT 范围 0-1，乘 100 报告。\n\n"
            "AF3 不能直接算 LDDT (因为推理时不知道 ground truth)，所以预测一个**分布**:\n\n"
            "$$\\mathrm{pLDDT}_\\ell^h = \\mathrm{Linear}_{b\\_\\text{plddt}}\\big(\\mathrm{LN}(s_\\ell)\\big) \\to \\text{50 bins over } [0, 100]$$\n\n"
            "(softmax 后做 bin 中心加权得到标量预测)。\n\n"
            "**PAE (predicted aligned error)**: 对每对 token (i, j) 预测「如果把预测结构对齐到 token i 的局部坐标系，"
            "token j 的位置误差是多少」。形式化:\n\n"
            "$$\\mathrm{AE}(i, j) = \\| T_i^\\text{pred} \\cdot x_j^\\text{pred} - T_i^\\text{gt} \\cdot x_j^\\text{gt} \\|$$\n\n"
            "其中 $T_i$ 是 token i 的局部 frame (用前面 `expressCoordinatesInFrame` 算的)。"
            "PAE 是 N×N 矩阵 (有方向)，可分到 64 个 bin。**iPTM** (interface predicted TM-score)"
            "由 PAE 在 chain 边界附近的统计算出，是判断复合物界面置信的关键。\n\n"
            "**PDE (predicted distance error)**: 对每对 token 预测「距离误差的分布」:\n\n"
            "$$\\mathrm{DE}(i, j) = \\big| \\|x_i^\\text{pred} - x_j^\\text{pred}\\| - \\|x_i^\\text{gt} - x_j^\\text{gt}\\| \\big|$$\n\n"
            "**对称**: $\\mathrm{DE}(i, j) = \\mathrm{DE}(j, i)$。所以 PDE head 内部要"
            "先做 `z + z.transpose(-2, -3)` 对称化再 LN + Linear。\n\n"
            "**resolved**: 训练时把 PDB 里有坐标的原子标 1、没有的标 0 (例如 missing residue)。"
            "这是 per-atom 二分类，告诉用户「模型预测的这个原子能否在湿实验里被观察到」。\n\n"
            "**Distogram (Alg 1 line 17)**: 训练目标。对每对 token (i, j) 直接预测真实距离落在 64 个 bin 的哪一个。"
            "由 pair 张量 z 直接投出 logits，对称化保证 $\\mathrm{Dist}(i, j) = \\mathrm{Dist}(j, i)$。\n\n"
            "## 本章范围\n\n"
            "完整的 ConfidenceHead 是个比较重的复合模块: 内部跑一个小型 PairformerStack +"
            "四个分类头。完整 forward 需要带 atom 级索引映射的特征字典，单元测试太繁琐 ——"
            "所以我们:\n\n"
            "1. **5.1** 测最简单的入口 `DistogramHead.forward` —— 它就是一个 Linear + 对称化。\n"
            "2. **5.2** 用 shape 检查覆盖整个 `ConfidenceHead.__init__` —— 验证你的"
            "   `__init__` 把所有子模块都装对了 (这是最容易出错的部分)。\n\n"
            "ConfidenceHead 的完整 `forward` 与 `memory_efficient_forward` 在 `overview.ipynb` 里"
            "端到端验证。\n\n"
            "## 本章模块\n\n"
            "| 文件 | 类 | 作用 |\n"
            "|---|---|---|\n"
            "| `confidence/distogram_head.py` | `DistogramHead.forward` | distogram logits (Alg 1 line 17) |\n"
            "| `confidence/confidence_head.py` | `ConfidenceHead.__init__` | 装配 4 个 head |"
        ),

        section_md(
            "## 5.1 DistogramHead (Algorithm 1 line 17)\n\n"
            "AF3 训练目标之一: **预测每对 token 之间真实距离落在哪个 bin**。Distogram 模型给出一个"
            "$[N, N, B]$ 张量 (B 个距离 bin 的 logits)，softmax 后得到每对 token 距离的概率分布。\n\n"
            "DistogramHead 极简: 就是把 pair 张量过一个 Linear 投到 64 个 bin。但有两个关键点:\n\n"
            "### 1. 零初始化\n\n"
            "`self.linear = Linear(c_z, no_bins, initializer=\"zeros\")` —— 训练初期输出是 0，"
            "softmax 后是均匀分布，logits 完全由训练学到。如果用默认初始化，初始预测就有偏，"
            "训练动态不稳。\n\n"
            "### 2. 对称化\n\n"
            "距离矩阵在数学上对称: $d(i, j) = d(j, i)$。但 pair 张量在 Pairformer 里**并不严格对称**。"
            "DistogramHead 通过 `logits = logits + logits.transpose(-2, -3)` 显式强制对称化 ——"
            "保证 distogram 预测合法。\n\n"
            "**任务**: 打开 `confidence/distogram_head.py` 填 forward 的 TODO 块 (3 步: linear → 对称化 → 返回)。"
        ),
        section_code(
            "from confidence.distogram_head import DistogramHead\n"
            "from confidence.control_values.confidence_checks import (\n"
            "    c_z, no_bins, test_inputs,\n"
            "    test_module_shape, test_module_forward,\n"
            ")\n\n"
            "dh = DistogramHead(c_z=c_z, no_bins=no_bins)\n"
            "test_module_shape(dh, 'distogram_head', control_folder)\n"
            "test_module_forward(\n"
            "    dh, 'distogram_head',\n"
            "    inputs=(test_inputs['z'],),\n"
            "    output_names='out',\n"
            "    control_folder=control_folder,\n"
            ")\n"
            "print('DistogramHead ✓')"
        ),

        section_md(
            "## 5.2 ConfidenceHead 装配检查\n\n"
            "完整的 ConfidenceHead 行为相当复杂:\n\n"
            "```text\n"
            "  x_pred_rep_coords  (per-token 代表原子 coord)\n"
            "      │\n"
            "      ├── cdist → 距离矩阵\n"
            "      │     │\n"
            "      │     └── one_hot 分箱 (distance_bin_*)\n"
            "      │           │\n"
            "      │           └── linear_no_bias_d / _wo_onehot → 加到 z_pair\n"
            "      ▼\n"
            "  z_pair + s_trunk\n"
            "      │\n"
            "      └── PairformerStack(c_s=c_s, c_z=c_z, n_blocks=n_blocks)\n"
            "      │\n"
            "      ├── pae_ln + linear_no_bias_pae  →  PAE logits  (per pair, b_pae bins)\n"
            "      ├── pde_ln + linear_no_bias_pde  →  PDE logits  (per pair, b_pde bins)\n"
            "      ├── plddt_ln + plddt_weight     →  pLDDT logits (per atom, b_plddt bins)\n"
            "      └── resolved_ln + resolved_weight → resolved (per atom, 2 classes)\n"
            "```\n\n"
            "`__init__` 里要拼 11 个左右子模块 + 2 个特殊的 weight 张量 (`plddt_weight` 和"
            "`resolved_weight` 是 atom-slot 级权重，shape `[max_atoms_per_token, c_s, b_*]`)。"
            "**任何子模块漏建 / 命名错 / 维度错都会让 state_dict 对不上 Protenix 权重**。\n\n"
            "因此我们用 `test_module_shape` 做整体形状检查 —— 它枚举所有命名参数的 shape，"
            "与保存的参考字典对比，任何不一致都会立刻报。\n\n"
            "完整 `forward` 与 `memory_efficient_forward` 还需要原子级索引映射 (`atom_to_token_idx`"
            "等)，构造测试 dict 太繁琐 —— 留给端到端 `overview.ipynb`。"
        ),
        section_code(
            "from confidence.confidence_head import ConfidenceHead\n"
            "from confidence.control_values.confidence_checks import test_module_shape\n\n"
            "ch = ConfidenceHead(\n"
            "    n_blocks=1,\n"
            "    c_s=32, c_z=c_z,\n"
            "    c_s_inputs=32,\n"
            "    b_pae=8, b_pde=8, b_plddt=10, b_resolved=2,\n"
            "    max_atoms_per_token=5,\n"
            "    pairformer_dropout=0.0,\n"
            "    distance_bin_start=3.25, distance_bin_end=8.25, distance_bin_step=1.25,\n"
            ")\n"
            "test_module_shape(ch, 'confidence_head_init', control_folder)\n"
            "print('ConfidenceHead.__init__ ✓')"
        ),

        section_md(
            "## 章节小结\n\n"
            "本章你交付了:\n\n"
            "1. **`DistogramHead.forward`** —— pair → 距离 bin logits，零初始化 + 对称化。\n"
            "2. **`ConfidenceHead.__init__`** —— 4 个置信度 head + 距离 bin 投影 + 内部 PairformerStack"
            "   的完整装配，通过 state_dict shape 检查验证。\n\n"
            "完整 `ConfidenceHead.forward` 与 `memory_efficient_forward` 的实现 TODO 已在 "
            "`confidence_head.py` 写好详细伪代码，但在端到端 `overview.ipynb` 里才整体测试。\n\n"
            "**全部章节走完，下一站**: `overview.ipynb` —— 把所有零件装成完整 Protenix，"
            "加载字节官方权重，跑一次 7r6r 蛋白的端到端推理。pLDDT≈33 / pTM≈0.21 是"
            "tiny 模型在 CPU 上 5 步采样的预期结果 (论文 base 模型当然更高，但跑一次 30 分钟，"
            "学习用足够了)。"
        ),
    ]
    return nb


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

NOTEBOOKS = [
    ("attention",         build_attention),
    ("pairformer",        build_pairformer),
    ("feature_embedding", build_feature_embedding),
    ("diffusion",         build_diffusion),
    ("confidence",        build_confidence),
]


def main() -> None:
    for chapter, builder in NOTEBOOKS:
        path = os.path.join(SOL, chapter, f"{chapter}.ipynb")
        nb = builder()
        with open(path, "w") as f:
            nbformat.write(nb, f)
        print(f"  wrote {path}  ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
