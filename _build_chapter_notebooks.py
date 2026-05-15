"""Generate per-chapter learning notebooks under ``solutions/<chapter>/``.

为每章生成学习引导 notebook，结构与 AF2 教学项目对齐:

    1. 基本环境 (cwd / sys.path / autoreload)
    2. 章节简介
    3. 每个模块一对 (指引 markdown + 调用 control_values 测试的 code cell)

Run from the repository root::

    python _build_chapter_notebooks.py

This script is not used at runtime — it's only here so the notebooks are
reproducible. The committed ``.ipynb`` files are what students open.

脚本只负责生成 notebook，正式使用时学生打开 ``.ipynb`` 即可。
"""
from __future__ import annotations

import os
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


HERE = os.path.dirname(os.path.abspath(__file__))
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
            "1. 把工作目录切到 `solutions/` (这样 `from attention.mha import ...` 这种导入能直接生效)。\n"
            "2. 启用 `autoreload`，编辑 .py 文件保存后 notebook 里立刻可用，不用重启 kernel。\n"
            "3. 设 `LAYERNORM_TYPE=torch`，避免 CUDA-only 算子的 import 失败。\n\n"
            "First time you open the notebook, run the 3 cells below: cd into `solutions/`, "
            "turn on autoreload, force the pure-PyTorch LayerNorm path."
        ),
        new_code_cell(
            "import os, sys\n"
            "if os.path.basename(os.getcwd()) != 'solutions':\n"
            "    if os.path.isdir('solutions'):\n"
            "        os.chdir('solutions')\n"
            "    else:\n"
            "        # already inside a chapter folder — climb out\n"
            "        while os.path.basename(os.getcwd()) != 'solutions' and os.getcwd() != '/':\n"
            "            os.chdir('..')\n"
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
            "AF3 的所有 attention 流派 (token / atom / pair-biased / 局部窗口)，底层都用同一个"
            "多头注意力 + 一组初始化好的线性层。本章把这些底座的代码一行一行填出来。\n\n"
            "本章涉及的模块 (全部位于 `attention/`):\n\n"
            "| 文件 | 类 / 函数 | 作用 |\n"
            "|---|---|---|\n"
            "| `linear.py` | `Linear._init_params`, `Linear.forward`, `BiasInitLinear.__init__` | 自定义初始化的线性层 |\n"
            "| `layer_norm.py` | `OpenFoldLayerNorm` | 与 Protenix 融合算子参数名对齐的 LN |\n"
            "| `mha.py` | `_attention`, `Attention._prep_qkv`, `Attention._wrap_up`, `Attention.__init__`, `Attention.forward` | 多头注意力 |\n"
            "| `transition.py` | `AdaptiveLayerNorm`, `Transition` | AF3 算法 26 / 11 |\n"
            "| `attention_pair_bias.py` | `AttentionPairBias` | AF3 算法 24 |\n\n"
            "每个测试在调用前会把模块参数替换成 `linspace(-1, 1, numel)`，"
            "只要你的实现与参考相同，输出就会完全相同 (`torch.allclose`)。"
        ),

        # --- Linear ---
        section_md(
            "## 1.1 Linear\n\n"
            "打开 `attention/linear.py`，把 `Linear._init_params`、`Linear.forward` 和"
            "`BiasInitLinear.__init__` 三个 TODO 填好。\n\n"
            "Linear 在 AF3 不同位置初始化方式不同 (`default` / `relu` / `zeros`)，"
            "Linear.forward 的 precision 路径用于强制 fp32 计算 (坐标投影、噪声水平条件)。"
            "BiasInitLinear 是为 sigmoid 门设定初始开度的特殊 Linear。"
        ),
        section_code(
            "from attention.linear import Linear, LinearNoBias, BiasInitLinear\n"
            "from attention.control_values.attention_checks import (\n"
            "    c_a, c_s, c_z, test_module_shape, test_module_forward,\n"
            "    test_inputs,\n"
            ")\n\n"
            "# Linear with default initializer\n"
            "lin = Linear(in_features=c_a, out_features=c_z)\n"
            "test_module_shape(lin, 'linear_default', control_folder)\n"
            "test_module_forward(lin, 'linear_default',\n"
            "                    inputs=(test_inputs['x_a'],),\n"
            "                    output_names='out',\n"
            "                    control_folder=control_folder)\n\n"
            "# LinearNoBias\n"
            "lin_nb = LinearNoBias(in_features=c_a, out_features=c_z)\n"
            "test_module_shape(lin_nb, 'linear_nobias', control_folder)\n"
            "test_module_forward(lin_nb, 'linear_nobias',\n"
            "                    inputs=(test_inputs['x_a'],),\n"
            "                    output_names='out',\n"
            "                    control_folder=control_folder)\n\n"
            "# BiasInitLinear with biasinit=-2.0\n"
            "bil = BiasInitLinear(in_features=c_s, out_features=c_a, bias=True, biasinit=-2.0)\n"
            "test_module_shape(bil, 'bias_init_linear', control_folder)\n"
            "test_module_forward(bil, 'bias_init_linear',\n"
            "                    inputs=(test_inputs['x_s'],),\n"
            "                    output_names='out',\n"
            "                    control_folder=control_folder)\n"
            "print('Linear ✓')"
        ),

        # --- LayerNorm ---
        section_md(
            "## 1.2 LayerNorm\n\n"
            "打开 `attention/layer_norm.py`，把 `OpenFoldLayerNorm.__init__` 和"
            "`OpenFoldLayerNorm.forward` 两个 TODO 填好。AdaLN 等需要"
            "`create_scale=False` 或 `create_offset=False`，因此 LN 必须支持可选的 scale/offset。"
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
            "## 1.3 `_attention` (核心点积数学)\n\n"
            "打开 `attention/mha.py`，把模块顶部的 `_attention(q, k, v, attn_bias, ...)` 函数 TODO 填好。\n\n"
            "这是所有 attention 流派最底层的数学：缩放点积 + softmax + 加权求和。"
            "函数没有可学参数，所以测试直接比较输出张量。"
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
            "## 1.4 Attention (多头注意力模块)\n\n"
            "继续在 `mha.py` 里，依次填:\n"
            "- `Attention.__init__` (5 个线性层 + 可选门控)\n"
            "- `Attention._prep_qkv` (Q/K/V 投影 + 拆头 + 缩放)\n"
            "- `Attention._wrap_up` (可选 sigmoid 门控 + 展平 + 输出投影)\n"
            "- `Attention.forward` (调度局部 / 全连接两条路径)"
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
            "## 1.5 AdaptiveLayerNorm + Transition\n\n"
            "打开 `attention/transition.py`，填两个 TODO:\n"
            "- `AdaptiveLayerNorm.__init__` 和 `.forward` (AF3 算法 26，FiLM 风格调制)\n"
            "- `Transition.__init__` 和 `.forward` (AF3 算法 11，SwiGLU FFN)"
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
            "## 1.6 AttentionPairBias (AF3 算法 24)\n\n"
            "打开 `attention/attention_pair_bias.py`，把 `__init__`、`local_multihead_attention`、"
            "`standard_multihead_attention`、`forward` 四个 TODO 填好。\n\n"
            "AttentionPairBias 是 AF3 最核心的复合块: AdaLN + 多头 attention + pair bias + adaLN-Zero 输出门。"
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
            "完成本章后，你已经实现了:\n"
            "- 三种自定义初始化策略的 Linear (default / relu / zeros) 与 sigmoid 门 BiasInitLinear\n"
            "- 与 Protenix 算子参数名对齐的 LayerNorm\n"
            "- 多头注意力的完整链路 (`_attention` 数学 + `_prep_qkv` 形状处理 + `_wrap_up` 门控)\n"
            "- AF3 三个高频组件: `AdaptiveLayerNorm` / `Transition` / `AttentionPairBias`\n\n"
            "这些是后续章节 Pairformer / Diffusion 的基石。"
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
            "Pairformer 是 AF3 的核心主干 (替代 AF2 的 Evoformer)。它由若干**三角更新**算子"
            "组合而成，让 pair 表示 $z_{ij}$ 之间的关系满足"
            "[三角不等式](https://en.wikipedia.org/wiki/Triangle_inequality)。\n\n"
            "本章涉及的模块 (全部位于 `pairformer/`):\n\n"
            "| 文件 | 类 | 作用 |\n"
            "|---|---|---|\n"
            "| `triangle_ops.py` | `OuterProductMean` | 算法 10 — MSA → pair |\n"
            "| `triangle.py` | `TriangleMultiplication{Outgoing,Incoming}` | 算法 11 / 12 |\n"
            "| `triangle.py` | `TriangleAttention` | 算法 13 / 14 |\n"
            "| `msa_stack.py` | `MSAPairWeightedAveraging` | MSAModule 内 pair → MSA 的反向通道 |\n"
            "| `pair_stack.py` | `PairformerBlock` | 算法 17 单块 (含全部三角操作) |"
        ),

        section_md(
            "## 2.1 OuterProductMean (算法 10)\n\n"
            "打开 `pairformer/triangle_ops.py`，填 `OuterProductMean.__init__`、`_opm`、`_forward` 三处 TODO。\n\n"
            "OPM 把 MSA 张量沿序列维做外积取均值，把 MSA 信息汇入 pair 通道。"
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
            "## 2.2 TriangleMultiplication (算法 11 / 12)\n\n"
            "打开 `pairformer/triangle.py`，填 `BaseTriangleMultiplicativeUpdate.__init__`、"
            "`TriangleMultiplicativeUpdate.__init__`、`_combine_projections`、`forward` 几处 TODO。\n\n"
            "外向 (outgoing) 与内向 (incoming) 走同一个类，只是子类用 `partialmethod` 固定 `_outgoing` 标志。"
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
            "## 2.3 TriangleAttention (算法 13 / 14)\n\n"
            "在 `pairformer/triangle.py` 里继续，填 `TriangleAttention.__init__` 与 `.forward`。\n\n"
            "三角自注意力对 pair 张量沿一条 residue 轴跑多头注意力，把另一条 residue 轴的信息作为 bias。"
            "起始节点 (starting=True) 与终止节点 (ending) 共用同一个类，end 模式通过物理转置实现。"
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
            "打开 `pairformer/msa_stack.py`，填 `MSAPairWeightedAveraging.__init__` 与 `.forward`。\n\n"
            "MSAModule 内由 pair 张量反过来更新 MSA 通道用的就是这一块——"
            "pair 的每对 (i, j) 产生权重，沿 j 轴对 MSA 的 value 加权平均。"
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
            "## 2.5 PairformerBlock (算法 17 — 一整块)\n\n"
            "打开 `pairformer/pair_stack.py`，把 `PairformerBlock.__init__` 与 `.forward` 两处 TODO 填好。\n\n"
            "一个 block 把前面 4 个三角操作 + pair transition 串起来，更新 pair 表示。"
            "若 `c_s > 0` 还会再叠一次 AttentionPairBias + Transition 更新 single 表示。\n\n"
            "本测试用 `c_s=0` 简化路径。"
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
            "本章完成后你已经能用三角更新算子把 pair 表示从 MSA 中提炼出来，并能复用"
            "上一章实现的 attention / transition 把它们组合成一个完整的 PairformerBlock。"
            "把若干个 block 堆叠起来就是 Algorithm 17 的 PairformerStack。"
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
            "AF3 的输入端: 把氨基酸 / 原子 / 链 / 模板 / 噪声水平 …… 各种"
            "异质特征**翻译**成主干能直接吃的张量。本章先实现两块通用的辅助:\n\n"
            "| 文件 | 类 | 作用 |\n"
            "|---|---|---|\n"
            "| `relative_position_encoding.py` | `RelativePositionEncoding` | 算法 3 — 每对 token 的相对位置 one-hot |\n"
            "| `relative_position_encoding.py` | `FourierEmbedding` | 算法 22 — 标量噪声水平的随机 Fourier 嵌入 |\n\n"
            "AtomAttentionEncoder / InputFeatureEmbedder 这些重的 wrapper 因为依赖太多上下文，"
            "我们这里不做单元测试，等学完 attention/pairformer/diffusion 后在端到端 notebook 里整段验证。"
        ),

        section_md(
            "## 3.1 RelativePositionEncoding (算法 3)\n\n"
            "打开 `feature_embedding/relative_position_encoding.py`，把 `generate_relp` 和 "
            "`forward` 两处 TODO 填好。\n\n"
            "`generate_relp` 是关键逻辑: 对每对 token 算 (同链 / 同残基 / 同 entity) 三个布尔门控，"
            "再用三组 clip 后的整数偏移做 one-hot。本测试只覆盖最后的线性投影 forward。"
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
            "## 3.2 FourierEmbedding (算法 22)\n\n"
            "同一个文件，填 `FourierEmbedding.__init__` 和 `.forward`。\n\n"
            "Fourier embedding 把扩散噪声水平这种标量映成 c 维向量: 用一组**固定的**随机系数 w 和 b "
            "(在 `__init__` 一次性抽样后作为不可训练参数保存) 算 `cos(2π · (t · w + b))`。\n\n"
            "因为 w 和 b 在构造时随机, 要让测试可重复, control values 的 `_generate.py` 调用"
            "`torch.manual_seed(0)` 后才生成参数; 你的测试 cell 不需要额外设种子, "
            "保存到 `.pt` 的参数会被替换成 linspace。"
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
            "这两块虽小但很关键: RelativePositionEncoding 是 Pairformer 与"
            "DiffusionConditioning 共用的 pair 起点; FourierEmbedding 把噪声水平注入 AdaLN 的 single 通道。"
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
            "AF3 的结构预测头是一个 [EDM](https://arxiv.org/abs/2206.00364) 风格的扩散模型: "
            "在原子坐标空间逐步去噪。本章实现扩散的三个核心模块, 以及 EDM 采样循环本身 "
            "(后者在端到端 notebook 里再用)。\n\n"
            "| 文件 | 类 / 函数 | 算法编号 |\n"
            "|---|---|---|\n"
            "| `diffusion_transformer.py` | `ConditionedTransitionBlock` | 25 |\n"
            "| `diffusion_transformer.py` | `DiffusionTransformerBlock` | 23 (单块) |\n"
            "| `diffusion_transformer.py` | `DiffusionTransformer` | 23 (整 stack) |\n"
            "| `diffusion_module.py` | `DiffusionConditioning`, `DiffusionModule.f_forward / forward` | 21 / 20 |\n"
            "| `sampler.py` | `sample_diffusion` | 18 |"
        ),

        section_md(
            "## 4.1 ConditionedTransitionBlock (算法 25)\n\n"
            "打开 `diffusion/diffusion_transformer.py`，把 `ConditionedTransitionBlock.forward` 的 TODO 填好。\n\n"
            "AdaLN 调制 + SwiGLU FFN + adaLN-Zero 输出门 (`linear_s` 的 sigmoid)，"
            "是 DiffusionTransformer 每个 block 的 FFN 分支。"
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
            "## 4.2 DiffusionTransformerBlock (算法 23 — 单块)\n\n"
            "在同一个文件里, 把 `DiffusionTransformerBlock.forward` 填好。\n\n"
            "结构: `AttentionPairBias(a, s, z)` (上一章已实现) + DropPath + 残差; 接着"
            "`ConditionedTransitionBlock` + DropPath + 残差。注意 `s` / `z` 是原样透传以便激活检查点存活。"
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
            "## 4.3 DiffusionTransformer (算法 23 — 整 stack)\n\n"
            "同一个文件, 把 `DiffusionTransformer.forward` 填好。\n\n"
            "把 `n_blocks` 个 DiffusionTransformerBlock 顺序应用; `s` / `z` 在 block 之间不变。"
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

        section_md(
            "## 章节小结\n\n"
            "本章实现了 AF3 扩散主干所需的 Transformer 三件套。剩下的两块——"
            "DiffusionConditioning (算法 21)、DiffusionModule (算法 20 EDM scaling) "
            "以及顶层 `sample_diffusion` (算法 18)——都已经写好详细的 TODO 伪代码, "
            "学完后续 notebook 后端到端推理就会一次跑通。"
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
            "推理完拿到坐标后还要给每个原子 / 每个 token 对一个**置信度估计**。AF3 的置信头输出 4 路:\n\n"
            "| 量 | 含义 | 维度 |\n"
            "|---|---|---|\n"
            "| pLDDT | per-atom local-distance-difference test | per atom |\n"
            "| PAE | predicted aligned error | per token pair |\n"
            "| PDE | predicted distance error | per token pair |\n"
            "| resolved | 可解析 / 未解析二分类 | per atom |\n\n"
            "ConfidenceHead 整体由前面章节的零件搭起来 (内部跑一个小型 PairformerStack), 我们"
            "本章只单独验证 DistogramHead——置信度系统的入口。"
        ),

        section_md(
            "## 5.1 DistogramHead (算法 1 第 17 行)\n\n"
            "打开 `confidence/distogram_head.py`，把 `forward` 的 TODO 填好。\n\n"
            "DistogramHead 把 pair 表示通过一个零初始化的线性层投到 64 个距离 bin 上, "
            "再对 (i, j) 做对称化。零初始化意味着训练初期输出近似均匀分布。"
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
            "## 章节小结\n\n"
            "ConfidenceHead 本身的 `forward` 与 `memory_efficient_forward` 在 `confidence_head.py` 里"
            "有详细 TODO, 端到端 notebook 会把它们整体跑一遍。本章重点只是让你理解"
            "DistogramHead 这个最简单的 head 长什么样子。"
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
