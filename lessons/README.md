# Lessons · 章节讲解

每个 markdown 文件对应一个章节，是配套 `tutorials/<chapter>/<chapter>.ipynb`
的「讲解正文」。讲清楚 AF3 的概念动机、数学公式、论文 Algorithm 编号、
以及它们如何对应到本仓库 `solutions/<chapter>/` 下的具体代码。

Each markdown corresponds to one chapter — the "lesson text" companion
to `tutorials/<chapter>/<chapter>.ipynb`. Walks through concept,
motivation, equations, paper Algorithm numbers, and how each maps to
our `solutions/<chapter>/` code.

## 灵感来源 · Inspiration

本系列在结构和切分上深度参考了
**[The Illustrated AlphaFold](https://elanapearl.github.io/blog/2024/the-illustrated-alphafold/)**
(Elana Pearl Simon, Stanford, 2024) —— 一份业内公认的 AF3 可视化讲解。
我们这套中文 lessons 是**原创内容**：保留她的章节切分和概念顺序，但
所有正文用我们自己的话重写，并把每一节绑到本仓库 `solutions/` 下的
具体类与函数；示意图请直接参考论文 (Abramson et al. 2024) Figure /
Supplementary Algorithm 编号。

This series is deeply inspired by **The Illustrated AlphaFold** in its
sectioning and concept ordering, but the prose is all original and tied
to this repo's code. For figures, refer directly to the AF3 paper
(Abramson et al. 2024).

## 章节顺序 · Chapter order

顺序与主 README "学习路径" 表保持一致。

| #  | 章节 · Chapter | 内容 · What it covers | AF3 Alg. |
|----|----------------|-----------------------|----------|
| 0  | `feature_extraction.md`  | JSON → tensors, MSA, templates (read-only tour) | — |
| 1  | `attention.md`           | MHA, LayerNorm, AdaLN, AttentionPairBias | Alg 11, 24, 26 |
| 2  | `pairformer.md`          | Pair stack + MSA module + triangle ops | Alg 8, 10, 11, 12, 13, 14, 16, 17 |
| 3  | `feature_embedding.md`   | InputFeatureEmbedder, AtomAttentionEncoder, Relative PE, Fourier | Alg 2, 3, 5, 22 |
| 4  | `diffusion.md`           | Diffusion module + transformer + sampler + frames | Alg 18, 19, 20, 21, 23, 25, 29 |
| 5  | `confidence.md`          | Confidence head + DistogramHead + pTM/iPTM/pLDDT | Alg 1 line 17, 31 |
| 6  | `model.md`               | Top-level Protenix assembly, inference loop | Alg 1 |

## 怎么读 · How to use

打开一个 lesson md → 边读边按它的章节顺序打开
`tutorials/<chapter>/<chapter>.ipynb` 的对应小节 → 在 `tutorials/<chapter>/*.py`
里填好 TODO → 跑 notebook 测试 cell 验证 → 跑章末 `generate_control_values.py
--verify --src tutorials --chapters <chapter>` 兜底。

## 参考资源 · References

- **AF3 主论文**: Abramson, J. *et al.* "Accurate structure prediction of biomolecular interactions with AlphaFold 3." *Nature* (2024). [DOI: 10.1038/s41586-024-07487-w](https://www.nature.com/articles/s41586-024-07487-w)
- **AF3 Supplementary**: 含所有 Algorithm 伪代码 (1 ~ 31)；建议下载放在手边对照看
- **The Illustrated AlphaFold**: https://elanapearl.github.io/blog/2024/the-illustrated-alphafold/ —— 可视化讲解，本系列灵感来源
- **ByteDance Protenix**: https://github.com/bytedance/Protenix —— AF3 的开源复现，我们 `solutions/` 的代码与之兼容、权重可直接加载

## 状态 · Status

| 章节 | 状态 |
|------|------|
| feature_extraction | ✅ |
| attention | ✅ |
| pairformer | ✅ |
| feature_embedding | ✅ |
| diffusion | ✅ |
| confidence | ✅ |
| model | ✅ |
