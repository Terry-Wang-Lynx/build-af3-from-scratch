# Lessons · 章节讲解

每个 markdown 文件对应一个章节，相当于配套"视频脚本"——讲清楚概念、动机、
和论文 Algorithm 编号的对应。

Each markdown corresponds to one chapter — a "lesson script" that walks
through the concept, the motivation, and the mapping to the AF3 paper's
Algorithm numbers.

## 章节顺序 · Chapter order

| #  | 章节 · Chapter | 内容 · What it covers | AF3 Alg. |
|----|----------------|-----------------------|----------|
| 0  | `series_introduction.md` | 项目概览，AF3 vs AF2     | —        |
| 1  | `attention.md`           | MHA, LayerNorm, AdaLN, AttentionPairBias | Alg 24 |
| 2  | `feature_extraction.md`  | JSON → tensors, MSA, templates | — |
| 3  | `feature_embedding.md`   | InputFeatureEmbedder, AtomAttentionEncoder, Relative PE | Alg 2, 5, 23 |
| 4  | `pairformer.md`          | Pair stack + MSA module + triangle ops | Alg 8, 10, 11, 12, 13, 14, 16, 17 |
| 5  | `diffusion.md`           | Diffusion module + transformer + sampler | Alg 18, 21, 23, 25 |
| 6  | `confidence.md`          | Confidence head + pTM/iPTM/pLDDT | Alg 26-31 |
| 7  | `model.md`               | Top-level assembly, inference loop | Alg 1 |

## 怎么读 · How to use

每个 markdown 写完后，对应章节下的 `tutorials/<chapter>/<chapter>.ipynb`
里就有一个 *narrative* 部分（从 markdown 抽出来的解释）+ 几个 *exercise*
代码 cell（带 TODO）。学生顺着 notebook 实现这一章的核心类，跑测试，
然后进入下一章。

After each lesson is written, the corresponding
`tutorials/<chapter>/<chapter>.ipynb` notebook contains a narrative
(extracted from the markdown) plus exercise code cells with `TODO`
blocks. Students implement the chapter's core classes, verify against
`control_values/`, then move on.

如果某一章的 markdown 还没写，可以直接看
`solutions/<chapter>/` 里的代码——文件名和类名都按论文的 Algorithm
来命名，搭配 AF3 论文 Supplementary 看就够了。

If a chapter's markdown isn't written yet, just read
`solutions/<chapter>/` directly — files and classes are named after the
paper's Algorithm numbers, so the AF3 SI is enough to follow along.

## 状态 · Status

| 章节 | 状态 |
|------|------|
| series_introduction | TBD |
| attention | TBD |
| feature_extraction | TBD |
| feature_embedding | TBD |
| pairformer | TBD |
| diffusion | TBD |
| confidence | TBD |
| model | TBD |
