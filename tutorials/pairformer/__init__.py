"""Pairformer trunk — pair stack, MSA module, template embedder, triangle ops.

主干模块 —— pair stack、MSA 模块、template embedder、三角形算子。

Each Pairformer block refines the pair representation ``z`` via triangle
multiplicative update, triangle attention, and transition, plus a
single-track attention biased by ``z``.

每个 Pairformer block 用三角形乘法、三角形注意力、Transition 来精炼 pair
表示 ``z``，并跑一次以 ``z`` 作为偏置的 single attention。

Files / 文件:
    triangle_ops.py       OpenfoldLinear, OuterProductMean, init helpers
    triangle.py           TriangleAttention, TriangleMultiplicativeUpdate (Alg 11/12/13/14)
    dropout.py            row-shared dropout + fused add
    pair_stack.py         PairformerBlock, PairformerStack  (Algorithm 17)
    msa_stack.py          MSAModule, MSABlock, MSAStack, MSAPairWeightedAveraging  (Alg 8/10)
    template_embedder.py  TemplateEmbedder  (Algorithm 16)
"""
