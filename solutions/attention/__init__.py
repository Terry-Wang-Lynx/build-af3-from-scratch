"""Attention primitives — linear, layer-norm, transition, multi-head attention.

注意力底层模块 —— 线性层、LayerNorm、Transition、多头注意力。

Files / 文件:
    linear.py              Linear, LinearNoBias, BiasInitLinear
    layer_norm.py          OpenFoldLayerNorm
    transition.py          Transition (FFN), AdaptiveLayerNorm
    mha.py                 Attention, DropPath
    attention_pair_bias.py AttentionPairBias  (Algorithm 24)
"""
