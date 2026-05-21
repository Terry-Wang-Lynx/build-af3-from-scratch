"""Feature embedding — turn raw features into the single (s) and pair (z) representations.

特征嵌入 —— 把原始特征转成 single (s) 与 pair (z) 表示。

Sits between ``feature_extraction`` (data) and ``pairformer`` (trunk). It
aggregates per-atom features into per-token representations, projects
trunk single+pair onto the atom level for the diffusion module, and
encodes relative residue / chain position into ``z``.

承接 ``feature_extraction``、对接 ``pairformer``：把原子级特征聚合成
token 级 single 表示；把主干 single+pair 投影到原子级供扩散模块使用；
把残基 / 链相对位置编码进 ``z``。

Files / 文件:
    input_embedder.py                InputFeatureEmbedder            (Algorithm 2)
    relative_position_encoding.py    RelativePositionEncoding, FourierEmbedding
    constraint_embedder.py           Constraint conditioning
    atom_attention.py                AtomTransformer / Encoder / Decoder  (Algorithm 5/6/7)
    local_attention.py               Sequence-local atom attention helpers
"""
