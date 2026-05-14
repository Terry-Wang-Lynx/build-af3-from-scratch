"""Data layer — parse the input JSON, fetch MSA, build tensor features.

数据层 —— 解析输入 JSON、加载 MSA、构造张量特征。

Turns a JSON spec of sequences + ligands + optional pre-computed MSA / templates
into the tensor dict the model consumes.

把序列 + 配体 + 可选 MSA / 模板的 JSON 描述转成模型可用的张量字典。

Sub-packages / 子包:
    core/                 CCD chemistry, atom array, residue tokenizer
    inference/            JSON parser → feature dict, inference dataloader
    msa/                  MSA loading + sampling
    template/             Template featurization
    esm/                  ESM-2 embedding featurizer (PLM models only)
    constraint/           Constraint embedder (pocket / contact / substructure)
    tools/                Logging, kalign wrapper, search helpers
"""
