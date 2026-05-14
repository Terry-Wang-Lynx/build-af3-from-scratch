"""Core data structures + featurization.

数据核心结构 + 特征化。

Files / 文件:
    ccd.py                Chemical Component Dictionary lookups / 化学组件字典查询
    parser.py             JSON spec → biotite AtomArray / JSON 输入解析为 AtomArray
    featurizer.py         AtomArray → tensor feature dict / AtomArray 转张量特征
    filter.py             quality filters (training only) / 质量过滤 (仅训练用)
    geometry_featurizer.py  per-token frame helpers / 每 token 的 frame 计算
    substructure_perms.py   chirality / symmetry equivalence (training only)
"""
