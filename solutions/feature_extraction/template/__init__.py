"""Template featurization for AF3.

模板特征化。

Files / 文件:
    template_parser.py     mmCIF templates → AtomArray + chain match
                           将模板 mmCIF 解析成 AtomArray 并与查询链对齐
    template_utils.py      distogram bin construction, backbone-frame helpers
                           距离-bin 构造、backbone frame 辅助
    template_featurizer.py Build per-template tensors consumed by TemplateEmbedder
                           构造 TemplateEmbedder 使用的每模板张量
"""
