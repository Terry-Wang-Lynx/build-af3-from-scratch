"""Top-level model assembly and inference driver.

顶层模型装配 + 推理入口。

``model.model.Protenix`` is Algorithm 1 from the AF3 paper — it composes the
input embedder, pairformer trunk, diffusion module, and confidence head into
one ``nn.Module``.

``model.model.Protenix`` 对应论文 Algorithm 1，把输入嵌入、Pairformer 主干、
扩散模块、置信度头组装成一个 ``nn.Module``。

Files / 文件:
    model.py        Protenix  (Algorithm 1)
    inference.py    CLI runner — load config + weights, run forward, dump CIF
    utils.py        Small helpers shared by the model
"""
