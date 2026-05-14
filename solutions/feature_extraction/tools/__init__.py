"""External-tool wrappers used by the data pipeline.

数据流水线用到的外部工具封装。

These are needed only if you build MSAs / templates from scratch; the
default path loads pre-computed MSAs from ``examples/``.

只有自己从头构造 MSA / 模板时才会用到；默认流程从 ``examples/`` 读取
预先算好的 MSA。

Files / 文件:
    common.py            shared subprocess helpers / 子进程封装
    kalign.py            kalign multi-sequence alignment binding
    search.py            HMMER / jackhmmer / kalign search drivers
    logger.py            MMCIFStatsLogger (used by parser.py)
    rewrite_biotite.py   small biotite patches used by parser.py
"""
