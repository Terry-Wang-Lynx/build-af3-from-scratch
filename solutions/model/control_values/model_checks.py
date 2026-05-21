"""Test helpers for ``model/`` — the integration chapter.

``model/`` 章 (顶层装配) 的测试辅助。

Unlike per-chapter unit tests, the ``model/`` chapter assembles every
sub-module into a single ``Protenix`` instance. The cheapest meaningful
test for the student's filled-in code is therefore a **state-dict
shape check**: build ``Protenix(cfg)`` with a small/tiny preset, then
verify every sub-module appears at the expected name with the expected
shape. Any wrong dim, missing module or misnamed attribute will surface
as a mismatch.

与其它章节单元测试不同，``model/`` 这一章把所有子模块装配成一个
``Protenix`` 实例。对学生填好的代码最便宜也最有效的测试就是
**state-dict 形状检查**：用 tiny 配置构造 ``Protenix(cfg)``，然后逐项
对照保存的 ``(name, shape)`` 字典 —— 任何维度错、模块漏建、命名不对
都会在这一步暴露。

This file does NOT depend on a downloaded checkpoint; only the
architecture is exercised.
本文件不依赖下载的 checkpoint，只验证架构。
"""
from __future__ import annotations

import os

from runtime.checks import (  # noqa: F401
    controlled_execution,
    controlled_forward,
    test_module_forward,
    test_module_method,
    test_module_shape,
)


CONTROL_FOLDER = os.path.dirname(os.path.abspath(__file__))


# Preset chosen for the state-dict shape test. ``protenix_tiny_default_v0.5.0``
# is the smallest preset committed to the repo (~110M params); building it
# still takes a few seconds but stays well under typical CI budgets.
# 用于 state-dict shape 测试的预设。``protenix_tiny_default_v0.5.0`` 是仓库里
# 最小的预设 (~110M 参数)，构造一次几秒钟。
MODEL_NAME = "protenix_tiny_default_v0.5.0"


def build_tiny_protenix():
    """Construct a Protenix model with the tiny preset and a minimal config.

    用 tiny 预设和最小配置构造一个 Protenix。

    Sets ``N_cycle = 1`` and reduces sampler steps so the *construction*
    itself stays light. The actual forward pass is not exercised here.
    设置 ``N_cycle = 1`` 并把采样步数调到最小，仅保证**构造**轻量；不在此处
    跑前向。
    """
    from copy import deepcopy

    from configs.configs_base import configs as base_cfg
    from configs.configs_data import data_configs
    from configs.configs_inference import inference_configs
    from configs.configs_model_type import model_configs
    from configs.parser import parse_configs
    from model.model import Protenix

    cfg = {**base_cfg, **{"data": data_configs}, **inference_configs}
    cfg.update({
        "project": "af3",
        "run_name": "ctrl",
        "base_dir": "/tmp/af3",
        "eval_interval": 0,
        "log_interval": 0,
        "input_json_path": "examples/example.json",
        "model_name": MODEL_NAME,
        "triangle_attention": "torch",
        "triangle_multiplicative": "torch",
        "enable_tf32": False,
        "enable_efficient_fusion": False,
    })
    overrides = deepcopy(model_configs[MODEL_NAME])

    def merge(d, s):
        for k, v in s.items():
            if isinstance(v, dict) and isinstance(d.get(k), dict):
                merge(d[k], v)
            else:
                d[k] = v
    merge(cfg, overrides)

    cfg = parse_configs(cfg, arg_str=None, fill_required_with_null=True)
    cfg.model.N_cycle = 1
    cfg.sample_diffusion.N_step = 1
    cfg.sample_diffusion.N_sample = 1

    return Protenix(cfg).eval()
