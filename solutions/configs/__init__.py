"""Config dictionaries and ConfigDict parser.

配置字典 + ConfigDict 解析器。

Files / 文件:
    configs_base.py         base config (default values for all knobs)
    configs_data.py         data-pipeline config
    configs_inference.py    inference-time config
    configs_model_type.py   per-model overrides (Tiny / Mini / Base / Mini-ESM …)
    parser.py               ``parse_configs`` — merge dicts into a ConfigDict
    extend_types.py         Typed-value sentinels (GlobalConfigValue, RequiredValue, …)
"""
