"""Standalone inference driver — 独立推理入口.

Usage / 用法 (run from ``solutions/``):

    cd solutions
    LAYERNORM_TYPE=torch python -m model.inference \\
        --model_name protenix_tiny_default_v0.5.0 \\
        --input_json examples/example.json \\
        --dump_dir   ./out \\
        --ckpt_dir   ../checkpoints \\
        --device     mps          # or cpu / cuda

Steps / 流程:
  1. Parse model + inference config.
     解析模型 + 推理配置.
  2. Build the model.
     构建模型.
  3. Load the model checkpoint.
     加载权重.
  4. Featurize the input JSON.
     把输入 JSON 转成张量.
  5. Run one forward pass + diffusion sampling.
     一次前向 + 扩散采样.
  6. Dump mmCIF + per-sample confidence summary JSON.
     输出 mmCIF + 每样本的 confidence summary JSON.

Pure PyTorch; pass ``--device mps`` on Apple Silicon.
纯 PyTorch；Apple Silicon 上加 ``--device mps``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from argparse import Namespace
from copy import deepcopy
from os.path import exists, join

# Force the pure-PyTorch LayerNorm path (no CUDA JIT compile).
os.environ.setdefault("LAYERNORM_TYPE", "torch")

import numpy as np
import torch
# ESM ckpts contain argparse.Namespace; allow it through torch.load's safe-globals filter.
torch.serialization.add_safe_globals([Namespace])

# Make `solutions/` importable so that `python -m model.inference` works.
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from configs.parser import parse_configs
from configs.configs_base import configs as _configs_base
from configs.configs_data import data_configs as _data_configs
from configs.configs_inference import inference_configs as _inference_configs
from configs.configs_model_type import model_configs as _model_configs
from feature_extraction.inference.infer_dataloader import get_inference_dataloader
from feature_extraction.utils import save_structure_cif
from model.model import Protenix
from runtime.seed import seed_everything
from runtime.torch_utils import round_values, to_device

logger = logging.getLogger("af3")


def pick_device(prefer: str = "auto") -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer in ("cuda", "auto") and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer in ("mps", "auto") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_config(model_name: str, overrides: dict | None = None):
    """Merge base + model-specific configs and fill in required scalars."""
    base = {**_configs_base, **{"data": _data_configs}, **_inference_configs}
    base.update({
        "project": "af3",
        "run_name": "inference",
        "base_dir": "/tmp/af3",
        "eval_interval": 0,
        "log_interval": 0,
        "input_json_path": "/dev/null",
        "model_name": model_name,
        "triangle_attention": "torch",
        "triangle_multiplicative": "torch",
        "enable_tf32": False,
        "enable_efficient_fusion": False,
    })
    if model_name in _model_configs:
        merged = deepcopy(_model_configs[model_name])

        def _merge(dst, src):
            for k, v in src.items():
                if isinstance(v, dict) and isinstance(dst.get(k), dict):
                    _merge(dst[k], v)
                else:
                    dst[k] = v

        _merge(base, merged)
    if overrides:
        base.update(overrides)
    return parse_configs(base, arg_str=None, fill_required_with_null=True)


def load_checkpoint(model: torch.nn.Module, ckpt_path: str) -> None:
    """Load a model checkpoint, stripping any ``module.`` DDP prefix."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    state = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state.items()
    }
    res = model.load_state_dict(state, strict=False)
    logger.info(
        "Loaded %s — missing=%d unexpected=%d",
        os.path.basename(ckpt_path),
        len(res.missing_keys),
        len(res.unexpected_keys),
    )


def dump_sample(
    *,
    name: str,
    seed: int,
    sample_idx: int,
    atom_array,
    atom_positions: torch.Tensor,
    entity_poly_type: dict,
    summary_confidence: dict,
    out_dir: str,
) -> None:
    """Write one sample's CIF + summary JSON."""
    pred_dir = join(out_dir, name, f"seed_{seed}", "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    cif_path = join(pred_dir, f"{name}_sample_{sample_idx}.cif")
    save_structure_cif(
        atom_array=atom_array,
        pred_coordinate=atom_positions,
        output_fpath=cif_path,
        entity_poly_type=entity_poly_type,
        pdb_id=name,
    )
    summary_path = join(pred_dir, f"{name}_summary_confidence_sample_{sample_idx}.json")
    rounded = round_values(deepcopy(summary_confidence))

    def _default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer, np.bool_)):
            return o.item()
        if isinstance(o, torch.Tensor):
            return o.cpu().tolist()
        return float(o)

    with open(summary_path, "w") as f:
        json.dump(rounded, f, indent=4, default=_default)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(
        description="AlphaFold 3 inference · AF3 推理"
    )
    ap.add_argument("--model_name", default="protenix_tiny_default_v0.5.0")
    ap.add_argument("--ckpt_dir", default=join(_HERE, "checkpoints"))
    ap.add_argument("--input_json", required=True)
    ap.add_argument("--dump_dir", default="/tmp/af3_out")
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--n_cycle", type=int, default=1)
    ap.add_argument("--n_step", type=int, default=5, help="Diffusion steps")
    ap.add_argument("--n_sample", type=int, default=1, help="Diffusion samples")
    args = ap.parse_args(argv)

    # ---- Validate paths early, before any expensive work ----
    # Forgetting the checkpoint download or mistyping the input JSON are the two
    # most common first-run mistakes. Catch them here (before seeding / building
    # the 100M+ param model / loading weights) and report via ap.error(), which
    # prints usage + a concrete message and exits with code 2 — no raw traceback.
    # 提前校验路径：最常见的两个首次运行错误（没下权重 / JSON 路径写错）在
    # 构建模型、加载权重之前就拦截，用 ap.error() 给出可读报错并以退出码 2 退出。
    if not exists(args.input_json):
        ap.error(f"--input_json not found: {args.input_json}")
    ckpt_path = join(args.ckpt_dir, f"{args.model_name}.pt")
    if not exists(args.ckpt_dir):
        ap.error(
            f"--ckpt_dir not found: {args.ckpt_dir}\n"
            "Download the checkpoint first (see the README 'Download a "
            "checkpoint' section)."
        )
    if not exists(ckpt_path):
        ap.error(
            f"checkpoint file not found: {ckpt_path}\n"
            f"(--ckpt_dir '{args.ckpt_dir}' exists but has no "
            f"'{args.model_name}.pt'; check --model_name / --ckpt_dir.)"
        )

    seed_everything(args.seed, deterministic=False)
    device = pick_device(args.device)
    logger.info("Device: %s", device)

    # ---- Config ----
    cfg = build_config(
        args.model_name,
        overrides={"dump_dir": args.dump_dir, "input_json_path": args.input_json, "seeds": [args.seed]},
    )
    cfg.model.N_cycle = args.n_cycle
    cfg.sample_diffusion.N_step = args.n_step
    cfg.sample_diffusion.N_sample = args.n_sample

    # ---- Model + weights ----
    logger.info("Building Protenix(%s)", args.model_name)
    model = Protenix(cfg).eval()
    n = sum(p.numel() for p in model.parameters())
    logger.info("Parameters: %.2f M", n / 1e6)

    # ckpt_path was validated to exist right after arg parsing.
    load_checkpoint(model, ckpt_path)
    model = model.to(device)

    # ---- Data ----
    os.makedirs(args.dump_dir, exist_ok=True)
    logger.info("Loading data from %s", args.input_json)
    loader = get_inference_dataloader(configs=cfg)

    # ---- Inference loop ----
    # The dataloader yields a `list` of `(feat_dict, atom_array, error_message)`
    # tuples (one per sample after collation).
    for batch in loader:
        data, atom_array, err = batch[0]
        name = data.get("sample_name", "sample")
        if err:
            logger.error("Data error for %s: %s", name, err)
            continue

        logger.info(
            "%s: N_token=%d, N_atom=%d, N_msa=%d",
            name,
            int(data["N_token"]),
            int(data["N_atom"]),
            int(data["N_msa"]),
        )

        data = to_device(data, device)
        feats = data["input_feature_dict"]
        entity_poly_type = {
            k: v for k, v in data["entity_poly_type"].items() if v != "non-polymer"
        }

        t0 = time.time()
        # bf16 autocast on CUDA; fp32 elsewhere (CPU / MPS).
        # CUDA 用 bf16 autocast 加速；CPU / MPS 走 fp32。
        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=False)
        )
        with torch.no_grad(), amp_ctx:
            pred, _, log = model(input_feature_dict=feats, mode="inference")
        logger.info("%s: forward in %.2fs", name, time.time() - t0)

        coords = pred["coordinate"]               # [N_sample, N_atom, 3]
        summaries = pred["summary_confidence"]    # list[dict], len N_sample

        # Sort samples by ranking_score (descending) — best sample first.
        ranking = [float(s["ranking_score"]) for s in summaries]
        order = sorted(range(len(ranking)), key=lambda i: -ranking[i])

        for out_idx, src in enumerate(order):
            s = summaries[src]
            logger.info(
                "  sample_%d (src=%d): plddt=%.2f ptm=%.3f rank=%.3f",
                out_idx, src,
                float(s["plddt"]), float(s["ptm"]), float(s["ranking_score"]),
            )
            dump_sample(
                name=name,
                seed=args.seed,
                sample_idx=out_idx,
                atom_array=atom_array,
                atom_positions=coords[src],
                entity_poly_type=entity_poly_type,
                summary_confidence=s,
                out_dir=args.dump_dir,
            )

    logger.info("Done. Output: %s", args.dump_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
