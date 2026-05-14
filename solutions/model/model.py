# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from typing import Any

import torch
import torch.nn as nn

import confidence as sample_confidence
from diffusion.sampler import InferenceNoiseScheduler, sample_diffusion
from confidence.confidence_head import ConfidenceHead
from diffusion.diffusion_module import DiffusionModule
from feature_embedding.constraint_embedder import ConstraintEmbedder
from feature_embedding.input_embedder import InputFeatureEmbedder
from feature_embedding.relative_position_encoding import RelativePositionEncoding
from confidence.distogram_head import DistogramHead
from pairformer.msa_stack import MSAModule
from pairformer.pair_stack import PairformerStack
from pairformer.template_embedder import TemplateEmbedder
from attention.linear import LinearNoBias
from pairformer.triangle_ops import LayerNorm
from runtime.logger import get_logger
from runtime.torch_utils import autocasting_disable_decorator

logger = get_logger(__name__)


def update_input_feature_dict(input_feature_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Lines 1-3 of Algorithm 5 compute d_lm, v_lm, and pad_info utilized in the AtomAttentionEncoder.
    Args:
            input_feature_dict (dict[str, Any]): input features
    Returns:
            input_feature_dict (dict[str, Any]): input features
    """
    from feature_embedding.local_attention import rearrange_qk_to_dense_trunk

    with torch.no_grad():
        # Prepare tensors in dense trunks for local operations
        q_trunked_list, k_trunked_list, pad_info = rearrange_qk_to_dense_trunk(
            q=[input_feature_dict["ref_pos"], input_feature_dict["ref_space_uid"]],
            k=[input_feature_dict["ref_pos"], input_feature_dict["ref_space_uid"]],
            dim_q=[-2, -1],
            dim_k=[-2, -1],
            n_queries=32,
            n_keys=128,
            compute_mask=True,
        )
        # Compute atom pair feature
        d_lm = (
            q_trunked_list[0][..., None, :] - k_trunked_list[0][..., None, :, :]
        )  # [..., n_blocks, n_queries, n_keys, 3]
        v_lm = (
            q_trunked_list[1][..., None].int() == k_trunked_list[1][..., None, :].int()
        ).unsqueeze(
            dim=-1
        )  # [..., n_blocks, n_queries, n_keys, 1]
        input_feature_dict["d_lm"] = d_lm
        input_feature_dict["v_lm"] = v_lm
        input_feature_dict["pad_info"] = pad_info
        return input_feature_dict


class Protenix(nn.Module):
    """Top-level AlphaFold 3 model — Algorithm 1 in the AF3 paper.

    顶层 AlphaFold 3 模型 —— 对应论文 Algorithm 1。

    The forward pass runs four stages:
    前向计算包含四个阶段:

    1. **Input embedding** — build the single (s) and pair (z) representations
       from sequence, MSA, and templates.
       从序列、MSA、模板构造 single (s) 与 pair (z) 表示.
    2. **Trunk recycling** — iterate MSAModule + Pairformer for ``N_cycle``
       rounds, refining (s, z).
       通过 ``N_cycle`` 轮 MSAModule + Pairformer 反复精炼 (s, z).
    3. **Diffusion sampling** — sample atom coordinates via the conditional
       diffusion module, guided by (s, z).
       以 (s, z) 为条件，用扩散模块采样原子坐标.
    4. **Confidence heads** — predict pLDDT, PAE, PDE, resolved, distogram.
       预测 pLDDT / PAE / PDE / resolved / distogram 等置信度.
    """

    def __init__(self, configs: Any) -> None:
        super().__init__()
        self.configs = configs
        torch.backends.cuda.matmul.allow_tf32 = configs.enable_tf32
        self.enable_efficient_fusion = configs.enable_efficient_fusion
        self.N_cycle = configs.model.N_cycle

        self.inference_noise_scheduler = InferenceNoiseScheduler(
            **configs.inference_noise_scheduler
        )

        # Model
        esm_configs = configs.get("esm", {})  # This is used in InputFeatureEmbedder
        self.input_embedder = InputFeatureEmbedder(
            **configs.model.input_embedder, esm_configs=esm_configs
        )
        self.relative_position_encoding = RelativePositionEncoding(
            **configs.model.relative_position_encoding
        )
        self.template_embedder = TemplateEmbedder(**configs.model.template_embedder)
        self.msa_module = MSAModule(
            **configs.model.msa_module,
            msa_configs=configs.data.get("msa", {}),
        )
        self.constraint_embedder = ConstraintEmbedder(
            **configs.model.constraint_embedder
        )
        self.pairformer_stack = PairformerStack(**configs.model.pairformer)
        self.diffusion_module = DiffusionModule(**configs.model.diffusion_module)
        self.distogram_head = DistogramHead(**configs.model.distogram_head)
        self.confidence_head = ConfidenceHead(**configs.model.confidence_head)

        self.c_s, self.c_z, self.c_s_inputs = (
            configs.c_s,
            configs.c_z,
            configs.c_s_inputs,
        )
        self.linear_no_bias_sinit = LinearNoBias(
            in_features=self.c_s_inputs, out_features=self.c_s
        )
        self.linear_no_bias_zinit1 = LinearNoBias(
            in_features=self.c_s, out_features=self.c_z
        )
        self.linear_no_bias_zinit2 = LinearNoBias(
            in_features=self.c_s, out_features=self.c_z
        )
        self.linear_no_bias_token_bond = LinearNoBias(
            in_features=1, out_features=self.c_z
        )
        self.linear_no_bias_z_cycle = LinearNoBias(
            in_features=self.c_z, out_features=self.c_z
        )
        self.linear_no_bias_s = LinearNoBias(
            in_features=self.c_s, out_features=self.c_s
        )
        self.layernorm_z_cycle = LayerNorm(self.c_z)
        self.layernorm_s = LayerNorm(self.c_s)

        # Zero init the recycling layer
        nn.init.zeros_(self.linear_no_bias_z_cycle.weight)
        nn.init.zeros_(self.linear_no_bias_s.weight)

    def get_pairformer_output(
        self,
        input_feature_dict: dict[str, Any],
        N_cycle: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Algorithm 1 lines 1-13 — input embedding + recycling.
        Algorithm 1 第 1-13 行 —— 输入嵌入 + 循环 (recycling)。

        Returns / 返回:
            (s_inputs, s, z)
            - s_inputs: [..., N_token, c_s_inputs]  raw single embedding 原始 single 表示
            - s:       [..., N_token, c_s]          single after Pairformer
            - z:       [..., N_token, N_token, c_z] pair after Pairformer
        """
        # Lines 1-5: input embedding
        s_inputs = self.input_embedder(input_feature_dict, inplace_safe=False)
        z_constraint = None
        if "constraint_feature" in input_feature_dict:
            z_constraint = self.constraint_embedder(input_feature_dict["constraint_feature"])

        s_init = self.linear_no_bias_sinit(s_inputs)
        z_init = (
            self.linear_no_bias_zinit1(s_init)[..., None, :]
            + self.linear_no_bias_zinit2(s_init)[..., None, :, :]
        )
        z_init = z_init + self.relative_position_encoding(input_feature_dict["relp"])
        z_init = z_init + self.linear_no_bias_token_bond(
            input_feature_dict["token_bonds"].unsqueeze(-1)
        )
        if z_constraint is not None:
            z_init = z_init + z_constraint

        # Lines 6-13: recycling
        z = torch.zeros_like(z_init)
        s = torch.zeros_like(s_init)
        for _ in range(N_cycle):
            z = z_init + self.linear_no_bias_z_cycle(self.layernorm_z_cycle(z))
            if self.template_embedder.n_blocks > 0:
                z = z + self.template_embedder(input_feature_dict, z)
            z = self.msa_module(input_feature_dict, z, s_inputs, pair_mask=None)
            s = s_init + self.linear_no_bias_s(self.layernorm_s(s))
            s, z = self.pairformer_stack(s, z, pair_mask=None)

        return s_inputs, s, z

    def sample_diffusion(self, **kwargs: Any) -> torch.Tensor:
        """
        Samples diffusion process based on the provided configurations.

        Returns:
            torch.Tensor: The result of the diffusion sampling process.
        """
        _configs = {
            key: self.configs.sample_diffusion.get(key)
            for key in [
                "gamma0",
                "gamma_min",
                "noise_scale_lambda",
                "step_scale_eta",
            ]
        }
        _configs.update(
            {
                "attn_chunk_size": (
                    self.configs.infer_setting.chunk_size if not self.training else None
                ),
                "diffusion_chunk_size": (
                    self.configs.infer_setting.sample_diffusion_chunk_size
                    if not self.training
                    else None
                ),
            }
        )
        _configs.update(
            {
                "guidance_configs": self.configs.sample_diffusion.to_dict().get(
                    "guidance"
                )
            }
        )
        return autocasting_disable_decorator(self.configs.skip_amp.sample_diffusion)(
            sample_diffusion
        )(**_configs, **kwargs)

    def run_confidence_head(self, *args: Any, **kwargs: Any) -> Any:
        """
        Runs the confidence head with optional automatic mixed precision (AMP) disabled.

        Returns:
            Any: The output of the confidence head.
        """
        return autocasting_disable_decorator(self.configs.skip_amp.confidence_head)(
            self.confidence_head
        )(*args, **kwargs)

    def main_inference_loop(
        self,
        input_feature_dict: dict[str, Any],
        label_dict: Any = None,
        N_cycle: int = 1,
        mode: str = "inference",
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """Single-seed inference loop. Returns (pred_dict, log_dict, time_tracker)."""
        del label_dict
        step_st = time.time()

        log_dict: dict[str, Any] = {}
        pred_dict: dict[str, Any] = {}
        time_tracker: dict[str, float] = {}

        s_inputs, s, z = self.get_pairformer_output(input_feature_dict, N_cycle)

        # Free MSA / template features once the trunk is done with them.
        for key in [k for k in list(input_feature_dict)
                    if k.startswith("template_") or k in {
                        "msa", "has_deletion", "deletion_value",
                        "profile", "deletion_mean"}]:
            del input_feature_dict[key]

        step_trunk = time.time()
        time_tracker["pairformer"] = step_trunk - step_st

        # ---- Diffusion sampling ----
        N_sample = self.configs.sample_diffusion["N_sample"]
        N_step = self.configs.sample_diffusion["N_step"]
        noise_schedule = self.inference_noise_scheduler(
            N_step=N_step, device=s_inputs.device, dtype=s_inputs.dtype
        )

        # Cache the conditioning that doesn't depend on the diffusion timestep.
        cache_pair_z = autocasting_disable_decorator(
            self.configs.skip_amp.sample_diffusion
        )(self.diffusion_module.diffusion_conditioning.prepare_cache)(
            input_feature_dict["relp"], z, False
        )
        cache_p_lm, cache_c_l = autocasting_disable_decorator(
            self.configs.skip_amp.sample_diffusion
        )(self.diffusion_module.atom_attention_encoder.prepare_cache)(
            ref_pos=input_feature_dict["ref_pos"],
            ref_charge=input_feature_dict["ref_charge"],
            ref_mask=input_feature_dict["ref_mask"],
            ref_element=input_feature_dict["ref_element"],
            ref_atom_name_chars=input_feature_dict["ref_atom_name_chars"],
            atom_to_token_idx=input_feature_dict["atom_to_token_idx"],
            d_lm=input_feature_dict["d_lm"],
            v_lm=input_feature_dict["v_lm"],
            pad_info=input_feature_dict["pad_info"],
            r_l=True,
            z=cache_pair_z,
            inplace_safe=False,
        )

        pred_dict["coordinate"] = self.sample_diffusion(
            denoise_net=self.diffusion_module,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=None,
            pair_z=cache_pair_z,
            p_lm=cache_p_lm,
            c_l=cache_c_l,
            N_sample=N_sample,
            noise_schedule=noise_schedule,
            inplace_safe=True,
            enable_efficient_fusion=self.enable_efficient_fusion,
        )

        step_diffusion = time.time()
        time_tracker["diffusion"] = step_diffusion - step_trunk

        # ---- Heads ----
        pred_dict["contact_probs"] = autocasting_disable_decorator(True)(
            sample_confidence.compute_contact_prob
        )(
            distogram_logits=self.distogram_head(z),
            **sample_confidence.get_bin_params(self.configs.loss.distogram),
        )

        (
            pred_dict["plddt"],
            pred_dict["pae"],
            pred_dict["pde"],
            pred_dict["resolved"],
        ) = self.run_confidence_head(
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=z,
            pair_mask=None,
            x_pred_coords=pred_dict["coordinate"],
        )
        step_confidence = time.time()
        time_tracker["confidence"] = step_confidence - step_diffusion
        time_tracker["model_forward"] = time.time() - step_st

        # ---- Summary confidence + per-sample dump payload ----
        (
            pred_dict["summary_confidence"],
            pred_dict["full_data"],
        ) = autocasting_disable_decorator(True)(
            sample_confidence.compute_full_data_and_summary
        )(
            configs=self.configs,
            pae_logits=pred_dict["pae"],
            plddt_logits=pred_dict["plddt"],
            pde_logits=pred_dict["pde"],
            contact_probs=pred_dict["contact_probs"],
            token_asym_id=input_feature_dict["asym_id"],
            token_has_frame=input_feature_dict["has_frame"],
            atom_coordinate=pred_dict["coordinate"],
            atom_to_token_idx=input_feature_dict["atom_to_token_idx"],
            atom_is_polymer=1 - input_feature_dict["is_ligand"],
            N_recycle=N_cycle,
            interested_atom_mask=None,
            return_full_data=True,
            mol_id=None,
            elements_one_hot=None,
        )

        return pred_dict, log_dict, time_tracker

    def forward(
        self,
        input_feature_dict: dict[str, Any],
        label_full_dict: Any = None,
        label_dict: Any = None,
        mode: str = "inference",
        **_ignored: Any,
    ) -> tuple[dict[str, torch.Tensor], None, dict[str, Any]]:
        """Inference-only forward pass.

        Only ``mode='inference'`` is supported. The label arguments are
        accepted for signature compatibility and ignored.
        仅支持 ``mode='inference'``；label_* 参数仅为签名兼容，运行时忽略。
        """
        assert mode == "inference", "only `mode='inference'` is supported."
        del label_full_dict, label_dict

        input_feature_dict = self.relative_position_encoding.generate_relp(input_feature_dict)
        input_feature_dict = update_input_feature_dict(input_feature_dict)

        pred_dict, log_dict, time_tracker = self.main_inference_loop(
            input_feature_dict=input_feature_dict,
            label_dict=None,
            N_cycle=self.N_cycle,
            mode=mode,
        )
        log_dict.update({"time": time_tracker})

        return pred_dict, None, log_dict
