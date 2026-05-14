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

from typing import Any, Optional, Union

import torch
from ml_collections.config_dict import ConfigDict

from confidence.bins import (
    compute_contact_prob,
    get_bin_params,
    logits_to_score,
)
from confidence.clash import calculate_clash, calculate_vdw_clash
from confidence.scores import (
    calculate_chain_based_gpde,
    calculate_chain_based_plddt,
    calculate_chain_based_ptm,
    calculate_chain_pair_pae,
    calculate_iptm,
    calculate_ptm,
)



def traverse_and_aggregate(dict_list, aggregation_func=None):
    """Merge a list of dicts into one dict by joining leaf values into lists."""
    merged_dict = {}
    all_keys = set().union(*dict_list)
    for key in all_keys:
        agg_value = [m[key] for m in dict_list if key in m]
        if isinstance(agg_value[0], dict):
            merged_dict[key] = traverse_and_aggregate(agg_value, aggregation_func)
        else:
            if aggregation_func is not None:
                agg_value = aggregation_func(agg_value)
            merged_dict[key] = agg_value
    return merged_dict


def merge_per_sample_confidence_scores(
    summary_confidence_list: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Merge confidence scores from multiple samples into a single dictionary.

    Args:
        summary_confidence_list (list[dict[str, Any]]): List of dictionaries containing confidence scores for each sample.

    Returns:
        dict[str, Any]: Merged dictionary of confidence scores.
    """

    def stack_score(tensor_list: list[torch.Tensor]) -> torch.Tensor:
        if tensor_list[0].dim() == 0:
            tensor_list = [x.unsqueeze(0) for x in tensor_list]
        score = torch.stack(tensor_list, dim=0)
        return score

    return traverse_and_aggregate(summary_confidence_list, aggregation_func=stack_score)


def _compute_full_data_and_summary(
    configs: ConfigDict,
    pae_logits: torch.Tensor,
    plddt_logits: torch.Tensor,
    pde_logits: torch.Tensor,
    contact_probs: torch.Tensor,
    token_asym_id: torch.Tensor,
    token_has_frame: torch.Tensor,
    atom_coordinate: torch.Tensor,
    atom_to_token_idx: torch.Tensor,
    atom_is_polymer: torch.Tensor,
    N_recycle: int,
    interested_atom_mask: Optional[torch.Tensor] = None,
    elements_one_hot: Optional[torch.Tensor] = None,
    mol_id: Optional[torch.Tensor] = None,
    return_full_data: bool = False,
) -> tuple[list[dict], list[dict]]:
    """
    Compute full data and summary confidence scores for the given inputs.

    Args:
        configs: Configuration object.
        pae_logits (torch.Tensor): Logits for PAE (Predicted Aligned Error).
        plddt_logits (torch.Tensor): Logits for pLDDT (Predicted Local Distance Difference Test).
        pde_logits (torch.Tensor): Logits for PDE (Predicted Distance Error).
        contact_probs (torch.Tensor): Contact probabilities.
        token_asym_id (torch.Tensor): Asymmetric ID for tokens.
        token_has_frame (torch.Tensor): Indicator for tokens having a frame.
        atom_coordinate (torch.Tensor): Atom coordinates.
        atom_to_token_idx (torch.Tensor): Mapping from atoms to tokens.
        atom_is_polymer (torch.Tensor): Indicator for atoms being part of a polymer.
        N_recycle (int): Number of recycles.
        interested_atom_mask (Optional[torch.Tensor]): Mask for interested atoms. Defaults to None.
        elements_one_hot (Optional[torch.Tensor]): One-hot encoding for elements. Defaults to None.
        mol_id (Optional[torch.Tensor]): Molecular ID. Defaults to None.
        return_full_data (bool): Whether to return full data. Defaults to False.

    Returns:
        tuple[list[dict], list[dict]]:
            - summary_confidence: List of dictionaries containing summary confidence scores.
            - full_data: List of dictionaries containing full data if `return_full_data` is True.
    """
    atom_is_ligand = (1 - atom_is_polymer).long()
    token_is_ligand = torch.zeros_like(token_asym_id).scatter_add(
        0, atom_to_token_idx, atom_is_ligand
    )
    token_is_ligand = token_is_ligand > 0

    full_data = {}
    full_data["atom_plddt"] = logits_to_score(
        plddt_logits, **get_bin_params(configs.loss.plddt)
    )  # [N_s, N_atom]
    # Cpu offload for saving cuda memory
    pde_logits = pde_logits.to(plddt_logits.device)
    full_data["token_pair_pde"] = logits_to_score(
        pde_logits, **get_bin_params(configs.loss.pde)
    )  # [N_s, N_token, N_token]
    del pde_logits
    full_data["contact_probs"] = contact_probs.clone()  # [N_token, N_token]
    pae_logits = pae_logits.to(plddt_logits.device)
    full_data["token_pair_pae"], pae_prob = logits_to_score(
        pae_logits, **get_bin_params(configs.loss.pae), return_prob=True
    )  # [N_s, N_token, N_token]
    del pae_logits

    summary_confidence = {}
    summary_confidence["plddt"] = full_data["atom_plddt"].mean(dim=-1) * 100  # [N_s, ]
    summary_confidence["gpde"] = (
        full_data["token_pair_pde"] * full_data["contact_probs"]
    ).sum(dim=[-1, -2]) / full_data["contact_probs"].sum(dim=[-1, -2])

    summary_confidence["ptm"] = calculate_ptm(
        pae_prob, has_frame=token_has_frame, **get_bin_params(configs.loss.pae)
    )  # [N_s, ]
    summary_confidence["iptm"] = calculate_iptm(
        pae_prob,
        has_frame=token_has_frame,
        asym_id=token_asym_id,
        **get_bin_params(configs.loss.pae)
    )  # [N_s, ]

    # Add: 'chain_gpde', 'chain_pair_gpde'
    summary_confidence.update(
        calculate_chain_based_gpde(
            token_pair_pde=full_data["token_pair_pde"],
            contact_probs=full_data["contact_probs"],
            asym_id=token_asym_id,
        )
    )
    # Add: 'chain_pair_iptm', 'chain_pair_iptm_global' 'chain_iptm', 'chain_ptm'
    summary_confidence.update(
        calculate_chain_based_ptm(
            pae_prob,
            has_frame=token_has_frame,
            asym_id=token_asym_id,
            token_is_ligand=token_is_ligand,
            **get_bin_params(configs.loss.pae)
        )
    )
    # Add: 'chain_plddt', 'chain_pair_plddt'
    summary_confidence.update(
        calculate_chain_based_plddt(
            full_data["atom_plddt"], token_asym_id, atom_to_token_idx
        )
    )
    # Add: 'chain_pair_pae_mean', 'chain_pair_pae_min'
    summary_confidence.update(
        calculate_chain_pair_pae(
            token_pair_pae=full_data["token_pair_pae"],
            asym_id=token_asym_id,
            token_has_frame=token_has_frame,
        )
    )
    del pae_prob
    summary_confidence["has_clash"] = calculate_clash(
        atom_coordinate,
        token_asym_id,
        atom_to_token_idx,
        atom_is_polymer,
        configs.metrics.clash.af3_clash_threshold,
    )
    summary_confidence["num_recycles"] = torch.tensor(
        N_recycle, device=atom_coordinate.device
    )

    summary_confidence["disorder"] = torch.zeros_like(summary_confidence["ptm"])
    summary_confidence["ranking_score"] = (
        0.8 * summary_confidence["iptm"]
        + 0.2 * summary_confidence["ptm"]
        + 0.5 * summary_confidence["disorder"]
        - 100 * summary_confidence["has_clash"]
    )
    if interested_atom_mask is not None:
        token_idx = atom_to_token_idx[interested_atom_mask[0].bool()].long()
        asym_ids = token_asym_id[token_idx]
        assert len(torch.unique(asym_ids)) == 1
        interested_asym_id = asym_ids[0].item()
        N_chains = token_asym_id.max().long().item() + 1
        pb_ranking_score = summary_confidence["chain_pair_iptm_global"][
            :, interested_asym_id, torch.arange(N_chains) != interested_asym_id
        ]  # [N_s, N_chain - 1]
        summary_confidence["pb_ranking_score"] = pb_ranking_score[:, 0]
        if elements_one_hot is not None and mol_id is not None:
            vdw_clash = calculate_vdw_clash(
                pred_coordinate=atom_coordinate,
                asym_id=token_asym_id,
                mol_id=mol_id,
                is_polymer=atom_is_polymer,
                atom_token_idx=atom_to_token_idx,
                elements_one_hot=elements_one_hot,
                threshold=configs.metrics.clash.vdw_clash_threshold,
            )
            N_sample = atom_coordinate.shape[0]
            vdw_clash_per_sample_flag = (
                vdw_clash[:, interested_asym_id, :].reshape(N_sample, -1).max(dim=-1)[0]
            )
            summary_confidence["has_vdw_pl_clash"] = vdw_clash_per_sample_flag
            summary_confidence["pb_ranking_score_vdw_penalized"] = (
                summary_confidence["pb_ranking_score"] - 100 * vdw_clash_per_sample_flag
            )

    summary_confidence = break_down_to_per_sample_dict(
        summary_confidence, shared_keys=["num_recycles"]
    )

    if return_full_data:
        # save extra inputs that are used for computing summary_confidence
        full_data["token_has_frame"] = token_has_frame.clone()
        full_data["token_asym_id"] = token_asym_id.clone()
        full_data["atom_to_token_idx"] = atom_to_token_idx.clone()
        full_data["atom_is_polymer"] = atom_is_polymer.clone()
        full_data["atom_coordinate"] = atom_coordinate.clone()

        full_data = break_down_to_per_sample_dict(
            full_data,
            shared_keys=[
                "contact_probs",
                "token_has_frame",
                "token_asym_id",
                "atom_to_token_idx",
                "atom_is_polymer",
            ],
        )
        return summary_confidence, full_data
    else:
        return summary_confidence, [{}]


def break_down_to_per_sample_dict(
    input_dict: dict[str, Any], shared_keys: list[str] = []
) -> list[dict[str, Any]]:
    """
    Break down a dictionary containing tensors into a list of dictionaries, each corresponding to a sample.

    Args:
        input_dict (dict[str, Any]): Dictionary containing tensors.
        shared_keys (list[str]): List of keys that are shared across all samples. Defaults to an empty list.

    Returns:
        list[dict[str, Any]]: List of dictionaries, each containing data for a single sample.
    """
    per_sample_keys = [key for key in input_dict if key not in shared_keys]
    assert len(per_sample_keys) > 0
    N_sample = input_dict[per_sample_keys[0]].size(0)
    for key in per_sample_keys:
        assert input_dict[key].size(0) == N_sample

    per_sample_dict_list = []
    for i in range(N_sample):
        sample_dict = {key: input_dict[key][i] for key in per_sample_keys}
        sample_dict.update({key: input_dict[key] for key in shared_keys})
        per_sample_dict_list.append(sample_dict)

    return per_sample_dict_list


@torch.no_grad()


def compute_full_data_and_summary(
    configs: ConfigDict,
    pae_logits: torch.Tensor,
    plddt_logits: torch.Tensor,
    pde_logits: torch.Tensor,
    contact_probs: torch.Tensor,
    token_asym_id: torch.Tensor,
    token_has_frame: torch.Tensor,
    atom_coordinate: torch.Tensor,
    atom_to_token_idx: torch.Tensor,
    atom_is_polymer: torch.Tensor,
    N_recycle: int,
    return_full_data: bool = False,
    interested_atom_mask: Optional[torch.Tensor] = None,
    mol_id: Optional[torch.Tensor] = None,
    elements_one_hot: Optional[torch.Tensor] = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Wrapper of `_compute_full_data_and_summary` by enumerating over N samples"""

    N_sample = pae_logits.size(0)
    if contact_probs.dim() == 2:
        # Convert to [N_sample, N_token, N_token]
        contact_probs = contact_probs.unsqueeze(dim=0).expand(N_sample, -1, -1)
    else:
        assert contact_probs.dim() == 3
    assert (
        contact_probs.size(0) == plddt_logits.size(0) == pde_logits.size(0) == N_sample
    )

    summary_confidence = []
    full_data = []
    for i in range(N_sample):
        summary_confidence_i, full_data_i = _compute_full_data_and_summary(
            configs=configs,
            pae_logits=pae_logits[i : i + 1],
            plddt_logits=plddt_logits[i : i + 1],
            pde_logits=pde_logits[i : i + 1],
            contact_probs=contact_probs[i],
            token_asym_id=token_asym_id,
            token_has_frame=token_has_frame,
            atom_coordinate=atom_coordinate[i : i + 1],
            atom_to_token_idx=atom_to_token_idx,
            atom_is_polymer=atom_is_polymer,
            N_recycle=N_recycle,
            interested_atom_mask=interested_atom_mask,
            return_full_data=return_full_data,
            mol_id=mol_id,
            elements_one_hot=elements_one_hot,
        )
        summary_confidence.extend(summary_confidence_i)
        full_data.extend(full_data_i)
    return summary_confidence, full_data