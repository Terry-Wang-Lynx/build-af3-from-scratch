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

from confidence.external_clash import Clash



def calculate_vdw_clash(
    pred_coordinate: torch.Tensor,
    asym_id: torch.LongTensor,
    mol_id: torch.LongTensor,
    atom_token_idx: torch.LongTensor,
    is_polymer: torch.BoolTensor,
    elements_one_hot: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """
    Calculate Van der Waals (VDW) clash for predicted coordinates.

    Args:
        pred_coordinate (torch.Tensor): Predicted coordinates of atoms.
            Shape: [N_sample, N_atom, 3]
        asym_id (torch.LongTensor): Asymmetric ID for tokens.
            Shape: [N_token]
        mol_id (torch.LongTensor): Molecular ID.
            Shape: [N_atom]
        atom_token_idx (torch.LongTensor): Mapping from atoms to tokens.
            Shape: [N_atom]
        is_polymer (torch.BoolTensor): Indicator for atoms being part of a polymer.
            Shape: [N_atom]
        elements_one_hot (torch.Tensor): One-hot encoding for elements.
            Shape: [N_atom, N_elements]
        threshold (float): Threshold for VDW clash detection.

    Returns:
        torch.Tensor: VDW clash summary.
            Shape: [N_sample]
    """
    clash_calculator = Clash(vdw_clash_threshold=threshold, compute_af3_clash=False)
    # Check ligand-polymer VDW clash
    N_sample = pred_coordinate.shape[0]
    dummy_is_dna = torch.zeros_like(is_polymer)
    dummy_is_rna = torch.zeros_like(is_polymer)
    clash_dict = clash_calculator(
        pred_coordinate=pred_coordinate,
        asym_id=asym_id,
        atom_to_token_idx=atom_token_idx,
        mol_id=mol_id,
        is_ligand=1 - is_polymer,
        is_protein=is_polymer,
        is_dna=dummy_is_dna,
        is_rna=dummy_is_rna,
        elements_one_hot=elements_one_hot,
    )
    return clash_dict["summary"]["vdw_clash"]


def calculate_clash(
    pred_coordinate: torch.Tensor,
    asym_id: torch.LongTensor,
    atom_to_token_idx: torch.LongTensor,
    is_polymer: torch.BoolTensor,
    threshold: float,
) -> torch.Tensor:
    """Check complex clash

    Args:
        pred_coordinate (torch.Tensor): [N_sample, N_atom, 3]
        asym_id (torch.LongTensor): [N_token, ]
        atom_to_token_idx (torch.LongTensor): [N_atom, ]
        is_polymer (torch.BoolTensor): [N_atom, ]
        threshold: (float)

    Returns:
        torch.Tensor: [N_sample] whether there is a clash in the complex
    """
    N_sample = pred_coordinate.shape[0]
    dummy_is_dna = torch.zeros_like(is_polymer)
    dummy_is_rna = torch.zeros_like(is_polymer)
    clash_calculator = Clash(vdw_clash_threshold=threshold, compute_vdw_clash=False)
    clash_dict = clash_calculator(
        pred_coordinate,
        asym_id,
        atom_to_token_idx,
        1 - is_polymer,
        is_polymer,
        dummy_is_dna,
        dummy_is_rna,
    )
    return clash_dict["summary"]["af3_clash"].reshape(N_sample, -1).max(dim=-1)[0]