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



def get_bin_params(cfg: ConfigDict) -> dict:
    """
    Extract bin parameters from the configuration object.
    """
    return {"min_bin": cfg.min_bin, "max_bin": cfg.max_bin, "no_bins": cfg.no_bins}


def compute_contact_prob(
    distogram_logits: torch.Tensor,
    min_bin: float,
    max_bin: float,
    no_bins: int,
    thres=8.0,
) -> torch.Tensor:
    """
    Compute the contact probability from distogram logits.

    Args:
        distogram_logits (torch.Tensor): Logits for the distogram.
            Shape: [N_token, N_token, N_bins]
        min_bin (float): Minimum bin value.
        max_bin (float): Maximum bin value.
        no_bins (int): Number of bins.
        thres (float): Threshold distance for contact probability. Defaults to 8.0.

    Returns:
        torch.Tensor: Contact probability.
            Shape: [N_token, N_token]
    """
    distogram_prob = torch.nn.functional.softmax(
        distogram_logits, dim=-1
    )  # [N_token, N_token, N_bins]
    distogram_bins = get_bin_centers(min_bin, max_bin, no_bins)
    thres_idx = (distogram_bins < thres).sum()
    contact_prob = distogram_prob[..., :thres_idx].sum(-1)
    del distogram_prob
    return contact_prob


def get_bin_centers(min_bin: float, max_bin: float, no_bins: int) -> torch.Tensor:
    """
    Calculate the centers of the bins for a given range and number of bins.

    Args:
        min_bin (float): The minimum value of the bin range.
        max_bin (float): The maximum value of the bin range.
        no_bins (int): The number of bins.

    Returns:
        torch.Tensor: The centers of the bins.
            Shape: [no_bins]
    """
    bin_width = (max_bin - min_bin) / no_bins
    boundaries = torch.linspace(
        start=min_bin,
        end=max_bin - bin_width,
        steps=no_bins,
    )
    bin_centers = boundaries + 0.5 * bin_width
    return bin_centers


def logits_to_prob(logits: torch.Tensor, dim=-1) -> torch.Tensor:
    return torch.nn.functional.softmax(logits, dim=dim)


def logits_to_score(
    logits: torch.Tensor,
    min_bin: float,
    max_bin: float,
    no_bins: int,
    return_prob=False,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """
    Convert logits to a score using bin centers.

    Args:
        logits (torch.Tensor): Logits tensor.
            Shape: [..., no_bins]
        min_bin (float): Minimum bin value.
        max_bin (float): Maximum bin value.
        no_bins (int): Number of bins.
        return_prob (bool): Whether to return the probability distribution. Defaults to False.

    Returns:
        score (torch.Tensor): Converted score.
            Shape: [...]
        prob (torch.Tensor, optional): Probability distribution if `return_prob` is True.
            Shape: [..., no_bins]
    """
    prob = logits_to_prob(logits, dim=-1)
    bin_centers = get_bin_centers(min_bin, max_bin, no_bins).to(logits.device)
    score = prob @ bin_centers
    if return_prob:
        return score, prob
    else:
        return score


def calculate_normalization(N: int) -> float:
    # TM-score normalization constant
    return 1.24 * (max(N, 19) - 15) ** (1 / 3) - 1.8