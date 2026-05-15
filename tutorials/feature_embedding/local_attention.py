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

import math
from functools import partial
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from pairformer.triangle_ops import LayerNorm, trunc_normal_init_  # noqa: F401


def _attention(*args, **kwargs):
    # Lazy import to avoid circular dependency: af3.model.attention imports
    # from this module too. The kernel itself lives in attention.py.
    from attention.mha import _attention as _impl
    return _impl(*args, **kwargs)
from model.utils import (
    chunk_layer,
    flatten_final_dims,
    move_final_dim_to_dim,
    pad_at_dim,
    reshape_at_dim,
)



def rearrange_qk_to_dense_trunk(
    q: Union[torch.Tensor, list[torch.Tensor]],
    k: Union[torch.Tensor, list[torch.Tensor]],
    dim_q: Union[int, list[int]],
    dim_k: Union[int, list[int]],
    n_queries: int = 32,
    n_keys: int = 128,
    compute_mask: bool = True,
) -> tuple[
    Union[torch.Tensor, list[torch.Tensor]],
    Union[torch.Tensor, list[torch.Tensor]],
    dict[str, Any],
]:
    """Rearrange q/k into blocked tensors for local operations.

    Args:
        q (torch.Tensor): query tensor. Could be a tensor or a list of tensors.
            [..., n_q, ...] (n_q is at dimension dim_q)
        k (torch.Tensor | List[torch.Tensor]): key tensor. Could be a tensor or a list of tensors.
            [..., n_k, ...] (n_k is at dimension dim_k)
        dim_q (int): along which dimension to build the trunks. Could be an int or a list of int.
        dim_k (int): along which dimension to build the trunks. Could be an int or a list of int.
        n_queries (int, optional): local window size of query tensor.
        n_keys (int, optional): local window size of key/value tensor.

    Returns:
        tuple[Union[torch.Tensor, list[torch.Tensor]]]:
            q_trunked: torch.Tensor or list of tensors. Same as the input type.
                [..., n_trunks, n_queries, ...]
            k_trunked: torch.Tensor or list of tensors. Same as the input type.
                [..., n_trunks, n_keys, ...]
            padding_info (dict):
                mask_trunked: torch.Tensor
                    [n_trunks, n_queries, n_keys]
                q_pad: query padded dimension
    """

    assert n_keys >= n_queries
    assert n_queries & 0x01 == 0
    assert n_keys & 0x01 == 0

    def basic_checks(x, dim_x):
        if isinstance(x, list):
            x_is_list = True
            assert isinstance(dim_x, list)
        else:
            x_is_list = False
            x = [x]
            dim_x = [dim_x]
        n_x = x[0].size(dim_x[0])
        for i in range(len(dim_x)):
            if dim_x[i] < 0:
                dim_x[i] = len(x[i].shape) + dim_x[i]
            assert x[i].size(dim_x[i]) == n_x
        return x, dim_x, x_is_list, n_x, len(x)

    q, dim_q, q_is_list, n, num_q = basic_checks(q, dim_q)
    k, dim_k, k_is_list, n_k, num_k = basic_checks(k, dim_k)

    assert n == n_k
    n_trunks = int(math.ceil(n / n_queries))
    q_pad_length = n_trunks * n_queries - n

    q_new = [
        pad_at_dim(q[i], dim=dim_q[i], pad_length=(0, q_pad_length))
        for i in range(num_q)
    ]
    q_trunked = [
        reshape_at_dim(q_new[i], dim=dim_q[i], target_shape=(n_trunks, n_queries))
        for i in range(num_q)
    ]

    pad_left = (n_keys - n_queries) // 2
    pad_right = int((n_trunks - 1 / 2) * n_queries + n_keys / 2 - n + 1 / 2)

    k_new = [
        pad_at_dim(k[i], dim=dim_k[i], pad_length=(pad_left, pad_right))
        for i in range(num_k)
    ]
    k_trunked = [
        k_new[i].unfold(dim_k[i], size=n_keys, step=n_queries) for i in range(num_k)
    ]
    k_trunked = [
        move_final_dim_to_dim(k_trunked[i], dim=dim_k[i] + 1) for i in range(num_k)
    ]

    if compute_mask:
        pad_mask = q[0].new_ones(
            *(1,) * len(q[0].shape[:-2]),
            n + q_pad_length,
            n + pad_left + pad_right,
            requires_grad=False,
        )
        pad_mask[..., :n, :pad_left] = 0
        pad_mask[..., :n, (pad_left + n) :] = 0
        pad_mask[..., n:, :] = 0

        concat_split_data = optimized_concat_split(pad_mask, n_queries)
        pad_mask_trunked = (
            concat_split_data.unfold(
                -1, n_keys, pad_mask.size(-1) + n_queries
            ).transpose(-2, -3)
        ).bool()
    else:
        pad_mask_trunked = None

    if not q_is_list:
        q_trunked = q_trunked[0]
    if not k_is_list:
        k_trunked = k_trunked[0]

    padding_info = {
        "mask_trunked": pad_mask_trunked,
        "q_pad": q_pad_length,
        "k_pad_left": pad_left,
        "k_pad_right": pad_right,
    }

    return q_trunked, k_trunked, padding_info


def optimized_concat_split(attn_bias: torch.Tensor, n_queries: int) -> torch.Tensor:
    """Optimized concatenation and splitting of attention bias tensor.

    Args:
        attn_bias (torch.Tensor): The attention bias tensor.
            Shape: [..., D, E]
        n_queries (int): The number of queries in each split.

    Returns:
        torch.Tensor: The reshaped and permuted attention bias tensor.
            Shape: [..., n_queries, D // n_queries * E]
    """
    D = attn_bias.size(-2)
    E = attn_bias.size(-1)
    assert D % n_queries == 0
    num_splits = D // n_queries
    reshaped = attn_bias.reshape(*attn_bias.shape[:-2], num_splits, n_queries, E)
    permuted = reshaped.permute(*range(reshaped.dim() - 3), -2, -3, -1)
    output = permuted.reshape(*attn_bias.shape[:-2], n_queries, num_splits * E)
    return output


def rearrange_to_dense_trunk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    n_queries: int,
    n_keys: int,
    attn_bias: Optional[torch.Tensor] = None,
    inf: float = 1e10,
) -> tuple[Union[torch.Tensor, int]]:
    """Rearrange q/k/v/bias into blocked tensors for local attention.

    Args:
        q (torch.Tensor): query tensor
            [..., n_q, d]
        k (torch.Tensor): key tensor
            [..., n_kv, d]
        v (torch.Tensor): value tensor
            [..., n_kv, d]
        attn_bias (torch.Tensor, optional): attention bias
            [..., n_q, n_kv] or None
        n_queries (int, optional): local window size of query tensor.
        n_keys (int, optional): local window size of key/value tensor.
        inf (float, optional): used for attention masking. Defaults to 1e10.

    Returns:
        tuple[Union[torch.Tensor, int]]:
            q_trunked
                [..., n_trunks, n_queries, d]
            k_trunked / v_trunked
                [..., n_trunks, n_keys, d]
            attn_bias_trunked:  padded position filled with -inf
                [..., n_trunks, n_queries, n_keys]
            q_pad_length: query padded dimension
    """
    assert n_keys >= n_queries
    assert n_queries & 0x01 == 0
    assert n_keys & 0x01 == 0

    n, d = q.shape[-2:]

    q_trunked, kv_trunked, padding_info = rearrange_qk_to_dense_trunk(
        q=q,
        k=[k, v],
        dim_q=-2,
        dim_k=[-2, -2],
        n_queries=n_queries,
        n_keys=n_keys,
        compute_mask=False,
    )
    q_pad_length, pad_left, pad_right = (
        padding_info["q_pad"],
        padding_info["k_pad_left"],
        padding_info["k_pad_right"],
    )

    # Padded_width = n + pad_left + pad_right
    if attn_bias is None:
        attn_bias = q.new_zeros(
            *(1,) * len(q.shape[:-2]), n + q_pad_length, n + pad_left + pad_right
        )
        attn_bias[..., :n, 0:pad_left] = -inf
        attn_bias[..., :n, pad_left + n : :] = -inf
        attn_bias[..., n::, :] = -inf
    else:
        attn_bias = F.pad(attn_bias, (pad_left, pad_right, 0, q_pad_length), value=-inf)

    concat_split_data = optimized_concat_split(attn_bias, n_queries)
    attn_bias_trunked = concat_split_data.unfold(
        -1, n_keys, attn_bias.shape[-1] + n_queries
    ).transpose(-2, -3)
    return q_trunked, kv_trunked[0], kv_trunked[1], attn_bias_trunked, q_pad_length


def _local_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    n_queries: int,
    n_keys: int,
    attn_bias: Optional[torch.Tensor] = None,
    trunked_attn_bias: Optional[torch.Tensor] = None,
    inf: float = 1e10,
    use_efficient_implementation: bool = True,
    inplace_safe: bool = False,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """Local attention

    Args:
        q (torch.Tensor): query tensor
            [..., Q, d]
        k (torch.Tensor): key tensor
            [..., K, d]
        v (torch.Tensor): value tensor
            [..., K, d]
        n_queries (int): local window size of query.
        n_keys (int): local window size of key/value.
        attn_bias (torch.Tensor, optional): the input biases for attention. Defaults to None.
            [..., Q, K]
        trunked_attn_bias (torch.Tensor, optional): the input biases where shape has been rearranged to dense trunks. Defaults to None.
            [..., n_trunks, n_queries, n_keys]
        inf (float): inf number used for attention bias. Defaults to 1e10.
        use_efficient_implementation (bool): whether to use the torch.nn.functional.scaled_dot_product_attention, Defaults to True.
    Returns:
        torch.Tensor: standard attention output
            [..., Q, d]
    """
    assert q.shape == k.shape == v.shape  # local attention doesn't make sense if Q != K

    # Prepare for attention qkv, q: [..., n_trunks, n_queries, d], kv: [..., n_trunks, n_keys, d]

    # Rerrange to dense trunks
    # q: [*, n, d] -> [*, n_trunks, n_queries, d]
    # kv: [*, n, d] -> [*, n_trunks, n_keys, d]
    # attn_bias: [*, n, d] -> [*, n_trunks, n_queries, n_keys]
    (
        q_trunked,
        k_trunked,
        v_trunked,
        attn_bias_trunked,
        q_pad_length,
    ) = rearrange_to_dense_trunk(
        q=q,
        k=k,
        v=v,
        n_queries=n_queries,
        n_keys=n_keys,
        attn_bias=attn_bias,
        inf=inf,
    )

    # Apply attention
    # [..., n_trunks, n_queries, d]
    if trunked_attn_bias is not None:
        attn_bias_trunked = attn_bias_trunked + trunked_attn_bias

    if chunk_size is not None:
        attn_inputs = {
            "q": q_trunked,
            "k": k_trunked,
            "v": v_trunked,
            "attn_bias": attn_bias_trunked,
        }
        out = chunk_layer(
            partial(
                _attention,
                use_efficient_implementation=use_efficient_implementation,
                inplace_safe=inplace_safe,
            ),
            attn_inputs,
            chunk_size=chunk_size,
            no_batch_dims=len(attn_bias_trunked.shape[:-2]),
            _out=None,
        )
    else:
        out = _attention(
            q=q_trunked,
            k=k_trunked,
            v=v_trunked,
            attn_bias=attn_bias_trunked,
            use_efficient_implementation=use_efficient_implementation,
            inplace_safe=inplace_safe,
        )

    # Revert back to orignal shape and remove q_pad_length
    # [..., n_trunks, n_queries, d] ->  [..., n_trunks * n_queries, d] ->  [..., n, d]
    out = out.reshape(*out.shape[:-3], -1, out.shape[-1])
    if q_pad_length > 0:
        out = out[..., :-q_pad_length, :]
    return out


def create_local_attn_bias(
    n: int, n_queries: int, n_keys: int, inf: float = 1e10, device: torch.device = None
) -> torch.Tensor:
    """Create local attention bias based on query window n_queries and kv window n_keys.

    Args:
        n (int): the length of quiries
        n_queries (int): window size of quiries
        n_keys (int): window size of keys/values
        inf (float, optional): the inf to mask attention. Defaults to 1e10.
        device (torch.device, optional): cuda|cpu|None. Defaults to None.

    Returns:
        torch.Tensor: the diagonal-like global attention bias
    """
    n_trunks = int(math.ceil(n / n_queries))
    padded_n = n_trunks * n_queries
    attn_mask = torch.zeros(padded_n, padded_n, device=device)
    for block_index in range(0, n_trunks):
        i = block_index * n_queries
        j1 = max(0, n_queries * block_index - (n_keys - n_queries) // 2)
        j2 = n_queries * block_index + (n_queries + n_keys) // 2
        attn_mask[i : i + n_queries, j1:j2] = 1.0
    attn_bias = (1 - attn_mask) * -inf
    return attn_bias.to(device=device)[:n, :n]


def gather_pair_embedding_in_dense_trunk(
    x: torch.Tensor, idx_q: torch.Tensor, idx_k: torch.Tensor
) -> torch.Tensor:
    """
    Selectively gather elements from a tensor using two sets of indices.

    Args:
        x: [..., N_token, N_token, d]
        idx_q: [N_b, N_q]
        idx_k: [N_b, N_k]

    Return:
        y: [..., N_b, N_q, N_k, d]
            where y[..., b, i, j, :] = x[..., idx_q[b, i], idx_k[b, j], :]
    """
    idx_q = idx_q.long()
    idx_k = idx_k.long()
    assert len(idx_q.shape) == len(idx_k.shape) == 2

    # Get the shape parameters
    N_b, N_q = idx_q.shape
    N_k = idx_k.shape[1]

    # Expand idx_q and idx_k to match the shape required for advanced indexing
    idx_q_expanded = idx_q.unsqueeze(-1).expand(-1, -1, N_k)
    idx_k_expanded = idx_k.unsqueeze(1).expand(-1, N_q, -1)

    # Use advanced indexing to gather the desired elements
    y = x[..., idx_q_expanded, idx_k_expanded, :]

    return y


def broadcast_token_to_local_atom_pair(
    z_token: torch.Tensor,
    atom_to_token_idx: torch.Tensor,
    n_queries: int,
    n_keys: int,
    compute_mask: bool = True,
) -> torch.Tensor:
    """Broadcast token pair embedding to atom pair embedding

    Args:
        z_token (torch.Tensor): token pair embedding
            [..., N_token, N_token, d]
        atom_to_token_idx (torch.Tensor): map atom idx to token idx
            [N_atom]

    Returns:
        z_gathered_blocked (torch.Tensor): atom pair embedding, with local blocked shape
            [..., n_trunks, n_queries, n_keys, d]
        pad_mask (torch.Tensor):
            [n_trunks, n_queries, n_keys]
        q_pad_length (int)
    """

    # [N_atom] -> [n_trunks, n_queries] and [n_trunks, n_keys]
    atom_to_token_idx_q, atom_to_token_idx_k, pad_info = rearrange_qk_to_dense_trunk(
        atom_to_token_idx,
        atom_to_token_idx,
        dim_q=-1,
        dim_k=-1,
        n_queries=n_queries,
        n_keys=n_keys,
        compute_mask=compute_mask,
    )

    z_gathered_blocked = gather_pair_embedding_in_dense_trunk(
        z_token, idx_q=atom_to_token_idx_q, idx_k=atom_to_token_idx_k
    )

    return z_gathered_blocked, pad_info


# from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py