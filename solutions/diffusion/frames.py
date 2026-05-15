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

import torch
import torch.nn.functional as F

from model.utils import batched_gather


def expressCoordinatesInFrame(
    coordinate: torch.Tensor, frames: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Algorithm 29 Express coordinate in frame

    Args:
        coordinate (torch.Tensor): the input coordinate
            [..., N_atom, 3]
        frames (torch.Tensor): the input frames
            [..., N_frame, 3, 3]
        eps (float): Small epsilon value

    Returns:
        torch.Tensor: the transformed coordinate projected onto frame basis
            [..., N_frame, N_atom, 3]
    """
    ##########################################################################
    # TODO: Algorithm 29 — project ``coordinate`` onto each frame's local    #
    #   orthonormal basis. Each frame is defined by three atoms (a, b, c)   #
    #   with ``b`` at the origin.                                            #
    #                                                                        #
    #   Step 1 — Unbind the three frame atoms along the second-to-last axis.#
    #     The middle one ``b`` is the origin; ``a`` and ``c`` define the    #
    #     in-plane directions:                                               #
    #       a, b, c = torch.unbind(frames, dim=-2)                           #
    #                                       # each: [..., N_frame, 3]       #
    #                                                                        #
    #   Step 2 — Two raw direction vectors from ``b`` to ``a`` and ``c``,    #
    #     normalized (eps guards near-zero norms):                           #
    #       w1 = F.normalize(a - b, dim=-1, eps=eps)                         #
    #       w2 = F.normalize(c - b, dim=-1, eps=eps)                         #
    #                                                                        #
    #   Step 3 — Build an orthonormal basis (e1, e2, e3). Using the sum and  #
    #     difference of w1 / w2 makes the basis insensitive to the           #
    #     individual lengths and gives a numerically stable Gram-Schmidt:    #
    #       e1 = F.normalize(w1 + w2, dim=-1, eps=eps)                       #
    #       e2 = F.normalize(w2 - w1, dim=-1, eps=eps)                       #
    #       e3 = torch.cross(e1, e2, dim=-1)            # [..., N_frame, 3] #
    #                                                                        #
    #   Step 4 — Broadcast every input atom relative to each frame's origin  #
    #     ``b``. Insert a query-atom axis so ``d[..., f, i, :]`` is the      #
    #     displacement of atom ``i`` from the origin of frame ``f``:         #
    #       d = coordinate[..., None, :, :] - b[..., None, :]               #
    #                                       # [..., N_frame, N_atom, 3]     #
    #                                                                        #
    #   Step 5 — Project each displacement onto (e1, e2, e3) by inner        #
    #     product on the last dim:                                           #
    #       x_transformed = torch.cat(                                       #
    #           [                                                            #
    #               torch.sum(d * e1[..., None, :], dim=-1, keepdim=True),  #
    #               torch.sum(d * e2[..., None, :], dim=-1, keepdim=True),  #
    #               torch.sum(d * e3[..., None, :], dim=-1, keepdim=True),  #
    #           ],                                                            #
    #           dim=-1,                                                       #
    #       )                                       # [..., N_frame, N_atom, 3]#
    #       return x_transformed                                              #
    #                                                                        #
    # TODO: 算法 29 —— 把 ``coordinate`` 投影到每个 frame 的局部正交基上。     #
    #   每个 frame 由 3 个原子 (a, b, c) 定义，b 作为局部原点。                #
    #                                                                        #
    #   步骤 1 — 沿倒数第二维拆出三个 frame 原子。中间的 ``b`` 是原点；         #
    #     ``a`` / ``c`` 给出在平面上的两个方向:                                #
    #       a, b, c = torch.unbind(frames, dim=-2)                           #
    #                                       # 形状: [..., N_frame, 3]       #
    #                                                                        #
    #   步骤 2 — b -> a 与 b -> c 两条方向向量并归一化 (eps 保护近零模长):     #
    #       w1 = F.normalize(a - b, dim=-1, eps=eps)                         #
    #       w2 = F.normalize(c - b, dim=-1, eps=eps)                         #
    #                                                                        #
    #   步骤 3 — 构造正交基 (e1, e2, e3)。用 w1/w2 的和差代替直接 Gram-Schmidt #
    #     可让结果对个体长度不敏感、数值更稳:                                  #
    #       e1 = F.normalize(w1 + w2, dim=-1, eps=eps)                       #
    #       e2 = F.normalize(w2 - w1, dim=-1, eps=eps)                       #
    #       e3 = torch.cross(e1, e2, dim=-1)            # [..., N_frame, 3] #
    #                                                                        #
    #   步骤 4 — 输入坐标相对每个 frame 原点 ``b`` 求位移，                     #
    #     插入查询原子轴使 ``d[..., f, i, :]`` 表示原子 i 相对 frame f 的位移: #
    #       d = coordinate[..., None, :, :] - b[..., None, :]               #
    #                                       # [..., N_frame, N_atom, 3]     #
    #                                                                        #
    #   步骤 5 — 把每条位移分别投影到 (e1, e2, e3) 上，沿最后一维内积:          #
    #       x_transformed = torch.cat(                                       #
    #           [                                                            #
    #               torch.sum(d * e1[..., None, :], dim=-1, keepdim=True),  #
    #               torch.sum(d * e2[..., None, :], dim=-1, keepdim=True),  #
    #               torch.sum(d * e3[..., None, :], dim=-1, keepdim=True),  #
    #           ],                                                            #
    #           dim=-1,                                                       #
    #       )                                                                 #
    #       return x_transformed                                              #
    ##########################################################################

    # Extract frame atoms
    a, b, c = torch.unbind(frames, dim=-2)  # a, b, c shape: [..., N_frame, 3]
    w1 = F.normalize(a - b, dim=-1, eps=eps)
    w2 = F.normalize(c - b, dim=-1, eps=eps)
    # Build orthonormal basis
    e1 = F.normalize(w1 + w2, dim=-1, eps=eps)
    e2 = F.normalize(w2 - w1, dim=-1, eps=eps)
    e3 = torch.cross(e1, e2, dim=-1)  # [..., N_frame, 3]
    # Project onto frame basis
    d = coordinate[..., None, :, :] - b[..., None, :]  # [..., N_frame, N_atom, 3]
    x_transformed = torch.cat(
        [
            torch.sum(d * e1[..., None, :], dim=-1, keepdim=True),
            torch.sum(d * e2[..., None, :], dim=-1, keepdim=True),
            torch.sum(d * e3[..., None, :], dim=-1, keepdim=True),
        ],
        dim=-1,
    )  # [..., N_frame, N_atom, 3]
    return x_transformed

    ##########################################################################
    #               END OF YOUR CODE                                         #
    ##########################################################################


def gather_frame_atom_by_indices(
    coordinate: torch.Tensor, frame_atom_index: torch.Tensor, dim: int = -2
) -> torch.Tensor:
    """construct frames from coordinate

    Args:
        coordinate (torch.Tensor):  the input coordinate
            [..., N_atom, 3]
        frame_atom_index (torch.Tensor): indices of three atoms in each frame
            [..., N_frame, 3] or [N_frame, 3]
        dim (int): along which dimension to select the frame atoms
    Returns:
        torch.Tensor: the constructed frames
            [..., N_frame, 3[three atom], 3[three coordinate]]
    """
    if len(frame_atom_index.shape) == 2:
        # the navie case
        x1 = torch.index_select(
            coordinate, dim=dim, index=frame_atom_index[:, 0]
        )  # [..., N_frame, 3]
        x2 = torch.index_select(
            coordinate, dim=dim, index=frame_atom_index[:, 1]
        )  # [..., N_frame, 3]
        x3 = torch.index_select(
            coordinate, dim=dim, index=frame_atom_index[:, 2]
        )  # [..., N_frame, 3]
        return torch.stack([x1, x2, x3], dim=dim)
    else:
        assert (
            frame_atom_index.shape[:dim] == coordinate.shape[:dim]
        ), "batch size dims should match"

    x1 = batched_gather(
        data=coordinate,
        inds=frame_atom_index[..., 0],
        dim=dim,
        no_batch_dims=len(coordinate.shape[:dim]),
    )  # [..., N_frame, 3]
    x2 = batched_gather(
        data=coordinate,
        inds=frame_atom_index[..., 1],
        dim=dim,
        no_batch_dims=len(coordinate.shape[:dim]),
    )  # [..., N_frame, 3]
    x3 = batched_gather(
        data=coordinate,
        inds=frame_atom_index[..., 2],
        dim=dim,
        no_batch_dims=len(coordinate.shape[:dim]),
    )  # [..., N_frame, 3]
    return torch.stack([x1, x2, x3], dim=dim)
