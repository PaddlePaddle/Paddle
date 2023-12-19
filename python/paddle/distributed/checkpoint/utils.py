# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import copy
from typing import List, Tuple, Union

import numpy as np

import paddle
from paddle.framework import core


def get_coordinator(mesh: Union[np.array, List[List[int]]], rank: int):
    mesh = paddle.to_tensor(mesh)
    rand_coordinator = (mesh == rank).nonzero()
    assert rand_coordinator.shape[0] in (
        0,
        1,
    ), f"rand_coordinator.shape: {rand_coordinator.shape}"
    return (
        rand_coordinator[0].tolist() if rand_coordinator.shape[0] > 0 else None
    )


def compute_local_shape_and_global_offset(
    global_shape: List[int],
    process_mesh: core.ProcessMesh,
    dims_mapping: List[int],
) -> Tuple[Tuple[int], Tuple[int]]:
    mesh = np.array(process_mesh.process_ids).reshape(process_mesh.shape)
    # deal with cross mesh case
    if paddle.distributed.get_rank() not in mesh:
        return (None, None)
    rank_coordinator = get_coordinator(mesh, paddle.distributed.get_rank())
    local_shape = copy.copy(global_shape)
    global_offset = [0 for _ in global_shape]
    for i, dim in enumerate(dims_mapping):
        if dim == -1:
            continue
        else:
            assert (
                global_shape[i] % process_mesh.shape[dim] == 0
            ), f"i:{i}, global_shape[i]:{global_shape[i]}, process_mesh.shape[dim]:{process_mesh.shape[dim]}"
            local_shape[i] = global_shape[i] // process_mesh.shape[dim]
            chunk_idx = rank_coordinator[dim]
            global_offset[i] = chunk_idx * local_shape[i]

    return tuple(local_shape), tuple(global_offset)


def flatten_state_dict(state_dict):
    # TODO, {"model": {"w0": xxx}} -> {model.w0: xxx}
    return state_dict
