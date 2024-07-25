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
import paddle.distributed as dist
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


# NOTE(zhangbo): Refer to the BalancedSplit function in the reshard_utils.cc file.
def balanced_split(total_nums, num_of_pieces):
    has_remainder = total_nums % num_of_pieces != 0
    result = [(total_nums + num_of_pieces - 1) // num_of_pieces] * num_of_pieces
    if has_remainder:
        last_value = result[-1]
        result[-1] = last_value - (last_value * num_of_pieces - total_nums)
    return result


def compute_local_shape_and_global_offset(
    global_shape: List[int],
    process_mesh: core.ProcessMesh,
    placements: List[core.Placement],
) -> Tuple[Tuple[int], Tuple[int]]:
    mesh = np.array(process_mesh.process_ids).reshape(process_mesh.shape)
    # deal with cross mesh case
    if paddle.distributed.get_rank() not in mesh:
        return (None, None)
    rank_coordinator = get_coordinator(mesh, paddle.distributed.get_rank())
    local_shape = copy.copy(global_shape)
    global_offset = [0 for _ in global_shape]
    for dim, placement in enumerate(placements):
        if isinstance(placement, dist.Replicate):
            continue
        else:
            i = placement.get_dim()
            chunk_idx = rank_coordinator[dim]
            chunks = balanced_split(global_shape[i], process_mesh.shape[dim])
            local_shape[i] = chunks[chunk_idx]
            global_offset[i] = sum(chunks[:chunk_idx])

    return tuple(local_shape), tuple(global_offset)


def flatten_state_dict(state_dict):
    """
    Flatten the nested dict to a flat dict.
    {"model": {"w0": xxx}} -> {model.w0: xxx}
    """
    flatten_state_dict = {}
    mapping = {}

    def _flatten(key, value):
        if isinstance(value, dict):
            for k, v in value.items():
                assert isinstance(k, str), f"The key should be str, but is {k}"
                _flatten(key + (k,), v)
        elif isinstance(value, paddle.Tensor):
            flatten_key_str = ".".join(key)
            flatten_state_dict[flatten_key_str] = value
            mapping[flatten_key_str] = key
        else:
            raise ValueError(
                f"The value should be dict or paddle.Tensor, but is {value}"
            )

    _flatten((), state_dict)

    return flatten_state_dict, mapping


def unflatten_state_dict(flat_state_dict, mapping):
    """
    Unflatten the flat dict to a nested dict.
    {model.w0: xxx} -> {"model": {"w0": xxx}}
    """
    state_dict = {}
    for key, value in flat_state_dict.items():
        key_tuple = mapping[key]
        assert isinstance(
            key_tuple, tuple
        ), f"The key should be tuple, but is {key_tuple}"
        tmp = state_dict
        for i in range(len(key_tuple) - 1):
            key = key_tuple[i]
            tmp = tmp.setdefault(key, {})
        tmp[key_tuple[-1]] = value

    return state_dict
