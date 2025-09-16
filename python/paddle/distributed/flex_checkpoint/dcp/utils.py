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
from __future__ import annotations

import ast
import copy
import os
import re
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

import paddle
from paddle.distributed.fleet.utils.log_util import logger

from ..aoa.aoa_engine import (
    postprocess_transpose,
)
from .sharded_weight import (
    ShardedWeight,
    ShardedWeightDesc,
)

if TYPE_CHECKING:
    from paddle.framework import core


def get_coordinator(mesh: np.array | list[list[int]], rank: int):
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
    global_shape: list[int],
    process_mesh: core.ProcessMesh,
    placements: list[core.Placement],
) -> tuple[tuple[int], tuple[int]]:
    from paddle.distributed.auto_parallel.placement_type import (
        placemetns_to_dist_status,
    )

    mesh = np.array(process_mesh.process_ids).reshape(process_mesh.shape)
    # deal with cross mesh case
    if paddle.distributed.get_rank() not in mesh:
        return (None, None)
    rank_coordinator = get_coordinator(mesh, paddle.distributed.get_rank())
    local_shape = copy.copy(global_shape)
    global_offset = [0 for _ in global_shape]

    dims_mapping, _ = placemetns_to_dist_status(placements, len(global_shape))
    for tensor_dim, mesh_dims in enumerate(dims_mapping):
        if len(mesh_dims) == 0:
            continue
        local_offset = [0] * len(global_shape)
        for mesh_dim in mesh_dims:
            chunk_idx = rank_coordinator[mesh_dim]
            chunks = balanced_split(
                local_shape[tensor_dim], process_mesh.shape[mesh_dim]
            )
            local_shape[tensor_dim] = chunks[chunk_idx]
            local_offset[tensor_dim] = sum(chunks[:chunk_idx])

            if global_offset[tensor_dim] <= local_offset[tensor_dim]:
                global_offset[tensor_dim] = local_offset[tensor_dim]
            else:
                global_offset[tensor_dim] += local_offset[tensor_dim]

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
                _flatten((*key, k), v)
        elif isinstance(value, (paddle.Tensor, ShardedWeight)):
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
        assert isinstance(key_tuple, tuple), (
            f"The key should be tuple, but is {key_tuple}"
        )
        tmp = state_dict
        for i in range(len(key_tuple) - 1):
            key = key_tuple[i]
            tmp = tmp.setdefault(key, {})
        tmp[key_tuple[-1]] = value

    return state_dict


def get_max_id(path):
    numbers = []
    pattern = re.compile(r"^(\d+)_(\d+)\.distcp$")
    files = os.listdir(path)
    for file in files:
        match = pattern.match(file)
        if match:
            numbers.append(int(match.group(2)))
    return max(numbers) if numbers else None


def check_unique_id(unique_id, process_group):
    all_unique_id = []
    paddle.distributed.all_gather_object(
        all_unique_id, unique_id, process_group
    )
    for id in all_unique_id[1:]:
        assert id == all_unique_id[0], f"id:{id} !=  all_unique_id[0]"


def ravel_index(indices, shape):
    idx = 0
    for i, dim in zip(indices, shape):
        idx = idx * dim + i
    return idx


def unravel_index(idx, shape):
    indices = []
    for dim in reversed(shape):
        indices.append(idx % dim)
        idx //= dim
    return tuple(reversed(indices))


def minimal_nd_slice(shape, flat_start, flat_end):
    start_idx = unravel_index(flat_start, shape)
    end_idx = unravel_index(flat_end - 1, shape)
    min_slices = []
    for axis in range(len(shape)):
        if axis == 0:
            s = start_idx[axis]
            e = end_idx[axis] + 1
        else:
            if start_idx[axis - 1] == end_idx[axis - 1]:
                s = min(start_idx[axis], end_idx[axis])
                e = max(start_idx[axis], end_idx[axis]) + 1
            else:
                s = 0
                e = shape[axis]
        min_slices.append((s, e))
    return min_slices, start_idx, end_idx


def flat_range_in_min_slice(shape, min_slices, flat_start, flat_end):
    min_starts = tuple(s[0] for s in min_slices)
    min_flat_start = ravel_index(min_starts, shape)
    return flat_start - min_flat_start, flat_end - min_flat_start


def is_sharded_state_dict(o):
    if not isinstance(o, dict):
        return False

    values = list(o.values())
    has_sharded_weight = any(isinstance(v, ShardedWeight) for v in values)

    if has_sharded_weight:
        if not all(isinstance(v, ShardedWeight) for v in values):
            raise TypeError(
                "All values must be ShardedWeight if any value is ShardedWeight."
            )
        return True
    else:
        return False


def get_overlap_region(desc_offset, desc_shape, shard_offset, shard_shape):
    ndim = len(desc_offset)
    overlap_offset = []
    overlap_shape = []
    desc_starts = []
    shard_starts = []
    for i in range(ndim):
        desc_lo = desc_offset[i]
        desc_hi = desc_offset[i] + desc_shape[i]
        shard_lo = shard_offset[i]
        shard_hi = shard_offset[i] + shard_shape[i]
        # overlap
        lo = max(desc_lo, shard_lo)
        hi = min(desc_hi, shard_hi)
        if lo >= hi:
            return False, None, None, None, None
        overlap_offset.append(lo)
        overlap_shape.append(hi - lo)
        desc_starts.append(lo - desc_lo)
        shard_starts.append(lo - shard_lo)
    return True, overlap_offset, overlap_shape, desc_starts, shard_starts


def assign_sharded_slice(
    src_desc, src_shard, dst_desc, dst_shard, postprocess_list=None
):
    src_has, _, overlap_shape, src_desc_starts, src_shard_starts = (
        get_overlap_region(
            src_desc.global_offset,
            src_desc.local_shape,
            src_shard.global_offset,
            src_shard.local_shape,
        )
    )

    dst_has, _, overlap_shape2, dst_desc_starts, dst_shard_starts = (
        get_overlap_region(
            dst_desc.global_offset,
            dst_desc.local_shape,
            dst_shard.global_offset,
            dst_shard.local_shape,
        )
    )

    assert src_has or dst_has, "no overlap!"
    if overlap_shape != overlap_shape2:
        assert postprocess_list is not None, (
            "only post transpose operation could make overlap shape mismatch"
        )
        transposed_src_overlap_shape = postprocess_transpose(
            overlap_shape, postprocess_list
        )

        assert transposed_src_overlap_shape == overlap_shape2, (
            f"overlap shape mismatch: {transposed_src_overlap_shape} vs {overlap_shape2}"
        )
        axes = list(range(len(transposed_src_overlap_shape)))

        src_tensor_slice = paddle.slice(
            src_shard.local_tensor,
            axes=axes,
            starts=src_shard_starts,
            ends=[s + o for s, o in zip(src_shard_starts, overlap_shape)],
        )

        for ps in postprocess_list:
            is_list, result = is_list_string(ps)
            if is_list:
                src_tensor_slice = paddle.transpose(src_tensor_slice, result)

        dst_tensor_slice = paddle.slice(
            dst_shard.local_tensor,
            axes=axes,
            starts=dst_shard_starts,
            ends=[s + o for s, o in zip(dst_shard_starts, overlap_shape2)],
        )

    else:
        axes = list(range(len(overlap_shape)))

        src_tensor_slice = paddle.slice(
            src_shard.local_tensor,
            axes=axes,
            starts=src_shard_starts,
            ends=[s + o for s, o in zip(src_shard_starts, overlap_shape)],
        )

        dst_tensor_slice = paddle.slice(
            dst_shard.local_tensor,
            axes=axes,
            starts=dst_shard_starts,
            ends=[s + o for s, o in zip(dst_shard_starts, overlap_shape)],
        )

    paddle.assign(src_tensor_slice, dst_tensor_slice)


def merge_shard_info_list(list_of_dicts):
    merged = defaultdict(list)
    for info in list_of_dicts:
        for k, v in info.items():
            merged[k].extend(v)
    return dict(merged)


def build_shard_desc(val):
    return ShardedWeightDesc(
        key=val.key,
        local_shape=tuple(val.local_shape),
        global_shape=tuple(val.global_shape),
        global_offset=tuple(val.global_offset),
    )


def is_list_string(s):
    try:
        result = ast.literal_eval(s)
        return (True, result) if isinstance(result, list) else (False, None)
    except:
        return False, None


def write_to_file_if_empty(data, path):
    lock_path = f"{path}.lock"
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        try:
            if os.path.exists(path) and os.path.getsize(path) > 0:
                logger.info(
                    f"Process {os.getpid()} found the metadata file already written."
                )
                return
            paddle.save(data, path)
            logger.info(
                f"Process {os.getpid()} successfully wrote the metadata to the file."
            )
        finally:
            if os.path.exists(lock_path):
                os.remove(lock_path)
    except FileExistsError:
        logger.info(
            f"Process {os.getpid()} could not acquire the lock; another process is writing or has written the metadata."
        )
