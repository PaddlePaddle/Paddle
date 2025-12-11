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
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
from safetensors.numpy import safe_open

import paddle
from paddle.distributed.fleet.utils.log_util import logger

from ..aoa.aoa_engine import (
    postprocess_transpose,
)
from .metadata import (
    LocalTensorIndex,
    LocalTensorMetadata,
    Metadata,
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
    numbers = [0]
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


def is_sharded_state_dict(state_dict, use_dist=True, process_group=None):
    values = list(state_dict.values())
    is_all_sharded = all(isinstance(v, ShardedWeight) for v in values)
    has_sharded = any(isinstance(v, ShardedWeight) for v in values)

    if has_sharded and not is_all_sharded:
        raise TypeError(
            "All values must be ShardedWeight if any value is ShardedWeight."
        )

    if not use_dist:
        return is_all_sharded

    if is_all_sharded:
        flag = 1
    elif len(values) == 0:
        flag = 0
    else:
        flag = -1

    all_flags = []
    paddle.distributed.all_gather_object(all_flags, flag, process_group)

    assert all(f >= 0 for f in all_flags) or all(f <= 0 for f in all_flags), (
        "Not support mixed type of ShardedWeight and non-ShardedWeight in the same state_dict!"
    )
    return all(f >= 0 for f in all_flags)


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

    if postprocess_list is not None:
        for ps in postprocess_list:
            is_list, result = is_list_string(ps)
            if is_list:
                src_tensor_slice = paddle.transpose(src_tensor_slice, result)
            else:
                if isinstance(ps, str):
                    src_tensor_slice = paddle.cast(src_tensor_slice, ps)

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
        dtype=str(val.local_tensor.dtype).split(".")[-1],
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


def build_global_state_shard_info(sharded_state_dict, process_group):
    state_shard_info = defaultdict(list)
    for key, val in sharded_state_dict.items():
        desc = build_shard_desc(val)
        state_shard_info[key].append(desc)

    gathered_info = []

    use_dist = True if paddle.distributed.get_world_size() > 1 else False
    if use_dist:
        paddle.distributed.all_gather_object(
            gathered_info, dict(state_shard_info), process_group
        )
    else:
        gathered_info = [dict(state_shard_info)]

    return merge_shard_info_list(gathered_info)


def merge_state_dict_metadata(global_state_dict_metadata):
    assert isinstance(global_state_dict_metadata, list), (
        "The global_state_dict should be a list."
    )
    out = {}
    for state_dict in global_state_dict_metadata:
        for key, val in state_dict.items():
            if key not in out:
                out[key] = []

            if isinstance(val, list):
                for item in val:
                    if item not in out[key]:
                        out[key].append(item)
            else:
                if val not in out[key]:
                    out[key].append(val)

    return out


def recover_shard_tensor_from_shards(sharded_weights: list, sw):
    def _assign_slice(dst_tensor, dst_starts, dst_ends, src_tensor):
        axes = list(range(len(dst_starts)))
        view = paddle.slice(
            dst_tensor, axes=axes, starts=dst_starts, ends=dst_ends
        )
        paddle.assign(src_tensor, output=view)
        return dst_tensor

    dims = len(sw.global_offset)
    sw_glo_start = sw.global_offset
    sw_glo_end = [sw.global_offset[i] + sw.local_shape[i] for i in range(dims)]
    sw_shape = sw.local_shape

    for s in sharded_weights:
        s_glo_start = s.global_offset
        s_glo_end = [s.global_offset[i] + s.local_shape[i] for i in range(dims)]

        overlap = []
        for i in range(dims):
            ol_start = max(s_glo_start[i], sw_glo_start[i])
            ol_end = min(s_glo_end[i], sw_glo_end[i])
            if ol_start >= ol_end:
                break
            overlap.append((ol_start, ol_end))
        else:
            s_starts = [ol[0] - s_glo_start[i] for i, ol in enumerate(overlap)]
            s_ends = [ol[1] - s_glo_start[i] for i, ol in enumerate(overlap)]
            sw_starts = [
                ol[0] - sw_glo_start[i] for i, ol in enumerate(overlap)
            ]
            sw_ends = [ol[1] - sw_glo_start[i] for i, ol in enumerate(overlap)]

            axes = list(range(len(s_starts)))
            src = paddle.slice(
                s.local_tensor, axes=axes, starts=s_starts, ends=s_ends
            )
            _assign_slice(sw.local_tensor, sw_starts, sw_ends, src)

    return sw


def create_hf_ckpt_metadata(
    ckpt_path: str,
    process_group=None,
):
    dtype_mapping = {
        'U16': 'bfloat16',
        'U8': 'uint8',
        'I8': 'int8',
        'I16': 'int16',
        'BOOL': 'bool',
        'F16': 'float16',
        'F32': 'float32',
        'F64': 'float64',
        'BF16': 'bfloat16',
    }

    use_dist = paddle.distributed.get_world_size() > 1
    cur_rank = paddle.distributed.get_rank() if use_dist else 0

    accessible_files = os.listdir(ckpt_path)
    safetensors_files = [
        file for file in accessible_files if file.endswith(".safetensors")
    ]
    if use_dist:
        rank_visible_files = []
        local_files = {cur_rank: safetensors_files}
        paddle.distributed.all_gather_object(
            rank_visible_files, local_files, process_group
        )
        rank_visible_files = {
            rank: files for d in rank_visible_files for rank, files in d.items()
        }
    else:
        rank_visible_files = {0: safetensors_files}

    def assign_files(
        rank_visible_files: dict[int, list[str]],
    ) -> dict[int, list[str]]:
        all_files = set()
        for files in rank_visible_files.values():
            all_files.update(files)
        all_files = list(all_files)

        file2ranks = defaultdict(list)
        for rank, files in rank_visible_files.items():
            for f in files:
                file2ranks[f].append(rank)

        result = defaultdict(list)

        all_files.sort(key=lambda f: (len(file2ranks[f]), f))

        rank_load = dict.fromkeys(rank_visible_files, 0)

        for f in all_files:
            candidates = file2ranks[f]
            min_rank = min(candidates, key=lambda r: (rank_load[r], r))
            result[min_rank].append(f)
            rank_load[min_rank] += 1

        return {rank: result.get(rank, []) for rank in rank_visible_files}

    rank2file = assign_files(rank_visible_files)
    need_handle_files = rank2file[cur_rank]

    local_state_dict_metadata = defaultdict(set)
    local_storage_metadata = {}
    for file_name in need_handle_files:
        file_path = os.path.join(ckpt_path, file_name)
        with safe_open(file_path, framework="np") as f:
            for key in f.keys():
                t_s = f.get_slice(key)
                shape = tuple(t_s.get_shape())
                dtype = t_s.get_dtype()
                assert dtype in dtype_mapping, f"{dtype} is not supported yet."
                dtype = dtype_mapping[dtype]
                ltm = LocalTensorMetadata(
                    global_offset=(0,) * len(shape),
                    local_shape=shape,
                    dtype=dtype,
                    global_shape=shape,
                    is_flattened=False,
                )
                lti = LocalTensorIndex(
                    tensor_key=key,
                    global_offset=(0,) * len(shape),
                    is_flattened=False,
                    local_shape=shape,
                )
                local_state_dict_metadata[key].add(ltm)
                local_storage_metadata[lti] = file_name

    if use_dist:
        global_state_dict_metadata = []
        global_storage_metadata = []
        paddle.distributed.all_gather_object(
            global_state_dict_metadata,
            dict(local_state_dict_metadata),
            process_group,
        )
        paddle.distributed.all_gather_object(
            global_storage_metadata, local_storage_metadata, process_group
        )
    else:
        global_state_dict_metadata = [dict(local_state_dict_metadata)]
        global_storage_metadata = [local_storage_metadata]

    state_dict_metadata = defaultdict(set)
    for md in global_state_dict_metadata:
        for k, v in md.items():
            state_dict_metadata[k].update(v)
    state_dict_metadata = {k: list(v) for k, v in state_dict_metadata.items()}

    storage_metadata = {}
    for md in global_storage_metadata:
        storage_metadata.update(md)

    metadata = Metadata(
        state_dict_metadata=state_dict_metadata,
        storage_metadata=storage_metadata,
    )

    METADATA_FILE_NAME = "flex-ckpt.auto_generated.metadata"
    write_to_file_if_empty(
        metadata, os.path.join(ckpt_path, METADATA_FILE_NAME)
    )

    if use_dist:
        paddle.distributed.barrier(process_group)


def get_target_tensor(target_state_dict, read_item):
    use_dist = paddle.distributed.get_world_size() > 1
    if any(isinstance(k, tuple) for k in target_state_dict):
        key = (read_item.tensor_name, read_item.dst_global_offset)
    else:
        key = read_item.tensor_name

    tensor = target_state_dict[key]
    return tensor._local_value() if use_dist and tensor.is_dist() else tensor


def slice_tensor(tensor, slice_begin, slice_shape):
    if not slice_shape:
        assert not tensor.shape, (
            "Only 0-dimensional tensor supports empty slice_shape."
        )
        return tensor

    slice_end = [
        start + length for start, length in zip(slice_begin, slice_shape)
    ]
    axes = list(range(tensor.ndim))
    return paddle.slice(tensor, axes=axes, starts=slice_begin, ends=slice_end)


def extract_tensor_metadata(val):
    if isinstance(val, paddle.Tensor):
        # Case1: not initialized means this tensor is placed in another mesh which do not contain this rank
        if not val._is_initialized():
            return None, None
        if val.is_dist():
            local_tensor = val._local_value()
            # Note: The local_tensor must keep the same name with the original tensor. Otherwise, the StructuredToParameterName@@ mapping will be wrong.
            local_tensor.name = val.name
            # when val is scalar, the shape is []
            (
                local_shape,
                global_offset,
            ) = (
                compute_local_shape_and_global_offset(
                    val.shape,
                    val.process_mesh,
                    val.placements,
                )
                if len(val.shape) > 0
                else ((), ())
            )
            global_shape = val.shape
            if local_shape is None or global_offset is None:
                return None, None
        else:
            local_shape = tuple(val.shape)
            global_offset = (
                tuple([0] * len(val.shape)) if len(val.shape) > 0 else ()
            )
            global_shape = local_shape
            local_tensor = val
        is_flattened = False
        flattened_range = None
    elif isinstance(val, ShardedWeight):
        local_tensor = val.local_tensor
        local_shape = val.local_shape
        global_offset = val.global_offset
        global_shape = val.global_shape
        is_flattened = val.is_flattened
        flattened_range = val.flattened_range
    else:
        raise ValueError(
            f"The value of state_dict should be a paddle.Tensor, but got: {val}"
        )

    local_tensor_dtype = str(local_tensor.dtype).split('.')[1]
    if flattened_range is not None:
        flattened_range = (flattened_range.start, flattened_range.stop)
    else:
        flattened_range = None
    local_tensor_metadata = LocalTensorMetadata(
        tuple(global_offset),
        tuple(local_shape),
        local_tensor_dtype,
        tuple(global_shape),
        is_flattened,
        flattened_range,
    )
    assert (local_tensor is None) == (local_tensor_metadata is None), (
        "local_tensor and local_tensor_metadata must both be None or both not None!"
    )
    return local_tensor, local_tensor_metadata


def check_resumable_locally(
    path, state_dict, metadata_manager, use_dist, process_group
):
    local_load = True
    rank = paddle.distributed.get_rank() if use_dist else 0
    checkpoint_file = f"{rank}_0.distcp"
    file_path = os.path.join(path, checkpoint_file)

    if not os.path.isfile(file_path):
        local_load = False

    state_dict_metadata = {}
    for key, value in state_dict.items():
        _, local_tensor_metadata = extract_tensor_metadata(value)
        if local_tensor_metadata is not None:
            state_dict_metadata[key] = local_tensor_metadata

    if local_load:
        file_storage_info = metadata_manager.get_file_storage_info()
        cur_file_storage = {
            replace(index, replica_id=None)
            for index in file_storage_info.get(checkpoint_file, [])
        }

        for key, local_tensor_metadata in state_dict_metadata.items():
            local_tensor_index = LocalTensorIndex(
                tensor_key=key,
                global_offset=local_tensor_metadata.global_offset,
                is_flattened=local_tensor_metadata.is_flattened,
                flattened_range=local_tensor_metadata.flattened_range,
                local_shape=local_tensor_metadata.local_shape,
                replica_id=None,
            )
            if local_tensor_index not in cur_file_storage:
                local_load = False
                break

    if use_dist:
        global_local_loads = []
        paddle.distributed.all_gather_object(
            global_local_loads, local_load, process_group
        )
        return all(global_local_loads)
    else:
        return local_load
