# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from typing import TYPE_CHECKING

import paddle.distributed as dist

from .load_state_dict import _load_state_dict
from .metadata import LocalTensorIndex, LocalTensorMetadata, Metadata

if TYPE_CHECKING:
    from paddle.distributed.communication.group import Group

    from .sharded_weight import ShardedStateDict


def _check_1d_cover(intervals, global_range):
    intervals = sorted(intervals)
    pos = global_range[0]
    for start, end in intervals:
        if start > pos or end <= start:
            return False
        pos = end
    return pos >= global_range[1]


def check_shard_cover(shard_blocks, global_ranges):
    """
    shard_blocks: List of tuples, each tuple (start0, end0, start1, end1, ...)
    global_ranges: List of (start, end) for each dimension, e.g. [(0, 10), (0, 10)]
    """
    ndim = len(global_ranges)
    if ndim == 1:
        intervals = [(s[0], s[1]) for s in shard_blocks]
        return _check_1d_cover(intervals, global_ranges[0])
    else:
        grouped = {}
        for block in shard_blocks:
            k = (block[0], block[1])
            grouped.setdefault(k, []).append(block[2:])
        keys = list(grouped.keys())
        if not _check_1d_cover(keys, global_ranges[0]):
            return False
        for sub_blocks in grouped.values():
            if not check_shard_cover(sub_blocks, global_ranges[1:]):
                return False
        return True


def validate_sharded_state_dict_integrity(state_dict_shard_info):
    for tensor_key, shards in state_dict_shard_info.items():
        std_global_shape = shards[0][3]
        ndim = len(std_global_shape)
        for (
            global_offset,
            local_shape,
            dtype,
            global_shape,
            is_flattened,
        ) in shards:
            if global_shape != std_global_shape:
                raise ValueError(f"Inconsistent global_shape for {tensor_key}")
        blocks = []
        for shard in shards:
            block = []
            for d in range(ndim):
                (
                    global_offset,
                    local_shape,
                    dtype,
                    global_shape,
                    is_flattened,
                ) = shard
                start = global_offset[d]
                end = start + local_shape[d]
                block.append(start)
                block.append(end)
            blocks.append(tuple(block))
        global_ranges = [(0, global_shape[d]) for d in range(ndim)]
        if not check_shard_cover(blocks, global_ranges):
            raise ValueError(
                f"Invalid sharding for {tensor_key}, missing region!"
            )


def check_dtype_and_flatten(state_dict_shard_info):
    for key, value in state_dict_shard_info.items():
        flattened = False
        dtype_set = set()
        for (
            global_offset,
            local_shape,
            dtype,
            global_shape,
            is_flattened,
        ) in value:
            if is_flattened:
                flattened = True
            dtype_set.add(dtype)
        if len(dtype_set) > 1:
            raise ValueError(
                f"Inconsistent dtypes for {key}, cannot be reshard !"
            )
        if is_flattened:
            raise ValueError(f"Flattened tensor {key}, cannot be reshard !")


def validate_sharded_state_dict_boundaries(state_dict_shard_info):
    for tensor_key, shards in state_dict_shard_info.items():
        std_global_shape = shards[0][3]
        for shard in shards:
            global_offset, local_shape, dtype, global_shape, is_flattened = (
                shard
            )
            ndim = len(global_shape)
            assert len(local_shape) == ndim == len(global_offset), (
                f"{tensor_key}: shape/offset dims mismatch"
            )
            for d in range(ndim):
                gs = global_shape[d]
                ls = local_shape[d]
                go = global_offset[d]
                if not (0 <= go < gs):
                    raise ValueError(
                        f"{tensor_key}: global_offset[{d}]={go} out of range [0, {gs})"
                    )
                if not (ls > 0):
                    raise ValueError(
                        f"{tensor_key}: local_shape[{d}]={ls} must be positive"
                    )
                if not (go + ls <= gs):
                    raise ValueError(
                        f"{tensor_key}: offset+shape ({go}+{ls}) exceeds global_shape {gs} at dim {d}"
                    )


def check_src_state_dict_validity(state_dict_shard_info):
    check_dtype_and_flatten(state_dict_shard_info)
    validate_sharded_state_dict_integrity(state_dict_shard_info)


def check_dst_state_dict_validity(state_dict_shard_info):
    check_dtype_and_flatten(state_dict_shard_info)
    validate_sharded_state_dict_boundaries(state_dict_shard_info)


def check_src_dst_state_dict_validity(
    src_state_dict_shard_info, dst_state_dict_shard_info
):
    src_tensor_keys = set(src_state_dict_shard_info.keys())
    keys = list(dst_state_dict_shard_info)
    if any(isinstance(k, tuple) for k in keys):
        if not all(isinstance(k, tuple) for k in keys):
            raise ValueError("All keys must be tuples if any key is a tuple.")
        dst_tensor_keys = {k[0] for k in keys}
    else:
        dst_tensor_keys = set(keys)
    missing_keys = dst_tensor_keys - src_tensor_keys
    if len(missing_keys) > 0:
        raise ValueError(
            f"Missing tensors in destination state dict: {missing_keys} !"
        )
    dst_tensor_keys = set(dst_state_dict_shard_info.keys())
    for key in dst_tensor_keys:
        src_shards = src_state_dict_shard_info[
            key[0] if isinstance(key, tuple) else key
        ]
        dst_shards = dst_state_dict_shard_info[key]
        src_global_shape = src_shards[0][3]
        dst_global_shape = dst_shards[0][3]
        if src_global_shape != dst_global_shape:
            raise ValueError(f"Inconsistent global_shape for {key}!")


def merge_global_shard_info(global_shard_info):
    merged = {}
    for rank_shard_info in global_shard_info:
        for key, tensor_shard_info in rank_shard_info.items():
            if key not in merged:
                merged[key] = []
            merged[key].append(tensor_shard_info)
    return merged


def reshard_sharded_state_dict(
    src_sharded_state_dict: ShardedStateDict,
    dst_sharded_state_dict: ShardedStateDict,
    process_group: Group,
    coordinator_rank: int | None = 0,
    offload: bool | None = False,
    aoa_config: dist[str, list[str]] | None = None,
) -> None:
    local_src_state_dict_shard_info = {
        key: (
            tuple(value.global_offset),
            tuple(value.local_shape),
            str(value.local_tensor.dtype).split(".")[-1],
            tuple(value.global_shape),
            value.is_flattened,
        )
        for key, value in src_sharded_state_dict.items()
    }

    global_src_state_dict_shard_info = []
    dist.all_gather_object(
        global_src_state_dict_shard_info,
        local_src_state_dict_shard_info,
        group=process_group,
    )

    src_state_dict_shard_info = merge_global_shard_info(
        global_src_state_dict_shard_info
    )

    # check validity
    check_src_state_dict_validity(src_state_dict_shard_info)

    local_dst_state_dict_shard_info = {
        key: (
            value.global_offset,
            value.local_shape,
            str(value.local_tensor.dtype).split(".")[-1],
            value.global_shape,
            value.is_flattened,
        )
        for key, value in dst_sharded_state_dict.items()
    }

    global_dst_state_dict_shard_info = []
    dist.all_gather_object(
        global_dst_state_dict_shard_info,
        local_dst_state_dict_shard_info,
        group=process_group,
    )

    dst_state_dict_shard_info = merge_global_shard_info(
        global_dst_state_dict_shard_info
    )

    # check validity
    check_dst_state_dict_validity(dst_state_dict_shard_info)
    check_src_dst_state_dict_validity(
        src_state_dict_shard_info, dst_state_dict_shard_info
    )

    # build metadata
    state_dict_metadata = {
        tensor_name: [
            LocalTensorMetadata(
                global_offset=shard_info[0],
                local_shape=shard_info[1],
                dtype=shard_info[2],
            )
            for shard_info in shard_infos
        ]
        for tensor_name, shard_infos in src_state_dict_shard_info.items()
    }

    virtual_file_path = f"vfile_{dist.get_rank()}"
    local_storage_metadata = {
        LocalTensorIndex(
            tensor_key=value.key,
            global_offset=value.global_offset,
        ): virtual_file_path
        for key, value in src_sharded_state_dict.items()
    }

    global_storage_metadata: list[dict[LocalTensorIndex, str]] = []
    dist.all_gather_object(
        global_storage_metadata,
        local_storage_metadata,
        group=process_group,
    )

    # Merge storage metadata
    storage_metadata: dict[LocalTensorIndex, str] = {}
    for rank_storage_metadata in global_storage_metadata:
        storage_metadata.update(rank_storage_metadata)

    # Prepare metadata for loading
    metadata = Metadata(
        state_dict_metadata=state_dict_metadata,
        storage_metadata=storage_metadata,
        flat_mapping=None,
    )

    # Extract local tensors
    src_state_dict = {
        key: value.local_tensor for key, value in src_sharded_state_dict.items()
    }
    dst_state_dict = dst_sharded_state_dict
    # reshard using _load_state_dict
    _load_state_dict(
        target_state_dict=dst_state_dict,
        source_state_dict={virtual_file_path: src_state_dict},
        metadata_list=[metadata],
        coordinator_rank=coordinator_rank,
        process_group=process_group,
        offload=offload,
    )
