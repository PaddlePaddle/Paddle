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

import copy
import gc
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

import paddle
from paddle.distributed.communication.group import is_initialized
from paddle.distributed.fleet.utils.log_util import logger

from ..aoa.aoa_engine import (
    AOAEngine,
)
from .metadata import LocalTensorIndex, LocalTensorMetadata, Metadata
from .metadata_manager import MetadataManager
from .sharded_weight import (
    ShardedWeight,
    ShardedWeightDesc,
    make_replicated_sharded_weight,
)
from .utils import (
    assign_sharded_slice,
    build_global_state_shard_info,
    build_shard_desc,
    check_unique_id,
    compute_local_shape_and_global_offset,
    create_hf_ckpt_metadata,
    flat_range_in_min_slice,
    flatten_state_dict,
    get_max_id,
    is_sharded_state_dict,
    merge_state_dict_metadata,
    minimal_nd_slice,
    ravel_index,
)

if TYPE_CHECKING:
    from paddle import Tensor
    from paddle.distributed.collective import Group


@dataclass(frozen=True)
class ReadItem:
    """
    A communication operation for a Tensor between ranks.

    Attributes:
        tensor_name (str): Name of the tensor.
        src_global_offset (tuple[int]): Global offset in the source tensor.
        dst_global_offset (tuple[int] | None): Global offset in the destination tensor.
        dst_rank (list[int]): Destination ranks.
        src_rank (int): Source rank.
        dst_local_offset (tuple[int]): Local offset in the destination tensor partition.
        src_local_offset (tuple[int]): Local offset in the source tensor partition.
        slice_shape (tuple[int]): Shape of the slice to transfer.
        file_name (str): The name of the file from which the source tensor is read on the source rank.
        dtype (str): Data type of the tensor.
    """

    tensor_name: str
    src_global_offset: tuple[int]
    dst_global_offset: tuple[int] | None
    dst_rank: tuple[int]
    src_rank: int
    dst_local_offset: tuple[int]
    src_local_offset: tuple[int]
    slice_shape: tuple[int]
    file_name: str
    dtype: str
    comm_group: Group | None = None


PATH_TO_CHECKPOINT_FILES: dict[str, tuple[list, list]] = {}

_metadata_manager = MetadataManager()


def get_checkpoint_files(path, use_cache=True, unique_id=None):
    # if unique_id is None, all file ends with .metadata and .distcp is returned
    if unique_id is None:
        unique_id = ''
    global PATH_TO_CHECKPOINT_FILES
    if use_cache and path in PATH_TO_CHECKPOINT_FILES:
        return PATH_TO_CHECKPOINT_FILES[path]
    accessible_files = os.listdir(path)
    metadata_files = [
        file
        for file in accessible_files
        if file.endswith(f"{unique_id}.metadata")
    ]

    safetensors_files = [
        file for file in accessible_files if file.endswith(".safetensors")
    ]

    if len(safetensors_files) > 0:
        logger.info(
            f"Found HuggingFace-format checkpoint with files: {', '.join(safetensors_files)}"
        )
        metadata_files = [
            file
            for file in accessible_files
            if file.endswith(".auto_generated.metadata")
        ]
        if len(metadata_files) == 0:
            logger.info(
                f"No metadata file found in the checkpoint directory: {path}. Creating one now."
            )
            create_hf_ckpt_metadata(path)
            accessible_files = os.listdir(path)
            metadata_files = [
                file
                for file in accessible_files
                if file.endswith(".auto_generated.metadata")
            ]
            logger.info(
                f"Created metadata file: {metadata_files[0]} successfully."
            )
        return (metadata_files, safetensors_files)

    assert len(metadata_files) > 0, (
        f"No metadata file ends with '{unique_id}.metadata' found in the checkpoint directory: {path}."
    )
    local_data_files = [
        file
        for file in accessible_files
        if file.endswith(f"{unique_id}.distcp")
    ]
    assert len(local_data_files) > 0, (
        f"No data file ends with '{unique_id}.distcp' found in the checkpoint directory:{path}."
    )
    if use_cache:
        PATH_TO_CHECKPOINT_FILES[path] = (metadata_files, local_data_files)
    return (metadata_files, local_data_files)


def get_rank_to_files(
    metadata_list,
    local_data_files,
    state_dict,
    process_group,
    use_dist,
    mw_name_compatibility=True,
):
    """
    Get the mapping of rank to its accessible files.
    """

    # The necessary files to be read
    tensor_key_list = []
    necessary_files = []
    mw_name_compatibility_mapping = {}

    state_dict_param_names = {
        key if isinstance(key, str) else key[0] for key in state_dict.keys()
    }

    for metadata in metadata_list:
        for local_tensor_index, file_name in metadata.storage_metadata.items():
            tensor_key_list.append(local_tensor_index.tensor_key)
            if local_tensor_index.tensor_key in state_dict_param_names:
                necessary_files.append(file_name)

    all_necessary_files = []
    if use_dist:
        paddle.distributed.all_gather_object(
            all_necessary_files, necessary_files, process_group
        )
    else:
        all_necessary_files.append(necessary_files)

    global_necessary_files = [
        file for files in all_necessary_files for file in files
    ]

    global_necessary_files_set = set(global_necessary_files)
    if len(global_necessary_files_set) <= 0:
        logger.warning(
            "No necessary data files found in the checkpoint directory. Please check the metadata."
        )
        missing_keys = set(state_dict.keys())
        return {}, missing_keys, mw_name_compatibility_mapping

    # allgather all accessible files
    global_data_files = []
    if use_dist:
        paddle.distributed.all_gather_object(
            global_data_files, local_data_files, process_group
        )
    else:
        global_data_files.append(local_data_files)
    tmp = []
    for files in global_data_files:
        tmp += files
    global_data_files_set = set(tmp)
    logger.debug(
        f"necessary_data_files_set:{global_necessary_files_set}, global_data_files_set:{global_data_files_set}"
    )
    # check necessary files in global_data_files
    assert (
        global_data_files_set & global_necessary_files_set
        == global_necessary_files_set
    ), (
        f"The checkpoint files are not complete. Please check the checkpoint directory. global_data_files_set:{global_data_files_set}, necessary_data_files_set:{global_necessary_files_set}"
    )
    missing_keys = set(state_dict_param_names) - set(tensor_key_list)
    if len(missing_keys) > 0:
        if mw_name_compatibility:
            mw_name_compatibility_mapping = _modify_mw_name_for_compatibility(
                state_dict, missing_keys, tensor_key_list
            )
            if len(missing_keys) > 0:
                logger.warning(
                    f"Missing keys:{missing_keys}, check whether the checkpoint is complete."
                )
        else:
            logger.warning(
                f"Missing keys:{missing_keys}, check whether the checkpoint is complete."
            )

    rank_to_files = {}
    for rank, need_files in enumerate(all_necessary_files):
        seen = set()
        unique_need_files = [
            f for f in need_files if not (f in seen or seen.add(f))
        ]
        rank_to_files[rank] = unique_need_files
    logger.debug(f"mapping rank_to_files:{rank_to_files}")
    return rank_to_files, missing_keys, mw_name_compatibility_mapping


def _modify_mw_name_for_compatibility(
    state_dict, missing_keys, tensor_key_list
):
    """
    Adjust the master weight name within the optimizer's state_dict to ensure compatibility between semi-automatic parallel execution in both dynamic and static graph modes.
    Args:
        state_dict(Dict[str, paddle.Tensor]): The state_dict to load. It will be modified inplace after loading.
        missing_keys(Set[str]): A set of keys that are expected to be loaded but are missing.
        tensor_key_list(List[str]): A list of tensor keys from the source checkpoint (ckpt).
    """
    compatibility_set = set()
    mw_name_compatibility_mapping = {}
    compatibility_key = None
    for missing_key in missing_keys:
        parts = missing_key.split(".")
        # Determine compatibility key based on naming style
        if "master_weights" in parts:
            parts.remove("master_weights")
            compatibility_key = ".".join(parts) + "_fp32_master_0"
        elif parts[-1].endswith("_fp32_master_0"):
            parts[-1] = parts[-1].replace("_fp32_master_0", "")
            parts.insert(1, "master_weights")
            compatibility_key = ".".join(parts)
        if compatibility_key in tensor_key_list:
            logger.info(
                f"Modify master weights {missing_key} -> {compatibility_key}"
            )
            compatibility_set.add(missing_key)
            mw_name_compatibility_mapping[missing_key] = compatibility_key
            state_dict[compatibility_key] = state_dict.pop(missing_key)
    # update missing_keys
    missing_keys -= compatibility_set
    return mw_name_compatibility_mapping


def get_rank_to_read_files(rank_to_files, rank_to_local_data_files):
    cross_node_file_names = []
    rank_to_need_files = copy.deepcopy(rank_to_files)
    for rank, need_files in rank_to_need_files.items():
        local_data_files = rank_to_local_data_files[rank]
        file_need_to_remove = []
        for file in need_files:
            if file not in local_data_files:
                file_need_to_remove.append(file)
        for file in file_need_to_remove:
            need_files.remove(file)
        cross_node_file_names += file_need_to_remove

    not_read_file_ranks = []
    for rank, files in rank_to_need_files.items():
        if len(files) == 0:
            not_read_file_ranks.append(rank)
    for rank in not_read_file_ranks:
        rank_to_need_files.pop(rank)

    rank_load_files = _get_rank_to_read_files(rank_to_need_files)

    for rank in not_read_file_ranks:
        rank_load_files[rank] = []

    cur_load_files = []
    for rank, load_file in rank_load_files.items():
        cur_load_files += load_file

    unload_files = []
    for file in cross_node_file_names:
        if file not in cur_load_files:
            unload_files.append(file)

    file_to_ranks = {}
    for rank, files in rank_to_local_data_files.items():
        for file in files:
            if file not in file_to_ranks:
                file_to_ranks[file] = [rank]
            else:
                file_to_ranks[file].append(rank)

    seen = set()
    unload_files = [x for x in unload_files if not (x in seen or seen.add(x))]
    for file in unload_files:
        sub_rank_load_files = {}
        for rank in file_to_ranks[file]:
            sub_rank_load_files[rank] = rank_load_files[rank]
        min_rank = min(
            sub_rank_load_files,
            key=lambda rank: (len(sub_rank_load_files[rank]), rank),
        )
        rank_load_files[min_rank].append(file)

    cur_rank = paddle.distributed.get_rank()
    if cur_rank in rank_load_files:
        return rank_load_files[cur_rank]
    else:
        logger.warning(f"rank:{cur_rank} does not need to load checkpoint")
        return []


def _get_rank_to_read_files(rank_to_files):
    """
    Load files in a load-balanced manner.

    Args:
        rank_to_files (dict): mapping from rank to files.

    Example:
        Case1: all ranks access the same data files
            rank_to_files = {rank0:[0_0.distcp, 1_0.distcp, 2_0.distcp, 3_0.distcp], rank1:[0_0.distcp, 1_0.distcp, 2_0.distcp, 3_0.distcp]}
            rank0 return [0_0.distcp, 1_0.distcp], rank1 return [2_0.distcp, 3_0.distcp]
        Case2: all ranks access different data files but some overlapped
            rank_to_files = {rank0:[0_0.distcp, 1_0.distcp, 2_0.distcp], rank1:[2_0.distcp, 3_0.distcp]
            rank0 return [0_0.distcp, 1_0.distcp], rank1 return [2_0.distcp, 3_0.distcp]
        Case3: all ranks access different data files and no overlapped
            rank_to_files = {rank0:[0_0.distcp, 1_0.distcp], rank1:[2_0.distcp, 3_0.distcp]
            rank0 return [0_0.distcp, 1_0.distcp], rank1 return [2_0.distcp, 3_0.distcp]
    """
    file_to_ranks = {}
    for rank, files in rank_to_files.items():
        for file in files:
            if file not in file_to_ranks:
                file_to_ranks[file] = []
            file_to_ranks[file].append(rank)
    rank_to_not_read_files = copy.deepcopy(rank_to_files)
    rank_to_read_files = {rank: [] for rank in rank_to_not_read_files.keys()}
    for file, ranks in file_to_ranks.items():
        if len(ranks) == 1:
            rank = ranks[0]
            rank_to_read_files[rank].append(file)
            rank_to_not_read_files[rank].remove(file)
            if len(rank_to_not_read_files[rank]) == 0:
                rank_to_not_read_files.pop(rank)

    logger.debug(
        f"rank_to_read_files:{rank_to_read_files}, rank_to_not_read_files:{rank_to_not_read_files}"
    )

    def get_least_read_files_ranks(rank_to_read_files):
        nums = [
            (rank, len(files)) for rank, files in rank_to_read_files.items()
        ]
        nums = sorted(nums, key=lambda x: x[1])
        ranks = [rank for rank, num in nums if num == nums[0][1]]
        return ranks

    def get_read_rank_file(rank_to_not_read_files, ranks):
        if len(rank_to_not_read_files) == 0:
            return (None, None)
        nums = [
            (rank, len(files))
            for rank, files in rank_to_not_read_files.items()
            if rank in ranks
        ]
        # 'ranks' refer to the ranks that have read the fewest number of files so far. However, the files containing the weights required
        # . by these ranks may have already been completely read. In this case, they will not read any more files.
        if len(nums) == 0:
            nums = [
                (rank, len(files))
                for rank, files in rank_to_not_read_files.items()
            ]
        nums = sorted(nums, key=lambda x: x[1])
        rank = nums[0][0]
        return (rank, rank_to_not_read_files[rank][0])

    def update(rank_to_read_files, rank_to_not_read_files, rank_file):
        rank, file = rank_file
        if rank is None and file is None:
            return
        if rank not in rank_to_read_files:
            rank_to_read_files[rank] = []
        rank_to_read_files[rank].append(file)
        # update rank_to_not_read_files
        file_to_ranks = {}
        for r, files in rank_to_not_read_files.items():
            for f in files:
                if f not in file_to_ranks:
                    file_to_ranks[f] = []
                file_to_ranks[f].append(r)
        logger.debug(f"file_to_ranks:{file_to_ranks}")
        if file in file_to_ranks:
            for r in file_to_ranks[file]:
                rank_to_not_read_files[r].remove(file)
                if len(rank_to_not_read_files[r]) == 0:
                    rank_to_not_read_files.pop(r)

    while len(rank_to_not_read_files) > 0:
        ranks = get_least_read_files_ranks(rank_to_read_files)
        rank_file = get_read_rank_file(rank_to_not_read_files, ranks)
        update(rank_to_read_files, rank_to_not_read_files, rank_file)
        logger.debug(
            f"update rank_to_read_files:{rank_to_read_files}, rank_to_not_read_files:{rank_to_not_read_files}, ranks:{ranks}, rank_file:{rank_file}"
        )
    return rank_to_read_files


def get_load_infos(metadata_list, local_load_files, process_group, use_dist):
    load_info = {}
    cur_rank = paddle.distributed.get_rank()
    for metadata in metadata_list:
        for local_tensor_index, file_name in metadata.storage_metadata.items():
            if file_name in local_load_files:
                load_info[local_tensor_index] = (
                    cur_rank,
                    file_name,
                )
    load_info_list = []
    if use_dist:
        paddle.distributed.all_gather_object(
            load_info_list, load_info, process_group
        )
    else:
        load_info_list.append(load_info)
    load_infos = {}
    for load_info in load_info_list:
        for local_tensor_index, (rank, file_name) in load_info.items():
            assert local_tensor_index not in load_infos
            load_infos[local_tensor_index] = (rank, file_name)
    return load_infos


def compute_overlap(
    cur_chunk_metadata: LocalTensorMetadata,
    storage_local_tensor_metadata: LocalTensorMetadata,
):
    cur_offsets = []
    storage_offsets = []
    lengths = []
    for cur_len, cur_offset, storage_len, storage_offset in zip(
        cur_chunk_metadata.local_shape,
        cur_chunk_metadata.global_offset,
        storage_local_tensor_metadata.local_shape,
        storage_local_tensor_metadata.global_offset,
    ):
        begin_offset = max(cur_offset, storage_offset)
        end_offset = min(cur_offset + cur_len, storage_offset + storage_len)
        if begin_offset == cur_offset:
            cur_offsets.append(0)
            storage_offsets.append(begin_offset - storage_offset)
        elif begin_offset == storage_offset:
            cur_offsets.append(begin_offset - cur_offset)
            storage_offsets.append(0)
        else:
            raise ValueError(
                f"Invalid begin_offset:{begin_offset}, cur_offset:{cur_offset}, storage_offset:{storage_offset}"
            )
        lengths.append(end_offset - begin_offset)
        assert lengths[-1] >= 0, (
            f"Invalid length:{lengths[-1]}, end_offset:{end_offset}, begin_offset:{begin_offset}"
        )
    return cur_offsets, storage_offsets, lengths


def not_overlap(
    cur_chunk_metadata: LocalTensorMetadata,
    storage_local_tensor_metadata: LocalTensorMetadata,
):
    for cur_len, cur_offset, storage_len, storage_offset in zip(
        cur_chunk_metadata.local_shape,
        cur_chunk_metadata.global_offset,
        storage_local_tensor_metadata.local_shape,
        storage_local_tensor_metadata.global_offset,
    ):
        if (
            cur_offset >= (storage_offset + storage_len)
            or (cur_offset + cur_len) <= storage_offset
        ):
            return True
    return False


def get_read_items(
    metadata_list, state_dict, process_group, use_dist, load_infos
):
    storage_state_dict_metadata = {}
    for metadata in metadata_list:
        for (
            tensor_key,
            local_tensor_metadata,
        ) in metadata.state_dict_metadata.items():
            if tensor_key not in storage_state_dict_metadata:
                storage_state_dict_metadata[tensor_key] = []
            storage_state_dict_metadata[tensor_key] += local_tensor_metadata

    read_items = []
    global_shape = None
    for tensor_key, val in state_dict.items():
        tensor_name = None
        if isinstance(val, paddle.Tensor):
            if val.is_dist():
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
                global_shape = tuple(val.shape)
                if local_shape is None or global_offset is None:
                    continue
            else:
                local_shape = tuple(val.shape)
                global_offset = (
                    tuple([0] * len(val.shape)) if len(val.shape) > 0 else ()
                )
                global_shape = local_shape
            dtype = str(val.dtype).split(".")[1]
            tensor_name = tensor_key
        elif isinstance(val, ShardedWeight):
            local_shape, global_offset = (
                (val.local_shape, val.global_offset)
                if len(val.global_shape) > 0
                else ((), ())
            )
            dtype = str(val.local_tensor.dtype).split(".")[1]
            tensor_name = (
                tensor_key[0] if isinstance(tensor_key, tuple) else tensor_key
            )
        else:
            raise ValueError(
                f"Only support paddle.Tensor., val type:{type(val)}"
            )

        cur_chunk_metadata = LocalTensorMetadata(
            global_offset, local_shape, dtype, global_shape
        )

        for storage_local_tensor_metadata in storage_state_dict_metadata[
            tensor_name
        ]:
            if not_overlap(cur_chunk_metadata, storage_local_tensor_metadata):
                continue
            cur_offsets, storage_offsets, lengths = compute_overlap(
                cur_chunk_metadata, storage_local_tensor_metadata
            )
            storage_local_tensor_index = LocalTensorIndex(
                tensor_name,
                tuple(storage_local_tensor_metadata.global_offset),
            )
            src_rank, file_name = load_infos[storage_local_tensor_index]
            read_items.append(
                ReadItem(
                    tensor_name=tensor_name,
                    src_global_offset=tuple(
                        storage_local_tensor_metadata.global_offset
                    ),
                    dst_global_offset=global_offset,
                    dst_rank=(paddle.distributed.get_rank(),),
                    src_rank=src_rank,
                    dst_local_offset=tuple(cur_offsets),
                    src_local_offset=tuple(storage_offsets),
                    slice_shape=tuple(lengths),
                    file_name=file_name,
                    dtype=storage_local_tensor_metadata.dtype,
                ),
            )

    global_read_items = []
    tmp = []
    if use_dist:
        paddle.distributed.all_gather_object(tmp, read_items, process_group)
    else:
        tmp.append(read_items)
    for items in tmp:
        for item in items:
            global_read_items.append(item)
    return global_read_items


def _split_flat_shards(state_dict):
    flat_shards, nonflat_shards = {}, {}
    for key, shard in state_dict.items():
        if getattr(shard, "is_flattened", False):
            flat_shards[key] = shard
        else:
            nonflat_shards[key] = shard
    return flat_shards, nonflat_shards


def _unflatten_shards(flat_shards):
    load_dict, padding_info = {}, {}
    for key, flat_shard in flat_shards.items():
        local_shape = flat_shard.local_shape
        flat_start, flat_end = (
            flat_shard.flattened_range.start,
            flat_shard.flattened_range.stop,
        )
        min_slices, _, _ = minimal_nd_slice(local_shape, flat_start, flat_end)
        min_flat_start, min_flat_end = flat_range_in_min_slice(
            local_shape, min_slices, flat_start, flat_end
        )
        min_shape = tuple(e - s for s, e in min_slices)
        min_offset = tuple(
            g_off + s[0]
            for g_off, s in zip(flat_shard.global_offset, min_slices)
        )
        min_numel = math.prod(min_shape)
        flat_numel = flat_end - flat_start

        if min_numel == flat_numel:
            tensor = flat_shard.local_tensor.reshape_(min_shape)
            load_dict[key] = ShardedWeight(
                key=key,
                local_tensor=tensor,
                local_shape=min_shape,
                global_shape=flat_shard.global_shape,
                global_offset=min_offset,
                is_flattened=False,
                flattened_range=None,
            )
        else:
            pad_tensor = paddle.zeros(
                min_shape, dtype=flat_shard.local_tensor.dtype
            )
            load_dict[key] = ShardedWeight(
                key=key,
                local_tensor=pad_tensor,
                local_shape=min_shape,
                global_shape=flat_shard.global_shape,
                global_offset=min_offset,
                is_flattened=False,
                flattened_range=None,
            )
            padding_info[key] = {
                "src": pad_tensor,
                "flat_shard": flat_shard,
                "slice_range": (min_flat_start, min_flat_end),
                "min_shape": min_shape,
            }
    return load_dict, padding_info


def _handle_aoa(
    load_dict,
    destination_state_shard_info,
    path,
    process_group,
    worker_groups,
    coordinator_rank,
    unique_id,
    offload,
    aoa_config,
    safetensors,
):
    metadata_files, _ = get_checkpoint_files(path, unique_id=unique_id)
    assert len(metadata_files) == 1, "Only support one metadata file now."
    metadata = paddle.load(os.path.join(path, metadata_files[0]))
    state_dict_metadata = metadata.state_dict_metadata

    source_state_shard_info = {
        param_name: [
            ShardedWeightDesc(
                key=param_name,
                local_shape=tuple(meta.local_shape),
                global_shape=tuple(meta.global_shape),
                global_offset=tuple(meta.global_offset),
                dtype=meta.dtype,
            )
            for meta in local_tensor_metas
        ]
        for param_name, local_tensor_metas in state_dict_metadata.items()
    }

    aoa_engine = AOAEngine(
        source_state_shard_info=source_state_shard_info,
        destination_state_shard_info=destination_state_shard_info,
        aoa_config=aoa_config,
    )

    src_desc_to_sharded_tensor = {}
    dst_to_src_desc_mapping = {}
    new_load_dict = {}
    src_desc_to_postprocess_list = {}
    force_gc = []

    for param_name, tgt_shard in load_dict.items():
        tgt_desc = build_shard_desc(tgt_shard)
        shard_mappings = aoa_engine.find_shard_sources(tgt_desc)
        for mapping in shard_mappings:
            src_desc = mapping.source_slice
            dst_desc = mapping.target_slice
            idx = (src_desc.key, tuple(src_desc.global_offset))
            if mapping.postprocess_list is not None:
                src_desc_to_postprocess_list[src_desc] = (
                    mapping.postprocess_list
                )
            if (len(shard_mappings) == 1) and (
                src_desc.local_shape == dst_desc.local_shape
                and src_desc.global_shape == dst_desc.global_shape
                and src_desc.global_offset == dst_desc.global_offset
                and src_desc.dtype == dst_desc.dtype
                and mapping.postprocess_list is None
            ):
                new_load_dict[idx] = ShardedWeight(
                    key=src_desc.key,
                    local_tensor=tgt_shard.local_tensor,
                    local_shape=src_desc.local_shape,
                    global_shape=src_desc.global_shape,
                    global_offset=src_desc.global_offset,
                )
            else:
                local_tensor = paddle.empty(
                    src_desc.local_shape, dtype=src_desc.dtype
                )
                force_gc.append(local_tensor)
                if local_tensor.place != tgt_shard.local_tensor.place:
                    local_tensor = local_tensor.to(tgt_shard.local_tensor.place)
                new_load_dict[idx] = ShardedWeight(
                    key=src_desc.key,
                    local_tensor=local_tensor,
                    local_shape=src_desc.local_shape,
                    global_shape=src_desc.global_shape,
                    global_offset=src_desc.global_offset,
                )
                src_desc_to_sharded_tensor[src_desc] = new_load_dict[idx]
                dst_to_src_desc_mapping[dst_desc] = src_desc

    load_state_dict_impl(
        state_dict=new_load_dict,
        path=path,
        process_group=process_group,
        coordinator_rank=coordinator_rank,
        unique_id=unique_id,
        offload=offload,
        safetensors=safetensors,
        worker_groups=worker_groups,
    )

    for dst_desc, src_desc in dst_to_src_desc_mapping.items():
        src_tensor = src_desc_to_sharded_tensor[src_desc]
        dst_tensor = load_dict[dst_desc.key]
        postprocess_list = src_desc_to_postprocess_list.get(src_desc, None)
        assign_sharded_slice(
            src_desc, src_tensor, dst_desc, dst_tensor, postprocess_list
        )

    for tensor in force_gc:
        # force GC
        tensor._clear()
        del tensor


def _finish_unflatten(flat_shards, padding_info):
    for key, info in padding_info.items():
        src_tensor = info["src"]
        flat_shard = info["flat_shard"]
        start, end = info["slice_range"]
        src_flat = src_tensor.flatten()
        paddle.assign(src_flat[start:end], flat_shard.local_tensor)
        # force GC
        src_flat._clear()
        src_tensor._clear()
    for key, flat_shard in flat_shards.items():
        flat_shard.local_tensor.flatten_()


def load_state_dict(
    state_dict: dict[str, Tensor] | dict[str, ShardedWeight],
    path: str,
    process_group: Group | None = None,
    coordinator_rank: int = 0,
    unique_id: int | None = None,
    offload: bool = False,
    mw_name_compatibility: bool = True,
    aoa_config: dict[str, list[str]] | None = None,
    safetensors: bool = False,
    worker_groups: list[Group] | None = None,
) -> None:
    r"""
    Load the state_dict inplace from a checkpoint path.

    Args:
        state_dict(Dict[str, paddle.Tensor]): The state_dict to load. It will be modified inplace after loading.
        path(str): The directory to load checkpoint files.
        process_group(paddle.distributed.collective.Group): ProcessGroup to be used for cross-rank synchronization. Use the default process group which contains all cards.
        coordinator_rank(int): The rank used to coordinate the checkpoint. Rank0 is used by default.
        unique_id(int): The unique id of checkpoint, used to distinguish between different checkpoint versions. Default is None, in which case the id the max id of given path, and the newest version checkpoint is loaded.
        offload(bool): Whether to offload the checkpoint data from GPU to CPU.
        mw_name_compatibility(bool): Enable name compatibility between dynamic and static graph semi-automatic parallel. Default is True.
        aoa_config(dict[str, list[str]]): AOA config to change parameters. Default is None.
        safetensors(bool): Whether to use safetensors format. Default is False.
        worker_groups (list[paddle.distributed.collective.Group]): Communication groups used for tensor communications; if multiple are provided, an appropriate group is chosen; if None, the global group (all cards) is used.
    Example:
        .. code-block:: python

            >>> # doctest: +SKIP('run in distributed mode.')
            >>> import paddle
            >>> import paddle.distributed as dist
            >>> ckpt_path = "./checkpoint"
            >>> w1 = paddle.arange(32).reshape([4, 8])
            >>> mesh = dist.ProcessMesh([0, 1])
            >>> sharded_w1 = dist.shard_tensor(w1, mesh, [dist.Shard(0)])
            >>> state_dict = {"w1": sharded_w1}
            >>> dist.save_state_dict(state_dict, ckpt_path)
            >>> w1_to_load = paddle.zeros_like(w1)
            >>> sharded_w1_to_load = dist.shard_tensor(w1, mesh, [dist.Replicate()])
            >>> state_dict_to_load = {"w1": sharded_w1_to_load}
            >>> dist.load_state_dict(state_dict_to_load, ckpt_path)
            >>> print(f"state_dict_to_load:{state_dict_to_load}")
            state_dict_to_load:{'w1': Tensor(shape=[4, 8], dtype=int64, place=Place(gpu:0), stop_gradient=True, dist_attr={process_mesh: {shape: [2], process_ids: [0,1], dim_names: [d0]}, dims_mappings: [-1,-1], batch_dim: 0, dynamic_dims: [0,0], annotated: [dims_mapping: 1,process_mesh: 1], partial: [].}, GlobalDenseTensor=
            [[0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 ],
            [8 , 9 , 10, 11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20, 21, 22, 23],
            [24, 25, 26, 27, 28, 29, 30, 31]])}
            >>> # doctest: -SKIP
    """
    use_dist = paddle.distributed.get_world_size() > 1

    if use_dist and process_group is None and not is_initialized():
        # Init the default global process group
        paddle.distributed.init_parallel_env()

    if use_dist:
        paddle.distributed.barrier(process_group)

    if not is_sharded_state_dict(state_dict):
        load_state_dict_impl(
            state_dict=state_dict,
            path=path,
            process_group=process_group,
            coordinator_rank=coordinator_rank,
            unique_id=unique_id,
            offload=offload,
            mw_name_compatibility=mw_name_compatibility,
            safetensors=safetensors,
            worker_groups=worker_groups,
        )
        return

    if not use_dist:
        load_dict = {}
        for key, val in state_dict.items():
            assert val.local_shape == val.global_shape, (
                f"{key} is not replicated!"
            )
            load_dict[key] = val
        destination_state_shard_info = defaultdict(list)
        for key, val in load_dict.items():
            desc = build_shard_desc(val)
            destination_state_shard_info[key].append(desc)
    else:
        flat_shards, nonflat_shards = _split_flat_shards(state_dict)
        load_dict, padding_info = _unflatten_shards(flat_shards)
        load_dict.update(nonflat_shards)
        destination_state_shard_info = build_global_state_shard_info(
            state_dict, process_group
        )

    if aoa_config is not None:
        _handle_aoa(
            load_dict,
            destination_state_shard_info,
            path,
            process_group,
            worker_groups,
            coordinator_rank,
            unique_id,
            offload,
            aoa_config,
            safetensors,
        )
    else:
        load_state_dict_impl(
            state_dict=load_dict,
            path=path,
            process_group=process_group,
            coordinator_rank=coordinator_rank,
            unique_id=unique_id,
            offload=offload,
            mw_name_compatibility=mw_name_compatibility,
            safetensors=safetensors,
            worker_groups=worker_groups,
        )
    if use_dist:
        _finish_unflatten(flat_shards, padding_info)

    global _metadata_manager
    _metadata_manager.clear()
    gc.collect()


def restore_unflattened_state_dict(
    source_state_dict: dict[str, dict[str, Tensor]],
    process_group,
    worker_groups,
):
    global _metadata_manager
    use_dist = paddle.distributed.get_world_size() > 1

    flattened_tensors = {}
    already_unflattened_tensors = {}
    for file_name, state_dict in source_state_dict.items():
        for tensor_name, tensor in state_dict.items():
            key = (tensor_name, file_name)
            meta = _metadata_manager.local_tensor_metadata[key]
            if meta.is_flattened:
                flattened_tensors[key] = tensor
            else:
                already_unflattened_tensors[key] = tensor

    direct_reshape_tensors = {}
    direct_reshape_metas = {}
    reshard_needed_tensors = {}

    reshard_target_infos = {}

    for key, local_tensor in flattened_tensors.items():
        meta = _metadata_manager.local_tensor_metadata[key]

        flat_start, flat_end = meta.flattened_range
        slices, _, _ = minimal_nd_slice(meta.local_shape, flat_start, flat_end)

        unflattened_local_shape = tuple(e - s for s, e in slices)
        unflattened_global_offset = tuple(
            o + s[0] for o, s in zip(meta.global_offset, slices)
        )
        numel_in_slice = math.prod(unflattened_local_shape)

        unflattened_meta = LocalTensorMetadata(
            local_shape=unflattened_local_shape,
            global_shape=meta.global_shape,
            dtype=meta.dtype,
            global_offset=unflattened_global_offset,
            is_flattened=False,
            flattened_range=None,
        )

        if numel_in_slice == (flat_end - flat_start):
            direct_reshape_tensors[key] = local_tensor.reshape_(
                unflattened_local_shape
            )
            direct_reshape_metas[key] = unflattened_meta
        else:
            reshard_needed_tensors[key] = local_tensor
            reshard_target_infos[key] = (
                numel_in_slice,
                slices,
                unflattened_meta,
            )

    resharded_tensors = {}
    force_gc = []

    source_state_dict_for_reshard = defaultdict(dict)
    source_local_tensor_meta = defaultdict(list)
    source_storage_meta = {}
    destination_sharded_state_dict = {}
    name_mapping = {}

    for key, local_tensor in reshard_needed_tensors.items():
        tensor_name, file_name = key
        meta = _metadata_manager.local_tensor_metadata[key]
        numel, slices, unflattened_meta = reshard_target_infos[key]
        tensor_name_expand = f"{tensor_name}.global_offset.{meta.global_offset}"

        flat_start, flat_end = meta.flattened_range
        source_state_dict_for_reshard[file_name][tensor_name_expand] = (
            local_tensor
        )
        source_local_tensor_meta[tensor_name_expand].append(
            LocalTensorMetadata(
                local_shape=(flat_end - flat_start,),
                global_shape=(math.prod(meta.local_shape),),
                dtype=meta.dtype,
                global_offset=(flat_start,),
                is_flattened=False,
            )
        )
        source_storage_meta[
            LocalTensorIndex(
                tensor_key=tensor_name_expand, global_offset=(flat_start,)
            )
        ] = file_name

        tmp_target_tensor = paddle.zeros((numel,), dtype=local_tensor.dtype)
        global_offset_1d = (
            ravel_index(tuple(s[0] for s in slices), meta.local_shape),
        )

        destination_sharded_state_dict[
            (tensor_name_expand, global_offset_1d)
        ] = ShardedWeight(
            key=tensor_name_expand,
            local_tensor=tmp_target_tensor,
            local_shape=(numel,),
            global_shape=(math.prod(meta.local_shape),),
            global_offset=global_offset_1d,
        )
        name_mapping[key] = (tensor_name_expand, global_offset_1d)
        force_gc.append(local_tensor)

    global_state_dict_metadata, global_storage_metadata = [], []
    if use_dist:
        paddle.distributed.all_gather_object(
            global_state_dict_metadata, source_local_tensor_meta, process_group
        )
        paddle.distributed.all_gather_object(
            global_storage_metadata, source_storage_meta, process_group
        )
    else:
        global_state_dict_metadata = [source_local_tensor_meta]
        global_storage_metadata = [source_storage_meta]

    tmp_metadata = Metadata()
    tmp_metadata.state_dict_metadata = merge_state_dict_metadata(
        global_state_dict_metadata
    )
    tmp_metadata.storage_metadata = {
        k: v for d in global_storage_metadata for k, v in d.items()
    }

    _load_state_dict(
        target_state_dict=destination_sharded_state_dict,
        source_state_dict=source_state_dict_for_reshard,
        metadata_list=[tmp_metadata],
        process_group=process_group,
        worker_groups=worker_groups,
    )

    for key in reshard_needed_tensors:
        target_key = name_mapping[key]
        unflattened_meta = reshard_target_infos[key][2]

        final_tensor = destination_sharded_state_dict[target_key].local_tensor
        final_tensor.reshape_(unflattened_meta.local_shape)
        resharded_tensors[key] = final_tensor

    final_unflattened_state_dict = defaultdict(dict)
    final_local_tensor_meta = defaultdict(list)
    final_storage_meta = {}

    all_unflattened_tensors_with_meta = []

    for key, tensor in already_unflattened_tensors.items():
        all_unflattened_tensors_with_meta.append(
            (key, tensor, _metadata_manager.local_tensor_metadata[key])
        )

    for key, tensor in direct_reshape_tensors.items():
        all_unflattened_tensors_with_meta.append(
            (key, tensor, direct_reshape_metas[key])
        )

    for key, tensor in resharded_tensors.items():
        unflattened_meta = reshard_target_infos[key][2]
        all_unflattened_tensors_with_meta.append(
            (key, tensor, unflattened_meta)
        )

    for key, tensor, meta in all_unflattened_tensors_with_meta:
        tensor_name, file_name = key
        final_unflattened_state_dict[file_name][tensor_name] = tensor
        final_local_tensor_meta[tensor_name].append(meta)
        final_storage_meta[
            LocalTensorIndex(
                tensor_key=tensor_name,
                global_offset=meta.global_offset,
                is_flattened=False,
                flattened_range=None,
            )
        ] = file_name

    global_state_dict_metadata, global_storage_metadata = [], []
    if use_dist:
        paddle.distributed.all_gather_object(
            global_state_dict_metadata, final_local_tensor_meta, process_group
        )
        paddle.distributed.all_gather_object(
            global_storage_metadata, final_storage_meta, process_group
        )
    else:
        global_state_dict_metadata = [final_local_tensor_meta]
        global_storage_metadata = [final_storage_meta]

    final_metadata = Metadata()
    final_metadata.state_dict_metadata = merge_state_dict_metadata(
        global_state_dict_metadata
    )
    final_metadata.storage_metadata = {
        k: v for d in global_storage_metadata for k, v in d.items()
    }
    final_metadata.flat_mapping = _metadata_manager.get_flat_mapping()
    _metadata_manager.set_metadata_list([final_metadata])

    for tensor in force_gc:
        # force GC
        tensor._clear()

    return final_unflattened_state_dict


def load_state_dict_impl(
    state_dict: (
        dict[str, Tensor]
        | dict[str, ShardedWeight]
        | dict[tuple[str, tuple[int, ...]], ShardedWeight]
    ),
    path: str,
    process_group: Group | None = None,
    coordinator_rank: int = 0,
    unique_id: int | None = None,
    offload: bool = False,
    mw_name_compatibility: bool = True,
    safetensors: bool = False,
    worker_groups: list[Group] | None = None,
) -> None:
    with paddle.base.dygraph.guard():
        global _metadata_manager
        assert isinstance(state_dict, dict), (
            "The state_dict should be a dictionary."
        )
        first_key = next(iter(state_dict), None)
        if isinstance(first_key, tuple):
            flat_state_dict = state_dict
            mapping = {}
        else:
            flat_state_dict, mapping = flatten_state_dict(state_dict)

        if len(flat_state_dict) > 0:
            for val in flat_state_dict.values():
                assert isinstance(val, (paddle.Tensor, ShardedWeight)), (
                    f"The value of state_dict should be a paddle.Tensor, but got: {val}."
                )

        use_dist = True if paddle.distributed.get_world_size() > 1 else False

        if use_dist:
            # sync to avoid some ranks not write path yet
            paddle.distributed.barrier(process_group)
        if unique_id is None:
            unique_id = get_max_id(path)
        else:
            assert unique_id >= 0, f'{unique_id} should be >= 0'
        logger.info(f"The unique_id:{unique_id} is used.")

        if use_dist:
            check_unique_id(unique_id, process_group)

        metadata_files, local_data_files = get_checkpoint_files(
            path, unique_id=unique_id
        )

        metadata_list = []
        for file in metadata_files:
            metadata_list.append(paddle.load(os.path.join(path, file)))

        global _metadata_manager
        _metadata_manager.set_metadata_list(metadata_list)

        rank_to_files, missing_keys, mw_name_compatibility_mapping = (
            get_rank_to_files(
                _metadata_manager.get_metadata_list(),
                local_data_files,
                flat_state_dict,
                process_group,
                use_dist,
                mw_name_compatibility,
            )
        )
        if len(missing_keys) > 0:
            logger.warning(
                f"The following keys:{missing_keys} are not found in checkpoint path: {path}."
            )
        if len(rank_to_files) <= 0:
            return

        cur_rank = paddle.distributed.get_rank()
        global_local_data_files = []
        if use_dist:
            paddle.distributed.all_gather_object(
                global_local_data_files,
                {cur_rank: local_data_files},
                process_group,
            )
        else:
            global_local_data_files = [{cur_rank: local_data_files}]

        rank_to_local_data_files = {}
        for d in global_local_data_files:
            rank_to_local_data_files.update(d)

        local_load_files = get_rank_to_read_files(
            rank_to_files, rank_to_local_data_files
        )

        logger.info(f"Rank {cur_rank}: loading files from {local_load_files}.")

        source_state_dict = {}
        for file in local_load_files:
            if offload:
                state_dict_numpy = paddle.load(
                    os.path.join(path, file),
                    return_numpy=True,
                    safetensors=safetensors,
                )
                source_state_dict[file] = {
                    key: paddle.to_tensor(value, place=paddle.CPUPlace())
                    for key, value in state_dict_numpy.items()
                }
            else:
                source_state_dict[file] = paddle.load(
                    os.path.join(path, file), safetensors=safetensors
                )

        if use_dist:
            paddle.distributed.barrier(process_group)

        if _metadata_manager.has_flattened_tensors:
            logger.info("Restoring unflattened state dict.")
            source_state_dict = restore_unflattened_state_dict(
                source_state_dict, process_group, worker_groups
            )
            logger.info("Restored unflattened state dict.")

        _load_state_dict(
            flat_state_dict,
            source_state_dict,
            _metadata_manager.get_metadata_list(),
            process_group,
            coordinator_rank,
            offload,
            worker_groups,
        )

        for file_name, state_dict in source_state_dict.items():
            for key, value in state_dict.items():
                # force GC
                value._clear()

        del source_state_dict

        for flat_key, keys in mapping.items():
            if (
                mw_name_compatibility
                and flat_key in mw_name_compatibility_mapping
            ):
                flat_key = mw_name_compatibility_mapping[flat_key]
            tmp = state_dict
            for key in keys[:-1]:
                tmp = tmp[key]
            tmp[keys[-1]] = flat_state_dict[flat_key]


def slice_tensor(tensor, slice_begin, slice_shape):
    # If slice_shape is empty, the tensor is 0-dimensional (scalar); return it as is.
    if len(slice_shape) == 0:
        assert len(tensor.shape) == 0, (
            "Only 0-dimensional tensor supports empty slice_shape."
        )
        return tensor
    slice_end = [
        start + length for start, length in zip(slice_begin, slice_shape)
    ]
    axes = list(range(tensor.ndim))
    return paddle.slice(tensor, axes=axes, starts=slice_begin, ends=slice_end)


def get_target_tensor(target_state_dict, read_item):
    use_dist = True if paddle.distributed.get_world_size() > 1 else False
    if any(isinstance(k, tuple) for k in target_state_dict):
        key = (read_item.tensor_name, read_item.dst_global_offset)
    else:
        key = read_item.tensor_name
    target_tensor = (
        target_state_dict[key]._local_value()
        if use_dist and target_state_dict[key].is_dist()
        else target_state_dict[key]
    )
    return target_tensor


def process_local_copy_tasks(
    local_tasks, cur_rank, source_state_dict, target_state_dict
):
    """
    Complete local copy tasks.
    """
    logger.debug(
        f"Rank {cur_rank} starting local copy for {len(local_tasks)} tasks."
    )
    for task in local_tasks:
        if task.src_rank != cur_rank:
            continue

        src_tensor = source_state_dict[task.file_name][task.tensor_name]
        dst_tensor = get_target_tensor(target_state_dict, task)
        src_chunk_tensor = slice_tensor(
            src_tensor, task.src_local_offset, task.slice_shape
        )

        dst_chunk_tensor = slice_tensor(
            dst_tensor, task.dst_local_offset, task.slice_shape
        )
        if src_chunk_tensor.place == dst_chunk_tensor.place:
            paddle.assign(src_chunk_tensor, dst_chunk_tensor)
            logger.debug(f"Local copy (same device) for task {task}.")
        else:
            tmp = (
                src_chunk_tensor.cuda()
                if dst_chunk_tensor.place.is_gpu_place()
                else src_chunk_tensor.cpu()
            )
            paddle.assign(tmp, dst_chunk_tensor)
            del tmp
            logger.debug(f"Local copy (cross device) for task {task}.")


def split_read_items(
    read_items: list[ReadItem],
) -> (list[ReadItem], list[ReadItem]):
    local_read_items = []
    comm_read_items = []

    for item in read_items:
        assert len(item.dst_rank) == 1, (
            "Before read_items is split, each ReadItem describes a communication task between one rank and another."
        )
        if item.src_rank == item.dst_rank[0]:
            local_read_items.append(item)
        else:
            comm_read_items.append(item)

    return local_read_items, comm_read_items


def schedule_comm_read_items_single_group(
    comm_read_items: list[ReadItem],
) -> dict[str, list[ReadItem]]:
    order_rules = lambda read_item: (
        read_item.tensor_name,
        read_item.src_rank,
        read_item.src_global_offset,
        read_item.dst_rank,
        read_item.dst_local_offset,
        read_item.dst_global_offset
        if read_item.dst_global_offset is not None
        else (),
        read_item.src_local_offset,
        read_item.slice_shape,
        read_item.file_name,
        read_item.dtype,
    )
    comm_read_items = sorted(comm_read_items, key=order_rules)
    # Step 1: Group by tensor_name
    tensor_groups = defaultdict(list)
    for item in comm_read_items:
        tensor_groups[item.tensor_name].append(item)

    scheduled_items = defaultdict(list)

    # Step 2: For each tensor_name group, further group by all attributes except dst_rank
    for tensor_name, items in tensor_groups.items():
        grouped_items = defaultdict(list)
        for item in items:
            key = (
                item.src_global_offset,
                item.dst_global_offset,
                item.src_rank,
                item.dst_local_offset,
                item.src_local_offset,
                item.slice_shape,
                item.file_name,
                item.dtype,
            )
            grouped_items[key].append(item)

        # Step 3: Combine items with the same key into a single ReadItem with all dst_ranks
        for key, grouped_item in grouped_items.items():
            combined_dst_rank = []
            for item in grouped_item:
                combined_dst_rank.extend(item.dst_rank)
            combined_dst_rank = sorted(
                set(combined_dst_rank)
            )  # Remove duplicates

            # Create a new ReadItem with combined dst_ranks
            scheduled_item = ReadItem(
                tensor_name=tensor_name,
                src_global_offset=key[0],
                dst_global_offset=key[1],
                dst_rank=tuple(combined_dst_rank),
                src_rank=key[2],
                dst_local_offset=key[3],
                src_local_offset=key[4],
                slice_shape=key[5],
                file_name=key[6],
                dtype=key[7],
            )
            scheduled_items[tensor_name].append(scheduled_item)
    for key, items in scheduled_items.items():
        scheduled_items[key] = sorted(items, key=order_rules)

    return dict(sorted(scheduled_items.items()))


def schedule_comm_read_items_multi_group(
    comm_read_items: list[ReadItem],
    worker_groups: list[Group],
) -> list[list[ReadItem]]:
    group_members = {}
    name_to_groups = {}
    read_items = []

    order_rules = lambda read_item: (
        read_item.tensor_name,
        read_item.src_rank,
        read_item.src_global_offset,
        read_item.dst_rank,
        read_item.dst_local_offset,
        read_item.dst_global_offset
        if read_item.dst_global_offset is not None
        else (),
        read_item.src_local_offset,
        read_item.slice_shape,
        read_item.file_name,
        read_item.dtype,
    )

    def _find_min_group(need_ranks, group_members, name_to_groups):
        min_group = None
        min_size = None
        for name, ranks in group_members.items():
            if need_ranks <= ranks:
                if (min_size is None) or (len(ranks) < min_size):
                    min_size = len(ranks)
                    min_group = name_to_groups[name]
        assert min_group is not None, f"No group found for {need_ranks}!"
        return min_group

    for group in worker_groups:
        if len(group.ranks) <= 1:
            continue
        group_members[group.name] = set(group.ranks)
        name_to_groups[group.name] = group

    for read_item in comm_read_items:
        need_ranks = need_ranks = {*read_item.dst_rank, read_item.src_rank}
        group = _find_min_group(
            need_ranks,
            group_members,
            name_to_groups,
        )
        read_items.append(replace(read_item, comm_group=group))

    read_items = sorted(read_items, key=order_rules)

    def _build_group_conflict(group_members: dict[str, set]):
        member_to_groups = defaultdict(set)
        for g, members in group_members.items():
            for m in members:
                member_to_groups[m].add(g)
        group_conflict = defaultdict(set)
        for group_set in member_to_groups.values():
            for g1 in group_set:
                for g2 in group_set:
                    if g1 != g2:
                        group_conflict[g1].add(g2)
        return group_conflict

    def _dsatur_coloring(group_conflict: dict[str, set]) -> dict[str, int]:
        import heapq

        all_groups = sorted(group_conflict.keys())
        sorted_conflict = {g: sorted(group_conflict[g]) for g in all_groups}

        color_map = {}
        neighbor_colors = {g: set() for g in all_groups}
        uncolored = set(all_groups)

        degree = {g: len(sorted_conflict[g]) for g in all_groups}

        heap = []
        for g in all_groups:
            heapq.heappush(heap, (0, -degree[g], g))
        saturation = dict.fromkeys(all_groups, 0)

        while uncolored:
            while True:
                _, _, node = heapq.heappop(heap)
                if node in uncolored:
                    break
            used = neighbor_colors[node]
            color = 0
            while color in used:
                color += 1
            color_map[node] = color
            uncolored.remove(node)
            for neighbor in sorted_conflict[node]:
                if neighbor in uncolored:
                    if color not in neighbor_colors[neighbor]:
                        neighbor_colors[neighbor].add(color)
                        saturation[neighbor] += 1
                        heapq.heappush(
                            heap,
                            (
                                -saturation[neighbor],
                                -degree[neighbor],
                                neighbor,
                            ),
                        )
        return color_map

    def _assign_batches(tasks, group_color_map):
        batches = defaultdict(list)
        for t in tasks:
            g = t.comm_group.name
            batches[group_color_map[g]].append(t)
        return [sorted(batches[c], key=order_rules) for c in sorted(batches)]

    group_conflict = _build_group_conflict(group_members)
    group_color_map = _dsatur_coloring(group_conflict)
    results = _assign_batches(read_items, group_color_map)
    return results


def _load_state_dict(
    target_state_dict: dict,
    source_state_dict: dict,
    metadata_list,
    process_group=None,
    coordinator_rank=0,
    offload=False,
    worker_groups=None,
):
    if worker_groups is None:
        _load_state_dict_single_group(
            target_state_dict,
            source_state_dict,
            metadata_list,
            process_group,
            coordinator_rank,
            offload,
        )
    else:
        _load_state_dict_multi_group(
            target_state_dict,
            source_state_dict,
            metadata_list,
            process_group,
            coordinator_rank,
            offload,
            worker_groups,
        )

    del source_state_dict


def pre_process_and_build_comm_read_items(
    target_state_dict: dict,
    source_state_dict: dict,
    metadata_list,
    process_group=None,
    coordinator_rank=0,
    offload=False,
):
    use_dist = paddle.distributed.get_world_size() > 1
    cur_rank = paddle.distributed.get_rank() if use_dist else 0

    if offload:
        for file_name, state_dict in source_state_dict.items():
            source_state_dict[file_name] = {
                k: paddle.to_tensor(v, place=paddle.CPUPlace())
                if isinstance(v, np.ndarray)
                else v
                for k, v in state_dict.items()
            }

    local_load_files = list(source_state_dict.keys())
    logger.info("Start generating global ReadItems..")
    load_infos = get_load_infos(
        metadata_list, local_load_files, process_group, use_dist
    )

    read_items = get_read_items(
        metadata_list, target_state_dict, process_group, use_dist, load_infos
    )

    local_read_items, comm_read_items = split_read_items(read_items)

    logger.info(f"Generated {len(comm_read_items)} communication tasks.")
    logger.info(f"Generated {len(local_read_items)} local tasks.")

    processed_target_state_dict = {
        k: v.local_tensor if isinstance(v, ShardedWeight) else v
        for k, v in target_state_dict.items()
    }
    has_tuple_key = any(
        isinstance(k, tuple) for k in processed_target_state_dict
    )
    has_non_tuple_key = any(
        not isinstance(k, tuple) for k in processed_target_state_dict
    )
    assert not (has_tuple_key and has_non_tuple_key), (
        "target_state_dict contains a mix of tuple and non-tuple keys. Please ensure key types are consistent."
    )

    if not use_dist:
        assert len(comm_read_items) == 0, (
            "No communication task is needed when not using distributed training."
        )

    process_local_copy_tasks(
        local_read_items,
        cur_rank,
        source_state_dict,
        processed_target_state_dict,
    )

    logger.info(
        f"Rank {cur_rank} finished local copy and entered communication phase."
    )

    return processed_target_state_dict, comm_read_items


def _load_state_dict_single_group(
    target_state_dict: dict,
    source_state_dict: dict,
    metadata_list,
    process_group=None,
    coordinator_rank=0,
    offload=False,
):
    use_dist = paddle.distributed.get_world_size() > 1
    cur_rank = paddle.distributed.get_rank() if use_dist else 0

    processed_target_state_dict, comm_read_items = (
        pre_process_and_build_comm_read_items(
            target_state_dict,
            source_state_dict,
            metadata_list,
            process_group,
            coordinator_rank,
            offload,
        )
    )

    if len(comm_read_items) == 0:
        return
    paddle.distributed.barrier(process_group)

    tasks = schedule_comm_read_items_single_group(comm_read_items)

    logger.info(
        f"Communication tasks generated successfully, total {len(tasks)} tasks!"
    )

    for tensor_name, read_items in tasks.items():
        logger.debug(f"Beginning to send/recv tasks for tensor {tensor_name}.")

        source_tensors = {}
        destination_tensors = {}
        for item in read_items:
            logger.debug(f"Beginning to send/recv task {item}.")
            if item.src_rank == cur_rank:
                src_tensor = source_state_dict[item.file_name][item.tensor_name]
                if not src_tensor.place.is_gpu_place():
                    src_tensor = src_tensor.cuda()
                source_tensors[(tensor_name, item.file_name)] = src_tensor
            elif cur_rank in item.dst_rank:
                dst_tensor = get_target_tensor(
                    processed_target_state_dict, item
                )
                if not dst_tensor.place.is_gpu_place():
                    gpu_dst_tensor = dst_tensor.cuda()
                    gpu_dst_tensor.need_cross_device_copy = True
                    gpu_dst_tensor.target_tensor = dst_tensor
                    destination_tensors[
                        (tensor_name, cur_rank, item.dst_global_offset)
                    ] = gpu_dst_tensor
                else:
                    gpu_dst_tensor = dst_tensor
                    gpu_dst_tensor.target_tensor = dst_tensor
                    destination_tensors[
                        (tensor_name, cur_rank, item.dst_global_offset)
                    ] = dst_tensor

        for item in read_items:
            logger.debug(f"Beginning to send/recv task {item}.")
            if item.src_rank == cur_rank:
                src_tensor = source_tensors[(tensor_name, item.file_name)]
                src_chunk_tensor = slice_tensor(
                    src_tensor, item.src_local_offset, item.slice_shape
                )
                buffer_tensor = src_chunk_tensor.contiguous()
            elif cur_rank in item.dst_rank:
                dst_tensor = destination_tensors[
                    (tensor_name, cur_rank, item.dst_global_offset)
                ]
                dst_chunk_tensor = slice_tensor(
                    dst_tensor, item.dst_local_offset, item.slice_shape
                )
                buffer_tensor = paddle.zeros_like(dst_chunk_tensor)
                paddle.assign(dst_chunk_tensor, buffer_tensor)

            else:
                buffer_tensor = paddle.zeros(item.slice_shape, item.dtype)

            paddle.distributed.broadcast(
                buffer_tensor, src=item.src_rank, group=process_group
            )
            if cur_rank in item.dst_rank:
                paddle.assign(buffer_tensor, dst_chunk_tensor)
            del buffer_tensor

        for dst_tensor in destination_tensors.values():
            if getattr(dst_tensor, 'need_cross_device_copy', False):
                target_tensor = dst_tensor.target_tensor
                target_tensor.copy_(dst_tensor)
            else:
                target_tensor = dst_tensor.target_tensor
                paddle.assign(dst_tensor, target_tensor)
            del dst_tensor

        del source_tensors

        if use_dist:
            paddle.distributed.barrier(process_group)

    logger.info("All communication tasks completed.")


def _load_state_dict_multi_group(
    target_state_dict: dict,
    source_state_dict: dict,
    metadata_list,
    process_group=None,
    coordinator_rank=0,
    offload=False,
    worker_groups=None,
):
    assert paddle.distributed.get_world_size() > 1, (
        "Multi-group loading is only supported in distributed training."
    )
    cur_rank = paddle.distributed.get_rank()

    processed_target_state_dict, comm_read_items = (
        pre_process_and_build_comm_read_items(
            target_state_dict,
            source_state_dict,
            metadata_list,
            process_group,
            coordinator_rank,
            offload,
        )
    )

    results = schedule_comm_read_items_multi_group(
        comm_read_items, worker_groups
    )

    logger.info(
        f"Communication task scheduling completed, {len(results)}  batches in total."
    )
    for read_items in results:
        source_tensors = {}
        destination_tensors = {}
        for item in read_items:
            tensor_name = item.tensor_name
            if item.src_rank == cur_rank:
                src_tensor = source_state_dict[item.file_name][tensor_name]
                if not src_tensor.place.is_gpu_place():
                    src_tensor = src_tensor.cuda()
                source_tensors[(tensor_name, item.file_name)] = src_tensor
            elif cur_rank in item.dst_rank:
                dst_tensor = get_target_tensor(
                    processed_target_state_dict, item
                )
                if not dst_tensor.place.is_gpu_place():
                    gpu_dst_tensor = dst_tensor.cuda()
                    gpu_dst_tensor.need_cross_device_copy = True
                    gpu_dst_tensor.target_tensor = dst_tensor
                    destination_tensors[
                        (tensor_name, cur_rank, item.dst_global_offset)
                    ] = gpu_dst_tensor
                else:
                    gpu_dst_tensor = dst_tensor
                    gpu_dst_tensor.target_tensor = dst_tensor
                    destination_tensors[
                        (tensor_name, cur_rank, item.dst_global_offset)
                    ] = dst_tensor

        for item in read_items:
            logger.debug(f"Beginning to send/recv task {item}.")
            tensor_name = item.tensor_name
            if item.src_rank == cur_rank:
                src_tensor = source_tensors[(tensor_name, item.file_name)]
                src_chunk_tensor = slice_tensor(
                    src_tensor, item.src_local_offset, item.slice_shape
                )
                buffer_tensor = src_chunk_tensor.contiguous()
            elif cur_rank in item.dst_rank:
                dst_tensor = destination_tensors[
                    (tensor_name, cur_rank, item.dst_global_offset)
                ]
                dst_chunk_tensor = slice_tensor(
                    dst_tensor, item.dst_local_offset, item.slice_shape
                )
                buffer_tensor = paddle.zeros_like(dst_chunk_tensor)
                paddle.assign(dst_chunk_tensor, buffer_tensor)

            elif cur_rank in item.comm_group.ranks:
                buffer_tensor = paddle.zeros(item.slice_shape, item.dtype)
            else:
                buffer_tensor = None

            if cur_rank in item.comm_group.ranks:
                paddle.distributed.broadcast(
                    buffer_tensor, src=item.src_rank, group=item.comm_group
                )

            if cur_rank in item.dst_rank:
                paddle.assign(buffer_tensor, dst_chunk_tensor)
            del buffer_tensor

        for dst_tensor in destination_tensors.values():
            if getattr(dst_tensor, 'need_cross_device_copy', False):
                target_tensor = dst_tensor.target_tensor
                target_tensor.copy_(dst_tensor)
            else:
                target_tensor = dst_tensor.target_tensor
                paddle.assign(dst_tensor, target_tensor)
            del dst_tensor

        del source_tensors

    paddle.distributed.barrier(process_group)
    logger.info("All communication tasks completed.")


def compute_global_shape(local_tensor_indices):
    rank = len(local_tensor_indices[0].local_shape)
    global_shape = []
    for dim in range(rank):
        max_size = max(
            m.global_offset[dim] + m.local_shape[dim]
            for m in local_tensor_indices
        )
        global_shape.append(max_size)
    return global_shape


def load_merged_state_dict(
    path: str,
    prefix: str | None = None,
    unique_id: int | None = None,
    offload: bool = False,
    aoa_config: dict[str, list[str]] | None = None,
    safetensors: bool = False,
) -> dict[str, paddle.Tensor]:
    """
    Load the distributed checkpoint and merge it to unsharded state_dict.

    Args:
        path(str): The directory to load checkpoint files.
        prefix(str): The flat_mapping prefix of state_dict key. e.g., 'model', Default None.
        unique_id(int): The unique id of checkpoint, used to distinguish between different checkpoint versions. Default is None, in which case the id the max id of given path, and the newest version checkpoint is loaded.
        offload(bool): Whether to offload the checkpoint data from GPU to CPU, set to True if GPU memory is not enough.
        aoa_config(dict[str, list[str]]): AOA config to change parameters. Default is None.
        safetensors(bool): Whether to use safetensors format. Default is False.
    Returns:
        dict: Merged state_dict.

    Example:
        .. code-block:: python

            >>> # doctest: +SKIP('run in distributed mode.')
            >>> import paddle
            >>> import paddle.distributed as dist
            >>> ckpt_path = "./checkpoint"
            >>> w1 = paddle.arange(32).reshape([4, 8])
            >>> mesh = dist.ProcessMesh([0, 1])
            >>> sharded_w1 = dist.shard_tensor(w1, mesh, [dist.Shard(0)])
            >>> state_dict = {"w1": sharded_w1}
            >>> dist.save_state_dict(state_dict, ckpt_path) # save sharded checkpoint

            >>> # doctest: +SKIP('run in single-card mode.')
            >>> import paddle
            >>> import paddle.distributed as dist
            >>> ckpt_path = "./checkpoint"
            >>> unsharded_state_dict = dist.load_merged_state_dict(ckpt_path)  # load unsharded checkpoint
            >>> print(f"unsharded_state_dict:{unsharded_state_dict}")
            unsharded_state_dict:{'w1':
            [[0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 ],
             [8 , 9 , 10, 11, 12, 13, 14, 15],
             [16, 17, 18, 19, 20, 21, 22, 23],
             [24, 25, 26, 27, 28, 29, 30, 31]])}
            >>> # doctest: -SKIP
    """
    if unique_id is None:
        unique_id = get_max_id(path)
    else:
        assert unique_id >= 0, f'{unique_id} should be >= 0'

    metadata_files, local_data_files = get_checkpoint_files(
        path, unique_id=unique_id
    )

    metadata_list = []
    for file in metadata_files:
        metadata_list.append(paddle.load(os.path.join(path, file)))

    # create target state_dict by local_tensor_meta
    state_dict_to_save = {}
    for metadata in metadata_list:
        for (
            tensor_key,
            local_tensor_meta,
        ) in metadata.state_dict_metadata.items():
            if prefix is None or tensor_key.startswith(prefix):
                global_shape = compute_global_shape(local_tensor_meta)
                t = paddle.zeros(global_shape, dtype=local_tensor_meta[0].dtype)
                if offload:
                    t = t.cpu()
                state_dict_to_save[tensor_key] = t
            else:
                continue

    load_state_dict(
        state_dict_to_save,
        path,
        offload=offload,
        aoa_config=aoa_config,
        safetensors=safetensors,
    )

    # Update dictionary keys in place
    for key in list(
        state_dict_to_save.keys()
    ):  # Use list(data.keys()) to avoid runtime error
        if prefix and key.startswith(prefix):
            new_key = key[len(prefix) + 1 :]  # Remove the "str" prefix
            state_dict_to_save[new_key] = state_dict_to_save.pop(
                key
            )  # Add new key and remove the old one
    return state_dict_to_save


def divide_positions(m, n):
    '''
    Divide positions evenly among n processors with a base value and remainder handling.

    Parameters:
    m (int): Total number of tensor positions.
    n (int): Number of processors.

    Returns:
    list: A list of positions indicating where to split the tensors among processors.

    Raises:
    ValueError: If n is zero or if m is less than n.
    '''
    if n == 0:
        raise ValueError("n should be greater than zero")
    if m < n:
        raise ValueError(
            f"tensor number {m} should be greater than or equal to processor number {n}"
        )
    base_value = m // n
    remainder = m % n
    positions = [0]
    for i in range(1, n):
        if remainder > 0:
            positions.append(positions[-1] + base_value + 1)
            remainder -= 1
        else:
            positions.append(positions[-1] + base_value)
    positions.append(m)
    return positions


def endswith(key, prefix_list):
    for prefix in prefix_list:
        if key.endswith(prefix):
            return True
    return False


def merge_sharded_state_dict(
    load_path: str,
    save_path: str,
    prefix: str | None = None,
    safetensor_prefix: str = 'model',
    skip_postfix_list: list = [],
    process_group: Group | None = None,
    unique_id: int | None = None,
    offload: bool = False,
    aoa_config: dict[str, list[str]] | None = None,
    safetensors: bool = False,
) -> None:
    """
    Load the distributed checkpoint and merge it to unsharded state_dict then save as safetensors.

    Note:
        save files are:
            model-00001-of-00008.safetensors
            model-00002-of-00008.safetensors
            ...
            model-00008-of-00008.safetensors
            model.safetensors.index.json
        model is safetensor_prefix; 00008 is file_num which same ad dist total_size.

    Args:
        load_path(str): The directory to load checkpoint files.
        save_path(str): The directory to save merged_checkpoint files.
        prefix(str): The flat_mapping prefix of state_dict key. e.g., 'model', Default None.
        safetensor_prefix(str): The safetensors file prefix e.g., Default 'model'.
        skip_postfix_list(list(str)): The skip postfix list of state_dict key. e.g., ['moment1_0', 'beta1_pow_acc_0'], Default [].
        process_group(paddle.distributed.collective.Group): ProcessGroup to be used for cross-rank synchronization. Use the default process group which contains all cards.
        unique_id(int): The unique id of checkpoint, used to distinguish between different checkpoint versions. Default is None, in which case the id the max id of given path, and the newest version checkpoint is loaded.
        offload(bool): Whether to offload the checkpoint data from GPU to CPU, set to True if GPU memory is not enough.
        aoa_config(dict[str, list[str]]): AOA config to change parameters. Default is None.
        safetensors(bool): Whether to use safetensors format. Default is False.
    Returns:
        None.

    Example:
        .. code-block:: python

            >>> # doctest: +SKIP('run in distributed mode.')
            >>> import paddle
            >>> import paddle.distributed as dist
            >>> ckpt_path = "./checkpoint"
            >>> w1 = paddle.arange(32).reshape([4, 8])
            >>> mesh = dist.ProcessMesh([0, 1])
            >>> sharded_w1 = dist.shard_tensor(w1, mesh, [dist.Shard(0)])
            >>> state_dict = {"w1": sharded_w1}
            >>> dist.save_state_dict(state_dict, ckpt_path) # save sharded checkpoint

            >>> # doctest: +SKIP('run in single-card mode.')
            >>> import paddle
            >>> import paddle.distributed as dist
            >>> ckpt_path = "./checkpoint"
            >>> save_path = "./merged_checkpoint"
            >>> dist.flex_checkpoint.dcp.load_state_dict.merge_sharded_state_dict(ckpt_path, save_path)  # load unsharded and save to safetensors
            >>> # doctest: -SKIP
    """
    if unique_id is None:
        unique_id = get_max_id(load_path)
    else:
        assert unique_id >= 0, f'{unique_id} should be >= 0'

    use_dist = True if paddle.distributed.get_world_size() > 1 else False

    if use_dist and process_group is None and not is_initialized():
        # Init the default global process group
        paddle.distributed.init_parallel_env()

    if use_dist:
        # sync to avoid some ranks not write path yet
        paddle.distributed.barrier(process_group)

    metadata_files, local_data_files = get_checkpoint_files(
        load_path, unique_id=unique_id
    )

    metadata_list = []
    for file in metadata_files:
        metadata_list.append(paddle.load(os.path.join(load_path, file)))
    file_num = paddle.distributed.get_world_size()

    # create target state_dict by local_tensor_meta
    def slice_dict(d, start, end):
        """Slice the dictionary keys and return the corresponding sub-dictionary"""
        keys = list(d.keys())[start:end]
        return {k: d[k] for k in keys}

    all_state_dict = []
    local_state_dict_to_save = {}
    SaveSafetensor = SavePartialSafetensors(
        save_path, process_group, safetensor_prefix
    )

    for metadata in metadata_list:
        state_dict_metadata = metadata.state_dict_metadata
        origin_size = len(state_dict_metadata)
        rm_key_list = []
        for key in state_dict_metadata.keys():
            if endswith(key, skip_postfix_list):
                rm_key_list.append(key)
        for key in rm_key_list:
            state_dict_metadata.pop(key)
        cur_size = len(state_dict_metadata)
        logger.info(
            f"state_dict_metadata origin_size: {origin_size}, cur_size: {cur_size} skip {origin_size - cur_size}"
        )

        positions = divide_positions(len(state_dict_metadata), file_num)
        rank = paddle.distributed.get_rank()

        partial_state_dict_metadata = slice_dict(
            state_dict_metadata, positions[rank], positions[rank + 1]
        )
        for (
            tensor_key,
            local_tensor_meta,
        ) in partial_state_dict_metadata.items():
            if prefix is None or tensor_key.startswith(prefix):
                global_shape = compute_global_shape(local_tensor_meta)
                t = paddle.zeros(global_shape, dtype=local_tensor_meta[0].dtype)
                if offload:
                    t = t.cpu()
                local_state_dict_to_save[tensor_key] = (
                    make_replicated_sharded_weight(
                        key=tensor_key,
                        tensor=t,
                    )
                )
            else:
                continue

        logger.info(
            f"rank :{rank} , local_state_dict_to_save.size :{len(local_state_dict_to_save)}"
        )

        if paddle.distributed.get_rank() == 0:
            for ii in range(len(positions) - 1):
                shard_file = f"{safetensor_prefix}-{ii + 1:05d}-of-{file_num:05d}.safetensors"
                for key in list(state_dict_metadata.keys())[
                    positions[ii] : positions[ii + 1]
                ]:
                    SaveSafetensor.index["weight_map"][key] = shard_file
                    local_tensor_meta = state_dict_metadata[key]
                    shape_ = compute_global_shape(local_tensor_meta)
                    dtype_ = local_tensor_meta[0].dtype
                    SaveSafetensor.index["metadata"]["total_size"] += int(
                        np.prod(shape_)
                        * SaveSafetensor.paddle_dtype_map[str(dtype_)]
                    )

            weight_size = len(SaveSafetensor.index["weight_map"])
            logger.info(
                f"SaveSafetensor.index[weight_map] size = {weight_size}."
            )

    if paddle.distributed.get_rank() == 0:
        SaveSafetensor.save_index_json()

    if use_dist:
        paddle.distributed.barrier(process_group)
        paddle.distributed.all_gather_object(
            all_state_dict, len(local_state_dict_to_save), process_group
        )
    else:
        all_state_dict = [len(local_state_dict_to_save)]

    if paddle.distributed.get_rank() == 0:
        total_keys = sum(size for size in all_state_dict)
        total_meta_items = sum(
            len(metadata.state_dict_metadata.items())
            for metadata in metadata_list
        )

        assert total_meta_items == total_keys, (
            f'split state dict filed :{total_meta_items} should seem as {total_keys}'
        )
        assert file_num == len(all_state_dict), (
            f'file_num:{file_num} should seem as len(all_state_dict):{len(all_state_dict)}'
        )

    load_state_dict(
        local_state_dict_to_save,
        load_path,
        process_group,
        offload=offload,
        aoa_config=aoa_config,
        safetensors=safetensors,
    )

    # Update dictionary keys in place
    for key in list(
        local_state_dict_to_save.keys()
    ):  # Use list(data.keys()) to avoid runtime error
        if prefix and key.startswith(prefix):
            new_key = key[len(prefix) + 1 :]  # Remove the "str" prefix
            local_state_dict_to_save[new_key] = local_state_dict_to_save.pop(
                key
            )  # Add new key and remove the old one

    for key, value in local_state_dict_to_save.items():
        if isinstance(value, ShardedWeight):
            value_to_save = value.local_tensor
            local_state_dict_to_save[key] = value_to_save
    logger.info(
        f"rank :{rank} , SaveSafetensor.local_state_dict_to_save.size :{len(local_state_dict_to_save)}"
    )
    SaveSafetensor.save_single_safetenors(
        local_state_dict_to_save, paddle.distributed.get_rank()
    )


class SavePartialSafetensors:
    def __init__(self, output_path, process_group, prefix="model"):
        self.output_path = output_path
        self.process_group = process_group
        self.prefix = prefix
        self.paddle_dtype_map = {
            "float64": 8,
            "float32": 4,
            "float16": 2,
            "uint16": 2,
            "bfloat16": 2,
            "uint8": 1,
            "float8_e4m3fn": 1,
            "float8_e5m2": 1,
        }
        self.index = {"metadata": {"total_size": 0}, "weight_map": {}}
        self.safe_index_name = prefix + ".safetensors.index.json"
        self.total_files_size = paddle.distributed.get_world_size()
        self.save_index_file = os.path.join(
            self.output_path, self.safe_index_name
        )
        os.makedirs(os.path.dirname(self.save_index_file), exist_ok=True)
        self.index_save_called = False

    def save_single_safetenors(self, state_dict, rank):
        save_file_name = os.path.join(
            self.output_path,
            f"{self.prefix}-{rank + 1:05d}-of-{self.total_files_size:05d}.safetensors",
        )
        logger.info(f"save_file_name = {save_file_name}")
        paddle.framework.io._safe_save(
            state_dict,
            save_file_name,
        )

    def save_index_json(self):
        if self.index_save_called:
            raise RuntimeError(
                "save_index_json method can only be called once!"
            )

        self.index_save_called = True
        with open(self.save_index_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.index, indent=2) + "\n")
        logger.info(f"Model index file saved in {self.save_index_file}.")
