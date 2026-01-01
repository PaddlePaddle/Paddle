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

import gc
import json
import math
import os
from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
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
from .reshard_comm import CommunicatorFactory
from .resharder import StateDictResharder
from .sharded_weight import (
    ShardedWeight,
    ShardedWeightDesc,
    make_replicated_sharded_weight,
)
from .utils import (
    assign_sharded_slice,
    build_global_state_shard_info,
    build_shard_desc,
    check_resumable_locally,
    check_unique_id,
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

PATH_TO_CHECKPOINT_FILES: dict[str, tuple[list, list]] = {}
_UNINIT_TENSOR_MODES = ["send_recv", "grouped_send_recv"]

_metadata_manager = MetadataManager()


def get_checkpoint_files(
    path, use_cache=True, unique_id=None, process_group=None, safetensors=False
):
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

    if safetensors and len(metadata_files) == 0:
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
            create_hf_ckpt_metadata(path, process_group=process_group)
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
        or file.endswith(f"{unique_id}.safetensors")
    ]
    # Check that local_data_files does not contain both .distcp and .safetensors files at the same time
    if any(file.endswith('.distcp') for file in local_data_files) and any(
        file.endswith('.safetensors') for file in local_data_files
    ):
        raise ValueError(
            f"Checkpoint directory cannot contain both .distcp and .safetensors files simultaneously in {path}."
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
            if (
                local_tensor_index.replica_id is not None
                and local_tensor_index.replica_id != 0
            ):
                continue
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

    unexpected_keys = set(tensor_key_list) - set(state_dict_param_names)
    if len(unexpected_keys) > 0:
        logger.warning(
            f"Unexpected keys:{unexpected_keys}, these keys exist in checkpoint but not in state_dict."
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


class CheckpointLoadBalancer:
    """
    Responsible for balancing file reading tasks in distributed training.

    Objectives:
    1. Ensure each file is read exactly once globally.
    2. Prioritize reading from the rank that has the file locally (minimize network overhead).
    3. Balance the load (number of files read) across all ranks.
    """

    def __init__(
        self,
        rank_to_required_files,
        rank_to_available_files,
    ):
        """
        Args:
            rank_to_required_files: Mapping of rank -> list of files it logically needs to load.
            rank_to_available_files: Mapping of rank -> list of files physically present on its local storage.
        """
        self.rank_to_required = rank_to_required_files
        self.rank_to_available = rank_to_available_files

        # Final result: {rank: [files_to_read]}
        self.assignments = defaultdict(list)
        # Real-time load counter for decision making: {rank: file_count}
        self.load_counts = defaultdict(int)
        # Track assigned files to prevent duplicate reading
        self.assigned_files = set()

    def _assign(self, rank: int, file_name: str):
        """Execute the assignment and update internal state."""
        self.assignments[rank].append(file_name)
        self.load_counts[rank] += 1
        self.assigned_files.add(file_name)

    def _get_rank_with_min_load(self, candidates) -> int:
        """
        Select the rank with the minimum current load among candidates.
        If loads are equal, select the smaller rank ID for deterministic behavior.
        """
        return min(candidates, key=lambda r: (self.load_counts[r], r))

    def _balance_files(self, file_to_candidates):
        """
        Core load balancing algorithm.

        Strategy:
        1. Sort files by candidate count (Ascending). Process files with fewer options first (stronger constraints).
        2. For files with multiple options, greedily assign to the rank with the lowest current load.
        """
        # Sort items by number of candidates: process most constrained files first.
        sorted_items = sorted(
            file_to_candidates.items(),
            key=lambda x: (
                len(x[1]),
                x[0],
            ),  # When candidates are the same, use smaller file name
        )

        for file_name, candidates in sorted_items:
            if file_name in self.assigned_files:
                continue

            if not candidates:
                continue

            # Greedy selection: assign to the candidate with the least work so far
            chosen_rank = self._get_rank_with_min_load(candidates)
            self._assign(chosen_rank, file_name)

    def plan(self):
        """Execute the planning process and return assignments for all ranks."""

        # --- Phase 1: Handle "Local" Files ---
        # Identify files that ranks need AND possess locally.
        local_file_candidates = defaultdict(list)
        cross_node_files = set()

        for rank, files in self.rank_to_required.items():
            local_files_set = set(self.rank_to_available.get(rank, []))
            for file_name in files:
                if file_name in local_files_set:
                    local_file_candidates[file_name].append(rank)
                else:
                    cross_node_files.add(file_name)

        # Assign local files (prioritizing load balance if multiple ranks have the file locally)
        self._balance_files(local_file_candidates)

        # --- Phase 2: Handle "Cross-Node" Files ---
        # These files are required but not found locally on the requester.
        # We must assign them to *any* rank that physically has the file.
        remaining_file_candidates = defaultdict(list)

        # Only process files that haven't been assigned in Phase 1
        files_to_process = [
            f for f in cross_node_files if f not in self.assigned_files
        ]

        # Build global index: file -> [all ranks that physically have it]
        global_availability = defaultdict(list)
        for rank, files in self.rank_to_available.items():
            for f in files:
                global_availability[f].append(rank)

        for file_name in files_to_process:
            candidates = global_availability.get(file_name, [])
            if candidates:
                remaining_file_candidates[file_name] = candidates
            else:
                logger.warning(
                    f"File {file_name} is required but not found on any rank."
                )

        # Assign remaining files using the same greedy strategy
        self._balance_files(remaining_file_candidates)

        return self.assignments


def get_rank_to_read_files(
    rank_to_required,
    rank_to_available_files,
):
    """
    Public API to determine which files the current rank should read.

    Args:
        rank_to_required: Logical mapping of rank to files it needs.
        rank_to_available_files: Physical mapping of rank to files on disk.

    Returns:
        List of file names the current rank is responsible for loading.
    """
    balancer = CheckpointLoadBalancer(rank_to_required, rank_to_available_files)
    all_assignments = balancer.plan()

    current_rank = paddle.distributed.get_rank()
    my_files = all_assignments.get(current_rank, [])

    if not my_files:
        logger.warning(
            f"Rank:{current_rank} does not need to load any checkpoint files."
        )
    else:
        logger.debug(f"Rank:{current_rank} assigned files: {my_files}")

    return my_files


def _split_flat_shards(state_dict):
    flat_shards, nonflat_shards = {}, {}
    for key, shard in state_dict.items():
        if getattr(shard, "is_flattened", False):
            flat_shards[key] = shard
        else:
            nonflat_shards[key] = shard
    return flat_shards, nonflat_shards


def _unflatten_shards(flat_shards, comm_method):
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
    comm_method,
):
    global _metadata_manager

    use_dist = paddle.distributed.get_world_size() > 1
    if _metadata_manager.is_metadata_list_empty():
        metadata_files, _ = get_checkpoint_files(
            path,
            unique_id=unique_id,
            process_group=process_group,
            safetensors=safetensors,
        )
        assert len(metadata_files) == 1, "Only support one metadata file now."
        metadata = paddle.load(os.path.join(path, metadata_files[0]))
        _metadata_manager.set_metadata_list([metadata])
    metadata = _metadata_manager.get_metadata_list()[0]
    state_dict_metadata = metadata.state_dict_metadata
    using_not_init_tensor = (
        True if comm_method in _UNINIT_TENSOR_MODES else False
    )

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

    for param_name, tgt_shard in sorted(load_dict.items()):
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
            if len(shard_mappings) == 1 and mapping.postprocess_list is None:
                if src_desc.global_shape != dst_desc.global_shape:
                    logger.warning(
                        f"Shape mismatch for parameter '{param_name}': "
                        f"source global_shape={src_desc.global_shape}, "
                        f"destination global_shape={dst_desc.global_shape}, "
                        "Please check if this is caused by an AOA configuration."
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
                if using_not_init_tensor:
                    local_tensor._clear_to_zero_allocation()
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
        comm_method=comm_method,
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


def local_load_state_dict(
    state_dict: dict[str, Tensor] | dict[str, ShardedWeight],
    path: str,
    offload: bool = False,
    use_dist: bool = True,
):
    cur_rank = paddle.distributed.get_rank() if use_dist else 0
    expect_checkpoint_file = f"{cur_rank}_0.distcp"
    ckpt_file = os.path.join(path, expect_checkpoint_file)
    source_state_dict = {}
    if offload:
        state_dict_numpy = paddle.load(ckpt_file, return_numpy=True)
        source_state_dict = {
            key: paddle.to_tensor(value, place=paddle.CPUPlace())
            for key, value in state_dict_numpy.items()
        }
    else:
        source_state_dict = paddle.load(ckpt_file)
    for key, value in state_dict.items():
        if isinstance(value, ShardedWeight):
            local_tensor = value.local_tensor
        else:
            if not value._is_initialized():
                continue
            if value.is_dist():
                local_tensor = value._local_value()
            else:
                local_tensor = value

        assert key in source_state_dict, f"{key} is not in source_state_dict."
        source_tensor = source_state_dict[key]
        source_tensor = source_tensor.to(local_tensor.place)
        paddle.assign(source_tensor, local_tensor)


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
    comm_method: str = "broadcast",
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
        worker_groups (list[paddle.distributed.collective.Group]): Communication groups used for tensor communications; if multiple are provided, an appropriate group is chosen; if None, the process_group group is used.
        comm_method (str): Communication method for resharding. Choices are "send_recv", "broadcast", "multi_group_broadcast", and "grouped_send_recv". Default is "broadcast".
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
    global _metadata_manager
    use_dist = paddle.distributed.get_world_size() > 1

    valid_methods = [
        "send_recv",
        "broadcast",
        "multi_group_broadcast",
        "grouped_send_recv",
    ]
    assert comm_method in valid_methods, (
        f"Invalid communication method '{comm_method}'. "
        f"Please choose from {valid_methods}."
    )

    if use_dist and process_group is None and not is_initialized():
        # Init the default global process group
        paddle.distributed.init_parallel_env()

    if use_dist:
        paddle.distributed.barrier(process_group)

    if not safetensors and aoa_config is None:
        metadata_files, _ = get_checkpoint_files(path, unique_id=unique_id)
        assert len(metadata_files) == 1, "Only support one metadata file now."
        metadata = paddle.load(os.path.join(path, metadata_files[0]))
        _metadata_manager.set_metadata_list([metadata])
        resumable_locally = check_resumable_locally(
            path, state_dict, _metadata_manager, use_dist, process_group
        )
        if resumable_locally:
            logger.info(
                f"Checkpoint '{path}' resumable locally, skipping reshard."
            )
            local_load_state_dict(
                state_dict=state_dict,
                path=path,
                offload=offload,
                use_dist=use_dist,
            )
            logger.info("Checkpoint successfully loaded locally!")
            return

    if not is_sharded_state_dict(state_dict, use_dist, process_group):
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
            comm_method=comm_method,
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
        load_dict, padding_info = _unflatten_shards(flat_shards, comm_method)
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
            comm_method,
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
            comm_method=comm_method,
        )
    if use_dist:
        _finish_unflatten(flat_shards, padding_info)

    _metadata_manager.clear()
    gc.collect()


def restore_unflattened_state_dict(
    source_state_dict: dict[str, dict[str, Tensor]],
    process_group,
    weoker_groups,
    comm_method,
    offload,
):
    global _metadata_manager
    use_dist = paddle.distributed.get_world_size() > 1
    using_not_init_tensor = (
        True if comm_method in _UNINIT_TENSOR_MODES else False
    ) and use_dist

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
        local_shape = (flat_end - flat_start,)
        source_state_dict_for_reshard[file_name][tensor_name_expand] = (
            local_tensor
        )
        source_local_tensor_meta[tensor_name_expand].append(
            LocalTensorMetadata(
                local_shape=local_shape,
                global_shape=(math.prod(meta.local_shape),),
                dtype=meta.dtype,
                global_offset=(flat_start,),
                is_flattened=False,
            )
        )
        source_storage_meta[
            LocalTensorIndex(
                tensor_key=tensor_name_expand,
                global_offset=(flat_start,),
                local_shape=local_shape,
            )
        ] = file_name

        tmp_target_tensor = paddle.zeros((numel,), dtype=local_tensor.dtype)
        if using_not_init_tensor:
            tmp_target_tensor._clear_to_zero_allocation()

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
        worker_groups=weoker_groups,
        comm_method=comm_method,
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
        tensor = tensor.cpu() if offload else tensor
        final_unflattened_state_dict[file_name][tensor_name] = tensor
        final_local_tensor_meta[tensor_name].append(meta)
        final_storage_meta[
            LocalTensorIndex(
                tensor_key=tensor_name,
                global_offset=meta.global_offset,
                is_flattened=False,
                flattened_range=None,
                local_shape=meta.local_shape,
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
    comm_method: str = 'broadcast',
) -> None:
    with paddle.base.dygraph.guard():
        global _metadata_manager
        assert isinstance(state_dict, dict), (
            f"The state_dict should be a dictionary.But now the type is {type(state_dict)}."
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
            path,
            unique_id=unique_id,
            process_group=process_group,
            safetensors=safetensors,
        )

        if _metadata_manager.is_metadata_list_empty():
            metadata_list = []
            for file in metadata_files:
                metadata_list.append(paddle.load(os.path.join(path, file)))
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

        metadata = _metadata_manager.get_metadata_list()[0]
        storage_metadata = metadata.storage_metadata

        replica_indexes = [
            local_tensor_index
            for local_tensor_index in storage_metadata
            if local_tensor_index.replica_id is not None
            and local_tensor_index.replica_id != 0
        ]

        for local_tensor_index in replica_indexes:
            file_name = storage_metadata[local_tensor_index]
            if file_name in source_state_dict:
                tensor_key = local_tensor_index.tensor_key
                state_dict = source_state_dict[file_name]
                if tensor_key in state_dict:
                    state_dict.pop(tensor_key)

        metadata_copy = deepcopy(metadata)
        storage_metadata_copy = metadata_copy.storage_metadata
        for local_tensor_index in replica_indexes:
            storage_metadata_copy.pop(local_tensor_index)

        new_storage_metadata = {}
        for local_tensor_index, value in storage_metadata_copy.items():
            if local_tensor_index.replica_id == 0:
                local_tensor_index_new = replace(
                    local_tensor_index, replica_id=None
                )
                new_storage_metadata[local_tensor_index_new] = value
            else:
                new_storage_metadata[local_tensor_index] = value

        metadata_copy.storage_metadata = new_storage_metadata

        _metadata_manager.set_metadata_list([metadata_copy])

        if use_dist:
            paddle.distributed.barrier(process_group)

        if _metadata_manager.has_flattened_tensors:
            logger.info("Restoring unflattened state dict.")
            source_state_dict = restore_unflattened_state_dict(
                source_state_dict,
                process_group,
                worker_groups,
                comm_method,
                offload,
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
            comm_method,
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


def _load_state_dict(
    target_state_dict: dict,
    source_state_dict: dict,
    metadata_list,
    process_group=None,
    coordinator_rank=0,
    offload=False,
    worker_groups: list[Group] | None = None,
    comm_method: str = 'broadcast',
):
    use_dist = True if paddle.distributed.get_world_size() > 1 else False
    communicator = CommunicatorFactory.create(
        comm_method, worker_groups=worker_groups
    )
    resharder = StateDictResharder(
        target_state_dict=target_state_dict,
        source_state_dict=source_state_dict,
        metadata_list=metadata_list,
        communicator=communicator,
        process_group=process_group,
        offload=offload,
        use_dist=use_dist,
    )
    resharder.reshard()


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
        path, unique_id=unique_id, safetensors=safetensors
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
        load_path, unique_id=unique_id, safetensors=safetensors
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
