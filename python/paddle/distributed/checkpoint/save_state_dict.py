# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os
from typing import List

import paddle
from paddle.distributed.communication.group import is_initialized
from paddle.distributed.fleet.utils.log_util import logger

from .metadata import LocalTensorIndex, LocalTensorMetadata, Metadata
from .utils import compute_local_shape_and_global_offset, flatten_state_dict


def check_state_dict(state_dict, process_group):
    local_keys = list(state_dict.keys())
    gloabl_keys = []
    paddle.distributed.all_gather_object(gloabl_keys, local_keys, process_group)
    for keys in gloabl_keys[1:]:
        assert (
            keys == gloabl_keys[0]
        ), f"keys:{keys} != first_keys: {gloabl_keys[0]}"


def check_file_name(file_name, process_group):
    all_unique_id = []
    unique_id = int(file_name.split(".")[0].split("_")[1])
    paddle.distributed.all_gather_object(
        all_unique_id, unique_id, process_group
    )
    for id in all_unique_id[1:]:
        assert (
            id == all_unique_id[0]
        ), f"id:{id} !=  all_unique_id[0]:{file_name}"


def merge_state_dict(global_state_dict):
    assert isinstance(
        global_state_dict, List
    ), "The global_state_dict should be a list."
    out = {}
    for state_dict in global_state_dict:
        for key, val in state_dict.items():
            if key in out:
                if val in out[key]:
                    continue
                out[key].append(val)
            else:
                out[key] = [val]
    return out


def dedup_state_dict(global_state_dict):
    out = {}
    for state_dict in global_state_dict:
        for key, val in state_dict.items():
            if key in out:
                continue
            out[key] = val
    return out


def save_state_dict(
    state_dict, path, process_group=None, coordinator_rank=0, use_dist=True
) -> None:
    """
    Save the state_dict of model to path.

    Args:
        state_dict: The state_dict to save.
        path: The directory to save state_dict.
        process_group: ProcessGroup to be used for cross-rank synchronization. Use the default process group which contains all cards.
        coordinator_rank: The rank used to coordinate the checkpoint. Rank0 is used by default.
        use_dist: Whether to save the state_dict in distributed mode. Set True by default.

    Examples:
        .. code-block:: python

            import paddle
            ...

    """
    if not use_dist and (
        paddle.distributed.get_world_size() > 1 or coordinator_rank != 0
    ):
        raise ValueError(
            f"use_dist is False, please set coordinator_rank to 0 and paddle.distributed.get_world_size() to 1, world_size:{paddle.distributed.get_world_size()}, coordinator_rank:{coordinator_rank}"
        )
    assert isinstance(
        state_dict, dict
    ), "The state_dict should be a dictionary."
    state_dict = flatten_state_dict(state_dict)
    if len(state_dict) > 0:
        for val in state_dict.values():
            assert isinstance(
                val, paddle.Tensor
            ), "Only support dygraph Tensor now, support static DistributedTensor later"

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    if use_dist and process_group is None:
        # Init the default global process group
        not is_initialized() and paddle.distributed.init_parallel_env()

    unique_id = 0
    file_name = ""
    while True:
        file_name = f"{paddle.distributed.get_rank()}_{unique_id}.distcp"
        if not os.path.exists(os.path.join(path, file_name)):
            break
        unique_id += 1
    logger.info(f"file_name:{file_name}")
    if use_dist:
        check_file_name(file_name, process_group)
        # the parameter_name and order in state_dict should be the same
        check_state_dict(state_dict, process_group)
    metadata = Metadata()
    local_state_dict = {}
    local_tensor_metadata = {}
    local_storage_metadata = {}
    for key, val in state_dict.items():
        if isinstance(val, paddle.Tensor):
            # Case1: not initialized means this tensor is placed in another mesh which do not contain this rank
            if not val._is_initialized():
                continue
            if val.is_dist():
                (
                    local_shape,
                    global_offset,
                ) = compute_local_shape_and_global_offset(
                    val.shape,
                    val.dist_attr.process_mesh,
                    val.dist_attr.dims_mapping,
                )
                if not local_shape or not global_offset:
                    continue
                local_tensor = val._local_value()
            else:
                global_offset = [0] * len(val.shape)
                local_shape = val.shape
                local_tensor = val
            local_state_dict[key] = local_tensor
            local_tensor_metadata[key] = LocalTensorMetadata(
                global_offset, local_shape
            )
            local_storage_metadata[
                LocalTensorIndex(key, tuple(global_offset))
            ] = file_name
    global_tensor_metadata = []
    global_storage_metadata = []
    if use_dist:
        paddle.distributed.all_gather_object(
            global_tensor_metadata, local_tensor_metadata, process_group
        )
        paddle.distributed.all_gather_object(
            global_storage_metadata, local_storage_metadata, process_group
        )
    else:
        global_tensor_metadata.append(local_tensor_metadata)
        global_storage_metadata.append(local_storage_metadata)

    metadata.state_dict_metadata = merge_state_dict(global_tensor_metadata)
    metadata.storage_metadata = dedup_state_dict(global_storage_metadata)
    if coordinator_rank == paddle.distributed.get_rank():
        logger.info(f"global_tensor_metadata:{global_tensor_metadata}")
        logger.info(f"global_storage_metadata:{global_storage_metadata}")
        logger.info(f"metadata:{metadata}")
        paddle.save(metadata, os.path.join(path, f"{unique_id}.metadata"))
    logger.info(f"local_state_dict:{local_state_dict}")
    paddle.save(local_state_dict, os.path.join(path, file_name))
