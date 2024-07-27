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
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import paddle
from paddle.distributed.communication.group import is_initialized
from paddle.distributed.fleet.utils.log_util import logger

from .metadata import LocalTensorIndex, LocalTensorMetadata
from .utils import (
    compute_local_shape_and_global_offset,
    flatten_state_dict,
)


@dataclass(frozen=True)
class ReadItem:
    local_tensor_index: LocalTensorIndex
    rank: int
    dtype: str
    cur_offset: Tuple[int]
    storage_offset: Tuple[int]
    lengths: Tuple[int]


PATH_TO_CHECKPOINT_FILES: Dict[str, Tuple[list, list]] = {}


def get_checkpoint_files(path, use_cache=True):
    global PATH_TO_CHECKPOINT_FILES
    if use_cache and path in PATH_TO_CHECKPOINT_FILES:
        return PATH_TO_CHECKPOINT_FILES[path]
    accessible_files = os.listdir(path)
    metadata_files = [
        file for file in accessible_files if file.endswith(".metadata")
    ]
    assert (
        len(metadata_files) > 0
    ), f"No metadata file found in the checkpoint directory:{path}."
    local_data_files = [
        file for file in accessible_files if file.endswith(".distcp")
    ]
    assert (
        len(local_data_files) > 0
    ), f"No data file found in the checkpoint directory:{path}."
    if use_cache:
        PATH_TO_CHECKPOINT_FILES[path] = (metadata_files, local_data_files)
    return (metadata_files, local_data_files)


def get_rank_to_files(path, state_dict, process_group, use_dist):
    """
    Get the mapping of rank to its accessible files.
    """
    metadata_files, local_data_files = get_checkpoint_files(path)
    # The necessary files to be read
    tensor_key_list = []
    necessary_files = []
    for metadata_file in metadata_files:
        metadata = paddle.load(os.path.join(path, metadata_file))
        for local_tensor_index, file_name in metadata.storage_metadata.items():
            assert (
                local_tensor_index not in tensor_key_list
            ), f"Duplicate tensor_key:{local_tensor_index} found. Check whether the metadata_file:{metadata_file} contains the same tensor metadata."
            tensor_key_list.append(local_tensor_index.tensor_key)
            if local_tensor_index.tensor_key in state_dict:
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
            f"No necessary data files found in the checkpoint directory:{path}. Please check the metadata_files:{metadata_files}"
        )
        missing_keys = set(state_dict.keys())
        return {}, missing_keys

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
    ), f"The checkpoint files are not complete. Please check the checkpoint directory:{path}.global_data_files_set:{global_data_files_set}, necessary_data_files_set:{global_necessary_files_set}"
    missing_keys = set(state_dict.keys()) - set(tensor_key_list)
    if len(missing_keys) > 0:
        logger.warning(
            f"Missing keys:{missing_keys}, check whether the checkpoint is complete."
        )

    rank_to_files = {}
    for rank, local_files in enumerate(global_data_files):
        if len(local_files) > 0:
            local_files = [
                f for f in local_files if f in all_necessary_files[rank]
            ]
            rank_to_files[rank] = local_files
    logger.debug(f"mapping rank_to_files:{rank_to_files}")
    return rank_to_files, missing_keys


def get_local_load_files(rank_to_files):
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
    rank_to_not_read_files = copy.copy(rank_to_files)
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
    cur_rank = paddle.distributed.get_rank()
    if cur_rank in rank_to_read_files:
        return rank_to_read_files[cur_rank]
    else:
        logger.warning(f"rank:{cur_rank} does not need to load checkpoint")
        return []


def get_load_infos(path, local_load_files, process_group, use_dist):
    load_info = {}
    metadata_files, _ = get_checkpoint_files(path)
    for metadata_file in metadata_files:
        metadata = paddle.load(os.path.join(path, metadata_file))
        for local_tensor_index, file_name in metadata.storage_metadata.items():
            if file_name in local_load_files:
                load_info[local_tensor_index] = (
                    paddle.distributed.get_rank(),
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
        assert (
            lengths[-1] >= 0
        ), f"Invalid length:{lengths[-1]}, end_offset:{end_offset}, begin_offset:{begin_offset}"
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


def get_read_items(path, state_dict, process_group, use_dist):
    storage_state_dict_metadata = {}
    metadata_files, _ = get_checkpoint_files(path)
    for metadata_file in metadata_files:
        metadata = paddle.load(os.path.join(path, metadata_file))
        for (
            tensor_key,
            local_tensor_metadata,
        ) in metadata.state_dict_metadata.items():
            if tensor_key not in storage_state_dict_metadata:
                storage_state_dict_metadata[tensor_key] = []
            storage_state_dict_metadata[tensor_key] += local_tensor_metadata
    read_items = []
    logger.debug(f"storage_state_dict_metadata:{storage_state_dict_metadata}")
    for tensor_key, val in state_dict.items():
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
                if local_shape is None or global_offset is None:
                    continue
            else:
                local_shape = tuple(val.shape)
                global_offset = (
                    tuple([0] * len(val.shape)) if len(val.shape) > 0 else ()
                )
            cur_chunk_metadata = LocalTensorMetadata(
                global_offset, local_shape, str(val.dtype).split(".")[1]
            )
            assert (
                tensor_key in storage_state_dict_metadata
            ), f"tensor_key:{tensor_key} not found in storage_state_dict_metadata:{storage_state_dict_metadata}."
            for storage_local_tensor_metadata in storage_state_dict_metadata[
                tensor_key
            ]:
                if not_overlap(
                    cur_chunk_metadata, storage_local_tensor_metadata
                ):
                    continue
                cur_offsets, storage_offsets, lengths = compute_overlap(
                    cur_chunk_metadata, storage_local_tensor_metadata
                )
                storage_local_tensor_index = LocalTensorIndex(
                    tensor_key,
                    tuple(storage_local_tensor_metadata.global_offset),
                )
                read_items.append(
                    ReadItem(
                        storage_local_tensor_index,
                        paddle.distributed.get_rank(),
                        storage_local_tensor_metadata.dtype,
                        tuple(cur_offsets),
                        tuple(storage_offsets),
                        tuple(lengths),
                    )
                )
        else:
            raise ValueError(
                f"Only support paddle.Tensor., val type:{type(val)}"
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


def load_state_dict(
    state_dict,
    path,
    process_group=None,
    coordinator_rank=0,
) -> None:
    """
    Load the state_dict inplace from a checkpoint path.

    Args:
        state_dict(Dict[str, paddle.Tensor]): The state_dict to load. It will be modified inplace after loading.
        path(str): The directory to load checkpoint files.
        process_group(paddle.distributed.collective.Group): ProcessGroup to be used for cross-rank synchronization. Use the default process group which contains all cards.
        coordinator_rank(int): The rank used to coordinate the checkpoint. Rank0 is used by default.

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
    with paddle.base.dygraph.guard():
        assert isinstance(
            state_dict, dict
        ), "The state_dict should be a dictionary."
        flat_state_dict, mapping = flatten_state_dict(state_dict)
        if len(flat_state_dict) > 0:
            for val in flat_state_dict.values():
                assert isinstance(
                    val, paddle.Tensor
                ), f"The value of state_dict should be a paddle.Tensor, but got: {val}."

        use_dist = True if paddle.distributed.get_world_size() > 1 else False

        if use_dist and process_group is None and not is_initialized():
            # Init the default global process group
            paddle.distributed.init_parallel_env()

        if use_dist:
            # sync to avoid some ranks not write path yet
            paddle.distributed.barrier(process_group)

        rank_to_files, missing_keys = get_rank_to_files(
            path, flat_state_dict, process_group, use_dist
        )

        if len(missing_keys) > 0:
            logger.warning(
                f"The following keys:{missing_keys} are not found in checkpoint path: {path}."
            )
        if len(rank_to_files) <= 0:
            return
        local_load_files = get_local_load_files(rank_to_files)
        # load_infos: {LocalTensorIndex: (rank, file_name)}, which local tensor located in which file, and the file is load in which rank.
        load_infos = get_load_infos(
            path, local_load_files, process_group, use_dist
        )
        # read_items: [ReadItem(local_tensor_index, rank, cur_offsets, storage_offsets, lengths)],
        # slice the storage local tensor in (storage_offsets, lengths) to assign the current tensor in (cur_offsets, lengths) in rank.
        read_items = get_read_items(
            path, flat_state_dict, process_group, use_dist
        )
        storage_file_to_state_dict = {}
        logger.debug(
            f"before load, state_dict:{flat_state_dict},\n load_infos:{load_infos},\n read_items:{read_items}"
        )
        state_dict_in_cpu = []
        for k, v in flat_state_dict.items():
            if v.place.is_cpu_place():
                state_dict_in_cpu.append(k)
                flat_state_dict[k] = v.cuda()
        for item in read_items:
            assert (
                item.local_tensor_index in load_infos
            ), f"item:{item}, load_infos:{load_infos}"
            src_rank, file_name = load_infos[item.local_tensor_index]
            storage_chunk_tensor = None
            cur_chunk_tensor = None
            # The src rank need to load the state_dict.
            if src_rank == paddle.distributed.get_rank():
                if file_name not in storage_file_to_state_dict:
                    # The value in state_dict is not distributed tensor but a normal tensor.
                    storage_file_to_state_dict[file_name] = paddle.load(
                        os.path.join(path, file_name)
                    )
                storage_state_dict = storage_file_to_state_dict[file_name]
                assert item.local_tensor_index.tensor_key in storage_state_dict
                storage_local_tensor = storage_state_dict[
                    item.local_tensor_index.tensor_key
                ]
                storage_offsets = item.storage_offset
                storage_lengths = item.lengths
                storage_ends = [
                    storage_offset + storage_length
                    for storage_offset, storage_length in zip(
                        storage_offsets, storage_lengths
                    )
                ]
                # The storage_chunk_tensor and storage_local_tensor share the same memory.
                if len(storage_lengths) > 0:
                    storage_chunk_tensor = paddle.slice(
                        storage_local_tensor,
                        list(range(len(storage_lengths))),
                        storage_offsets,
                        storage_ends,
                    )
                else:
                    storage_chunk_tensor = storage_local_tensor
            # The read item rank need to be assigned
            if item.rank == paddle.distributed.get_rank():
                assert (
                    item.local_tensor_index.tensor_key in flat_state_dict
                ), f"item:{item}, state_dict:{flat_state_dict}"

                cur_local_tensor = (
                    flat_state_dict[
                        item.local_tensor_index.tensor_key
                    ]._local_value()
                    if use_dist
                    and flat_state_dict[
                        item.local_tensor_index.tensor_key
                    ].is_dist()
                    else flat_state_dict[item.local_tensor_index.tensor_key]
                )

                cur_offsets = item.cur_offset
                cur_lengths = item.lengths
                cur_ends = [
                    cur_offset + cur_length
                    for cur_offset, cur_length in zip(cur_offsets, cur_lengths)
                ]
                # The cur_chunk_tensor and cur_local_tensor share the same memory.
                if len(cur_lengths) > 0:
                    cur_chunk_tensor = paddle.slice(
                        cur_local_tensor,
                        list(range(len(cur_lengths))),
                        cur_offsets,
                        cur_ends,
                    )
                else:
                    cur_chunk_tensor = cur_local_tensor
            else:
                # Why we use item.dtype: In static mode, the state_dict maybe incomplete in pp, the dtype is stored in advance.
                cur_chunk_tensor = paddle.zeros(
                    item.lengths,
                    item.dtype,
                )

            # Src_rank represents the rank of data read from ckpt, item_rank is the rank of the parameter of the data to be loaded.
            if src_rank == item.rank:
                if src_rank == paddle.distributed.get_rank():
                    # Assign value locally: in the case of src_rank is cur_rank, it means that the ckpt and the parameters to be loaded are both in the current node.
                    paddle.assign(storage_chunk_tensor, cur_chunk_tensor)
            else:
                # Assign value remotely: src_rank broadcasts the ckpt, and the parameters to be loaded receive the data broadcast by src_rank.
                if src_rank == paddle.distributed.get_rank():
                    storage_chunk_tensor = storage_chunk_tensor.contiguous()
                    paddle.distributed.broadcast(
                        storage_chunk_tensor, src=src_rank, group=process_group
                    )
                else:
                    # The memory hold by cur_chunk_tensor may be non-contiguous, and the broadcast API does not support this type of tensor.
                    tmp_tensor = paddle.assign(cur_chunk_tensor)
                    paddle.distributed.broadcast(
                        tmp_tensor, src=src_rank, group=process_group
                    )
                    paddle.assign(tmp_tensor, cur_chunk_tensor)

        for k, v in flat_state_dict.items():
            if k in state_dict_in_cpu:
                value = state_dict
                for key in mapping[k]:
                    value = value[key]
                paddle.assign(v.cpu(), value)
