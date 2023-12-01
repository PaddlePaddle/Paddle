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
from dataclasses import dataclass
from typing import List, Tuple

import paddle
from paddle.distributed.communication.group import is_initialized

from .metadata import Metadata, ChunkMetadata, MetadataIndex
from .utils import compute_local_shape_and_global_offset

@dataclass(frozen=True)
class ReadItem:
    rank:int
    meta_index:MetadataIndex
    cur_offset:Tuple[int]
    storage_offset:Tuple[int]
    lengths:Tuple[int]


def get_local_load_files(path, state_dict, process_group):
    # step 1, get neccesary files to be read
    accessible_files = os.listdir(path)
    metadata_files = [file for file in accessible_files if file.endswith(".metadata")]
    assert len(metadata_files) > 0, "No metadata file found in the checkpoint directory:{path}."
    # The neccesary files to be read
    necessary_files = []
    for metadata_file in metadata_files:
        metadata = paddle.load(os.path.join(path, metadata_file))
        for metadata_index, file_name in metadata.storage_metadata.items():
            if metadata_index.param in state_dict:
                necessary_files.append(file_name)
    necessary_files_set = set(necessary_files)
    # allgather all accessible files
    local_data_files = [file for file in accessible_files if file.endswith(".distcp")]
    global_data_files = []
    paddle.distributed.all_gather_object(global_data_files, local_data_files, process_group)
    tmp = []
    for files in global_data_files:
        tmp += files
    global_data_files_set = set(tmp)
    print(f"necessary_files_set:{necessary_files_set}, global_data_files_set:{global_data_files_set}")
    # check neccesary files in global_data_files
    assert global_data_files_set & necessary_files_set == necessary_files_set, \
        f"The checkpoint files are not complete. Please check the checkpoint directory:{path}.global_data_files_set:{global_data_files_set}, necessary_files_set:{necessary_files_set}"
    # step 2, get mapping between ranks and local files
    rank_to_files = {}
    file_to_ranks = {}
    for rank, local_files in enumerate(global_data_files):
        if len(local_files) > 0:
            local_files = [f for f in local_files if f in necessary_files_set]
            rank_to_files[rank] = local_files
        for file in local_files:
            if file not in file_to_ranks:
                file_to_ranks[file] = []
            file_to_ranks[file].append(rank)
    print(f"mapping rank_to_files:{rank_to_files}, file_to_ranks:{file_to_ranks}")
    rank_to_read_files = {rank:[] for rank in rank_to_files.keys()}
    for file, ranks in file_to_ranks.items():
        if len(ranks) == 1:
            rank = ranks[0]
            rank_to_read_files[rank].append(file)
            rank_to_files[rank].remove(file)
            if len(rank_to_files[rank]) == 0:
                rank_to_files.pop(rank)
    
    print(f"start rank_to_read_files:{rank_to_read_files}, rank_to_files:{rank_to_files}")
    # step 3, update the rank_to_read_files
    def get_least_read_files_ranks(rank_to_read_files):
        nums = [(rank, len(files)) for rank, files in rank_to_read_files.items()]
        nums = sorted(nums, key=lambda x: x[1])
        ranks = [rank for rank, num in nums if num == nums[0][1]]
        return ranks
    def get_read_rank_file(rank_to_files, ranks):
        if len(rank_to_files) == 0:
            return (None, None)
        nums = [(rank, len(files)) for rank, files in rank_to_files.items() if rank in ranks]
        nums = sorted(nums, key=lambda x: x[1])
        rank = nums[0][0]
        return (rank, rank_to_files[rank][0])
    def update(rank_to_read_files, rank_to_files, rank_file):
        rank, file = rank_file
        if rank is None and file is None:
            return
        if rank not in rank_to_read_files:
            rank_to_read_files[rank] = []
        rank_to_read_files[rank].append(file)
        # update rank_to_files
        file_to_ranks = {}
        for r, files in rank_to_files.items():
            for f in files:
                if f not in file_to_ranks:
                    file_to_ranks[f] = []
                file_to_ranks[f].append(r)
        print(f"file_to_ranks:{file_to_ranks}")
        if file in file_to_ranks:
            for r in file_to_ranks[file]:
                rank_to_files[r].remove(file)
                if len(rank_to_files[r]) == 0:
                    rank_to_files.pop(r)
    # step 4, get final rank_to_read_files
    while len(rank_to_files) > 0:
        ranks = get_least_read_files_ranks(rank_to_read_files)
        rank_file = get_read_rank_file(rank_to_files, ranks)
        update(rank_to_read_files, rank_to_files, rank_file)
        print(f"update rank_to_read_files:{rank_to_read_files}, rank_to_files:{rank_to_files}, ranks:{ranks}, rank_file:{rank_file}")
    print(f"rank_to_read_files:{rank_to_read_files}")
    cur_rank = paddle.distributed.get_rank()
    if cur_rank in rank_to_read_files:
        print(f"cur_rank:{cur_rank}, rank_to_read_files[cur_rank]:{rank_to_read_files[cur_rank]}")
        return rank_to_read_files[cur_rank]
    else:
        print(f"rank:{cur_rank} does not need to load checkpoint")
        return []


def get_load_infos(path, local_load_files, process_group):
    load_info = {}
    accessible_files = os.listdir(path)
    metadata_files = [file for file in accessible_files if file.endswith(".metadata")]
    assert len(metadata_files) > 0, "No metadata file found in the checkpoint directory:{path}."
    for metadata_file in metadata_files:
        metadata = paddle.load(os.path.join(path, metadata_file))
        for meta_index, file_name in metadata.storage_metadata.items():
            if file_name in local_load_files:
                load_info[meta_index] = (paddle.distributed.get_rank(), file_name)
    load_info_list = []
    paddle.distributed.all_gather_object(load_info_list, load_info, process_group)
    load_infos = {}
    for load_info in load_info_list:
        for meta_index, (rank, file_name) in load_info.items():
            assert meta_index not in load_infos
            load_infos[meta_index] = (rank, file_name)
    return load_infos


def compute_overlap(cur_chunk_metadata:ChunkMetadata, storage_chunk_metadata:ChunkMetadata):
    cur_offsets = []
    storage_offsets = []
    lengths = []
    for cur_len, cur_offset, strorage_len, storage_offset in zip(
        cur_chunk_metadata.local_shape,
        cur_chunk_metadata.global_offset,
        storage_chunk_metadata.local_shape,
        storage_chunk_metadata.global_offset
    ):
        begin_offset = max(cur_offset, storage_offset)
        end_offset = min(cur_offset + cur_len, storage_offset + strorage_len)
        if begin_offset == cur_offset:
            cur_offsets.append(0)
            storage_offsets.append(begin_offset - storage_offset)
        elif begin_offset == storage_offset:
            cur_offsets.append(begin_offset - cur_offset)
            storage_offsets.append(0)
        else:
            assert False, "Should not reach here."
        lengths.append(end_offset - begin_offset)
        assert lengths[-1] >= 0, f"Invalid length:{lengths[-1]}, end_offset:{end_offset}, begin_offset:{begin_offset}"
    return cur_offsets, storage_offsets, lengths


def not_overlap(cur_chunk_metadata:ChunkMetadata, storage_chunk_metadata:ChunkMetadata):
    for cur_len, cur_offset, strorage_len, storage_offset in zip(
        cur_chunk_metadata.local_shape,
        cur_chunk_metadata.global_offset,
        storage_chunk_metadata.local_shape,
        storage_chunk_metadata.global_offset
    ):
        if cur_offset >= (storage_offset + strorage_len) or (cur_offset + cur_len) <= storage_offset:
            return True
    return False

def get_read_items(path, state_dict, process_group):
    accessible_files = os.listdir(path)
    metadata_files = [file for file in accessible_files if file.endswith(".metadata")]
    assert len(metadata_files) > 0, "No metadata file found in the checkpoint directory:{path}."
    param_to_chunkmetadata = {}
    for metadata_file in metadata_files:
        metadata = paddle.load(os.path.join(path, metadata_file))
        for param_name, chunk_metadata in metadata.state_dict_metadata.items():
            if param_name not in param_to_chunkmetadata:
                param_to_chunkmetadata[param_name] = []
            param_to_chunkmetadata[param_name] += chunk_metadata
    read_items = []
    print(f"param_to_chunkmetadata:{param_to_chunkmetadata}")
    for param_name, val in state_dict.items():
        if isinstance(val, paddle.Tensor):
            if val.is_dist():
                local_shape, global_offset = compute_local_shape_and_global_offset(val.shape, val.dist_attr.process_mesh, val.dist_attr.dims_mapping)
                if not local_shape or not global_offset:
                    continue
                cur_chunk_metadata = ChunkMetadata(local_shape, global_offset)
                assert param_name in param_to_chunkmetadata, f"param_name:{param_name} not found in param_to_chunkmetadata:{param_to_chunkmetadata}."
                for storage_chunk_metadata in param_to_chunkmetadata[param_name]:
                    if not_overlap(cur_chunk_metadata, storage_chunk_metadata):
                        continue
                    cur_offsets, storage_offsets, lengths = compute_overlap(cur_chunk_metadata, storage_chunk_metadata)
                    storage_meta_index = MetadataIndex(param_name, tuple(storage_chunk_metadata.global_offset))
                    read_items.append(ReadItem(paddle.distributed.get_rank(), storage_meta_index, tuple(cur_offsets), tuple(storage_offsets), tuple(lengths)))
            else:
                assert False, f"Only support distributed tensor., val type:{type(val)}"
        else:
            assert False, f"Only support paddle.Tensor., val type:{type(val)}"
    global_read_items = []
    tmp = []
    paddle.distributed.all_gather_object(tmp, read_items, process_group)
    for items in tmp:
        for item in items:
            global_read_items.append(item)
    return global_read_items

def flatten_state_dict(state_dict):
    # TODO, {"model": {"w0": xxx}} -> {model.w0: xxx}
    return state_dict


def load_state_dict(state_dict, path, process_group=None, coordinator_rank=0, use_dist=True) -> None:
    """
    Load the state_dict inplace from a checkpoint path.
    Args:
        state_dict: The state_dict to load. It will be modified inplace after loading.
        path: The directory to load checkpoint files.
        process_group: ProcessGroup to be used for cross-rank synchronization. Use the default process group which contains all cards.
        coordinator_rank: The rank used to coordinate the checkpoint. Rank0 is used by default.
        use_dict: Whether to load the state_dict in distributed mode. Set True by default.
    Example:
        .. code-block:: python
        import paddle
        ...
    """
    if process_group is None:
        # Init the default global process group
        not is_initialized() and paddle.distributed.init_parallel_env()
        # process_group = paddle.distributed.new_group(list(range(paddle.distributed.ParallelEnv().nranks)), backend="nccl")

    state_dict = flatten_state_dict(state_dict)
    local_load_files = get_local_load_files(path, state_dict, process_group)
    # load_infos: {MetaIndex: (rank, file_name)}
    load_infos = get_load_infos(path, local_load_files, process_group)
    read_items = get_read_items(path, state_dict, process_group)
    loaded_state_dict = {}
    print(f"before load, state_dict:{state_dict},\n load_infos:{load_infos},\n read_items:{read_items}")
    for item in read_items:
        assert item.meta_index in load_infos, f"item:{item}, load_infos:{load_infos}"
        src_rank, file_name = load_infos[item.meta_index]
        storage_chunk_tensor = None
        cur_sub_chunk_tensor = None
        # The src rank need to load the state_dict.
        if src_rank == paddle.distributed.get_rank():
            if file_name not in loaded_state_dict:
                # The load state_dict is not distributed tensor but a normal tensor.
                loaded_state_dict[file_name] = paddle.load(os.path.join(path, file_name))
            storage_state_dict = loaded_state_dict[file_name]
            assert item.meta_index.param in storage_state_dict
            storage_local_tensor = storage_state_dict[item.meta_index.param]
            storage_offsets = item.storage_offset
            storage_lengths = item.lengths
            storage_ends = [storage_offset + storage_length for storage_offset, storage_length in zip(storage_offsets, storage_lengths)]
            storage_chunk_tensor = paddle.slice(storage_local_tensor, list(range(len(storage_lengths))), storage_offsets, storage_ends)
        # The read item rank need to be assigned
        if item.rank == paddle.distributed.get_rank():
            assert item.meta_index.param in state_dict, f"item:{item}, state_dict:{state_dict}"
            cur_local_tensor = state_dict[item.meta_index.param]._local_value()
            cur_offsets = item.cur_offset
            cur_lengths = item.lengths
            cur_ends = [cur_offset + cur_length for cur_offset, cur_length in zip(cur_offsets, cur_lengths)]
            cur_sub_chunk_tensor = paddle.slice(cur_local_tensor, list(range(len(cur_lengths))), cur_offsets, cur_ends)
        else:
            cur_sub_chunk_tensor = paddle.zeros(item.lengths, dtype=state_dict[item.meta_index.param].dtype)

        if src_rank == item.rank:
            # assign value locally
            paddle.assign(storage_chunk_tensor, cur_sub_chunk_tensor)
        else:
            # assign value remotely
            if src_rank == paddle.distributed.get_rank():
                paddle.distributed.broadcast(storage_chunk_tensor, src=src_rank, group=process_group)
            else:
                paddle.distributed.broadcast(cur_sub_chunk_tensor, src=src_rank, group=process_group)

    local_state_dict = { k:v._local_value() for k, v in state_dict.items()}
    print(f"after load, local_state_dict:{local_state_dict} \n state_dict:{state_dict}")


def test_get_local_load_files():
    path = "./output"
    # build state_dict
    import paddle.distributed as dist
    w1 = paddle.zeros([4,2], dtype=paddle.int64)
    w2 = paddle.zeros([2,2], dtype=paddle.int64)
    mesh = dist.ProcessMesh([0,1,2,3])
    sharded_w1 = dist.shard_tensor(w1, mesh, [dist.Shard(0), dist.Replicate()])
    sharded_w2 = dist.shard_tensor(w2, mesh, [dist.Replicate(), dist.Replicate()])
    state_dict = {"w1": sharded_w1, "w2": sharded_w2}
    load_state_dict(state_dict, path)
    
    


def test_load_state_dict():
    test_get_local_load_files()

if __name__ == "__main__":
    test_load_state_dict()
