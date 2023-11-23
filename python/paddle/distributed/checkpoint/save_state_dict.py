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
from typing import List, Dict
import numpy as np

import paddle
from paddle.distributed.communication.group import is_initialized
from metadata import Metadata, ChunkMetadata, MetadataIndex
from utils import merge_state_dict, dedup_state_dict, compute_local_shape_and_global_offset

def check_state_dict(state_dict, process_group):
    local_keys = list(state_dict.keys())
    gloabl_keys = []
    paddle.distributed.all_gather_object(gloabl_keys, local_keys, process_group)
    for keys in gloabl_keys[1:]:
        assert keys == gloabl_keys[0], f"keys:{keys} != first_keys: {gloabl_keys[0]}"

def check_file_name(file_name, process_group):
    all_unique_id = []
    unique_id = int(file_name.split(".")[0].split("_")[1])
    paddle.distributed.all_gather_object(all_unique_id, unique_id, process_group)
    for id in all_unique_id[1:]:
        assert id == all_unique_id[0], f"id:{id} !=  all_unique_id[0]:{file_name}"

# def merge_state_dict(global_state_dict):
#     assert isinstance(global_state_dict, List), "The global_state_dict should be a list."
#     out = {}
#     for state_dict in global_state_dict:
#         for key, val in state_dict.items():
#             if key in out and val not in out[key]:
#                 out[key].append(val)
#             else:
#                 out[key] = [val]
#     return out

# def dedup_state_dict(global_state_dict):
#     out = {}
#     for state_dict in global_state_dict:
#         for key, val in state_dict.items():
#             if key in out:
#                 continue
#             out[key] = val
#     return out

def save_state_dict(state_dict, path, process_group=None, coordinator_rank=0, use_dist=True) -> None:
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
    assert isinstance(state_dict, dict), "The state_dict should be a dictionary."
    if len(state_dict) > 0:
        for val in state_dict.values():
            assert isinstance(val, (paddle.Tensor, paddle.base.framework.EagerParamBase)), "Only support dygraph Tensor now, support static DistributedTensor later"
    #
    if process_group is None:
        not is_initialized() and paddle.distributed.init_parallel_env()
        process_group = paddle.distributed.new_group(list(range(paddle.distributed.ParallelEnv().nranks)), backend="nccl")
    # calculate (global offset, local shape) of each DTensor
    local_state_dict = {}
    metadata = Metadata()
    unique_id = 0
    file_name = ""
    while(True):
        file_name = f"{paddle.distributed.get_rank()}_{unique_id}.distcp"
        if not os.path.exists(os.path.join(path, file_name)):
            break
        unique_id += 1
    print(f"file_name:{file_name}")
    check_file_name(file_name, process_group)
    # the parameter_name and order in state_dict should be the same
    check_state_dict(state_dict, process_group)
    local_chunk_metadata = {}
    local_storage_metadata = {}
    for key, val in state_dict.items():
        if isinstance(val, paddle.Tensor):
            if val.is_dist():
                local_tensor = val.get_tensor().get_tensor()
                local_shape, global_offset = compute_local_shape_and_global_offset(val.shape, val.dist_attr.process_mesh, val.dist_attr.dims_mapping)
                # gather local_shape and global_offset from all ranks of each parameter
                local_chunk_metadata[key] = ChunkMetadata(local_shape, global_offset)
                local_storage_metadata[MetadataIndex(key, tuple(global_offset))] = file_name
            else:
                local_tensor = val
            local_state_dict[key] = local_tensor
    global_chunk_metadata = []
    global_storage_metadata = []
    paddle.distributed.all_gather_object(global_chunk_metadata, local_chunk_metadata, process_group)
    paddle.distributed.all_gather_object(global_storage_metadata, local_storage_metadata, process_group)
    metadata.state_dict_metadata = merge_state_dict(global_chunk_metadata)
    metadata.storage_metadata = dedup_state_dict(global_storage_metadata)
    if coordinator_rank == paddle.distributed.get_rank():
        print(f"metadata:{metadata}")
        paddle.save(metadata, os.path.join(path, f"{unique_id}.metadata"))
    print(f"local_state_dict:{local_state_dict}")
    for k,v in local_state_dict.items():
        # the phi::DenseTensor only support convert to np.array
        local_state_dict[k] = np.array(v)
        print(f"local_state_dict name:{k}, val:{local_state_dict[k]}, type:{type(local_state_dict[k])}")
    paddle.save(local_state_dict, os.path.join(path, file_name))
    


def test_save_state_dict():
    import paddle.distributed as dist
    w1 = paddle.arange(8).reshape([4, 2])
    w2 = paddle.arange(8, 12).reshape([2, 2])
    mesh = dist.ProcessMesh([0,1], dim_names=["x"])
    w1_dist_attr = dist.DistAttr(mesh, sharding_specs=["x", None])
    sharded_w1 = dist.shard_tensor(w1, dist_attr=w1_dist_attr)
    w2_dist_attr = dist.DistAttr(mesh, sharding_specs=[None, None])
    sharded_w2 = dist.shard_tensor(w2, dist_attr=w2_dist_attr)
    state_dict = {"w1": sharded_w1, "w2": sharded_w2}
    save_state_dict(state_dict, "./output")

if __name__ == "__main__":
    test_save_state_dict()
