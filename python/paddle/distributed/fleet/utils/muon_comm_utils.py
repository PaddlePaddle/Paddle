# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.distributed as dist
from paddle.distributed.communication.batch_isend_irecv import (
    _coalescing_manager as batch_isend_irecv_coalescing_manager,
)


def gather_varlen(input, dst, group, all_shape_and_dtype):
    """Gather variable-length tensors from all ranks to *dst*.

    The destination rank pre-allocates a single contiguous buffer for all
    incoming data to avoid memory fragmentation from intermediate concat.
    Non-destination ranks send their local slice and return None.

    Args:
        input: Local tensor slice (may be None if this rank contributes nothing).
        dst: Global rank of the destination.
        group: The process group.
        all_shape_and_dtype: List of (shape, dtype) tuples, one per rank.
            shape is None (or shape[0] == 0) when a rank has no data.

    Returns:
        Concatenated 1-D tensor on the destination rank; None elsewhere.
    """
    tasks = []

    if group.ranks[group.rank] == dst:
        # Destination: allocate one contiguous buffer and receive all slices.
        total_len = sum([s[0] for s, _ in all_shape_and_dtype if s is not None])
        dtype = all_shape_and_dtype[0][1]
        output_tensor = paddle.empty([total_len], dtype=dtype)

        task_info_list = []
        current_offset = 0

        with batch_isend_irecv_coalescing_manager(group, tasks):
            for src in range(group.nranks):
                shape = all_shape_and_dtype[src][0]
                if shape is None or shape[0] == 0:
                    continue
                length = shape[0]
                if src != group.rank:
                    recv_tensor = paddle.empty(
                        shape, dtype=all_shape_and_dtype[src][1]
                    )
                    task = dist.irecv(
                        recv_tensor, group.ranks[src], group=group
                    )
                    tasks.append(task)
                    task_info_list.append(
                        (task, recv_tensor, current_offset, length)
                    )
                else:
                    output_tensor[current_offset : current_offset + length] = (
                        input
                    )
                current_offset += length

        for task, recv_tensor, offset, length in task_info_list:
            task.wait()
            output_tensor[offset : offset + length] = recv_tensor
            del recv_tensor

        return output_tensor

    else:
        # Sender: push local slice to dst and return None.
        with batch_isend_irecv_coalescing_manager(group, tasks):
            if input is not None and input.shape[0] != 0:
                task = dist.isend(input, dst, group=group)
                tasks.append(task)

        for task in tasks:
            task.wait()

        return None


def get_sharding_info(buffer, param_name, world_size, rank):
    """Compute per-rank element counts and local offset for a sharded parameter.

    ShardingV2 splits the flat param storage evenly across ranks.  This
    function intersects each rank's slice of that storage with the parameter's
    global range to produce the element count each rank owns.

    Args:
        buffer: The FusedCommBuffer that contains the parameter.
        param_name: Name of the parameter.
        world_size: Number of ranks in the sharding group.
        rank: Local rank in the sharding group.

    Returns:
        indices: List of element counts per rank (length == world_size).
        my_slice_offset: Offset of this rank's slice within the full flat param.
    """
    grad_view = buffer._sharding_param_grad_view[param_name]

    param_global_start = grad_view._index
    param_global_end = grad_view._index + grad_view._padded_size

    # ShardingV2 splits the storage buffer evenly across ranks.
    shard_size = buffer.param_storage.shape[0] // world_size

    indices = []
    my_slice_offset = 0
    current_relative_offset = 0

    for r in range(world_size):
        r_start = r * shard_size
        r_end = (r + 1) * shard_size
        start = max(param_global_start, r_start)
        end = min(param_global_end, r_end)
        length = max(0, end - start)
        indices.append(length)
        if r == rank:
            my_slice_offset = current_relative_offset
        current_relative_offset += length

    return indices, my_slice_offset


def should_use_muon(name, shape):
    """Return True if a parameter should receive Muon (orthogonal) updates.

    Muon applies to 2-D weight matrices, and also to 3-D fused MoE expert
    tensors [n_experts, H, I] where each expert's 2-D slice is updated
    independently via Newton-Schulz.  Embeddings, biases, and LM-head
    weights fall back to AdamW.
    """
    if len(shape) not in (2, 3):
        return False
    name = name.lower()
    if "embed" in name or "bias" in name or "lm_head" in name:
        return False
    return True
