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

"""
Muon-specific parameter annotation utilities for DygraphShardingOptimizerV2.

When the inner optimizer is Muon, each parameter slice needs to be annotated
with metadata (original shape, sharding indices, etc.) so that `_muon_update`
can correctly gather/scatter the full 2-D matrix across sharding ranks.
"""

import numpy as np

from ...utils.muon_comm_utils import (
    get_sharding_info,
    should_use_muon,
)


def annotate_muon_params(param, original_p, hcg, param2bucket):
    """Annotate a single parameter slice with Muon sharding metadata.

    Args:
        param: The local 1-D shard held by this rank.
        original_p: The original (unsliced) parameter from the model.
        hcg: The hybrid communicate group.
        param2bucket: Mapping from param name to (FusedCommBuffer, ...).

    Returns:
        True if the param was annotated (and should be kept), False if it
        should be skipped (uninitialised or sentinel).
    """
    if not should_use_muon(original_p.name, original_p.shape):
        return True

    # Skip uninitialised slices and shape-[1] sentinels.
    if not param._is_initialized():
        return False
    if list(param.shape) == [1] and list(original_p.shape) != [1]:
        return False

    # Annotate whether this rank holds a partial shard or the full weight.
    param.is_sharded_gather = int(param.numel()) < int(original_p.numel())
    param.original_shape = original_p.shape
    param.split_axis = getattr(original_p, "split_axis", None)
    param.needs_qkv_split = getattr(original_p, "needs_qkv_split", False)
    param.head_num = getattr(original_p, "head_num", 0)
    param.kv_head_num = getattr(original_p, "kv_head_num", 0)
    param.is_muon = True

    # MoE experts use a dedicated expert-parallel sharding group.
    if getattr(original_p, "no_sync", False):
        sharding_group = hcg.get_moe_sharding_parallel_group()
    else:
        sharding_group = hcg.get_sharding_parallel_group()

    sharding_rank = sharding_group.rank
    if sharding_rank == -1:
        sharding_rank = 0
    sharding_world_size = sharding_group.nranks

    if param.is_sharded_gather:
        # Compute per-rank element counts for the variable-length gather.
        target_buffer = param2bucket[param.name][0]
        indices, my_offset = get_sharding_info(
            target_buffer,
            param.name,
            sharding_world_size,
            sharding_rank,
        )
        param.sharding_indices = indices
        param.sharding_my_offset = my_offset

    return True


def sort_muon_params_grads(params_grads):
    """Sort params_grads so that largest fully-owned params come first.

    This improves GPU memory allocator locality by processing large contiguous
    allocations before smaller fragmented ones.
    """
    params_grads.sort(
        key=lambda x: (
            getattr(x[0], "is_sharded_gather", False),
            np.prod(getattr(x[0], "original_shape", []))
            if getattr(x[0], "original_shape", None)
            else 0,
        ),
        reverse=True,
    )
