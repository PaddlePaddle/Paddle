# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""gin_overlap: the sharding-stage1 gradient reduce_scatter in stages on NCCL GIN one-sided puts, released inside
the overlap windows of the backward pass by a C++ scheduler (paddle.base.core.gin_overlap.GinGradCommScheduler).

Turned on by ``strategy.hybrid_configs["sharding_configs"].gin_overlap``; DygraphShardingOptimizerV2 and
MuonShardingOptimizer then build the scheduler (build_grad_comm_scheduler). A framework that knows the overlap
windows (for example the DeepEP hooks of a MoE layer) drives it through active_scheduler():

  - ``release(ops, intra_sum_blocks, final_sum_blocks)``: in a window, release what the ops in ``ops`` can release
    now; the two counts are the grid sizes of the INTRA_SUM / FINAL_SUM kernels.
  - ``release_async(intra_sum_blocks, final_sum_blocks)``: after the last window of the step, hand the rest over to
    the proxy thread.
  - ``completed_steps()`` / ``has_expert_buckets()``: read-only.

Without a driver every bucket is released at the end of reduce_gradients. Op constants: INTRA_COPY, INTRA_SUM, PUT,
FINAL_SUM, GATHER (bit flags, OR-ed together).
"""

from .grad_comm_scheduler import (  # noqa: F401
    FINAL_SUM,
    GATHER,
    INTRA_COPY,
    INTRA_SUM,
    PUT,
    GinGradCommScheduler,
    active_scheduler,
    build_grad_comm_scheduler,
)

__all__ = []
