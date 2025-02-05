# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from .parallel_layers import (  # noqa: F401
    ColumnParallelLinear,
    LayerDesc,
    ParallelCrossEntropy,
    PipelineLayer,
    RNGStatesTracker,
    RowParallelLinear,
    SharedLayerDesc,
    VocabParallelEmbedding,
    get_rng_state_tracker,
    model_parallel_random_seed,
)
from .pipeline_parallel import (  # noqa: F401
    PipelineParallel,
    PipelineParallelMicroStepLocations,
    PipelineParallelWithInterleave,
    PipelineParallelWithInterleaveFthenB,
    VPPFhenBInBalancedMemory,
    register_global_pipeline_parallel_hook,
)
from .segment_parallel import SegmentParallel  # noqa: F401
from .sharding_parallel import ShardingParallel  # noqa: F401
from .tensor_parallel import TensorParallel  # noqa: F401

__all__ = []
