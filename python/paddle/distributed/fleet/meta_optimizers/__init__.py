# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
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

from .amp_optimizer import AMPOptimizer  # noqa: F401
from .asp_optimizer import ASPOptimizer  # noqa: F401
from .dgc_optimizer import (  # noqa: F401
    DGCMomentumOptimizer,
    DGCOptimizer,
)
from .dygraph_optimizer import (  # noqa: F401
    HeterParallelOptimizer,
    HybridParallelGradScaler,
    HybridParallelOptimizer,
)
from .fp16_allreduce_optimizer import FP16AllReduceOptimizer  # noqa: F401
from .gradient_merge_optimizer import GradientMergeOptimizer  # noqa: F401
from .lamb_optimizer import LambOptimizer  # noqa: F401
from .lars_optimizer import LarsOptimizer  # noqa: F401
from .localsgd_optimizer import (  # noqa: F401
    AdaptiveLocalSGDOptimizer,
    LocalSGDOptimizer,
)
from .pipeline_optimizer import PipelineOptimizer  # noqa: F401
from .ps_optimizer import ParameterServerOptimizer  # noqa: F401
from .qat_optimizer import QATOptimizer  # noqa: F401
from .raw_program_optimizer import RawProgramOptimizer  # noqa: F401
from .recompute_optimizer import RecomputeOptimizer  # noqa: F401
from .sharding_optimizer import ShardingOptimizer  # noqa: F401
from .tensor_parallel_optimizer import TensorParallelOptimizer  # noqa: F401
