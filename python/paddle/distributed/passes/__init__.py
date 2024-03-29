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

from .pass_base import new_pass, PassManager, PassContext

from .auto_parallel_gradient_merge import (  # noqa: F401
    parse_program,
    GradientMergePass,
)
from .auto_parallel_sharding import (  # noqa: F401
    ShardingPass,
    is_sharding_param_broadcast_op,
    partition_by_use_order,
    partition_by_greedy_even,
    partition_parameters,
    re_order_program,
    group_param,
    ShardingInfo,
    VarGroup,
)
from .auto_parallel_amp import (  # noqa: F401
    AMPLists,
    AMPState,
    AMPPass,
)
from .auto_parallel_master_grad import (  # noqa: F401
    get_output_in_varlist,
    MasterGradPass,
)
from .auto_parallel_fp16 import (  # noqa: F401
    set_op_dtype_to_fp16,
    set_auto_cast_attr,
    FP16State,
    cast_startup_program,
    FP16Pass,
)
from .auto_parallel_recompute import (  # noqa: F401
    RecomputeState,
    RecomputePass,
)
from .auto_parallel_quantization import QuantizationPass  # noqa: F401
from .auto_parallel_data_parallel_optimization import (  # noqa: F401
    DataParallelOptimizationPass,
    GradientsGroup,
)
from .auto_parallel_grad_clip import (  # noqa: F401
    ClipHelper,
    ClipGradByGlobalNormPass,
)
from .auto_parallel_fused_linear_promotion import (  # noqa: F401
    FusedLinearPromotionPass,
)
from .auto_parallel_supplement_explicit_dependencies import (  # noqa: F401
    AutoParalSupplementDepPass,
)
from .auto_parallel_pipeline import is_reshard_op, PipelinePass  # noqa: F401
from .auto_parallel_sequence_parallel_optimization import (  # noqa: F401
    SequenceParallelOptimizationPass,
)
from .allreduce_matmul_grad_overlapping import (  # noqa: F401
    AllreduceMatmulGradOverlappingPass,
)
from .cpp_pass import (  # noqa: F401
    FuseElementwiseAddActPass,
    FuseBatchNormActPass,
    FuseBatchNormAddActPass,
    FuseReluDepthwiseConvPass,
    FusedAttentionPass,
    FusedFeedforwardPass,
    FuseGemmEpiloguePass,
    FuseAdamWPass,
    FuseDotProductAttentionPass,
    FuseOptimizerPass,
    InplaceAddtoOpPass,
    FuseResUnitPass,
    BuildCINNPass,
)
from .fuse_all_reduce import (  # noqa: F401
    find_adjacent_match_sequences,
    insert_fuse_all_reduce_ops,
    has_same_attrs,
    filter_all_collective_op_indices,
    find_all_fuse_all_reduce_groups,
    split_fuse_all_reduce_groups_by_deps,
    insert_coalesce_tensor_ops,
    insert_fuse_all_reduce_by_memory_size,
    FuseAllReducePass,
)
from .pipeline_scheduler_pass import (  # noqa: F401
    PipelineFThenBPass,
    Pipeline1F1BPass,
    PipelineEager1F1BPass,
    PipelineVirtualPipelinePass,
    apply_pass,
)
from .ps_trainer_pass import (  # noqa: F401
    AppendSendOpsPass,
    DistributedOpsPass,
    DeleteOptimizesPass,
    DeleteExtraOptimizerPass,
    FakeInitOpsPass,
    PsGpuPass,
    PsTranspilePass,
    SplitHeterWorkerOpsPass,
    SplitTrainerOpsPass,
    SetHeterPipelineOptPass,
    SplitFlOpsPass,
)
from .ps_server_pass import (  # noqa: F401
    AddLrDecayTablePass,
    AddListenAndServPass,
    AddRpcGlobalFlagsPass,
    AddOptimizerPass,
    AddGeoOptimizerPass,
    BuildPserverStartupProgramPass,
    DeleteUnusedInStartupPass,
)


__all__ = [
    'new_pass',
    'PassManager',
    'PassContext',
]
