# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import atexit  # noqa: F401

from .value_patch import monkey_patch_value_in_dist

monkey_patch_value_in_dist()
from paddle.base.core import Placement, ReduceType
from paddle.distributed.fleet.base.topology import ParallelMode
from paddle.distributed.fleet.dataset import InMemoryDataset, QueueDataset

from . import (
    cloud_utils,  # noqa: F401
    io,
    rpc,  # noqa: F401
)
from .auto_parallel import shard_op  # noqa: F401
from .auto_parallel.api import (
    DistAttr,
    DistModel,
    ShardingStage1,
    ShardingStage2,
    ShardingStage3,
    Strategy,
    dtensor_from_fn,
    in_auto_parallel_align_mode,  # noqa: F401
    reshard,
    shard_dataloader,
    shard_layer,
    shard_optimizer,
    shard_scaler,
    shard_tensor,
    to_static,
    unshard_dtensor,
)
from .auto_parallel.high_level_api import to_distributed
from .auto_parallel.interface import get_mesh, set_mesh
from .auto_parallel.intermediate.parallelize import parallelize
from .auto_parallel.intermediate.pipeline_parallel import SplitPoint
from .auto_parallel.intermediate.tensor_parallel import (
    ColWiseParallel,
    PrepareLayerInput,
    PrepareLayerOutput,
    RowWiseParallel,
    SequenceParallelBegin,
    SequenceParallelDisable,
    SequenceParallelEnable,
    SequenceParallelEnd,
)
from .auto_parallel.local_layer import LocalLayer
from .auto_parallel.placement_type import (
    Partial,
    Replicate,
    Shard,
)
from .auto_parallel.process_mesh import ProcessMesh
from .checkpoint.load_state_dict import load_state_dict
from .checkpoint.save_state_dict import save_state_dict
from .collective import (
    is_available,
    new_group,
    split,
)
from .communication import (  # noqa: F401
    P2POp,
    ReduceOp,
    all_gather,
    all_gather_object,
    all_reduce,
    alltoall,
    alltoall_single,
    barrier,
    batch_isend_irecv,
    broadcast,
    broadcast_object_list,
    destroy_process_group,
    gather,
    get_backend,
    get_group,
    irecv,
    is_initialized,
    isend,
    recv,
    reduce,
    reduce_scatter,
    scatter,
    scatter_object_list,
    send,
    stream,
    wait,
)
from .entry_attr import (
    CountFilterEntry,
    ProbabilityEntry,
    ShowClickEntry,
)
from .fleet import BoxPSDataset  # noqa: F401
from .launch.main import launch
from .parallel import (  # noqa: F401
    DataParallel,
    ParallelEnv,
    get_rank,
    get_world_size,
    init_parallel_env,
)
from .parallel_with_gloo import (
    gloo_barrier,
    gloo_init_parallel_env,
    gloo_release,
)
from .sharding import (  # noqa: F401
    group_sharded_parallel,
    save_group_sharded_model,
)
from .spawn import spawn

__all__ = [
    "io",
    "spawn",
    "launch",
    "scatter",
    "gather",
    "scatter_object_list",
    "broadcast",
    "broadcast_object_list",
    "ParallelEnv",
    "new_group",
    "init_parallel_env",
    "gloo_init_parallel_env",
    "gloo_barrier",
    "gloo_release",
    "QueueDataset",
    "split",
    "CountFilterEntry",
    "ShowClickEntry",
    "get_world_size",
    "get_group",
    "all_gather",
    "all_gather_object",
    "InMemoryDataset",
    "barrier",
    "all_reduce",
    "alltoall",
    "alltoall_single",
    "send",
    "reduce",
    "recv",
    "ReduceOp",
    "wait",
    "get_rank",
    "ProbabilityEntry",
    "ParallelMode",
    "is_initialized",
    "destroy_process_group",
    "isend",
    "irecv",
    "reduce_scatter",
    "is_available",
    "get_backend",
    "ProcessMesh",
    "DistAttr",
    "shard_tensor",
    "dtensor_from_fn",
    "reshard",
    "shard_layer",
    "shard_dataloader",
    "ReduceType",
    "Placement",
    "Shard",
    "Replicate",
    "Partial",
    "save_state_dict",
    "load_state_dict",
    "shard_optimizer",
    "shard_scaler",
    "ShardingStage1",
    "ShardingStage2",
    "ShardingStage3",
    "to_static",
    "Strategy",
    "DistModel",
    "LocalLayer",
    "unshard_dtensor",
    "parallelize",
    "SequenceParallelEnd",
    "SequenceParallelBegin",
    "SequenceParallelEnable",
    "SequenceParallelDisable",
    "ColWiseParallel",
    "RowWiseParallel",
    "PrepareLayerOutput",
    "PrepareLayerInput",
    "SplitPoint",
    "set_mesh",
    "get_mesh",
    "to_distributed",
]
