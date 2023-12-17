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
from . import io
from .spawn import spawn
from .launch.main import launch
from .parallel import (  # noqa: F401
    init_parallel_env,
    get_rank,
    get_world_size,
    ParallelEnv,
    DataParallel,
)
from .parallel_with_gloo import (
    gloo_init_parallel_env,
    gloo_barrier,
    gloo_release,
)

from paddle.distributed.fleet.dataset import InMemoryDataset, QueueDataset
from paddle.distributed.fleet.base.topology import ParallelMode

from .collective import (
    split,
    new_group,
    is_available,
)
from .communication import (  # noqa: F401
    stream,
    ReduceOp,
    all_gather,
    all_gather_object,
    all_reduce,
    alltoall,
    alltoall_single,
    broadcast,
    broadcast_object_list,
    reduce,
    send,
    scatter,
    gather,
    scatter_object_list,
    isend,
    recv,
    irecv,
    batch_isend_irecv,
    P2POp,
    reduce_scatter,
    is_initialized,
    destroy_process_group,
    get_group,
    wait,
    barrier,
    get_backend,
)

from .auto_parallel.process_mesh import ProcessMesh

from paddle.base.core import ReduceType, Placement
from .auto_parallel.placement_type import (
    Shard,
    Replicate,
    Partial,
)

from .auto_parallel import shard_op  # noqa: F401

from .auto_parallel.api import (
    DistAttr,
    shard_tensor,
    dtensor_from_fn,
    reshard,
    shard_layer,
    shard_optimizer,
    to_static,
    Strategy,
)

from .fleet import BoxPSDataset  # noqa: F401

from .entry_attr import (  # noqa: F401
    ProbabilityEntry,
    CountFilterEntry,
    ShowClickEntry,
)

from . import cloud_utils  # noqa: F401

from .sharding import (  # noqa: F401
    group_sharded_parallel,
    save_group_sharded_model,
)

from . import rpc  # noqa: F401

from .checkpoint.save_state_dict import save_state_dict
from .checkpoint.load_state_dict import load_state_dict

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
    "ReduceType",
    "Placement",
    "Shard",
    "Replicate",
    "Partial",
    "save_state_dict",
    "load_state_dict",
    "shard_optimizer",
    "to_static",
    "Strategy",
]
