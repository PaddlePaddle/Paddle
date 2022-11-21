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

from . import io
from .spawn import spawn  # noqa: F401
from .launch.main import launch  # noqa: F401
from .parallel import init_parallel_env  # noqa: F401
from .parallel import get_rank  # noqa: F401
from .parallel import get_world_size  # noqa: F401

from .parallel_with_gloo import gloo_init_parallel_env
from .parallel_with_gloo import gloo_barrier
from .parallel_with_gloo import gloo_release

from paddle.distributed.fleet.dataset import InMemoryDataset  # noqa: F401
from paddle.distributed.fleet.dataset import QueueDataset  # noqa: F401
from paddle.distributed.fleet.base.topology import ParallelMode  # noqa: F401

from .collective import all_gather  # noqa: F401
from .collective import all_gather_object  # noqa: F401
from .collective import barrier  # noqa: F401
from .collective import split  # noqa: F401
from .collective import new_group  # noqa: F401
from .collective import wait  # noqa: F401

from .communication import (
    stream,
    ReduceOp,
    all_reduce,
    alltoall,
    alltoall_single,
    broadcast,
    reduce,
    send,
    scatter,
    isend,
    recv,
    irecv,
    batch_isend_irecv,
    P2POp,
    reduce_scatter,
    is_initialized,
    destroy_process_group,
    get_group,
)  # noqa: F401

from .auto_parallel import shard_op  # noqa: F401
from .auto_parallel import shard_tensor  # noqa: F401

from .fleet import BoxPSDataset  # noqa: F401

from .entry_attr import ProbabilityEntry  # noqa: F401
from .entry_attr import CountFilterEntry  # noqa: F401
from .entry_attr import ShowClickEntry  # noqa: F401

from paddle.fluid.dygraph.parallel import ParallelEnv  # noqa: F401

from . import cloud_utils  # noqa: F401

from .sharding import group_sharded_parallel  # noqa: F401
from .sharding import save_group_sharded_model  # noqa: F401

from . import rpc

__all__ = [  # noqa
    "io",
    "spawn",
    "launch",
    "scatter",
    "broadcast",
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
    "rpc",
]
