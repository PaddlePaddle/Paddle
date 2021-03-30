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

from .spawn import spawn  # noqa: F401

from .parallel import init_parallel_env  # noqa: F401
from .parallel import get_rank  # noqa: F401
from .parallel import get_world_size  # noqa: F401

from paddle.distributed.fleet.dataset import InMemoryDataset  # noqa: F401
from paddle.distributed.fleet.dataset import QueueDataset  # noqa: F401

from .collective import broadcast  # noqa: F401
from .collective import all_reduce  # noqa: F401
from .collective import reduce  # noqa: F401
from .collective import all_gather  # noqa: F401
from .collective import scatter  # noqa: F401
from .collective import barrier  # noqa: F401
from .collective import ReduceOp  # noqa: F401
from .collective import split  # noqa: F401
from .fleet import BoxPSDataset  # noqa: F401

from .entry_attr import ProbabilityEntry  # noqa: F401
from .entry_attr import CountFilterEntry  # noqa: F401
#TODO: remove ParallelEnv
from paddle.fluid.dygraph.parallel import ParallelEnv  # noqa: F401

from . import cloud_utils  # noqa: F401
from . import utils  # noqa: F401

__all__ = [     #noqa
           'spawn',
           'init_parallel_env',
           'get_rank',
           'get_world_size',
           'InMemoryDataset',
           'QueueDataset',
           'broadcast',
           'all_reduce',
           'reduce',
           'all_gather',
           'scatter',
           'barrier',
           'ReduceOp',
           'BoxPSDataset'
]
