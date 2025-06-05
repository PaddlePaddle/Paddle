#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

# TODO: define distributed api under this directory,
from . import metrics  # noqa: F401
from .base.distributed_strategy import DistributedStrategy
from .base.role_maker import (
    PaddleCloudRoleMaker,
    Role,
    UserDefinedRoleMaker,
)
from .base.topology import CommunicateTopology, HybridCommunicateGroup
from .base.util_factory import UtilBase
from .data_generator.data_generator import (
    MultiSlotDataGenerator,
    MultiSlotStringDataGenerator,
)
from .dataset import (  # noqa: F401
    BoxPSDataset,
    DatasetBase,
    FileInstantDataset,
    InMemoryDataset,
    QueueDataset,
)
from .fleet import Fleet
from .model import distributed_model
from .optimizer import distributed_optimizer
from .scaler import distributed_scaler
from .utils import log_util

__all__ = [
    "CommunicateTopology",
    "UtilBase",
    "HybridCommunicateGroup",
    "MultiSlotStringDataGenerator",
    "UserDefinedRoleMaker",
    "DistributedStrategy",
    "Role",
    "MultiSlotDataGenerator",
    "PaddleCloudRoleMaker",
    "Fleet",
]

fleet = Fleet()
_final_strategy = fleet._final_strategy
_get_applied_meta_list = fleet._get_applied_meta_list
_get_applied_graph_list = fleet._get_applied_graph_list
init = fleet.init
is_first_worker = fleet.is_first_worker
worker_index = fleet.worker_index
worker_num = fleet.worker_num
node_num = fleet.node_num
rank = fleet.worker_index
nranks = fleet.worker_num
world_size = fleet.worker_num
# device id in current trainer
local_device_ids = fleet.local_device_ids
# device ids in world
world_device_ids = fleet.world_device_ids
# rank in node
local_rank = fleet.local_rank
rank_in_node = local_rank
is_worker = fleet.is_worker
is_coordinator = fleet.is_coordinator
init_coordinator = fleet.init_coordinator
make_fl_strategy = fleet.make_fl_strategy
get_fl_client = fleet.get_fl_client
worker_endpoints = fleet.worker_endpoints
server_num = fleet.server_num
server_index = fleet.server_index
server_endpoints = fleet.server_endpoints
is_server = fleet.is_server
util = UtilBase()
barrier_worker = fleet.barrier_worker
all_reduce = fleet.all_reduce
init_worker = fleet.init_worker
init_server = fleet.init_server
run_server = fleet.run_server
stop_worker = fleet.stop_worker
distributed_optimizer = distributed_optimizer
save_inference_model = fleet.save_inference_model
save_persistables = fleet.save_persistables
save_cache_model = fleet.save_cache_model
check_save_pre_patch_done = fleet.check_save_pre_patch_done
save_one_table = fleet.save_one_table
save_dense_params = fleet.save_dense_params
load_model = fleet.load_model
load_inference_model = fleet.load_inference_model
load_one_table = fleet.load_one_table
set_date = fleet.set_date
print_table_stat = fleet.print_table_stat
minimize = fleet.minimize
distributed_model = distributed_model
shrink = fleet.shrink
get_hybrid_communicate_group = fleet.get_hybrid_communicate_group
distributed_scaler = distributed_scaler
set_log_level = log_util.set_log_level
get_log_level_code = log_util.get_log_level_code
get_log_level_name = log_util.get_log_level_name
check_memory_usage = log_util.check_memory_usage
save_cache_table = fleet.save_cache_table
collective_perf = fleet.collective_perf
from .. import auto_parallel as auto  # noqa: F401
