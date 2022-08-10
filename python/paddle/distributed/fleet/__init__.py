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
from .base.role_maker import Role  # noqa: F401
from .base.role_maker import UserDefinedRoleMaker  # noqa: F401
from .base.role_maker import PaddleCloudRoleMaker  # noqa: F401
from .base.distributed_strategy import DistributedStrategy  # noqa: F401
from .base.util_factory import UtilBase  # noqa: F401
from .dataset import DatasetBase  # noqa: F401
from .dataset import InMemoryDataset  # noqa: F401
from .dataset import QueueDataset  # noqa: F401
from .dataset import FileInstantDataset  # noqa: F401
from .dataset import BoxPSDataset  # noqa: F401
from .data_generator.data_generator import MultiSlotDataGenerator  # noqa: F401
from .data_generator.data_generator import MultiSlotStringDataGenerator  # noqa: F401
from . import metrics  # noqa: F401
from .base.topology import CommunicateTopology
from .base.topology import HybridCommunicateGroup  # noqa: F401
from .fleet import Fleet
from .model import distributed_model
from .optimizer import distributed_optimizer
from .scaler import distributed_scaler

__all__ = [  #noqa
    "CommunicateTopology", "UtilBase", "HybridCommunicateGroup",
    "MultiSlotStringDataGenerator", "UserDefinedRoleMaker",
    "DistributedStrategy", "Role", "MultiSlotDataGenerator",
    "PaddleCloudRoleMaker", "Fleet"
]

Fleet = Fleet()
_final_strategy = Fleet._final_strategy
_get_applied_meta_list = Fleet._get_applied_meta_list
_get_applied_graph_list = Fleet._get_applied_graph_list
init = Fleet.init
is_first_worker = Fleet.is_first_worker
worker_index = Fleet.worker_index
worker_num = Fleet.worker_num
node_num = Fleet.node_num
rank = Fleet.worker_index
nranks = Fleet.worker_num
world_size = Fleet.worker_num
# device id in current trainer
local_device_ids = Fleet.local_device_ids
# device ids in world
world_device_ids = Fleet.world_device_ids
# rank in node
local_rank = Fleet.local_rank
rank_in_node = local_rank
is_worker = Fleet.is_worker
is_coordinator = Fleet.is_coordinator
init_coordinator = Fleet.init_coordinator
make_fl_strategy = Fleet.make_fl_strategy
get_fl_client = Fleet.get_fl_client
worker_endpoints = Fleet.worker_endpoints
server_num = Fleet.server_num
server_index = Fleet.server_index
server_endpoints = Fleet.server_endpoints
is_server = Fleet.is_server
util = UtilBase()
barrier_worker = Fleet.barrier_worker
init_worker = Fleet.init_worker
init_server = Fleet.init_server
run_server = Fleet.run_server
stop_worker = Fleet.stop_worker
distributed_optimizer = distributed_optimizer
save_inference_model = Fleet.save_inference_model
save_persistables = Fleet.save_persistables
save_cache_model = Fleet.save_cache_model
check_save_pre_patch_done = Fleet.check_save_pre_patch_done
save_one_table = Fleet.save_one_table
save_dense_params = Fleet.save_dense_params
load_model = Fleet.load_model
load_inference_model = Fleet.load_inference_model
load_one_table = Fleet.load_one_table
minimize = Fleet.minimize
distributed_model = distributed_model
step = Fleet.user_defined_optimizer.step
clear_grad = Fleet.user_defined_optimizer.clear_grad
set_lr = Fleet.user_defined_optimizer.set_lr
get_lr = Fleet.user_defined_optimizer.get_lr
state_dict = Fleet.user_defined_optimizer.state_dict
set_state_dict = Fleet.user_defined_optimizer.set_state_dict
shrink = Fleet.shrink
get_hybrid_communicate_group = Fleet.get_hybrid_communicate_group
distributed_scaler = distributed_scaler
