#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import copy

from paddle.distributed import fleet
from paddle.framework import in_dynamic_mode

from .meta_optimizers import HeterParallelOptimizer, HybridParallelOptimizer
from .utils.log_util import logger


def _dygraph_distributed_optimizer(optimizer, strategy=None):
    """
    Optimizer for distributed training.
    For the distributed training, this method would rebuild a new instance of DistributedOptimizer.
    Which has basic Optimizer function and special features for distributed training.
    Args:
        optimizer(Optimizer): The executor to run for init server.
        strategy(DistributedStrategy): Extra properties for distributed optimizer.
            It is recommended to use DistributedStrategy in fleet.init(). The strategy
            here is for compatibility. If the strategy in fleet.distributed_optimizer()
            is not None, then it will overwrite the DistributedStrategy in fleet.init(),
            which will take effect in distributed training.
    Returns:
        Fleet: instance of fleet.
    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.distributed.fleet as fleet
            >>> fleet.init(is_collective=True)
            >>> strategy = fleet.DistributedStrategy()
            >>> linear = paddle.nn.Linear(10, 10)
            >>> optimizer = paddle.optimizer.SGD(learning_rate=0.001, parameters=linear.parameters())
            >>> optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)

    """
    fleet_env = fleet.fleet
    fleet_env.user_defined_optimizer = optimizer

    if strategy is not None:
        if fleet_env._is_collective:
            logger.warning(
                "It is recommended to use DistributedStrategy "
                "in fleet_env.init(). The strategy here is only for compatibility. "
                "If the strategy in fleet_env.distributed_optimizer() is "
                "not None, then it will overwrite the DistributedStrategy in fleet_env.init(), "
                "which will take effect in distributed training."
            )
        fleet_env._user_defined_strategy = copy.deepcopy(strategy)

    fleet_env._context = {}

    if fleet_env.worker_num() > 1:
        if not fleet_env._user_defined_strategy.heter_ccl_mode:
            hp_optim = HybridParallelOptimizer(
                optimizer, fleet_env._hcg, fleet_env._user_defined_strategy
            )

            if fleet_env._user_defined_strategy.hybrid_configs[
                "pp_configs"
            ].dp_comm_overlap:
                # grad all-reduce of dp and sep with be fused
                hp_optim._dp_enable = False
                hp_optim._sep_enable = False

            if fleet_env._user_defined_strategy.hybrid_configs[
                "pp_configs"
            ].sharding_comm_overlap:
                hp_optim._sharding_enable = False
                assert (
                    not hp_optim._sep_enable
                ), "sep parallel can not coexist with sharding_comm_overlap"

            return hp_optim
        else:
            return HeterParallelOptimizer(
                optimizer, fleet_env._user_defined_strategy
            )
    else:
        return optimizer


def distributed_optimizer(*args, **kwargs):
    if in_dynamic_mode():
        return _dygraph_distributed_optimizer(*args, **kwargs)
    else:
        return fleet.fleet.distributed_optimizer(*args, **kwargs)
