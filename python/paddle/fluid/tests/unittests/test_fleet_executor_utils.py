# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import paddle
import paddle.fluid.core as core
from paddle.distributed.fleet.fleet_executor_utils import TaskNode, FleetExecutorUtils

paddle.enable_static()


class TestFleetExecutorUtils(unittest.TestCase):

    def test_construct_program(self):
        # TODO(liyurui): These functions are not ready now.
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.sharding_configs = {
            "dp_degree": 2,
            "mp_degree": 2,
            "pp_degree": 2
        }
        fleet_executor_utils = FleetExecutorUtils(
            dist_strategy=strategy.sharding_configs,
            rank=0,
            nrank=1,
            max_run_times=1)
        op_list = {"lr": [], "fwd": [], "bwd": [], "opt": []}
        program_map = fleet_executor_utils.convert_op_list_to_program(
            op_list, paddle.static.Program())
        task_node_map = fleet_executor_utils.construct_task_nodes_1f1b(
            program_map)


if __name__ == "__main__":
    unittest.main()
