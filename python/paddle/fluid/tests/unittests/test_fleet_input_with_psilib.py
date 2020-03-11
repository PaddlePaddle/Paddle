#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
from six.moves import reduce

import paddle.fluid as fluid
from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig
from paddle.fluid.incubate.fleet.base.role_maker import UserDefinedRoleMaker
from paddle.fluid.incubate.fleet.base.role_maker import UserDefinedCollectiveRoleMaker
from paddle.fluid.incubate.fleet.base.role_maker import Role
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.incubate.fleet.parameter_server.pslib import fleet
from paddle.fluid.incubate.fleet.parameter_server.distributed_strategy import StrategyFactory
from dist_simnet_bow import train_network


class FleetPSLibTest(unittest.TestCase):
    def test_transpile(self):
        role = role_maker.UserDefinedRoleMaker(
            current_id=0,
            role=role_maker.Role.SERVER,
            worker_num=2,
            server_endpoints=["127.0.0.1:36011", "127.0.0.1:36012"])

        optimizer = fluid.optimizer.SGD(0.1)
        # case5
        self.assertRaises(Exception, fleet.distributed_optimizer, optimizer,
                          "Adam")
        fleet.init(role)

        avg_cost, _, _ = train_network(128, False, True)

        # case1
        strategy = StrategyFactory.create_async_strategy()
        fleet.distributed_optimizer(optimizer, strategy)

        # case2
        strategy = {}
        fleet.distributed_optimizer(optimizer, strategy)

        # case3
        self.assertRaises(Exception, fleet.distributed_optimizer, optimizer,
                          "Adam")

        # case4
        self.assertRaises(Exception, fleet.distributed_optimizer, "Adam",
                          "Adam")


if __name__ == '__main__':
    unittest.main()
