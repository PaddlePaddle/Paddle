#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import time
import threading
import numpy

import paddle
import paddle.fluid as fluid
from paddle.fluid.communicator import Communicator
from paddle.fluid.transpiler.distribute_transpiler import DistributedMode

import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig


class TestCommunicator(unittest.TestCase):
    def net(self):
        x = fluid.layers.data(name='x', shape=[13], dtype='float32')
        y_predict = fluid.layers.fc(input=x, size=1, act=None)
        y = fluid.layers.data(name='y', shape=[1], dtype='float32')

        cost = fluid.layers.square_error_cost(input=y_predict, label=y)
        avg_cost = fluid.layers.mean(cost)
        return avg_cost

    def test_communicator_geo(self):
        role = role_maker.UserDefinedRoleMaker(
            current_id=0,
            role=role_maker.Role.WORKER,
            worker_num=2,
            server_endpoints=["127.0.0.1:6001", "127.0.0.1:6002"])

        fleet.init(role)
        avg_cost = self.net()

        optimizer = fluid.optimizer.SGD(0.01)

        strategy = DistributeTranspilerConfig()
        strategy.sync_mode = False
        strategy.runtime_split_send_recv = True
        strategy.geo_sgd_mode = True
        strategy.wait_port = False
        optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(avg_cost)

        fleet.init_worker()
        time.sleep(10)
        fleet.stop_worker()


# class TestCommunicatorGEO(unittest.TestCase):
#     def test_communicator_init_and_start(self):
#         prog = fluid.Program()

#         envs = {}
#         envs["communicator_thread_pool_size"] = "5"
#         envs["communicator_send_wait_times"] = "5"

#         kwargs = {}
#         kwargs["push_vars"] = {}
#         kwargs["trainers"] = 10
#         kwargs["push_nums"] = 10

#         comm = Communicator(prog, DistributedMode.GEO, kwargs, envs)

if __name__ == '__main__':
    unittest.main()
