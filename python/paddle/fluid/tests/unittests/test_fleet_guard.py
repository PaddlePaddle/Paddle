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
import os
import unittest

import paddle.fluid as fluid
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.incubate.fleet.collective import fleet, Collective, DistributedStrategy


class TestCloudRoleMaker(unittest.TestCase):
    def network(self):
        with fluid.unique_name.guard():
            # Change g_program, so the rest layers use `g_program`
            images = self._get_data(name='pixel', shape=[784], dtype='float32')
            label = self._get_data(name='label', shape=[1], dtype='int64')
            hidden1 = layers.fc(input=images, size=128, act='relu')
            hidden2 = layers.fc(input=hidden1, size=64, act='relu')
            predict = layers.fc(input=[hidden2, hidden1],
                                size=10,
                                act='softmax',
                                param_attr=["sftmax.w1", "sftmax.w2"])
            cost = layers.cross_entropy(input=predict, label=label)
            avg_cost = layers.mean(cost)
            return avg_cost

    def get_dist_strategy(self, num_threads):
        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.num_threads = args.num_threads

        dist_strategy = DistributedStrategy()
        dist_strategy.mode = "collective"
        dist_strategy.nccl_comm_num = 2
        dist_strategy.exec_strategy = exec_strategy

        return dist_strategy

    def fleet_optimize(self, cost, num_threads):
        fleet = Collective()
        with fluid.fleet_guard(fleet_a):
            role = role_maker.PaddleCloudRoleMaker(is_collective=True)
            fleet.init(role)  # use fleet_a

            strategy = get_dist_strategy(num_threads=num_threads)

            optimizer = fluid.optimizer.SGD(0.1)
            optimizer = fleet.distributed_optimizer(optimizer, strategy)
            optimizer.minimize(cost)

        return fleet, optimizer

    def test_guard(self):
        start_up = fluid.Program()
        train_a = fluid.Program()
        train_b = fluid.Program()

        fleet_a = None
        opt_a = None
        fleet_b = None
        opt_b = None

        with program_guard(train_a, startup):
            avg_cost = self.network()
            fleet_a, opt_a = fleet_optimize(avg_cost, num_threads=1)

        with program_guard(train_b, startup):
            avg_cost = self.network()
            fleet_b, opt_b = fleet_optimize(avg_cost, num_threads=2)

        self.assertNotEqual(opt_a._strategy.exec_strategy.num_threads == 1)
        self.assertNotEqual(opt_b._strategy.exec_strategy.num_threads == 2)
