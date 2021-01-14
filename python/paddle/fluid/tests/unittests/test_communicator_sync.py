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

import os
import paddle
paddle.enable_static()

import paddle.fluid as fluid

import paddle.distributed.fleet.base.role_maker as role_maker
import paddle.distributed.fleet as fleet


class TestCommunicator(unittest.TestCase):
    def net(self):
        x = fluid.layers.data(name='x', shape=[1], dtype='float32')
        y = fluid.layers.data(name='y', shape=[1], dtype='float32')
        cost = fluid.layers.square_error_cost(input=x, label=y)
        avg_cost = fluid.layers.mean(cost)
        return avg_cost

    def test_communicator_sync(self):
        os.environ["TRAINING_ROLE"] = "TRAINER"
        os.environ["PADDLE_PSERVER_NUMS"] = "2"
        os.environ["PADDLE_TRAINERS_NUM"] = "2"
        os.environ["POD_IP"] = "127.0.0.1"
        os.environ["PADDLE_PORT"] = "36001"
        os.environ["PADDLE_TRAINER_ID"] = "0"
        os.environ["PADDLE_TRAINERS_NUM"] = "2"
        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = \
            "127.0.0.1:36001,127.0.0.2:36001"

        fleet.init(role_maker.PaddleCloudRoleMaker())
        avg_cost = self.net()

        optimizer = fluid.optimizer.SGD(0.01)

        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.a_sync = False
        strategy.a_sync_configs = {"launch_barrier": False}

        optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(avg_cost)

        os.environ["TEST_MODE"] = "1"
        fleet.init_worker()
        time.sleep(10)
        fleet.stop_worker()


if __name__ == '__main__':
    unittest.main()
