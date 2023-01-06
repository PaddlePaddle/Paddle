# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os
import unittest

import paddle
import paddle.distributed.fleet as fleet
import paddle.distributed.fleet.base.role_maker as role_maker

paddle.enable_static()


class TestFleetBase_1(unittest.TestCase):
    def setUp(self):
        os.environ["POD_IP"] = "127.0.0.1"
        os.environ["PADDLE_TRAINER_ENDPOINTS"] = "127.0.0.1:36001"
        os.environ["PADDLE_TRAINERS_NUM"] = "2"
        os.environ[
            "PADDLE_PSERVERS_IP_PORT_LIST"
        ] = "127.0.0.1:36001,127.0.0.2:36001"

    def test_collective_minimize(self):
        input_x = paddle.fluid.layers.data(
            name="x", shape=[32], dtype='float32'
        )
        input_y = paddle.fluid.layers.data(name="y", shape=[1], dtype='int64')

        fc_1 = paddle.fluid.layers.fc(input=input_x, size=64, act='tanh')
        fc_2 = paddle.fluid.layers.fc(input=fc_1, size=64, act='tanh')
        prediction = paddle.fluid.layers.fc(input=[fc_2], size=2, act='softmax')
        cost = paddle.nn.functional.cross_entropy(
            input=prediction, label=input_y, reduction='none', use_softmax=False
        )
        avg_cost = paddle.mean(x=cost)

        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        strategy = fleet.DistributedStrategy()
        optimizer = paddle.fluid.optimizer.SGD(learning_rate=0.001)
        optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
        optimizer.minimize(avg_cost)


class TestFleetBase(unittest.TestCase):
    def setUp(self):
        os.environ["POD_IP"] = "127.0.0.1"
        os.environ["PADDLE_TRAINER_ENDPOINTS"] = "127.0.0.1:36001"
        os.environ["PADDLE_TRAINERS_NUM"] = "2"
        os.environ[
            "PADDLE_PSERVERS_IP_PORT_LIST"
        ] = "127.0.0.1:36001,127.0.0.2:36001"

    def test_fleet_get_applied_optimizer(self):
        input_x = paddle.fluid.layers.data(
            name="x", shape=[32], dtype='float32'
        )
        input_y = paddle.fluid.layers.data(name="y", shape=[1], dtype='int64')

        fc_1 = paddle.fluid.layers.fc(input=input_x, size=64, act='tanh')
        fc_2 = paddle.fluid.layers.fc(input=fc_1, size=64, act='tanh')
        prediction = paddle.fluid.layers.fc(input=[fc_2], size=2, act='softmax')
        cost = paddle.nn.functional.cross_entropy(
            input=prediction, label=input_y, reduction='none', use_softmax=False
        )
        avg_cost = paddle.mean(x=cost)

        fleet.init(is_collective=True)

        meta_list = fleet._get_applied_meta_list()
        graph_list = fleet._get_applied_graph_list()
        # not called minimize function
        self.assertEqual(len(meta_list), 0)
        self.assertEqual(len(graph_list), 0)

        strategy = fleet.DistributedStrategy()
        optimizer = paddle.fluid.optimizer.SGD(learning_rate=0.001)
        optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
        optimizer.minimize(avg_cost)

        meta_list = fleet._get_applied_meta_list()
        graph_list = fleet._get_applied_graph_list()
        self.assertEqual(len(meta_list), 0)
        self.assertEqual(len(graph_list), 1)


if __name__ == "__main__":
    unittest.main()
