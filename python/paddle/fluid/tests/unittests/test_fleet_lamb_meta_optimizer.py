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

import unittest
import paddle
from paddle import fluid
import os
import paddle.distributed.fleet as fleet
import paddle.distributed.fleet.base.role_maker as role_maker


class TestFleetLambMetaOptimizer(unittest.TestCase):
    def setUp(self):
        os.environ["POD_IP"] = "127.0.0.1"
        os.environ["PADDLE_TRAINER_ENDPOINTS"] = "127.0.0.1:36001"
        os.environ["PADDLE_TRAINERS_NUM"] = "2"
        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = \
                       "127.0.0.1:36001,127.0.0.2:36001"

    def net(self, main_prog, startup_prog):
        with fluid.program_guard(main_prog, startup_prog):
            with fluid.unique_name.guard():
                input_x = paddle.fluid.layers.data(
                    name="x", shape=[32], dtype='float32')
                input_y = paddle.fluid.layers.data(
                    name="y", shape=[1], dtype='int64')

                fc_1 = paddle.fluid.layers.fc(input=input_x,
                                              size=64,
                                              act='tanh')
                fc_2 = paddle.fluid.layers.fc(input=fc_1, size=256, act='tanh')
                prediction = paddle.fluid.layers.fc(input=[fc_2],
                                                    size=2,
                                                    act='softmax')
                cost = paddle.fluid.layers.cross_entropy(
                    input=prediction, label=input_y)
                avg_cost = paddle.fluid.layers.mean(x=cost)

                strategy = paddle.distributed.fleet.DistributedStrategy()
                strategy.lamb = True
                strategy.lamb_configs = {
                    'lamb_weight_decay': 0.01,
                    'exclude_from_weight_decay': [],
                }

        return avg_cost, strategy

    def test_lamb_optimizer(self):
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        startup_prog = fluid.Program()
        train_prog = fluid.Program()
        avg_cost, strategy = self.net(train_prog, startup_prog)
        optimizer = paddle.fluid.optimizer.Adam(learning_rate=0.01)
        optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
        optimizer.minimize(avg_cost)

        ops = [op.type for op in avg_cost.block.ops]
        self.assertIn('lamb', ops)

    def test_lamb_not_apply_with_momentum(self):
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        startup_prog = fluid.Program()
        train_prog = fluid.Program()
        avg_cost, strategy = self.net(train_prog, startup_prog)
        optimizer = paddle.fluid.optimizer.Momentum(
            learning_rate=0.1, momentum=0.9)
        optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
        optimizer.minimize(avg_cost)

        ops = [op.type for op in avg_cost.block.ops]
        self.assertNotIn('lamb', ops)

    def test_lamb_exclude_fn(self):
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        startup_prog = fluid.Program()
        train_prog = fluid.Program()
        avg_cost, strategy = self.net(train_prog, startup_prog)
        optimizer = paddle.fluid.optimizer.Adam(learning_rate=0.01)
        strategy.lamb_configs = {
            'lamb_weight_decay': 0.01,
            'exclude_from_weight_decay': ['.b_0'],
        }
        optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
        optimizer.minimize(avg_cost)

        ops_with_bias = [
            op for op in avg_cost.block.ops
            if op.type == 'lamb' and op.attr('op_role_var')[0].endswith('.b_0')
        ]
        for op in ops_with_bias:
            self.assertEqual(op.attr('weight_decay'), 0)


if __name__ == "__main__":
    unittest.main()
