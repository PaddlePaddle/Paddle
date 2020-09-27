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

paddle.enable_static()


class TestFleetCombineOptimizer(unittest.TestCase):
    def setUp(self):
        os.environ["PADDLE_TRAINER_ID"] = "1"
        os.environ[
            "PADDLE_TRAINER_ENDPOINTS"] = "127.0.0.1:36001,127.0.0.1:36002"

    def net(self, main_prog, startup_prog):
        with fluid.program_guard(main_prog, startup_prog):
            with fluid.unique_name.guard():
                role = role_maker.PaddleCloudRoleMaker(is_collective=True)
                fleet.init(role)
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
                strategy.dgc = True
                strategy.dgc_configs = {
                    "rampup_begin_step": 128,
                    "rampup_step": 100,
                    "sparsity": [0.996, 0.999]
                }
        return avg_cost, strategy

    def optimizer(self, loss, strategy, train_prog, startup_prog):
        with fluid.program_guard(train_prog, startup_prog):
            with fluid.unique_name.guard():
                optimizer = paddle.fluid.optimizer.Momentum(
                    learning_rate=0.01, momentum=0.9)
                optimizer = fleet.distributed_optimizer(
                    optimizer, strategy=strategy)
                optimizer.minimize(loss)

    def set_strategy(self, strategy, name):
        if name == 'amp':
            strategy.amp = True
        elif name == 'dgc':
            strategy.dgc = True
        elif name == 'recompute':
            strategy.recompute = True
            strategy.recompute_configs = {
                "checkpoints": ["fc_0.tmp_2", "fc_1.tmp_2"]
            }

    def test_dgc_recompute_optimizer(self):
        train_prog = fluid.Program()
        startup_prog = fluid.Program()
        avg_cost, strategy = self.net(train_prog, startup_prog)

        self.set_strategy(strategy, 'dgc')
        self.set_strategy(strategy, 'recompute')

        self.optimizer(avg_cost, strategy, train_prog, startup_prog)

        ops = [op.type for op in avg_cost.block.ops]
        outs = [
            op.output('Out')[0] for op in avg_cost.block.ops if op.type == 'mul'
        ]
        self.assertIn('dgc', ops)
        self.assertIn('dgc_momentum', ops)

        self.assertIn('subprog', ''.join(outs))

    def test_amp_recompute_optimizer(self):
        train_prog = fluid.Program()
        startup_prog = fluid.Program()
        avg_cost, strategy = self.net(train_prog, startup_prog)

        self.set_strategy(strategy, 'amp')
        self.set_strategy(strategy, 'recompute')

        self.optimizer(avg_cost, strategy, train_prog, startup_prog)

        ops = [op.type for op in avg_cost.block.ops]
        outs = [
            op.output('Out')[0] for op in avg_cost.block.ops if op.type == 'mul'
        ]
        print(train_prog)
        self.assertIn('cast', ops)
        self.assertIn('check_finite_and_unscale', ops)

        self.assertIn('subprog', ''.join(outs))


if __name__ == "__main__":
    unittest.main()
