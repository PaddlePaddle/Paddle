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

import paddle.distributed.fleet as fleet
import paddle.distributed.fleet.base.role_maker as role_maker
import unittest
import paddle
import os


class TestFleetAMPOptimizer(unittest.TestCase):
    def setUp(self):
        os.environ["PADDLE_TRAINER_ID"] = "0"
        os.environ["PADDLE_TRAINER_ENDPOINTS"] = "127.0.0.1:36001"

    def test_amp_optimizer(self):
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        input_x = paddle.fluid.layers.data(
            name="x", shape=[32], dtype='float32')
        input_y = paddle.fluid.layers.data(name="y", shape=[1], dtype='int64')

        fc_1 = paddle.fluid.layers.fc(input=input_x, size=64, act='tanh')
        fc_2 = paddle.fluid.layers.fc(input=fc_1, size=64, act='tanh')
        prediction = paddle.fluid.layers.fc(input=[fc_2], size=2, act='softmax')
        cost = paddle.fluid.layers.cross_entropy(
            input=prediction, label=input_y)
        avg_cost = paddle.fluid.layers.mean(x=cost)

        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.amp = True
        strategy.amp_configs = {
            "init_loss_scaling": 32768,
            "decr_every_n_nan_or_inf": 2,
            "incr_every_n_steps": 1000,
            "incr_ratio": 2.0,
            "use_dynamic_loss_scaling": True,
            "decr_ratio": 0.5,
            "custom_white_list": ['softmax'],
            "custom_black_list": ['tanh'],
        }

        optimizer = paddle.fluid.optimizer.SGD(learning_rate=0.01)
        optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
        optimizer.minimize(avg_cost)

        ops = [op.type for op in avg_cost.block.ops]
        self.assertIn('cast', ops)
        self.assertIn('isfinite', ops)


if __name__ == "__main__":
    unittest.main()
