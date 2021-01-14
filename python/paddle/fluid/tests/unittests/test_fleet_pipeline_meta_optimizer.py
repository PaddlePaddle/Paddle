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
import os

paddle.enable_static()


class TestFleetMetaOptimizer(unittest.TestCase):
    def setUp(self):
        os.environ["PADDLE_TRAINER_ID"] = "1"
        os.environ[
            "PADDLE_TRAINER_ENDPOINTS"] = "127.0.0.1:36001,127.0.0.1:36002"

    def test_pipeline_optimizer(self):
        import paddle.distributed.fleet as fleet
        import paddle.distributed.fleet.base.role_maker as role_maker
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        with paddle.fluid.device_guard("gpu:0"):
            input_x = paddle.fluid.layers.data(
                name="x", shape=[32], dtype='float32')
            input_y = paddle.fluid.layers.data(
                name="y", shape=[1], dtype='int64')
            fc_1 = paddle.fluid.layers.fc(input=input_x, size=64, act='tanh')

        with paddle.fluid.device_guard("gpu:1"):
            fc_2 = paddle.fluid.layers.fc(input=fc_1, size=64, act='tanh')
            prediction = paddle.fluid.layers.fc(input=[fc_2],
                                                size=2,
                                                act='softmax')
            cost = paddle.fluid.layers.cross_entropy(
                input=prediction, label=input_y)
            avg_cost = paddle.fluid.layers.mean(x=cost)

        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.pipeline = True
        strategy.pipeline_configs = {'micro_batch': 2}

        optimizer = paddle.fluid.optimizer.Adam(0.01)
        optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
        optimizer.minimize(avg_cost)


if __name__ == "__main__":
    unittest.main()
