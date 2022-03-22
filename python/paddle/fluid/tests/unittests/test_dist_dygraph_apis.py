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
import os
import paddle
import paddle.nn as nn


class SimpleNet(nn.Layer):
    def __init__(self, in_size, out_size):
        super(SimpleNet, self).__init__()
        self.linear = nn.Linear(in_size, out_size)
        self.softmax = nn.Softmax(axis=-1)

    def forward(self, input):
        y = self.linear(input)
        pred = self.softmax(y)
        return pred


class TestDygraphFleetApis(unittest.TestCase):
    def setUp(self):
        os.environ["PADDLE_TRAINER_ID"] = "1"
        os.environ[
            "PADDLE_TRAINER_ENDPOINTS"] = "127.0.0.1:36001,127.0.0.1:36002"

    def test_pipeline_optimizer(self):
        import paddle.distributed.fleet as fleet
        strategy = fleet.DistributedStrategy()
        strategy.amp = True
        strategy.recompute = True
        fleet.init(is_collective=True, strategy=strategy)
        net = SimpleNet(8, 8)
        net = dist.fleet.distributed_model(net)


if __name__ == "__main__":
    unittest.main()
