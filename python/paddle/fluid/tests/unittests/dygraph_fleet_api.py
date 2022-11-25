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
import random
import numpy as np

import paddle
from paddle.fluid.framework import _test_eager_guard


class TestDygraphFleetAPI(unittest.TestCase):

    def setUp(self):
        paddle.seed(2022)
        random.seed(2022)
        np.random.seed(2022)
        self.config()

    def config(self):
        self.dtype = "float32"
        self.shape = (2, 10, 5)

    def test_dygraph_fleet_api(self):
        import paddle.distributed.fleet as fleet
        import paddle.distributed as dist

        strategy = fleet.DistributedStrategy()
        strategy.amp = True
        strategy.recompute = True
        fleet.init(is_collective=True, strategy=strategy)
<<<<<<< HEAD
        net = paddle.nn.Sequential(
            paddle.nn.Linear(10, 1), paddle.nn.Linear(1, 2)
        )
=======
        net = paddle.nn.Sequential(paddle.nn.Linear(10, 1),
                                   paddle.nn.Linear(1, 2))
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
        net = dist.fleet.distributed_model(net)
        data = np.random.uniform(-1, 1, [30, 10]).astype('float32')
        data = paddle.to_tensor(data)
        net(data)


if __name__ == "__main__":
    with _test_eager_guard():
        pass
    unittest.main()
