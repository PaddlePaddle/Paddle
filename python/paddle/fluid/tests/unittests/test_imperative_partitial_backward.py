# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
import numpy as np


class TestImperativePartitialBackward(unittest.TestCase):
    def test_partitial_backward(self):
        with fluid.dygraph.guard():
            x = np.random.randn(2, 4, 5).astype("float32")
            x = fluid.dygraph.to_variable(x)
            fc1 = fluid.dygraph.FC("fc1", 10, num_flatten_dims=2)
            fc2 = fluid.dygraph.FC("fc2", 10, num_flatten_dims=2)

            y = fc1(x[:, :2])
            z = fc2(x[:, 2:])
            loss = fluid.layers.reduce_mean(y)
            loss.backward()

            for param in fc1.parameters():
                self.assertIsNotNone(param._ivar._grad_ivar())

            for param in fc2.parameters():
                self.assertIsNone(param._ivar._grad_ivar())

            optimizer = fluid.optimizer.AdamOptimizer()
            _, params_grads = optimizer.minimize(loss)

            self.assertListEqual(
                sorted([p.name for p in fc1.parameters()]),
                sorted([p_g[0].name for p_g in params_grads]))

            fc1.clear_gradients()
            fc2.clear_gradients()


if __name__ == '__main__':
    unittest.main()
