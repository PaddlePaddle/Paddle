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
from paddle.fluid.framework import _test_eager_guard


class TestImperativePartitialBackward(unittest.TestCase):
    def func_partitial_backward(self):
        with fluid.dygraph.guard():
            x = np.random.randn(2, 4, 5).astype("float32")
            x = fluid.dygraph.to_variable(x)
            linear1 = fluid.dygraph.Linear(5, 10)
            linear2 = fluid.dygraph.Linear(5, 10)

            y = linear1(x[:, :2])
            z = linear2(x[:, 2:])
            loss = fluid.layers.reduce_mean(y)
            loss.backward()

            for param in linear1.parameters():
                self.assertIsNotNone(param._grad_ivar())

            for param in linear2.parameters():
                self.assertIsNone(param._grad_ivar())

            optimizer = fluid.optimizer.AdamOptimizer(parameter_list=(
                linear1.parameters() + linear2.parameters()))
            _, params_grads = optimizer.minimize(loss)

            self.assertListEqual(
                sorted([p.name for p in linear1.parameters()]),
                sorted([p_g[0].name for p_g in params_grads]))

            linear1.clear_gradients()
            linear2.clear_gradients()

    def test_partitial_backward(self):
        with _test_eager_guard():
            self.func_partitial_backward()
        self.func_partitial_backward()


if __name__ == '__main__':
    unittest.main()
