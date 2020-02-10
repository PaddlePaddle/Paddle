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

import numpy as np
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.core as core
import unittest

from paddle.fluid.dygraph.jit import dygraph_to_static_output

np.random.seed(1)


def dyfunc(a, b):
    with fluid.dygraph.guard():
        x = fluid.dygraph.to_variable(a)
        y = fluid.dygraph.to_variable(b)
        x.stop_gradient = False
        y.stop_gradient = False

        inputs = {'X': [x], 'Y': [y]}
        loss = core.ops.elementwise_mul(inputs)['Out'][0]

        loss.backward()
        x_grad = x.gradient()
        y_grad = y.gradient()
        return x_grad, y_grad


@dygraph_to_static_output
def dyfunc_to_static(a, b):
    return dyfunc(a, b)


class TestBasicModel(unittest.TestCase):
    def test_dygraph_static_same_output(self):
        a = np.random.uniform(
            low=0.1, high=1, size=(3, 4, 5)).astype(np.float32)
        b = np.random.uniform(
            low=0.1, high=1, size=(3, 4, 5)).astype(np.float32)
        dy_output = dyfunc(a, b)
        static_output = dyfunc_to_static(a, b)
        self.assertTrue(np.array_equal(dy_output[0], static_output[0]))
        self.assertTrue(np.array_equal(dy_output[1], static_output[1]))


if __name__ == '__main__':
    unittest.main()
