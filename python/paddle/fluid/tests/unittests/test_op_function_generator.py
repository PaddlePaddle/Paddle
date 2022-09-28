#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.fluid.framework import default_main_program, Program, convert_np_dtype_to_dtype_, _non_static_mode, in_dygraph_mode
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.core as core
from paddle.fluid.dygraph.jit import TracedLayer
import numpy as np
from paddle import _C_ops, _legacy_C_ops


class TestTracedLayer(fluid.dygraph.Layer):

    def __init__(self, name_scope):
        super(TestTracedLayer, self).__init__(name_scope)

    def forward(self, input):
        return _legacy_C_ops.relu(input)


class TestVariable(unittest.TestCase):

    def setUp(self):
        self.shape = [512, 768]
        self.dtype = np.float32
        self.array = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)

    def test_elementwise_add(self):
        with fluid.dygraph.guard():
            a = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
            b = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
            x = fluid.dygraph.to_variable(a)
            y = fluid.dygraph.to_variable(b)
            x.stop_gradient = False

            res1 = layers.elementwise_add(x, y)
            res2 = _legacy_C_ops.elementwise_add(x, y)

            np.testing.assert_array_equal(res1.numpy(), res2.numpy())

    def test_elementwise_mul(self):
        with fluid.dygraph.guard():
            a = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
            b = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
            x = fluid.dygraph.to_variable(a)
            y = fluid.dygraph.to_variable(b)

            res1 = layers.elementwise_mul(x, y)
            res2 = _legacy_C_ops.elementwise_mul(x, y)

            np.testing.assert_array_equal(res1.numpy(), res2.numpy())

    def test_relu(self):
        with fluid.dygraph.guard():
            a = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
            x = fluid.dygraph.to_variable(a)

            res1 = layers.relu(x)
            res2 = _legacy_C_ops.relu(x)

            np.testing.assert_array_equal(res1.numpy(), res2.numpy())

    def test_trace_backward(self):
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
        with fluid.dygraph.guard():
            a = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
            b = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
            x = fluid.dygraph.to_variable(a)
            y = fluid.dygraph.to_variable(b)
            x.stop_gradient = False
            y.stop_gradient = False

            loss = _legacy_C_ops.elementwise_mul(x, y)

            loss.backward()
            x_grad = x.gradient()
            y_grad = y.gradient()

            np.testing.assert_array_equal(x_grad, loss.gradient() * b)
            np.testing.assert_array_equal(y_grad, loss.gradient() * a)
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": False})

    def test_traced_layer(self):
        if in_dygraph_mode():
            return
        with fluid.dygraph.guard():
            layer = TestTracedLayer("test_traced_layer")
            a = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
            x = fluid.dygraph.to_variable(a)
            res_dygraph, static_layer = TracedLayer.trace(
                layer, inputs=x)  # dygraph out
            res_static_graph = static_layer([x])[0]

            np.testing.assert_array_equal(res_dygraph.numpy(), res_static_graph)


if __name__ == '__main__':
    unittest.main()
