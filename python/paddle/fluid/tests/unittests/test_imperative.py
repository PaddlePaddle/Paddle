# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import contextlib
import unittest
import numpy as np

import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid.imperative.nn import FC
from test_imperative_base import new_program_scope


class MyLayer(fluid.imperative.PyLayer):
    def __init__(self):
        super(MyLayer, self).__init__()

    def forward(self, inputs):
        x = fluid.layers.relu(inputs)
        self._x_for_debug = x
        x = fluid.layers.elementwise_mul(x, x)
        x = fluid.layers.reduce_sum(x)
        return [x]


class MLP(fluid.imperative.PyLayer):
    def __init__(self):
        super(MLP, self).__init__()
        self._fc1 = FC(3,
                       fluid.ParamAttr(
                           initializer=fluid.initializer.Constant(value=0.1)))
        self._fc2 = FC(4,
                       fluid.ParamAttr(
                           initializer=fluid.initializer.Constant(value=0.1)))

    def forward(self, inputs):
        x = self._fc1(inputs)
        x = self._fc2(x)
        x = fluid.layers.reduce_sum(x)
        return x


class TestImperative(unittest.TestCase):
    def test_layer(self):
        with fluid.imperative.guard():
            cl = core.Layer()
            cl.forward([])
            l = fluid.imperative.PyLayer()
            self.assertRaises(NotImplementedError, l.forward, [])

    def test_layer_in_out(self):
        np_inp = np.array([1.0, 2.0, -1.0], dtype=np.float32)
        with fluid.imperative.guard():
            var_inp = fluid.imperative.base.to_variable(np_inp)
            l = MyLayer()
            x = l(var_inp)[0]
            self.assertIsNotNone(x)
            dy_out = x._numpy()
            x._backward()
            dy_grad = l._x_for_debug._gradient()

        with new_program_scope():
            inp = fluid.layers.data(
                name="inp", shape=[3], append_batch_size=False)
            l = MyLayer()
            x = l(inp)[0]
            param_grads = fluid.backward.append_backward(
                x, parameter_list=[l._x_for_debug.name])[0]
            exe = fluid.Executor(fluid.CPUPlace())

            static_out, static_grad = exe.run(
                feed={inp.name: np_inp},
                fetch_list=[x.name, param_grads[1].name])

        self.assertTrue(np.allclose(dy_out, static_out))
        self.assertTrue(np.allclose(dy_grad, static_grad))

    def test_mlp(self):
        np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        with fluid.imperative.guard():
            var_inp = fluid.imperative.base.to_variable(np_inp)
            mlp = MLP()
            out = mlp(var_inp)
            dy_out = out._numpy()
            out._backward()
            dy_grad = mlp._fc1._w._gradient()

        with new_program_scope():
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            mlp = MLP()
            out = mlp(inp)
            param_grads = fluid.backward.append_backward(
                out, parameter_list=[mlp._fc1._w.name])[0]
            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(fluid.default_startup_program())

            static_out, static_grad = exe.run(
                feed={inp.name: np_inp},
                fetch_list=[out.name, param_grads[1].name])

        self.assertTrue(np.allclose(dy_out, static_out))
        self.assertTrue(np.allclose(dy_grad, static_grad))


if __name__ == '__main__':
    unittest.main()
