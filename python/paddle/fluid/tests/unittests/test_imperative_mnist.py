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
from paddle.fluid.imperative.nn import Conv2D


@contextlib.contextmanager
def new_program_scope():
    prog = fluid.Program()
    startup_prog = fluid.Program()
    scope = fluid.core.Scope()
    with fluid.scope_guard(scope):
        with fluid.program_guard(prog, startup_prog):
            yield


class MNIST(fluid.imperative.PyLayer):
    def __init__(self):
        super(MNIST, self).__init__()

        groups = 1
        dilation = [1, 1]
        pad = [0, 0]
        stride = [1, 1]
        input_size = [2, 3, 5, 5]  # NCHW
        assert np.mod(input_size[1], groups) == 0
        f_c = input_size[1] // groups
        filter_size = [6, f_c, 3, 3]

        self._conv2d = Conv2D(
            num_channels=3,
            num_filters=20,
            filter_size=3,
            stride=stride,
            padding=pad,
            dilation=dilation,
            groups=groups,
            use_cudnn=False)

    def forward(self, inputs):
        x = self._conv2d(inputs)
        return x


class TestImperativeMnist(unittest.TestCase):
    #  def test_layer(self):
    #  with fluid.imperative.guard():
    #  cl = core.Layer()
    #  cl.forward([])
    #  l = fluid.imperative.PyLayer()
    #  l.forward([])

    #  def test_layer_in_out(self):
    #  np_inp = np.array([1.0, 2.0, -1.0], dtype=np.float32)
    #  with fluid.imperative.guard():
    #  l = MyLayer()
    #  x = l(np_inp)[0]
    #  self.assertIsNotNone(x)
    #  dy_out = x._numpy()
    #  x._backward()
    #  dy_grad = l._x_for_debug._gradient()

    #  with new_program_scope():
    #  inp = fluid.layers.data(
    #  name="inp", shape=[3], append_batch_size=False)
    #  l = MyLayer()
    #  x = l(inp)[0]
    #  param_grads = fluid.backward.append_backward(
    #  x, parameter_list=[l._x_for_debug.name])[0]
    #  exe = fluid.Executor(fluid.CPUPlace())

    #  static_out, static_grad = exe.run(
    #  feed={inp.name: np_inp},
    #  fetch_list=[x.name, param_grads[1].name])

    #  self.assertTrue(np.allclose(dy_out, static_out))
    #  self.assertTrue(np.allclose(dy_grad, static_grad))

    def test_mnist_cpu_float32(self):
        with fluid.imperative.guard():
            mnist = MNIST()

            data = np.random.rand(2, 3, 5, 5).astype('float32')
            mnist(data)
        #  np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        #  with fluid.imperative.guard():
        #  mlp = MLP()
        #  out = mlp(np_inp)
        #  dy_out = out._numpy()
        #  out._backward()
        #  dy_grad = mlp._fc1._w._gradient()

        #  with new_program_scope():
        #  inp = fluid.layers.data(
        #  name="inp", shape=[2, 2], append_batch_size=False)
        #  mlp = MLP()
        #  out = mlp(inp)
        #  param_grads = fluid.backward.append_backward(
        #  out, parameter_list=[mlp._fc1._w.name])[0]
        #  exe = fluid.Executor(fluid.CPUPlace())
        #  exe.run(fluid.default_startup_program())

        #  static_out, static_grad = exe.run(
        #  feed={inp.name: np_inp},
        #  fetch_list=[out.name, param_grads[1].name])

        #  self.assertTrue(np.allclose(dy_out, static_out))
        #  self.assertTrue(np.allclose(dy_grad, static_grad))


if __name__ == '__main__':
    unittest.main()
