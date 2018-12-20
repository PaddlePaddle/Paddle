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
from paddle.fluid.imperative.nn import Conv2D, Pool2D, FC
from paddle.fluid.imperative.base import to_variable


class SimpleImgConvPool(fluid.imperative.PyLayer):
    def __init__(self,
                 num_channels,
                 filter_size,
                 num_filters,
                 pool_size,
                 pool_stride,
                 pool_padding=0,
                 pool_type='max',
                 global_pooling=False,
                 conv_stride=1,
                 conv_padding=0,
                 conv_dilation=1,
                 conv_groups=1,
                 act=None,
                 use_cudnn=False,
                 param_attr=None,
                 bias_attr=None):
        super(SimpleImgConvPool, self).__init__()

        #  groups = 1
        #  dilation = [1, 1]
        #  pad = [0, 0]
        #  stride = [1, 1]
        #  input_size = [2, 3, 5, 5]  # NCHW
        #  assert np.mod(input_size[1], groups) == 0
        #  f_c = input_size[1] // groups
        #  filter_size = [6, f_c, 3, 3]

        self._conv2d = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=conv_stride,
            padding=conv_padding,
            dilation=conv_dilation,
            groups=conv_groups,
            param_attr=None,
            bias_attr=None,
            use_cudnn=use_cudnn)

        self._pool2d = Pool2D(
            pool_size=pool_size,
            pool_type=pool_type,
            pool_stride=pool_stride,
            pool_padding=pool_padding,
            global_pooling=global_pooling,
            use_cudnn=use_cudnn)

    def forward(self, inputs):
        x = self._conv2d(inputs)
        x = self._pool2d(x)
        return x


class MNIST(fluid.imperative.PyLayer):
    def __init__(self, param_attr=None, bias_attr=None):
        super(MNIST, self).__init__(param_attr=param_attr, bias_attr=bias_attr)

        self._simple_img_conv_pool_1 = SimpleImgConvPool(
            1, 5, 20, 2, 2, act="relu")

        self._simple_img_conv_pool_2 = SimpleImgConvPool(
            20, 5, 50, 2, 2, act="relu")

        pool_2_shape = 50 * 8 * 8
        SIZE = 10
        scale = (2.0 / (pool_2_shape**2 * SIZE))**0.5
        self._fc = FC(-1,
                      10,
                      param_attr=fluid.param_attr.ParamAttr(
                          initializer=fluid.initializer.NormalInitializer(
                              loc=0.0, scale=scale)))

    def forward(self, inputs):
        x = self._simple_img_conv_pool_1(inputs)
        x = self._simple_img_conv_pool_2(x)
        x = self._fc(x)
        return x


class TestImperativeMnist(unittest.TestCase):
    def test_mnist_cpu_float32(self):
        with fluid.imperative.guard():
            mnist = MNIST()

            x_data = np.random.rand(128, 1, 28, 28).astype('float32')
            img = to_variable(x_data)
            y_data = np.random.rand(128, 1).astype('int64')
            label = to_variable(y_data)
            label._stop_gradient = True

            predict = mnist(img)
            print(predict.shape, predict.dtype, label.shape, label.dtype)
            out = fluid.layers.cross_entropy(predict, label)
            print(out.shape, out.dtype)
            out._backward()
            filter_grad = mnist._simple_img_conv_pool_1._conv2d._filter_param._gradient(
            )
            print(filter_grad)
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
