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
import six

import paddle
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid.optimizer import SGDOptimizer
from paddle.fluid.imperative.nn import Conv2D, Pool2D, FC
from paddle.fluid.imperative.base import to_variable
from test_imperative_base import new_program_scope


class SimpleImgConvPool(fluid.imperative.PyLayer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
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
        super(MNIST, self).__init__()

        self._simple_img_conv_pool_1 = SimpleImgConvPool(
            1, 20, 5, 2, 2, act="relu")

        self._simple_img_conv_pool_2 = SimpleImgConvPool(
            20, 50, 5, 2, 2, act="relu")

        pool_2_shape = 50 * 8 * 8
        SIZE = 10
        scale = (2.0 / (pool_2_shape**2 * SIZE))**0.5
        self._fc = FC(10,
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
        seed = 90

        with fluid.imperative.guard():
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed

            #  mnist = Conv2D(1, 20, 5)
            mnist = MNIST()
            sgd = SGDOptimizer(learning_rate=1e-3)
            train_reader = paddle.batch(
                paddle.dataset.mnist.train(), batch_size=128)

            dy_param_init_value = {}
            for batch_id, data in enumerate(train_reader()):
                if batch_id >= 2:
                    break

                x_data = np.array(
                    [x[0].reshape(1, 28, 28) for x in data]).astype('float32')
                y_data = np.array([x[1] for x in data]).astype('int64').reshape(
                    128, 1)

                img = to_variable(x_data)
                label = to_variable(y_data)
                label._stop_gradient = True

                cost = mnist(img)
                loss = fluid.layers.reduce_mean(cost)
                dy_out = loss._numpy()

                if batch_id == 0:
                    for param in fluid.default_main_program().global_block(
                    ).all_parameters():
                        dy_param_init_value[param.name] = param._numpy()

                loss._backward()
                sgd.minimize(loss)
                dy_param_value = {}
                for param in fluid.default_main_program().global_block(
                ).all_parameters():
                    dy_param_value[param.name] = param._numpy()

        with new_program_scope():
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed

            exe = fluid.Executor(fluid.CPUPlace())

            #  mnist = Conv2D(1, 20, 5)
            mnist = MNIST()
            sgd = SGDOptimizer(learning_rate=1e-3)
            train_reader = paddle.batch(
                paddle.dataset.mnist.train(), batch_size=128)

            img = fluid.layers.data(
                name='pixel', shape=[1, 28, 28], dtype='float32')
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')
            cost = mnist(img)
            loss = fluid.layers.reduce_mean(cost)
            sgd.minimize(loss)

            # initialize params and fetch them
            static_param_init_value = {}
            static_param_name_list = []
            for param in fluid.default_startup_program().global_block(
            ).all_parameters():
                static_param_name_list.append(param.name)

            out = exe.run(fluid.default_startup_program(),
                          fetch_list=static_param_name_list)

            for i in range(len(static_param_name_list)):
                static_param_init_value[static_param_name_list[i]] = out[i]

            for batch_id, data in enumerate(train_reader()):
                if batch_id >= 2:
                    break

                x_data = np.array(
                    [x[0].reshape(1, 28, 28) for x in data]).astype('float32')
                y_data = np.array([x[1] for x in data]).astype('int64').reshape(
                    [128, 1])

                fetch_list = [loss.name]
                fetch_list.extend(static_param_name_list)
                out = exe.run(fluid.default_main_program(),
                              feed={"pixel": x_data,
                                    "label": y_data},
                              fetch_list=fetch_list)

                static_param_value = {}
                static_out = out[0]
                for i in range(1, len(out)):
                    static_param_value[static_param_name_list[i - 1]] = out[i]

        for key, value in six.iteritems(static_param_init_value):
            self.assertTrue(
                np.allclose(value.all(), dy_param_init_value[key].all()))
        self.assertTrue(np.allclose(static_out.all(), dy_out.all()))
        for key, value in six.iteritems(static_param_value):
            self.assertTrue(np.allclose(value.all(), dy_param_value[key].all()))


if __name__ == '__main__':
    unittest.main()
