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

import unittest
import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.fluid.optimizer import SGDOptimizer
from paddle.fluid import Conv2D, Pool2D, FC
from paddle.fluid.dygraph.base import to_variable


class SimpleImgConvPool(fluid.Layer):
    def __init__(self,
                 name_scope,
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
        super(SimpleImgConvPool, self).__init__(name_scope)

        self._conv2d = Conv2D(
            self.full_name(),
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
            self.full_name(),
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


class MNIST(fluid.Layer):
    def __init__(self, name_scope):
        super(MNIST, self).__init__(name_scope)

        self._simple_img_conv_pool_1 = SimpleImgConvPool(
            self.full_name(), 1, 20, 5, 2, 2, act="relu")

        self._simple_img_conv_pool_2 = SimpleImgConvPool(
            self.full_name(), 20, 50, 5, 2, 2, act="relu")

        pool_2_shape = 50 * 4 * 4
        SIZE = 10
        scale = (2.0 / (pool_2_shape**2 * SIZE))**0.5
        self._fc = FC(self.full_name(),
                      10,
                      param_attr=fluid.param_attr.ParamAttr(
                          initializer=fluid.initializer.NormalInitializer(
                              loc=0.0, scale=scale)),
                      act="softmax")

    def forward(self, inputs):
        x = self._simple_img_conv_pool_1(inputs)
        x = self._simple_img_conv_pool_2(x)
        x = self._fc(x)
        return x


class TestDygraphCheckpoint(unittest.TestCase):
    def save_load_persistables(self):
        seed = 90
        epoch_num = 1

        with fluid.dygraph.guard():
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed

            mnist = MNIST("mnist")
            sgd = SGDOptimizer(learning_rate=1e-3)
            train_reader = paddle.batch(
                paddle.dataset.mnist.train(), batch_size=128, drop_last=True)

            dy_param_init_value = {}

            step = 0
            for epoch in range(epoch_num):
                for batch_id, data in enumerate(train_reader()):
                    dy_x_data = np.array(
                        [x[0].reshape(1, 28, 28)
                         for x in data]).astype('float32')
                    y_data = np.array(
                        [x[1] for x in data]).astype('int64').reshape(128, 1)

                    img = to_variable(dy_x_data)
                    label = to_variable(y_data)
                    label._stop_gradient = True

                    cost = mnist(img)
                    loss = fluid.layers.cross_entropy(cost, label)
                    avg_loss = fluid.layers.mean(loss)

                    dy_out = avg_loss._numpy()

                    avg_loss._backward()
                    sgd.minimize(avg_loss)
                    fluid.dygraph.save_persistables(mnist, "save_dir")
                    mnist.clear_gradients()

                    for param in mnist.parameters():
                        dy_param_init_value[param.name] = param._numpy()

                    mnist.load_dict(
                        fluid.dygraph.load_persistables(mnist, "save_dir"))

                    restore = mnist.parameters()

                    self.assertEqual(len(dy_param_init_value), len(restore))
                    for value in restore:
                        self.assertTrue(
                            np.allclose(value, dy_param_init_value[value.name]))
                        self.assertTrue(np.isfinite(value.all()))
                        self.assertFalse(np.isnan(value.any()))

                    step += 1

                    if step > 20:
                        break


if __name__ == '__main__':
    unittest.main()
