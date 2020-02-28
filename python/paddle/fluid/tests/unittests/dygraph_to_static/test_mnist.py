#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from time import time
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.optimizer import AdamOptimizer
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear

from paddle.fluid.dygraph.jit import dygraph_to_static_graph

import unittest


class SimpleImgConvPool(fluid.dygraph.Layer):
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
            act=act,
            use_cudnn=use_cudnn)

        self._pool2d = Pool2D(
            pool_size=pool_size,
            pool_type=pool_type,
            pool_stride=pool_stride,
            pool_padding=pool_padding,
            global_pooling=global_pooling,
            use_cudnn=use_cudnn)

    @dygraph_to_static_graph
    def forward(self, inputs):
        x = self._conv2d(inputs)
        x = self._pool2d(x)
        return x


class MNIST(fluid.dygraph.Layer):
    def __init__(self):
        super(MNIST, self).__init__()

        self._simple_img_conv_pool_1 = SimpleImgConvPool(
            1, 20, 5, 2, 2, act="relu")

        self._simple_img_conv_pool_2 = SimpleImgConvPool(
            20, 50, 5, 2, 2, act="relu")

        self.pool_2_shape = 50 * 4 * 4
        SIZE = 10
        scale = (2.0 / (self.pool_2_shape**2 * SIZE))**0.5
        self._fc = Linear(
            self.pool_2_shape,
            10,
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.NormalInitializer(
                    loc=0.0, scale=scale)),
            act="softmax")

    @dygraph_to_static_graph
    def forward(self, inputs, label=None):
        x = self.inference(inputs)
        if label is not None:
            acc = fluid.layers.accuracy(input=x, label=label)
            loss = fluid.layers.cross_entropy(x, label)
            avg_loss = fluid.layers.mean(loss)
            return x, acc, avg_loss
        else:
            return x

    @dygraph_to_static_graph
    def inference(self, inputs):
        x = self._simple_img_conv_pool_1(inputs)
        x = self._simple_img_conv_pool_2(x)
        x = fluid.layers.reshape(x, shape=[-1, self.pool_2_shape])
        x = self._fc(x)
        return x


class TestMNIST(unittest.TestCase):
    def setUp(self):
        self.epoch_num = 1
        self.batch_size = 64
        self.place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        self.train_reader = paddle.batch(
            paddle.dataset.mnist.train(),
            batch_size=self.batch_size,
            drop_last=True)


class TestMNISTWithStaticMode(TestMNIST):
    """
    Tests model when using `dygraph_to_static_graph` to convert dygraph into static
    model. It allows user to add customized code to train static model, such as `with`
    and `Executor` statement.
    """

    def test_train(self):

        main_prog = fluid.Program()
        with fluid.program_guard(main_prog):
            mnist = MNIST()
            adam = AdamOptimizer(
                learning_rate=0.001, parameter_list=mnist.parameters())

            exe = fluid.Executor(self.place)
            start = time()

            img = fluid.data(
                name='img', shape=[None, 1, 28, 28], dtype='float32')
            label = fluid.data(name='label', shape=[None, 1], dtype='int64')
            label.stop_gradient = True

            prediction, acc, avg_loss = mnist(img, label)
            adam.minimize(avg_loss)
        exe.run(fluid.default_startup_program())

        for epoch in range(self.epoch_num):
            for batch_id, data in enumerate(self.train_reader()):
                dy_x_data = np.array([x[0].reshape(1, 28, 28)
                                      for x in data]).astype('float32')
                y_data = np.array(
                    [x[1] for x in data]).astype('int64').reshape(-1, 1)

                out = exe.run(main_prog,
                              fetch_list=[avg_loss, acc],
                              feed={'img': dy_x_data,
                                    'label': y_data})
                if batch_id % 100 == 0:
                    print(
                        "Loss at epoch {} step {}: loss: {:}, acc: {}, cost: {}"
                        .format(epoch, batch_id,
                                np.array(out[0]),
                                np.array(out[1]), time() - start))
                    if batch_id == 300:
                        # The accuracy of mnist should converge over 0.9 after 300 batch.
                        accuracy = np.array(out[1])
                        self.assertGreater(
                            accuracy,
                            0.9,
                            msg="The accuracy {} of mnist should converge over 0.9 after 300 batch."
                            .format(accuracy))
                        break


# TODO: TestCase with cached program is required when building program in `for` loop.

if __name__ == "__main__":
    unittest.main()
