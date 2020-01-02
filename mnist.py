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

import contextlib

import numpy as np

import paddle
from paddle import fluid
from paddle.fluid.optimizer import MomentumOptimizer
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear

from model import Model, shape_hints


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
                 conv_groups=None,
                 act=None,
                 use_cudnn=False,
                 param_attr=None,
                 bias_attr=None):
        super(SimpleImgConvPool, self).__init__('SimpleConv')

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


class MNIST(Model):
    def __init__(self):
        super(MNIST, self).__init__()

        self._simple_img_conv_pool_1 = SimpleImgConvPool(
            1, 20, 5, 2, 2, act="relu")

        self._simple_img_conv_pool_2 = SimpleImgConvPool(
            20, 50, 5, 2, 2, act="relu")

        pool_2_shape = 50 * 4 * 4
        SIZE = 10
        scale = (2.0 / (pool_2_shape**2 * SIZE))**0.5
        self._fc = Linear(800,
                          10,
                          param_attr=fluid.param_attr.ParamAttr(
                              initializer=fluid.initializer.NormalInitializer(
                                  loc=0.0, scale=scale)),
                          act="softmax")

    @shape_hints(inputs=[None, 1, 28, 28])
    def forward(self, inputs):
        if self.mode == 'test':  # XXX demo purpose
            x = self._simple_img_conv_pool_1(inputs)
            x = self._simple_img_conv_pool_2(x)
            x = fluid.layers.flatten(x, axis=1)
            x = self._fc(x)
        else:
            x = self._simple_img_conv_pool_1(inputs)
            x = self._simple_img_conv_pool_2(x)
            x = fluid.layers.flatten(x, axis=1)
            x = self._fc(x)
        return x


@contextlib.contextmanager
def null_guard():
    yield


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--dynamic':
        guard = fluid.dygraph.guard()
    else:
        guard = null_guard()

    with guard:
        train_loader = fluid.io.xmap_readers(
            lambda b: [np.array([x[0] for x in b]).reshape(-1, 1, 28, 28),
                       np.array([x[1] for x in b]).reshape(-1, 1)],
            paddle.batch(paddle.dataset.mnist.train(),
                         batch_size=4, drop_last=True), 1, 1)
        test_loader = fluid.io.xmap_readers(
            lambda b: [np.array([x[0] for x in b]).reshape(-1, 1, 28, 28),
                       np.array([x[1] for x in b]).reshape(-1, 1)],
            paddle.batch(paddle.dataset.mnist.test(),
                         batch_size=4, drop_last=True), 1, 1)
        model = MNIST()
        sgd = MomentumOptimizer(learning_rate=1e-3, momentum=0.9,
                                parameter_list=model.parameters())
        # sgd = SGDOptimizer(learning_rate=1e-3)
        model.prepare(sgd, 'cross_entropy')

        for e in range(2):
            for idx, batch in enumerate(train_loader()):
                out, loss = model.train(batch[0], batch[1], device='gpu',
                                        device_ids=[0, 1, 2, 3])
                print("=============== output =========")
                print(out)
                print("=============== loss ===========")
                print(loss)
                if idx > 10:
                    model.save("test.{}".format(e))
                    break
            print("==== switch to test mode =====")
            for idx, batch in enumerate(test_loader()):
                out = model.test(batch[0], device='gpu',
                                 device_ids=[0, 1, 2, 3])
                print(out)
                if idx > 10:
                    break

        model.load("test.1")
