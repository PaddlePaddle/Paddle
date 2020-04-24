# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import division
from __future__ import print_function

import unittest

import os

import numpy as np
import contextlib

import paddle
from paddle import fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear
from paddle.io import BatchSampler, DataLoader

from hapi.model import Model, Input, Loss, set_device
from hapi.loss import CrossEntropy
from hapi.metrics import Accuracy
from hapi.callbacks import ProgBarLogger
from hapi.datasets import MNIST as MnistDataset


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
        self._fc = Linear(
            800,
            10,
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.NormalInitializer(
                    loc=0.0, scale=scale)),
            act="softmax")

    def forward(self, inputs):
        inputs = fluid.layers.reshape(inputs, [-1, 1, 28, 28])
        x = self._simple_img_conv_pool_1(inputs)
        x = self._simple_img_conv_pool_2(x)
        x = fluid.layers.flatten(x, axis=1)
        x = self._fc(x)
        return x


class TestMnistDataset(MnistDataset):
    def __init__(self):
        super(TestMnistDataset, self).__init__(mode='test')

    def __getitem__(self, idx):
        return self.images[idx],

    def __len__(self):
        return len(self.images)


def get_predict_accuracy(pred, gt):
    pred = np.argmax(pred, -1)
    gt = np.array(gt)

    correct = pred[:, np.newaxis] == gt

    return np.sum(correct) / correct.shape[0]


class TestModel(unittest.TestCase):
    def fit(self, dynamic):
        device = set_device('gpu')
        fluid.enable_dygraph(device) if dynamic else None

        im_shape = (-1, 784)
        batch_size = 128

        inputs = [Input(im_shape, 'float32', name='image')]
        labels = [Input([None, 1], 'int64', name='label')]

        train_dataset = MnistDataset(mode='train')
        val_dataset = MnistDataset(mode='test')
        test_dataset = TestMnistDataset()

        model = MNIST()
        optim = fluid.optimizer.Momentum(
            learning_rate=0.01, momentum=.9, parameter_list=model.parameters())
        loss = CrossEntropy()
        model.prepare(optim, loss, Accuracy(), inputs, labels, device=device)
        cbk = ProgBarLogger(50)

        model.fit(train_dataset,
                  val_dataset,
                  epochs=2,
                  batch_size=batch_size,
                  callbacks=cbk)

        eval_result = model.evaluate(val_dataset, batch_size=batch_size)

        output = model.predict(
            test_dataset, batch_size=batch_size, stack_outputs=True)

        np.testing.assert_equal(output[0].shape[0], len(test_dataset))

        acc = get_predict_accuracy(output[0], val_dataset.labels)

        np.testing.assert_allclose(acc, eval_result['acc'])

    def test_multiple_gpus_static(self):
        self.fit(False)

    def test_multiple_gpus_dygraph(self):
        self.fit(True)


if __name__ == '__main__':
    unittest.main()
