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

from hapi.model import Model, Input, set_device
from hapi.loss import Loss, CrossEntropy
from hapi.vision.models import LeNet
from hapi.metrics import Accuracy
from hapi.callbacks import ProgBarLogger
from hapi.datasets import MNIST


class MnistDataset(MNIST):
    def __init__(self, mode, return_label=True):
        super(MnistDataset, self).__init__(mode=mode)
        self.return_label = return_label

    def __getitem__(self, idx):
        img = np.reshape(self.images[idx], [1, 28, 28])
        if self.return_label:
            return img, np.array(self.labels[idx]).astype('int64')
        return img,

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
        test_dataset = MnistDataset(mode='test', return_label=False)

        model = LeNet()
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
