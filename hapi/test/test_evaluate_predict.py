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
import cv2
import numpy as np

import paddle
from paddle import fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear
from paddle.fluid.dygraph.container import Sequential
from paddle.io import BatchSampler, DataLoader

from hapi.model import Model, Input, set_device
from hapi.loss import Loss
from hapi.metrics import Accuracy
from hapi.datasets import MNIST
from hapi.vision.models import LeNet
from hapi.download import get_weights_path_from_url


class LeNetDygraph(fluid.dygraph.Layer):
    """LeNet model from
    `"LeCun Y, Bottou L, Bengio Y, et al. Gradient-based learning applied to document recognition[J]. Proceedings of the IEEE, 1998, 86(11): 2278-2324.`_

    Args:
        num_classes (int): output dim of last fc layer. If num_classes <=0, last fc layer 
                            will not be defined. Default: 10.
        classifier_activation (str): activation for the last fc layer. Default: 'softmax'.
    """

    def __init__(self, num_classes=10, classifier_activation='softmax'):
        super(LeNetDygraph, self).__init__()
        self.num_classes = num_classes
        self.features = Sequential(
            Conv2D(
                1, 6, 3, stride=1, padding=1),
            Pool2D(2, 'max', 2),
            Conv2D(
                6, 16, 5, stride=1, padding=0),
            Pool2D(2, 'max', 2))

        if num_classes > 0:
            self.fc = Sequential(
                Linear(400, 120),
                Linear(120, 84),
                Linear(
                    84, 10, act=classifier_activation))

    def forward(self, inputs):
        x = self.features(inputs)

        if self.num_classes > 0:
            x = fluid.layers.flatten(x, 1)
            x = self.fc(x)
        return x


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


def low_level_lenet_dygraph_train(model, dataloader):
    optim = fluid.optimizer.Adam(
        learning_rate=0.001, parameter_list=model.parameters())
    model.train()
    for inputs, labels in dataloader:
        outputs = model(inputs)
        loss = fluid.layers.cross_entropy(outputs, labels)
        avg_loss = fluid.layers.reduce_sum(loss)
        avg_loss.backward()
        optim.minimize(avg_loss)
        model.clear_gradients()


def low_level_dynamic_evaluate(model, dataloader):
    with fluid.dygraph.no_grad():
        model.eval()
        cnt = 0
        for inputs, labels in dataloader:
            outputs = model(inputs)

            cnt += (np.argmax(outputs.numpy(), -1)[:, np.newaxis] ==
                    labels.numpy()).astype('int').sum()

    return cnt / len(dataloader.dataset)


class TestEvaluatePredict(unittest.TestCase):
    def setUp(self):
        self.device = set_device('gpu')
        self.train_dataset = MnistDataset(mode='train')
        self.val_dataset = MnistDataset(mode='test')
        self.test_dataset = MnistDataset(mode='test', return_label=False)

        fluid.enable_dygraph(self.device)
        train_dataloader = fluid.io.DataLoader(
            self.train_dataset, places=self.device, batch_size=64)
        val_dataloader = fluid.io.DataLoader(
            self.val_dataset, places=self.device, batch_size=64)
        self.lenet_dygraph = LeNetDygraph()
        low_level_lenet_dygraph_train(self.lenet_dygraph, train_dataloader)
        self.acc1 = low_level_dynamic_evaluate(self.lenet_dygraph,
                                               val_dataloader)

    def evaluate(self, dynamic):
        fluid.enable_dygraph(self.device) if dynamic else None

        inputs = [Input([-1, 1, 28, 28], 'float32', name='image')]
        labels = [Input([None, 1], 'int64', name='label')]

        if fluid.in_dygraph_mode():
            feed_list = None
        else:
            feed_list = [x.forward() for x in inputs + labels]

        self.train_dataloader = fluid.io.DataLoader(
            self.train_dataset,
            places=self.device,
            batch_size=64,
            feed_list=feed_list)
        self.val_dataloader = fluid.io.DataLoader(
            self.val_dataset,
            places=self.device,
            batch_size=64,
            feed_list=feed_list)
        self.test_dataloader = fluid.io.DataLoader(
            self.test_dataset,
            places=self.device,
            batch_size=64,
            feed_list=feed_list)

        model = LeNet()
        model.load_dict(self.lenet_dygraph.state_dict())
        model.prepare(metrics=Accuracy(), inputs=inputs, labels=labels)

        result = model.evaluate(self.val_dataloader)

        np.testing.assert_allclose(result['acc'], self.acc1)

    def predict(self, dynamic):
        fluid.enable_dygraph(self.device) if dynamic else None

        inputs = [Input([-1, 1, 28, 28], 'float32', name='image')]
        labels = [Input([None, 1], 'int64', name='label')]

        if fluid.in_dygraph_mode():
            feed_list = None
        else:
            feed_list = [x.forward() for x in inputs + labels]

        self.train_dataloader = fluid.io.DataLoader(
            self.train_dataset,
            places=self.device,
            batch_size=64,
            feed_list=feed_list)
        self.val_dataloader = fluid.io.DataLoader(
            self.val_dataset,
            places=self.device,
            batch_size=64,
            feed_list=feed_list)
        self.test_dataloader = fluid.io.DataLoader(
            self.test_dataset,
            places=self.device,
            batch_size=64,
            feed_list=feed_list)

        model = LeNet()
        model.load_dict(self.lenet_dygraph.state_dict())
        model.prepare(metrics=Accuracy(), inputs=inputs, labels=labels)

        output = model.predict(self.test_dataloader, stack_outputs=True)

        np.testing.assert_equal(output[0].shape[0], len(self.test_dataset))

        acc = get_predict_accuracy(output[0], self.val_dataset.labels)

        np.testing.assert_allclose(acc, self.acc1)

    def test_evaluate_dygraph(self):
        self.evaluate(True)

    def test_evaluate_static(self):
        self.evaluate(False)

    def test_predict_dygraph(self):
        self.predict(True)

    def test_predict_static(self):
        self.predict(False)


if __name__ == '__main__':
    unittest.main()
