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

import numpy as np
import contextlib

from paddle import fluid

from paddle.incubate.hapi.model import Model, Input, set_device
from paddle.incubate.hapi.loss import CrossEntropy
from paddle.incubate.hapi.vision.models import LeNet
from paddle.incubate.hapi.metrics import Accuracy
from paddle.incubate.hapi.callbacks import ProgBarLogger
from paddle.incubate.hapi.datasets import MNIST


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


def compute_accuracy(pred, gt):
    pred = np.argmax(pred, -1)
    gt = np.array(gt)

    correct = pred[:, np.newaxis] == gt

    return np.sum(correct) / correct.shape[0]


@unittest.skipIf(not fluid.is_compiled_with_cuda(),
                 'CPU testing is not supported')
class TestDistTraning(unittest.TestCase):
    def test_static_multiple_gpus(self):
        device = set_device('gpu')

        im_shape = (-1, 1, 28, 28)
        batch_size = 128

        inputs = [Input(im_shape, 'float32', name='image')]
        labels = [Input([None, 1], 'int64', name='label')]

        train_dataset = MnistDataset(mode='train')
        val_dataset = MnistDataset(mode='test')
        test_dataset = MnistDataset(mode='test', return_label=False)

        model = LeNet()
        optim = fluid.optimizer.Momentum(
            learning_rate=0.001, momentum=.9, parameter_list=model.parameters())
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

        acc = compute_accuracy(output[0], val_dataset.labels)

        np.testing.assert_allclose(acc, eval_result['acc'])


if __name__ == '__main__':
    unittest.main()
