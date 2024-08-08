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

import shutil
import tempfile
import unittest

import numpy as np

import paddle
from paddle import Model
from paddle.metric import Accuracy
from paddle.nn.layer.loss import CrossEntropyLoss
from paddle.static import InputSpec
from paddle.vision.datasets import MNIST
from paddle.vision.models import LeNet


class MnistDataset(MNIST):
    def __init__(self, mode, return_label=True, sample_num=None):
        super().__init__(mode=mode)
        self.return_label = return_label
        if sample_num:
            self.images = self.images[:sample_num]
            self.labels = self.labels[:sample_num]

    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx]
        img = np.reshape(img, [1, 28, 28])
        if self.return_label:
            return img, np.array(self.labels[idx]).astype('int64')
        return (img,)

    def __len__(self):
        return len(self.images)


class TestCallbacks(unittest.TestCase):
    def setUp(self):
        self.save_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.save_dir)

    def test_earlystopping(self):
        paddle.seed(2020)
        for dynamic in [True, False]:
            paddle.enable_static() if not dynamic else None
            device = paddle.set_device('cpu')
            sample_num = 100
            train_dataset = MnistDataset(mode='train', sample_num=sample_num)
            val_dataset = MnistDataset(mode='test', sample_num=sample_num)

            net = LeNet()
            optim = paddle.optimizer.Adam(
                learning_rate=0.001, parameters=net.parameters()
            )

            inputs = [InputSpec([None, 1, 28, 28], 'float32', 'x')]
            labels = [InputSpec([None, 1], 'int64', 'label')]

            model = Model(net, inputs=inputs, labels=labels)
            model.prepare(
                optim,
                loss=CrossEntropyLoss(reduction="sum"),
                metrics=[Accuracy()],
            )
            callbacks_0 = paddle.callbacks.EarlyStopping(
                'loss',
                mode='min',
                patience=1,
                verbose=1,
                min_delta=0,
                baseline=None,
                save_best_model=True,
            )
            callbacks_1 = paddle.callbacks.EarlyStopping(
                'acc',
                mode='auto',
                patience=1,
                verbose=1,
                min_delta=0,
                baseline=0,
                save_best_model=True,
            )
            callbacks_2 = paddle.callbacks.EarlyStopping(
                'loss',
                mode='auto_',
                patience=1,
                verbose=1,
                min_delta=0,
                baseline=None,
                save_best_model=True,
            )
            callbacks_3 = paddle.callbacks.EarlyStopping(
                'acc_',
                mode='max',
                patience=1,
                verbose=1,
                min_delta=0,
                baseline=0,
                save_best_model=True,
            )
            model.fit(
                train_dataset,
                val_dataset,
                batch_size=64,
                save_freq=10,
                save_dir=self.save_dir,
                epochs=10,
                verbose=0,
                callbacks=[
                    callbacks_0,
                    callbacks_1,
                    callbacks_2,
                    callbacks_3,
                ],
            )
            # Test for no val_loader
            model.fit(
                train_dataset,
                batch_size=64,
                save_freq=10,
                save_dir=self.save_dir,
                epochs=10,
                verbose=0,
                callbacks=[callbacks_0],
            )


if __name__ == '__main__':
    unittest.main()
