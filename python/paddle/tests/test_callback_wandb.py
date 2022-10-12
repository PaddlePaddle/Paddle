# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os
import shutil
import stat
import tempfile
import unittest

import paddle.vision.transforms as T
from paddle.static import InputSpec
from paddle.vision.datasets import MNIST

import paddle
from paddle.fluid.framework import _test_eager_guard


class MnistDataset(MNIST):

    def __len__(self):
        return 512


class TestWandbCallbacks(unittest.TestCase):

    def setUp(self):
        self.save_dir = tempfile.mkdtemp()

    def tearDown(self):
        os.chmod(self.save_dir, stat.S_IWUSR)
        shutil.rmtree(self.save_dir)

    def func_wandb_callback(self):
        inputs = [InputSpec([-1, 1, 28, 28], 'float32', 'image')]
        labels = [InputSpec([None, 1], 'int64', 'label')]

        transform = T.Compose([T.Transpose(), T.Normalize([127.5], [127.5])])
        train_dataset = MnistDataset(mode='train', transform=transform)
        eval_dataset = MnistDataset(mode='test', transform=transform)

        net = paddle.vision.models.LeNet()
        model = paddle.Model(net, inputs, labels)

        optim = paddle.optimizer.Adam(0.001, parameters=net.parameters())
        model.prepare(optimizer=optim,
                      loss=paddle.nn.CrossEntropyLoss(),
                      metrics=paddle.metric.Accuracy())

        callback = paddle.callbacks.WandbCallback(project='random',
                                                  dir=self.save_dir,
                                                  anonymous='must',
                                                  mode='offline')
        model.fit(train_dataset,
                  eval_dataset,
                  batch_size=64,
                  callbacks=callback)

    def test_wandb_callback(self):
        with _test_eager_guard():
            self.func_wandb_callback()
        self.func_wandb_callback()


if __name__ == '__main__':
    unittest.main()
