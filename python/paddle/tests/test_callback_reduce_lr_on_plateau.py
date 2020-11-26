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

import sys
import unittest
import time
import random
import tempfile
import shutil
import numpy as np

import paddle
import paddle.vision.transforms as T
from paddle import Model
from paddle.static import InputSpec
from paddle.vision.models import LeNet
from paddle.hapi.callbacks import config_callbacks
from paddle.vision.datasets import MNIST
from paddle.metric import Accuracy
from paddle.nn.layer.loss import CrossEntropyLoss


class TestCallbacks(unittest.TestCase):
    def test_reduce_lr_on_plateau(self):
        sample_num = 200
        transform = T.Compose([T.Transpose(), T.Normalize([127.5], [127.5])])
        train_dataset = MNIST(mode='train', transform=transform)
        val_dataset = MNIST(mode='test', transform=transform)
        net = LeNet()
        optim = paddle.optimizer.Adam(
            learning_rate=0.001, parameters=net.parameters())
        inputs = [InputSpec([None, 1, 28, 28], 'float32', 'x')]
        labels = [InputSpec([None, 1], 'int64', 'label')]
        model = Model(net, inputs=inputs, labels=labels)
        model.prepare(optim, loss=CrossEntropyLoss(), metrics=[Accuracy()])
        callbacks = paddle.callbacks.ReduceLROnPlateau(patience=3, verbose=1)
        model.fit(train_dataset,
                  val_dataset,
                  batch_size=64,
                  log_freq=200,
                  save_freq=10,
                  epochs=20,
                  callbacks=[callbacks])


if __name__ == '__main__':
    unittest.main()
