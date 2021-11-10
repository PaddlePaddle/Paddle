# copyright (c) 2020 paddlepaddle authors. all rights reserved.
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

import os
os.environ['FLAGS_cudnn_deterministic'] = '1'

import unittest

import numpy as np

import paddle
from paddle import fluid

from paddle import Model
from paddle.static import InputSpec
from paddle.nn.layer.loss import CrossEntropyLoss
from paddle.vision.models import LeNet
from paddle.vision.datasets import MNIST
import paddle.vision.transforms as T


@unittest.skipIf(not fluid.is_compiled_with_cuda(),
                 'CPU testing is not supported')
class TestHapiWithAmp(unittest.TestCase):
    def get_model(self, amp_config):
        net = LeNet()
        inputs = InputSpec([None, 1, 28, 28], "float32", 'x')
        labels = InputSpec([None, 1], "int64", "y")
        model = Model(net, inputs, labels)
        optim = paddle.optimizer.Adam(
            learning_rate=0.001, parameters=model.parameters())
        model.prepare(
            optimizer=optim,
            loss=CrossEntropyLoss(reduction="sum"),
            amp_configs=amp_config)
        return model

    def run_model(self, model):
        transform = T.Compose([T.Transpose(), T.Normalize([127.5], [127.5])])
        train_dataset = MNIST(mode='train', transform=transform)
        model.fit(train_dataset,
                  epochs=1,
                  batch_size=64,
                  num_iters=2,
                  log_freq=1)

    def run_amp(self, amp_level):
        for dynamic in [True, False]:
            if not dynamic and amp_level['level'] == 'O2':
                amp_level['use_fp16_guard'] = False
            print('dynamic' if dynamic else 'static', amp_level)

            paddle.seed(2021)
            paddle.enable_static() if not dynamic else paddle.disable_static()
            paddle.set_device('gpu')
            model = self.get_model(amp_level)
            self.run_model(model)

    def test_pure_fp16(self):
        amp_config = {
            "level": "O2",
            "init_loss_scaling": 128,
        }
        self.run_amp(amp_config)


if __name__ == '__main__':
    unittest.main()
