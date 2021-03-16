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

import unittest

import numpy as np

import paddle
from paddle import fluid

from paddle import Model
from paddle.static import InputSpec
from paddle.nn.layer.loss import CrossEntropyLoss
from paddle.vision.datasets import MNIST
from paddle.vision.models import LeNet


class MnistDataset(MNIST):
    def __init__(self, mode, return_label=True, sample_num=None):
        super(MnistDataset, self).__init__(mode=mode)
        self.return_label = return_label
        if sample_num:
            self.images = self.images[:sample_num]
            self.labels = self.labels[:sample_num]

    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx]
        img = np.reshape(img, [1, 28, 28])
        if self.return_label:
            return img, np.array(self.labels[idx]).astype('int64')
        return img,

    def __len__(self):
        return len(self.images)


@unittest.skipIf(not fluid.is_compiled_with_cuda(),
                 'CPU testing is not supported')
class TestDistTraningUsingAMP(unittest.TestCase):
    def test_amp_training(self):
        if not fluid.is_compiled_with_cuda():
            self.skipTest('module not tested when ONLY_CPU compling')
        for dynamic in [True, False]:
            for amp_mode in ['O1', 'O2']:
                paddle.enable_static() if not dynamic else None
                device = paddle.set_device('gpu')
                net = LeNet()
                mnist_data = MnistDataset(mode='train', sample_num=2048)
                inputs = InputSpec([None, 1, 28, 28], "float32", 'x')
                label = InputSpec([None, 1], "int64", "y")
                model = Model(net, inputs, label)
                optim = paddle.optimizer.Adam(
                    learning_rate=0.001, parameters=model.parameters())
                amp_configs = {"incr_ratio": 2, "custom_black_list": {'conv2d'}}
                model.prepare(
                    optimizer=optim,
                    loss=CrossEntropyLoss(reduction="sum"),
                    amp_mode=amp_mode,
                    amp_configs=amp_configs)
                model.fit(mnist_data, batch_size=64, verbose=0)


if __name__ == '__main__':
    unittest.main()
