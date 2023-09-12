# copyright (c) 2022 paddlepaddle authors. all rights reserved.
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

import unittest

import numpy as np

import paddle
import paddle.nn.functional as F
from paddle.io import Dataset
from paddle.nn import Conv2D, Linear, ReLU, Sequential
from paddle.quantization import QAT, QuantConfig
from paddle.quantization.quanters import FakeQuanterWithAbsMaxObserver
from paddle.quantization.quanters.abs_max import (
    FakeQuanterWithAbsMaxObserverLayer,
)


class RandomDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        data = np.random.random([3, 32, 32]).astype('float32')
        return data

    def __len__(self):
        return self.num_samples


class Model(paddle.nn.Layer):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.features = Sequential(
            Conv2D(3, 6, 3, stride=1, padding=1),
            ReLU(),
            paddle.nn.MaxPool2D(2, stride=2),
            Conv2D(6, 16, 5, stride=1, padding=0),
            ReLU(),
            paddle.nn.MaxPool2D(2, stride=2),
        )

        if num_classes > 0:
            self.fc = Sequential(
                Linear(576, 120), Linear(120, 84), Linear(84, 10)
            )

    def forward(self, inputs):
        x = self.features(inputs)
        if self.num_classes > 0:
            x = paddle.flatten(x, 1)
            x = self.fc(x)
        out = F.relu(x)
        return out


class TestQAT(unittest.TestCase):
    def test_qat(self):
        nums_batch = 100
        batch_size = 32
        dataset = RandomDataset(nums_batch * batch_size)
        loader = paddle.io.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=0,
        )
        model = Model()
        quanter = FakeQuanterWithAbsMaxObserver(moving_rate=0.9)
        q_config = QuantConfig(activation=quanter, weight=quanter)
        qat = QAT(q_config)
        print(model)
        quant_model = qat.quantize(model)
        print(quant_model)
        quanter_count = 0
        for _layer in quant_model.sublayers(True):
            if isinstance(_layer, FakeQuanterWithAbsMaxObserverLayer):
                quanter_count += 1
        self.assertEqual(quanter_count, 14)

        for _, data in enumerate(loader):
            out = quant_model(data)
            out.backward()


if __name__ == '__main__':
    unittest.main()
