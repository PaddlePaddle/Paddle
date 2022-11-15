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

import paddle
import paddle.nn.functional as F
import numpy as np
from paddle import fluid
from paddle.nn import Conv2D, Linear, ReLU, Sequential
from paddle.quantization import QAT, Stub, TRTQuantConfig
from paddle.quantization.quanters import (
    ActLSQPlusQuanter,
    WeightLSQPlusQuanter,
    FakeQuanterWithAbsMaxObserver,
)
from paddle.io import Dataset

observer = FakeQuanterWithAbsMaxObserver(moving_rate=0.9)


class RandomDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        data = np.random.random([3, 32, 32]).astype('float32')
        return data

    def __len__(self):
        return self.num_samples


class LeNetDygraph(paddle.nn.Layer):
    def __init__(self, num_classes=10):
        super(LeNetDygraph, self).__init__()
        self.num_classes = num_classes
        self.features = Sequential(
            Conv2D(3, 6, 3, stride=1, padding=1),
            ReLU(),
            paddle.fluid.dygraph.Pool2D(2, 'max', 2),
            Conv2D(6, 16, 5, stride=1, padding=0),
            ReLU(),
            paddle.fluid.dygraph.Pool2D(2, 'max', 2),
        )

        if num_classes > 0:
            self.fc = Sequential(
                Linear(576, 120), Linear(120, 84), Linear(84, 10)
            )
        self.quant = Stub(observer)
        self.quant_out = Stub()

    def forward(self, inputs):
        x = self.features(inputs)
        if self.num_classes > 0:
            x = fluid.layers.flatten(x, 1)
            x = self.fc(x)
        out = F.relu(x)
        out = self.quant(out)
        out = F.relu(out)
        out = self.quant_out(out)

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
        model = LeNetDygraph()

        act_quanter = ActLSQPlusQuanter(
            quant_bits=8,
            all_postive=False,
            symmetric=False,
            batch_init=20,
            reduce_type=None,
        )
        weight_quanter = WeightLSQPlusQuanter(
            quant_bits=8,
            all_postive=False,
            per_channel=False,
            batch_init=20,
            quant_linear=False,
            reduce_type=None,
        )

        q_config = TRTQuantConfig(activation=act_quanter, weight=weight_quanter)
        q_config.add_group(
            paddle.nn.Conv2D, activation=act_quanter, weight=weight_quanter
        )
        q_config.add_group(
            paddle.nn.Linear, activation=None, weight=weight_quanter
        )
        q_config.add_group(
            ["layer1_name", "layer2_name"], activation=act_quanter, weight=None
        )
        q_config.add_group(paddle.quantization.Stub, activation=act_quanter)
        q_config.add_group(paddle.nn.ReLU, activation=observer)
        qat = QAT(q_config)
        quant_model = qat.quantize(model)
        for i, data in enumerate(loader):
            out = quant_model(data)
            out.backward()


if __name__ == '__main__':
    unittest.main()
