# copyright (c) 2023 paddlepaddle authors. all rights reserved.
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
import tempfile
import unittest

import paddle
import paddle.nn.functional as F
from paddle.nn import Conv2D, Linear, ReLU, Sequential
from paddle.quantization import PTQ, QuantConfig
from paddle.quantization.observers import GroupWiseWeightObserver


class LeNetDygraph(paddle.nn.Layer):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.features = Sequential(
            Conv2D(1, 6, 3, stride=1, padding=1),
            ReLU(),
            paddle.nn.MaxPool2D(2, 2),
            Conv2D(6, 16, 5, stride=1, padding=0),
            ReLU(),
            paddle.nn.MaxPool2D(2, 2),
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


class TestPTQGroupWise(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.temp_dir.name, 'ptq')

    def tearDown(self):
        self.temp_dir.cleanup()

    def _get_model_for_ptq(self):
        observer = GroupWiseWeightObserver(quant_bits=4, group_size=128)
        model = LeNetDygraph()
        model.eval()
        q_config = QuantConfig(activation=None, weight=observer)
        ptq = PTQ(q_config)
        quant_model = ptq.quantize(model)
        return quant_model, ptq

    def _count_layers(self, model, layer_type):
        count = 0
        for _layer in model.sublayers(True):
            if isinstance(_layer, layer_type):
                count += 1
        return count

    def test_quantize(self):
        ptq_model, _ = self._get_model_for_ptq()
        image = paddle.rand([1, 1, 32, 32], dtype="float32")
        out = ptq_model(image)
        self.assertIsNotNone(out)


if __name__ == '__main__':
    unittest.main()
