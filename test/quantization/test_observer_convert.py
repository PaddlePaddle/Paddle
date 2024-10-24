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
from paddle.nn import Linear, Sequential
from paddle.quantization import PTQ, QuantConfig
from paddle.quantization.observers import (
    AbsmaxObserver,
    GroupWiseWeightObserver,
)


class LinearDygraph(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.fc = Sequential(
            Linear(128, 128), Linear(128, 128), Linear(128, 128)
        )

    def forward(self, inputs):
        out = self.fc(inputs)
        return out


class TestPTQGroupWise(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.temp_dir.name, 'ptq')

    def tearDown(self):
        self.temp_dir.cleanup()

    def _get_model_for_ptq_groupwise(self):
        observer = GroupWiseWeightObserver(quant_bits=4, group_size=128)
        model = LinearDygraph()
        model.eval()
        q_config = QuantConfig(activation=None, weight=observer)
        ptq = PTQ(q_config)
        quant_model = ptq.quantize(model)
        inputs = paddle.rand([128, 128], dtype="float32")
        out = model(inputs)
        return quant_model, ptq

    def _get_model_for_ptq_absmax(self):
        observer = AbsmaxObserver(quant_bits=8)
        model = LinearDygraph()
        model.eval()
        q_config = QuantConfig(activation=observer, weight=observer)
        ptq = PTQ(q_config)
        quant_model = ptq.quantize(model)
        inputs = paddle.rand([128, 128], dtype="float32")
        out = model(inputs)
        return quant_model, ptq

    def test_quantize(self):
        ptq_model, ptq = self._get_model_for_ptq_groupwise()
        inputs = paddle.rand([128, 128], dtype="float32")
        out = ptq_model(inputs)
        self.assertIsNotNone(out)
        converted_model = ptq.convert(ptq_model)
        out = converted_model(inputs)
        self.assertIsNotNone(out)

    def test_quantize_absmax(self):
        ptq_model, ptq = self._get_model_for_ptq_absmax()
        inputs = paddle.rand([128, 128], dtype="float32")
        out = ptq_model(inputs)
        self.assertIsNotNone(out)
        converted_model = ptq.convert(ptq_model)
        out = converted_model(inputs)
        self.assertIsNotNone(out)


if __name__ == '__main__':
    unittest.main()
