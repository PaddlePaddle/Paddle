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
from paddle import fluid
from paddle.nn import Conv2D, Linear, ReLU, Sequential
from paddle.quantization import QAT, TRTQuantConfig
from paddle.quantization.quanters import (
    QDropFakeQuanter,
    SoftRoundingFakeQuanter,
)
import paddle.nn.functional as F


class LeNetDygraph(paddle.nn.Layer):
    def __init__(self, num_classes=10):
        super(LeNetDygraph, self).__init__()
        self.num_classes = num_classes
        self.features = Sequential(
            Conv2D(1, 6, 3, stride=1, padding=1),
            ReLU(),
            paddle.fluid.dygraph.Pool2D(2, 'max', 2),
            Conv2D(6, 16, 5, stride=1, padding=0),
            ReLU(),
            paddle.fluid.dygraph.Pool2D(2, 'max', 2),
        )

        if num_classes > 0:
            self.fc = Sequential(
                Linear(400, 120), Linear(120, 84), Linear(84, 10)
            )

    def forward(self, inputs):
        x = self.features(inputs)
        if self.num_classes > 0:
            x = fluid.layers.flatten(x, 1)
            x = self.fc(x)
        out = F.relu(x)
        out = F.relu(out)
        return out


# class DistModel(Layer):
#    def __init__(self, student, teacher):
#        super(DistModel, self).__init()
#        self._student = student
#        self._teacher = teacher
#        self.loss =
#
#    def forward(*args, **kvargs):
#        student_outs = self.student(*args, **kvargs)
#        teacher_outs = self.teacher(*args, **kvargs)


class TestQDrop(unittest.TestCase):
    def test_qat(self):
        model = LeNetDygraph()

        act_quanter = QDropFakeQuanter()
        weight_quanter = SoftRoundingFakeQuanter()

        q_config = TRTQuantConfig(activation=act_quanter, weight=weight_quanter)

        qat = QAT(q_config)

        quant_model = qat.quantize(model)

        print(quant_model)


if __name__ == '__main__':
    unittest.main()
