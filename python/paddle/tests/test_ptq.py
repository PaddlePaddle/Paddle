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
from paddle.quantization import QuantConfig, PTQ, TRTQuantConfig
from paddle.quantization.quanters import ActLSQPlusQuanter, WeightLSQPlusQuanter
from paddle.quantization.observers import AbsmaxObserver
#from paddle.quantization.observers import PerChannelAbsmaxObserver
#from paddle.quantization.observers import HistObserver
from paddle.quantization.observers import KLObserver
from paddle.quantization import Stub
import paddle.nn.functional as F

observer = AbsmaxObserver(quant_bits=8)


class LeNetDygraph(paddle.nn.Layer):

    def __init__(self, num_classes=10):
        super(LeNetDygraph, self).__init__()
        self.num_classes = num_classes
        self.features = Sequential(Conv2D(1, 6, 3, stride=1, padding=1), ReLU(),
                                   paddle.fluid.dygraph.Pool2D(2, 'max', 2),
                                   Conv2D(6, 16, 5, stride=1, padding=0),
                                   ReLU(),
                                   paddle.fluid.dygraph.Pool2D(2, 'max', 2))

        if num_classes > 0:
            self.fc = Sequential(Linear(400, 120), Linear(120, 84),
                                 Linear(84, 10))
        self.quan = Stub(observer)
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


class TestPTQ(unittest.TestCase):

    def test_ptq(self):
        model = LeNetDygraph()

        # weight_observer = PerChannelAbsmaxObserver(quant_bits=8)

        #hist_observer = HistObserver(quant_bits=8)
        kl_observer = KLObserver()
        weight_observer = kl_observer
        hist_observer = kl_observer

        q_config = TRTQuantConfig(activation=observer, weight=weight_observer)
        q_config.add_group(paddle.nn.Conv2D,
                           activation=hist_observer,
                           weight=weight_observer)
        q_config.add_group(paddle.nn.Linear,
                           activation=kl_observer,
                           weight=weight_observer)
        q_config.add_group(paddle.quantization.Stub, activation=kl_observer)
        q_config.add_group(paddle.nn.ReLU, activation=kl_observer)
        print(q_config)
        ptq = PTQ(q_config)
        quant_model = ptq.quantize(model)
        print(quant_model)
        onnx_model = ptq.convert(quant_model)
        print(onnx_model)


if __name__ == '__main__':
    unittest.main()
