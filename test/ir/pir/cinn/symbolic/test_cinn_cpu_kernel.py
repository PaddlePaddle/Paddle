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

import sys
import unittest
from os.path import dirname

import numpy as np

sys.path.append(dirname(dirname(__file__)))

import utils

import paddle
from paddle.static import InputSpec


class CINNCosSubGraphNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        if x.shape[-1] > 1:
            y += 1

        return y


class TestCinnCos(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def prepare_data(self):
        self.x = paddle.uniform([80, 128], dtype="float32", min=-0.5, max=0.5)
        self.x.stop_gradient = True
        self.y = paddle.uniform([128], dtype="float32", min=-0.5, max=0.5)
        self.y.stop_gradient = True

    def train(self, use_cinn):
        net = CINNCosSubGraphNet()
        net.eval()

        input_spec = [
            InputSpec(shape=[None, None], dtype='float32'),
            InputSpec(shape=[None], dtype='float32'),
        ]
        net = utils.apply_to_static(net, use_cinn, input_spec)

        out = net(self.x, self.y)
        return out

    def test_train(self):
        cinn_out = self.train(use_cinn=True)
        dy_out = self.train(use_cinn=False)

        np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-5)


if __name__ == '__main__':
    unittest.main()
