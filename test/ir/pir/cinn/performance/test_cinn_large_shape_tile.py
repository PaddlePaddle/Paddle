# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

sys.path.append(dirname(dirname(__file__)))

import numpy as np
import utils

import paddle


class CINNCosSubGraphNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, z):
        tmp = x * y
        tmp1 = paddle.reshape(tmp, [80, 32, 4])
        tmp2 = paddle.sum(tmp1, axis=2)
        tmp3 = paddle.reshape(tmp2, [80, 1, 32, 1])
        tmp4 = tmp3 * z
        return tmp4


class TestCinnCos(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def prepare_data(self):
        self.x = paddle.uniform([80, 128], dtype="float32", min=-0.5, max=0.5)
        self.x.stop_gradient = True
        self.y = paddle.uniform([128], dtype="float32", min=-0.5, max=0.5)
        self.y.stop_gradient = True
        self.z = paddle.uniform(
            [80, 32768, 32, 4], dtype="float32", min=-0.5, max=0.5
        )
        self.z.stop_gradient = True

    def train(self, use_cinn):
        net = CINNCosSubGraphNet()
        net.eval()
        net = utils.apply_to_static(net, use_cinn)
        out = net(self.x, self.y, self.z)
        return out

    def test_train(self):
        cinn_out = self.train(use_cinn=True)
        dy_out = self.train(use_cinn=False)

        np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-6)


if __name__ == '__main__':
    unittest.main()
