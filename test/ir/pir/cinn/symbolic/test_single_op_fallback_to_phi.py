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

import numpy as np

import paddle
from paddle.static import InputSpec

sys.path.append(dirname(dirname(__file__)))
from utils import apply_to_static


class MatmulReshapeMatmulNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    # (64, 96) * (96, 32) -> (64, 32)
    # (64, 32) -> reshape -> (16, 128)
    # (16, 128) * (128, 16) -> (16, 16)
    def forward(self, x, y, z):
        out = paddle.matmul(x, y)
        out = paddle.reshape(out, [16, -1])
        out = paddle.matmul(out, z)
        return out


class TestSingleOpFallbackToPhi(unittest.TestCase):
    """
    Test Pir API + @to_static + CINN.
    """

    def setUp(self):
        paddle.seed(2022)
        self.prepare_data()

    def prepare_data(self):
        self.x = paddle.randn([64, 96], dtype="float32")
        self.x.stop_gradient = False
        self.y = paddle.randn([96, 32], dtype="float32")
        self.y.stop_gradient = False
        self.z = paddle.randn([128, 16], dtype="float32")
        self.z.stop_gradient = False

    def eval(self, use_cinn):
        paddle.seed(2022)
        net = MatmulReshapeMatmulNet()
        if use_cinn:
            net = apply_to_static(net, use_cinn)
        net.eval()
        out = net(self.x, self.y, self.z)
        return out

    def test_eval(self):
        cinn_out = self.eval(use_cinn=True)
        dy_out = self.eval(use_cinn=False)
        np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-8)


class TestSingleOpFallbackToPhiDynamic(TestSingleOpFallbackToPhi):
    def eval(self, use_cinn):
        paddle.seed(2022)
        net = MatmulReshapeMatmulNet()
        if use_cinn:
            input_spec = [
                InputSpec(shape=[None, None], dtype="float32"),
                InputSpec(shape=[None, None], dtype="float32"),
                InputSpec(shape=[None, None], dtype="float32"),
            ]
            net = apply_to_static(net, use_cinn, input_spec)
        net.eval()
        out = net(self.x, self.y, self.z)
        return out


if __name__ == '__main__':
    unittest.main()
