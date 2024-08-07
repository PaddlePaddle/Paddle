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

sys.path.append(dirname(dirname(__file__)))
import utils


def exp_sub_concat(x):
    y = paddle.exp(x)
    z = y - x
    out = paddle.concat([z, x], 0)
    return out


class CheckInferSymbolicNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.fn = exp_sub_concat

    def forward(self, x):
        out = self.fn(x)
        return out


class TestCheckInferSymbolic(unittest.TestCase):
    """
    Test Pir API + @to_static + CINN.
    """

    def setUp(self):
        paddle.seed(2022)
        self.prepare_data()

    def prepare_data(self):
        self.shape = [64, 96]
        self.x = paddle.randn(self.shape, dtype="float32")
        self.x.stop_gradient = False

    def eval(self, use_cinn):
        paddle.seed(2022)
        net = CheckInferSymbolicNet()
        if use_cinn:
            net = utils.apply_to_static(net, use_cinn)
        net.eval()
        out = net(self.x)
        return out

    def test_eval(self):
        cinn_out = self.eval(use_cinn=True)
        dy_out = self.eval(use_cinn=False)
        np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-8)


if __name__ == '__main__':
    unittest.main()
