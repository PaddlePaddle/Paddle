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
import utils


def relu6(x):
    return paddle.nn.functional.relu6(x)


class CINNSubGraphNet(paddle.nn.Layer):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        out = self.fn(x)
        return out


class TestCinnSubGraprelu6(unittest.TestCase):
    """
    Test Pir API + @to_static + CINN.
    """

    def setUp(self):
        paddle.seed(2022)
        self.prepare_data()

    def prepare_data(self):
        self.x_shape = [32, 32]
        self.x = paddle.randn(self.x_shape, dtype="float32")
        self.x.stop_gradient = False

    def check_jit_kernel_info(self, static_fn):
        utils.check_jit_kernel_number(static_fn, 1)

    def eval_symbolic(self, use_cinn):
        paddle.seed(2022)
        net = CINNSubGraphNet(relu6)
        input_spec = [
            InputSpec(shape=[None, 32], dtype='float32'),
        ]
        net = utils.apply_to_static(net, use_cinn, input_spec)
        net.eval()
        out = net(self.x)
        if use_cinn:
            self.check_jit_kernel_info(net.forward)
        return out

    def test_eval_symbolic(self):
        cinn_out = self.eval_symbolic(use_cinn=True)
        dy_out = self.eval_symbolic(use_cinn=False)
        np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-8)


if __name__ == '__main__':
    unittest.main()
