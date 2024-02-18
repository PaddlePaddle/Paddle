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
from os.path import dirname

sys.path.append(dirname(dirname(__file__)))

import unittest

import numpy as np
import utils

import paddle
from paddle.static import InputSpec


def exp_sub(x):
    y = paddle.exp(x)
    z = y - x
    return z


class CINNSubGraphNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.fn = exp_sub

    def forward(self, x):
        out = self.fn(x)
        return out


class TestCinnSubGraphBase(unittest.TestCase):
    """
    Test Pir API + @to_static + CINN.
    """

    def setUp(self):
        paddle.seed(2022)
        self.prepare_data()

    def prepare_data(self):
        # Setting "FLAGS_cinn_convert_dynamic_dim_to_static_dim=S0:64" can convert S0 to 64
        # "FLAGS_cinn_convert_dynamic_dim_to_static_dim=S0:64,S1:96" can change S0 and S1 at the same time
        self.shape = [64, 96]
        self.x = paddle.randn(self.shape, dtype="float32")
        self.x.stop_gradient = False

    def check_jit_kernel_info(self, static_fn):
        utils.check_jit_kernel_number(static_fn, 1)
        utils.check_jit_kernel_structure(static_fn, {utils.JIT_KERNEL_NAME: 1})

    def eval(self, use_cinn):
        paddle.seed(2022)
        net = CINNSubGraphNet()
        input_spec = [
            InputSpec(shape=[None, 96], dtype='float32'),
        ]
        net = utils.apply_to_static(net, use_cinn, input_spec)
        net.eval()
        out = net(self.x)
        if use_cinn:
            self.check_jit_kernel_info(net.forward)
        return out

    def test_eval(self):
        cinn_out = self.eval(use_cinn=True)
        dy_out = self.eval(use_cinn=False)
        np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-8)


if __name__ == '__main__':
    unittest.main()
