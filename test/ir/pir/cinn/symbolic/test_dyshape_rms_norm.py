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
from paddle import nn
from paddle.static import InputSpec

sys.path.append(dirname(dirname(__file__)))

import utils


class LlamaRMSNorm(nn.Layer):
    def __init__(self):
        super().__init__()
        self.hidden_size = 768
        self.dtype = "bfloat16"
        self.weight = paddle.randn([128], dtype=self.dtype)
        self.weight.stop_gradient = False
        self.bias = paddle.randn([128], dtype=self.dtype)
        self.bias.stop_gradient = False

        self.data_format = "NHWC"

    def forward(self, x):
        return paddle.nn.functional.group_norm(
            x,
            num_groups=32,
            epsilon=1e-6,
            weight=self.weight,
            bias=self.bias,
            data_format=self.data_format,
        )


class TestLlamaRMSNorm(unittest.TestCase):
    def setUp(self):
        paddle.seed(2022)
        self.shape = [80, 128, 256, 128]
        self.dtype = "bfloat16"
        self.data_format = "NHWC"
        self.prepare_data()

    def prepare_data(self):
        self.x = paddle.randn(self.shape, dtype=self.dtype)
        self.x.stop_gradient = False

    def eval(self, use_cinn):
        net = LlamaRMSNorm()
        input_spec = [
            InputSpec(shape=[None, None, None, 128], dtype='bfloat16'),
        ]
        net = utils.apply_to_static(net, use_cinn, input_spec)
        net.eval()
        out = net(self.x)

        return out

    def test_eval(self):
        cinn_out = self.eval(use_cinn=True)
        dy_out = self.eval(use_cinn=False)
        np.testing.assert_allclose(
            cinn_out.numpy(), dy_out.numpy(), atol=1e-6, rtol=1e-6
        )


if __name__ == '__main__':
    unittest.main()
