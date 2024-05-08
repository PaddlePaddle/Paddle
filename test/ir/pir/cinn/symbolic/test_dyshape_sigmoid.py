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

import os
import sys
import unittest
from os.path import dirname

import numpy as np

os.environ["FLAGS_prim_forward_blacklist"] = "pd_op.flatten"

import paddle
from paddle import nn
from paddle.static import InputSpec

sys.path.append(dirname(dirname(__file__)))

import utils

#paddle.set_default_dtype("float16")

class CastLayer(nn.Layer):
    def __init__(self):
        super().__init__()
        self.hidden_size = 320
        self.weight = paddle.create_parameter(
            shape=[self.hidden_size],
            dtype="float32",
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.bias = paddle.create_parameter(
            shape=[self.hidden_size],
            dtype="float32",
            default_initializer=nn.initializer.Constant(1.0),
        )
        

    def forward(self, x, y):
        t1 = x + y
        # t2 = paddle.flatten(t1, 1, 2)
        t3 = paddle.nn.functional.layer_norm( t1, normalized_shape=[320], weight=self.weight, bias=self.bias, )
        return t3


class TestCast(unittest.TestCase):
    def setUp(self):
        paddle.seed(2022)
        self.prepare_data()

    def prepare_data(self):
        self.shape = [256, 32, 32, 320]
        self.x = paddle.randn(self.shape, dtype="float16")
        self.x.stop_gradient = True
        
        self.y = paddle.randn([320], dtype="float16")
        self.y.stop_gradient = True

    def check_jit_kernel_info(self, static_fn):
        utils.check_jit_kernel_number(static_fn, 1)
        utils.check_jit_kernel_structure(static_fn, {utils.JIT_KERNEL_NAME: 1})

    def eval(self, use_cinn):
        net = CastLayer()
        input_spec = [
            InputSpec(shape=[None, None, None, 320], dtype='float16'),
            InputSpec(shape=[320], dtype='float16')
        ]
        net = utils.apply_to_static(net, use_cinn, input_spec)
        net.eval()
        out = net(self.x, self.y)
        if use_cinn:
            self.check_jit_kernel_info(net.forward)
        return out

    def test_eval(self):
        cinn_out = self.eval(use_cinn=True)
        # dy_out = self.eval(use_cinn=False)
        # np.testing.assert_allclose(
        #     cinn_out.numpy(), dy_out.numpy(), atol=1e-6, rtol=1e-6
        # )


if __name__ == '__main__':
    unittest.main()
