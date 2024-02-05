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


class IfSubgraph(nn.Layer):
    def __init__(self):
        super().__init__()

    def _make_causal_mask(self, mask_shape):
        batch_size, seq_len = mask_shape

        return paddle.tril(paddle.ones((batch_size, seq_len), dtype="bool"))

    def forward(self, attention_mask):
        if attention_mask.shape[-1] > 1:
            causal_mask = self._make_causal_mask(attention_mask.shape)
            attention_mask = attention_mask & causal_mask
        attention_mask = paddle.where(
            attention_mask, 0.0, paddle.finfo("float32").min
        ).astype("float32")
        return attention_mask


class TestIfSubgraph(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def prepare_data(self):
        self.attention_mask = paddle.ones([1, 2048], dtype="bool")
        self.attention_mask.stop_gradient = False

    def check_jit_kernel_info(self, static_fn):
        utils.check_jit_kernel_number(static_fn, 1)
        utils.check_jit_kernel_structure(static_fn, {utils.JIT_KERNEL_NAME: 1})

    def eval(self, use_cinn):
        net = IfSubgraph()
        input_spec = [
            InputSpec(shape=[1, 2048], dtype="bool"),
        ]
        net = utils.apply_to_static(net, use_cinn, input_spec)
        net.eval()
        out = net(self.attention_mask)
        if use_cinn:
            self.check_jit_kernel_info(net.forward)
        return out

    def test_eval(self):
        dy_out = self.eval(use_cinn=False)
        if utils.unittest_use_cinn():
            cinn_out = self.eval(use_cinn=True)
            np.testing.assert_allclose(
                cinn_out.numpy(), dy_out.numpy(), atol=1e-6, rtol=1e-6
            )


if __name__ == '__main__':
    unittest.main()
