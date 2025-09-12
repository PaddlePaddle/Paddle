# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import utils

import paddle
from paddle import nn


class SubGraph(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = paddle.full([1, 32, 2, 10], 1.0, dtype="float32")
        z = paddle.transpose(y.reshape([4, 8, 2, 10]), perm=[0, 2, 3, 1])
        return x + z


class TestFuldFullOps(unittest.TestCase):
    def setUp(self):
        paddle.seed(2022)
        self.prepare_data()

    def prepare_data(self):
        self.x = paddle.randn([4, 2, 1, 8], dtype="float32")

    def check_jit_kernel_info(self, static_fn):
        utils.check_jit_kernel_number(static_fn, 1)
        utils.check_jit_kernel_structure(static_fn, {utils.JIT_KERNEL_NAME: 1})

    def eval(self, use_cinn):
        paddle.seed(2022)
        net = SubGraph()
        net = utils.apply_to_static(net, use_cinn)
        net.eval()
        out = net(self.x)

        if use_cinn:
            self.check_jit_kernel_info(net.forward)
        return out

    def test_eval(self):
        cinn_outs = self.eval(use_cinn=True)
        dy_outs = self.eval(use_cinn=False)

        for cinn_out, dy_out in zip(cinn_outs, dy_outs):
            np.testing.assert_allclose(
                cinn_out.numpy(), dy_out.numpy(), atol=1e-6, rtol=1e-6
            )


if __name__ == '__main__':
    unittest.main()
