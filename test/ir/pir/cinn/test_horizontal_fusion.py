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
import unittest

import numpy as np
import utils

import paddle

# NOTE(SigureMo): Disable the CSE optimization to avoid op number change.
paddle.set_flags({"FLAGS_enable_cse_in_dy2st": False})


class HorizontalSubGraph(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        tmp1 = paddle.sum(x, axis=-1)
        tmp2 = paddle.sum(x * x, axis=-1)
        return tmp1 * tmp2


class TestHorizontalGraph(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def prepare_data(self):
        self.x = paddle.randn([256, 128], dtype="float32")
        self.x.stop_gradient = True

    def check_jit_kernel_info(self, static_fn):
        utils.check_jit_kernel_number(static_fn, 1)
        utils.check_jit_kernel_structure(static_fn, {utils.JIT_KERNEL_NAME: 1})

    def eval(self, use_cinn):
        net = HorizontalSubGraph()
        net.eval()
        net = utils.apply_to_static(net, use_cinn)
        out = net(self.x)
        if use_cinn:
            self.check_jit_kernel_info(net.forward)
        return out

    def test_eval(self):
        cinn_out = self.eval(use_cinn=True)
        dy_out = self.eval(use_cinn=False)
        np.testing.assert_allclose(
            cinn_out.numpy(), dy_out.numpy(), atol=1e-5, rtol=1e-5
        )


if __name__ == '__main__':
    # Fix YieldStore Segment fault.
    # unittest.main()
    pass
