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
from paddle.base import core
from paddle.static import InputSpec

sys.path.append(dirname(dirname(__file__)))
sys.path.append("../")


import utils


class TransposeReshapeNet(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        out = paddle.incubate.nn.functional.swiglu(x, y)

        return out


class TestTransposeReshape(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def prepare_data(self):
        self.x = paddle.randn([4, 32, 11008], dtype="float16")
        self.y = paddle.randn([4, 32, 11008], dtype="float16")

    def check_jit_kernel_info(self, static_fn):
        utils.check_jit_kernel_number(static_fn, 1)

    def eval(self, use_cinn=False, mode="jit"):
        net = TransposeReshapeNet()
        if mode == "eager":
            out = out = net(self.x, self.y)
        else:
            input_spec = [
                InputSpec(shape=[None, None, 11008], dtype="float16"),
                InputSpec(shape=[None, None, 11008], dtype="float16"),
            ]
            net = utils.apply_to_static(net, use_cinn, input_spec)
            net.eval()
            out = net(self.x, self.y)
            if use_cinn:
                self.check_jit_kernel_info(net.forward)
        return out

    def test_eval(self):
        dy_out = self.eval(mode="eager")
        core._set_prim_all_enabled(True)
        cinn_out = self.eval(use_cinn=True)
        np.testing.assert_allclose(
            cinn_out.numpy(), dy_out.numpy(), atol=1e-2, rtol=1e-2
        )
        core._set_prim_all_enabled(False)


if __name__ == '__main__':
    unittest.main()
