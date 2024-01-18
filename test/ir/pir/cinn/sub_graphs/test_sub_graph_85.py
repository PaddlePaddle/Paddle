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

# repo: PaddleDetection
# model: configs^mot^fairmot^fairmot_dla34_30e_1088x608_airplane_single_dy2st_train
# method||__neg__,api||paddle.tensor.ops.exp,method||__mul__,method||__neg__,api||paddle.tensor.ops.exp,method||__mul__,method||__add__,method||__add__,method||__add__,method||__mul__
import unittest

import numpy as np

import paddle


class SIR79(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.var_576 = self.create_parameter(
            shape=[1],
            dtype=paddle.float32,
        )
        self.var_572 = self.create_parameter(
            shape=[1],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_566,  # (shape: [], dtype: paddle.float32, stop_gradient: False)
        var_570,  # (shape: [], dtype: paddle.float32, stop_gradient: False)
    ):
        var_573 = self.var_572.__neg__()
        var_574 = paddle.tensor.ops.exp(var_573)
        var_575 = var_574.__mul__(var_570)
        var_577 = self.var_576.__neg__()
        var_578 = paddle.tensor.ops.exp(var_577)
        var_579 = var_578.__mul__(var_566)
        var_580 = var_575.__add__(var_579)
        var_581 = self.var_572.__add__(self.var_576)
        var_582 = var_580.__add__(var_581)
        var_583 = var_582.__mul__(0.5)
        return var_583


class TestSIR79(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[1], dtype=paddle.float32),
            paddle.rand(shape=[1], dtype=paddle.float32),
        )
        self.net = SIR79()

    def train(self, net, to_static, with_prim=False, with_cinn=False):
        paddle.set_flags({'FLAGS_prim_all': with_prim})
        if to_static:
            if with_cinn:
                build_strategy = paddle.static.BuildStrategy()
                build_strategy.build_cinn_pass = True
                net = paddle.jit.to_static(
                    net, build_strategy=build_strategy, full_graph=True
                )
            else:
                net = paddle.jit.to_static(net, full_graph=True)
        outs = net(*self.inputs)
        return outs

    def test_ast_prim_cinn(self):
        st_out = self.train(self.net, to_static=True)
        cinn_out = self.train(
            self.net, to_static=True, with_prim=True, with_cinn=True
        )
        for st, cinn in zip(
            paddle.utils.flatten(st_out), paddle.utils.flatten(cinn_out)
        ):
            np.testing.assert_allclose(st.numpy(), cinn.numpy(), atol=1e-8)


if __name__ == '__main__':
    unittest.main()
