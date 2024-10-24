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

from dygraph_to_static_utils import (
    Dy2StTestBase,
    test_ast_only,
    test_pir_only,
)

import paddle


class HighOrderNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.bilinear = paddle.nn.Bilinear(
            in1_features=5, in2_features=4, out_features=1000
        )

    def forward(self, x, y):
        y = self.bilinear(x, y)
        z = paddle.pow(y, 2)
        x_grad = paddle.grad(z, x, create_graph=True)[0]
        x_grad_grad = paddle.grad(x_grad, x, create_graph=True)[0]
        return x_grad_grad.mean()


class TestBackwardHasNoGradError(Dy2StTestBase):
    @test_ast_only
    @test_pir_only
    def test_backward_has_no_grad_error(self):
        net = HighOrderNet()
        static_net = paddle.jit.to_static(net, full_graph=True)

        x = layer1 = paddle.rand((5, 5)).astype('float32')
        x.stop_gradient = False
        y = layer1 = paddle.rand((5, 4)).astype('float32')
        y.stop_gradient = False

        with self.assertRaisesRegex(
            ValueError,
            "op 'pd_op.bilinear_grad' has no grad op, consider enable prim to decompose it.",
        ):
            x_grad_grad = static_net(x, y)
            x_grad_grad.backward()


if __name__ == "__main__":
    unittest.main()
