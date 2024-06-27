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
        self.linear = paddle.nn.Linear(3, 4, bias_attr=False)

    def forward(self, x):
        y = self.linear(x)
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

        x = paddle.to_tensor([[1, 1, 1], [1, 1, 1]], 'float32')
        x.stop_gradient = False

        with self.assertRaisesRegex(
            ValueError,
            "op 'pd_op.matmul_double_grad' has no grad op, consider enable prim to decompose it.",
        ):
            x_grad_grad = static_net(x)
            x_grad_grad.backward()


if __name__ == "__main__":
    unittest.main()
