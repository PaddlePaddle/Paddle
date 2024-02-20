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

import paddle


def apply_to_static(net, use_cinn):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(
        net,
        build_strategy=build_strategy,
        full_graph=True,
    )


def exp_sub_concat(x1, x2, x3):
    y1 = paddle.concat([x1, x3], 0)
    y2 = paddle.concat([x2, x3], 0)
    # out = paddle.concat([y2, y3], 0)
    return y1


class TestSubstituteDimExprNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.fn = exp_sub_concat

    def forward(self, x1, x2, x3):
        out = self.fn(x1, x2, x3)
        return out


class TestSubstituteDimExprBasedOnConstraint(unittest.TestCase):
    """
    Test Pir API + @to_static + CINN.
    """

    def setUp(self):
        paddle.seed(2022)
        self.prepare_data()

    def prepare_data(self):
        self.shape1 = [64, 96]
        self.shape2 = [64, 96]
        self.shape3 = [64, -1]
        self.x1 = paddle.randn(self.shape1, dtype="float32")
        self.x2 = paddle.randn(self.shape2, dtype="float32")
        self.x3 = paddle.randn(self.shape3, dtype="float32")
        self.x1.stop_gradient = False
        self.x2.stop_gradient = False
        self.x3.stop_gradient = False

    def eval(self, use_cinn):
        paddle.seed(2022)
        net = TestSubstituteDimExprNet()
        if use_cinn:
            net = apply_to_static(net, use_cinn)
        net.eval()
        out = net(self.x1, self.x2, self.x3)
        return out

    def test_eval(self):
        cinn_out = self.eval(use_cinn=True)
        dy_out = self.eval(use_cinn=False)
        np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-8)


if __name__ == '__main__':
    unittest.main()
