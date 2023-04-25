# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import ParamAttr
from paddle.nn import BatchNorm, Linear


class SimpleNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.linear0 = Linear(100, 50)
        self.linear1 = Linear(50, 10)

        param_attr0 = ParamAttr(name="aaaprefix_bn_scale")
        bias_attr0 = ParamAttr(name="aaaprefix_bn_offset")
        self.bn0 = BatchNorm(50, param_attr=param_attr0, bias_attr=bias_attr0)

        param_attr1 = ParamAttr(name="bn_scale")
        bias_attr1 = ParamAttr(name="bn_offset")
        self.bn1 = BatchNorm(10, param_attr=param_attr1, bias_attr=bias_attr1)

    def forward(self, x):
        x1 = self.linear0(x)
        x2 = self.bn0(x1)
        x3 = self.linear1(x2)
        x4 = self.bn1(x3)
        dx = paddle.grad(x4, x)
        return dx[0]


class TestGradNameParse(unittest.TestCase):
    def test_grad_name_parse(self):
        net = SimpleNet()
        opt = paddle.optimizer.Adam(
            learning_rate=0.1,
            parameters=net.parameters(),
            weight_decay=paddle.regularizer.L1Decay(0.01),
        )
        net = paddle.jit.to_static(net)
        inp = paddle.rand([100, 100], dtype="float32")
        inp.stop_gradient = False
        out = net(inp)
        loss = out.mean()
        loss.backward()

        for name, param in net.bn1.named_parameters():
            if name in ["bn_scale", "bn_offset"]:
                assert param.shape == param.grad.shape

        opt.minimize(loss)


class TestXGradNameParse(unittest.TestCase):
    def test_x_grad_name_parse(self):
        def tanh_high_order_grad(x):
            y = paddle.tanh(x)
            return paddle.grad(y, x, create_graph=True)[0]

        x1 = paddle.ones((1,))
        x1.stop_gradient = False
        y1 = tanh_high_order_grad(x1)
        g1 = paddle.grad(y1, x1)

        x2 = paddle.ones((1,))
        x2.stop_gradient = False
        y2 = paddle.jit.to_static(tanh_high_order_grad)(x2)
        g2 = paddle.grad(y2, x2)

        np.testing.assert_equal(g1[0].numpy(), g2[0].numpy())


class TestGradNone(unittest.TestCase):
    def test_grad_none(self):
        @paddle.jit.to_static
        def matmul_high_order_grad(x, y):
            z = paddle.matmul(x, y)
            g = paddle.grad(z, [x, y], create_graph=False)
            return g[0]

        x = paddle.ones([1])
        x.stop_gradient = False
        y = paddle.ones([1])
        y.stop_gradient = False
        g = matmul_high_order_grad(x, y)
        o = paddle.grad(g, x)
        np.testing.assert_equal(y.grad, None)


if __name__ == "__main__":
    unittest.main()
