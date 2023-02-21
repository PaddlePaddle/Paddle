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
import paddle.nn.functional as F
from paddle.fluid import core


def apply_to_static(net, use_cinn):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(net, build_strategy=build_strategy)


class PrimeNet(paddle.nn.Layer):
    def __init__(self):
        super(PrimeNet, self).__init__()
        self.fc = paddle.nn.Linear(4, 4)

    def forward(self, x):
        y = self.fc(x)
        out = F.softmax(y)
        return out


class TestPrimForward(unittest.TestCase):
    """
    This case only tests prim_forward + to_static + cinn. Thus we need to
    set this flag as False to avoid prim_backward.
    core.set_prim_backward(False)
    """

    def setUp(self):
        paddle.seed(2022)
        self.x = paddle.randn([2, 4])
        self.x.stop_gradient = False

    def train(self, use_prim):
        paddle.seed(2022)
        net = PrimeNet()
        sgd = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=net.parameters()
        )
        core._set_prim_forward_enabled(use_prim)
        if use_prim:
            net = apply_to_static(net, use_prim)

        res = []
        for _ in range(10):
            out = net(self.x)
            loss = paddle.mean(out)
            loss.backward()
            sgd.step()
            sgd.clear_grad()

            res.append(out.numpy())

        self.check_prim(net, use_prim)

        return res

    def check_prim(self, net, use_prim):
        if not use_prim:
            return
        fwd_ops = [op.type for op in net.forward.main_program.block(0).ops]
        # Ensure that softmax is splitted into small ops
        self.assertTrue('softmax' not in fwd_ops)

    def test_cinn_prim_forward(self):
        dy_res = self.train(use_prim=False)
        cinn_res = self.train(use_prim=True)

        for i in range(len(dy_res)):
            np.testing.assert_allclose(
                cinn_res[i], dy_res[i], rtol=1e-7, atol=1e-7
            )


class TestPrimForwardAndBackward(unittest.TestCase):
    """
    Test PrimeNet with @to_static + prim forward + prim backward + cinn v.s Dygraph
    """

    def setUp(self):
        paddle.seed(2022)
        self.x = paddle.randn([2, 4])
        self.x.stop_gradient = False

    def train(self, use_prim):
        paddle.seed(2022)
        net = PrimeNet()
        sgd = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=net.parameters()
        )
        core._set_prim_all_enabled(use_prim)
        if use_prim:
            net = apply_to_static(net, use_prim)

        res = []
        for _ in range(10):
            out = net(self.x)
            loss = paddle.mean(out)
            loss.backward()
            sgd.step()
            sgd.clear_grad()

            res.append(out.numpy())

        self.check_prim(net, use_prim)

        return res

    def check_prim(self, net, use_prim):
        if not use_prim:
            return
        fwd_ops = [op.type for op in net.forward.main_program.block(0).ops]
        # Ensure that softmax is splitted into small ops
        self.assertTrue('softmax' not in fwd_ops)

    def test_cinn_prim(self):
        dy_res = self.train(use_prim=False)
        cinn_res = self.train(use_prim=True)

        for i in range(len(dy_res)):
            np.testing.assert_allclose(
                cinn_res[i], dy_res[i], rtol=1e-6, atol=1e-6
            )


if __name__ == '__main__':
    unittest.main()
