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
        self.fc = paddle.nn.Linear(10, 10)

    def forward(self, x):
        return F.dropout(self.fc(x))


class TestPrimForwardAndBackward(unittest.TestCase):
    def setUp(self):
        self.x = paddle.ones([100, 100, 10])
        self.x.stop_gradient = False

    def train(self, use_prim=True, use_cinn=False):
        net = PrimeNet()
        sgd = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=net.parameters()
        )

        core._set_prim_all_enabled(use_prim)
        net = apply_to_static(net, use_cinn)

        fwd, rev = 0.0, 0.0
        for _ in range(10):
            out = net(self.x)
            loss = paddle.sum(out)
            loss.backward()
            fwd += loss.numpy()
            rev += self.x.grad.sum().numpy()
            sgd.step()
            self.x.clear_gradient()
        return fwd, rev, net

    def check_prim(self, net):
        ops = [op.type for op in net.forward.main_program.block(0).ops]
        # Ensure that dropout is splitted into small ops
        self.assertTrue('dropout' not in ops)

    def test_cinn_prim_forward(self):
        paddle.seed(1999)
        dy_fwd, dy_rev, _ = self.train(use_prim=False, use_cinn=False)
        paddle.seed(1999)
        cinn_fwd, cinn_rev, prim_net = self.train(use_prim=True, use_cinn=True)

        self.check_prim(prim_net)
        np.testing.assert_allclose(dy_fwd, cinn_fwd, rtol=1e-2, atol=0)
        np.testing.assert_allclose(dy_rev, cinn_rev, rtol=1e-2, atol=0)


if __name__ == '__main__':
    unittest.main()
