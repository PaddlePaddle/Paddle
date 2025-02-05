# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import nn
from paddle.base import core, framework
from paddle.nn import BatchNorm

np.random.seed(2023)


class PrimeNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2D(2, 4, (3, 3), bias_attr=False)
        self.bn = BatchNorm(4, act="relu")

    def forward(self, x):
        y = self.conv(x)
        out = self.bn(y)
        res = F.max_pool2d(out, kernel_size=2, stride=2, padding=0)
        return res


class TestPrimAMPO1(unittest.TestCase):
    """
    Test PrimeNet with @to_static + prim v.s Dygraph in AMPO1.
    """

    def setUp(self):
        paddle.seed(2022)
        self.x = paddle.randn([4, 2, 6, 6], dtype="float32")
        self.x.stop_gradient = False

    def train(self, use_prim):
        core._set_prim_all_enabled(use_prim)
        paddle.seed(2022)
        net = PrimeNet()
        sgd = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=net.parameters()
        )

        if use_prim:
            net = paddle.jit.to_static(
                net, build_strategy=False, full_graph=True
            )
        with paddle.amp.auto_cast(level='O1'):
            out = net(self.x)
            loss = paddle.mean(out)
            loss.backward()
            sgd.step()
            sgd.clear_grad()
            return loss

    def test_amp_01(self):
        if not isinstance(framework._current_expected_place(), core.CPUPlace):
            expected = self.train(False)
            actual = self.train(True)
            np.testing.assert_allclose(
                expected,
                actual,
                rtol=1e-3,
                atol=1e-3,
            )

    def test_amp_O1_infer(self):
        if not isinstance(framework._current_expected_place(), core.CPUPlace):
            net = PrimeNet()
            core._set_prim_all_enabled(False)
            net.eval()
            static_net = paddle.jit.to_static(
                net, build_strategy=False, full_graph=True
            )
            res = static_net(self.x)

            # set prim all enabled
            core._set_prim_all_enabled(True)
            net.eval()
            static_net = paddle.jit.to_static(
                net, build_strategy=False, full_graph=True
            )
            with paddle.amp.auto_cast(level='O1'):
                res_amp = static_net(self.x)

            np.testing.assert_allclose(
                res,
                res_amp,
                rtol=1e-3,
                atol=1e-3,
            )


if __name__ == '__main__':
    unittest.main()
