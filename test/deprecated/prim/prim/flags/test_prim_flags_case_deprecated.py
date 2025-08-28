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

import os
import unittest

import paddle
import paddle.nn.functional as F
from paddle.base import core
from paddle.base.core import (
    __check_and_set_prim_all_enabled as check_and_set_prim_all_enabled,
)


def apply_to_static(net):
    return paddle.jit.to_static(net, backend=None, full_graph=True)


class PrimeNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = F.softmax(x)
        res = paddle.exp(out)
        return res


class TestPrimForwardAndBackward(unittest.TestCase):
    """
    Test PrimeNet with @to_static + prim forward + prim backward + cinn v.s Dygraph
    """

    def setUp(self):
        paddle.seed(2022)
        self.x = paddle.randn([2, 4])
        self.x.stop_gradient = False
        self.flag = None

    def reset_env_flag(self):
        if os.getenv("FLAGS_prim_backward"):
            del os.environ["FLAGS_prim_backward"]
        if os.getenv("FLAGS_prim_forward"):
            del os.environ["FLAGS_prim_forward"]
        if os.getenv("FLAGS_prim_all"):
            del os.environ["FLAGS_prim_all"]
        core._set_prim_all_enabled(False)

    def train(self):
        net = PrimeNet()
        net = apply_to_static(net)

        out = net(self.x)
        loss = paddle.mean(out)
        loss.backward()

        self.check_prim(net)

    def check_prim(self, net):
        ops = [
            op.type
            for op in net.forward.program_cache.last()[-1][-1]
            .train_program.block(0)
            .ops
        ]

        if self.flag in ["prim_all"]:
            self.assertTrue('softmax' not in ops)
            self.assertTrue('exp_grad' not in ops)
        elif self.flag in ["prim_forward"]:
            self.assertTrue('softmax' not in ops)
            self.assertTrue('exp_grad' in ops)
        elif self.flag in ["prim_backward"]:
            self.assertTrue('softmax' in ops)
            self.assertTrue('exp_grad' not in ops)
        else:
            raise TypeError

    def test_prim_all(self):
        """prim forward + prim backward"""
        self.reset_env_flag()
        os.environ["FLAGS_prim_all"] = "True"
        check_and_set_prim_all_enabled()
        self.flag = "prim_all"
        _ = self.train()

    def test_prim_forward(self):
        """only prim forward"""
        self.reset_env_flag()
        os.environ["FLAGS_prim_forward"] = "True"
        check_and_set_prim_all_enabled()
        self.flag = "prim_forward"
        _ = self.train()

    def test_prim_backward(self):
        """only prim backward"""
        self.reset_env_flag()
        os.environ["FLAGS_prim_backward"] = "True"
        check_and_set_prim_all_enabled()
        self.flag = "prim_backward"
        _ = self.train()


if __name__ == '__main__':
    unittest.main()
