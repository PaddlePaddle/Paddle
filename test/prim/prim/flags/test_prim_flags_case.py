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


def apply_to_static(net, use_cinn):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(
        net, build_strategy=build_strategy, full_graph=True
    )


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
        os.environ["FLAGS_prim_backward"] = "False"
        os.environ["FLAGS_prim_forward"] = "False"
        if os.getenv("FLAGS_prim_all"):
            del os.environ["FLAGS_prim_all"]
        core.check_and_set_prim_all_enabled()

    def train(self, use_cinn):
        net = PrimeNet()
        net = apply_to_static(net, use_cinn)

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

        if self.flag in ["prim_all", "cinn_prim_all"]:
            self.assertTrue('softmax' not in ops)
            self.assertTrue('exp_grad' not in ops)
        elif self.flag in ["prim_forward", "cinn_prim_forward"]:
            self.assertTrue('softmax' not in ops)
            self.assertTrue('exp_grad' in ops)
        elif self.flag in ["prim_backward", "cinn_prim_backward"]:
            self.assertTrue('softmax' in ops)
            self.assertTrue('exp_grad' not in ops)
        elif self.flag == "cinn":
            self.assertTrue('softmax' in ops)
            self.assertTrue('exp_grad' in ops)
        else:
            raise TypeError

    def test_cinn_prim_all(self):
        """cinn + prim forward + prim backward"""
        self.reset_env_flag()
        os.environ["FLAGS_prim_all"] = "True"
        self.flag = "cinn_prim_all"
        _ = self.train(use_cinn=True)

    def test_prim_all(self):
        """prim forward + prim backward"""
        self.reset_env_flag()
        os.environ["FLAGS_prim_all"] = "True"
        self.flag = "prim_all"
        _ = self.train(use_cinn=False)

    def test_cinn_prim_forward(self):
        """cinn + prim forward"""
        self.reset_env_flag()
        os.environ["FLAGS_prim_forward"] = "True"
        self.flag = "cinn_prim_forward"
        _ = self.train(use_cinn=True)

    def test_prim_forward(self):
        """only prim forward"""
        self.reset_env_flag()
        os.environ["FLAGS_prim_forward"] = "True"
        self.flag = "prim_forward"
        _ = self.train(use_cinn=False)

    def test_cinn_prim_backward(self):
        """cinn + prim_backward"""
        self.reset_env_flag()
        os.environ["FLAGS_prim_backward"] = "True"
        self.flag = "cinn_prim_backward"
        _ = self.train(use_cinn=True)

    def test_prim_backward(self):
        """only prim backward"""
        self.reset_env_flag()
        os.environ["FLAGS_prim_backward"] = "True"
        self.flag = "prim_backward"
        _ = self.train(use_cinn=False)

    def test_cinn(self):
        """only cinn"""
        self.reset_env_flag()
        self.flag = "cinn"
        _ = self.train(use_cinn=True)


if __name__ == '__main__':
    unittest.main()
