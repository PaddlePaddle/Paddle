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
from dygraph_to_static_util import test_and_compare_with_new_ir

import paddle


class Net(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.relu = paddle.nn.functional.relu
        self.fc = paddle.nn.Linear(4, 4)

    def forward(self, x):
        y = paddle.full_like(x, 1.0)
        y.stop_gradient = False
        z = self.fc(x) * y
        out = y + z
        out = self.relu(out)

        return out


def apply_to_static(net, use_cinn):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(net, build_strategy=build_strategy)


class TestCINN(unittest.TestCase):
    def setUp(self):
        self.x = paddle.randn([2, 4])
        self.x.stop_gradient = False

    def train(self, use_cinn):
        paddle.seed(2022)
        net = Net()
        sgd = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=net.parameters()
        )
        if use_cinn:
            net = apply_to_static(net, use_cinn)

        res = []
        for step in range(10):
            out = net(self.x)
            loss = paddle.mean(out)
            loss.backward()
            sgd.step()
            sgd.clear_grad()

            res.append(out.numpy())

            if use_cinn and paddle.device.is_compiled_with_cinn():
                self.assertTrue(
                    paddle.framework.core.is_run_with_cinn(),
                    msg="The test was not running with CINN! Please check.",
                )
            else:
                self.assertFalse(
                    paddle.framework.core.is_run_with_cinn(),
                    msg="The test should not running with CINN when the whl package was not compiled with CINN! Please check.",
                )

        return res

    @test_and_compare_with_new_ir(False)
    def test_cinn(self):
        dy_res = self.train(use_cinn=False)
        cinn_res = self.train(use_cinn=True)

        for i in range(len(dy_res)):
            np.testing.assert_array_equal(cinn_res[i], dy_res[i])


if __name__ == '__main__':
    unittest.main()
