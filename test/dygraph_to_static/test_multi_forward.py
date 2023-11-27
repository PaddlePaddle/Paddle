#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from dygraph_to_static_utils_new import (
    Dy2StTestBase,
    test_legacy_and_pir_exe_and_pir_api,
)

import paddle


class MyLayer(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


class TestBackward(Dy2StTestBase):
    @test_legacy_and_pir_exe_and_pir_api
    def test_order_0(self):
        """
        loss = 1 * w * 1 + 2 * w * 2
        delta_w = 5
        """
        model = paddle.jit.to_static(
            function=MyLayer(),
            input_spec=[
                paddle.static.InputSpec(
                    shape=[None, None], dtype=paddle.float32
                )
            ],
        )
        model.clear_gradients()
        inp = paddle.ones([1, 1])
        out1 = model(inp * 1)
        out2 = model(inp * 2)
        loss = out2 * 2 + out1 * 1
        loss.backward()
        self.assertEqual(model.linear.weight.grad, 5)

    @test_legacy_and_pir_exe_and_pir_api
    def test_order_1(self):
        """
        loss = 2 * w * 2  + 1 * w * 1
        delta_w = 5
        """
        model = paddle.jit.to_static(
            function=MyLayer(),
            input_spec=[
                paddle.static.InputSpec(
                    shape=[None, None], dtype=paddle.float32
                )
            ],
        )
        model.clear_gradients()
        inp = paddle.ones([1, 1])
        out1 = model(inp * 1)
        out2 = model(inp * 2)
        loss = out1 * 1 + out2 * 2
        loss.backward()
        self.assertEqual(model.linear.weight.grad, 5)


if __name__ == '__main__':
    unittest.main()
