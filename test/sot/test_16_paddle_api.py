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

from test_case_base import TestCaseBase

import paddle
from paddle.nn.functional import relu


def paddle_api_method_call(x: paddle.Tensor):
    m = x + 2
    m = paddle.nn.functional.relu(m)
    return m


def paddle_api_function_call(x: paddle.Tensor):
    m = x + 2
    m = relu(m)
    return m


def paddle_api_function_call_concat(
    x: paddle.Tensor, y: paddle.Tensor, axis: int
):
    return paddle.concat([x, y], axis=axis)


def paddle_api_function_breakgraph_when_type_error(
    x: paddle.Tensor, axis: paddle.Tensor
):
    return paddle.nn.functional.softmax(x, axis=axis)


class TestPaddleApiCall(TestCaseBase):
    def test_paddle_api_method_call(self):
        self.assert_results(paddle_api_method_call, paddle.to_tensor(2.0))
        self.assert_results(paddle_api_method_call, paddle.to_tensor(-5.0))
        self.assert_results(paddle_api_method_call, paddle.to_tensor(0.0))

    def test_paddle_api_function_call(self):
        self.assert_results(paddle_api_function_call, paddle.to_tensor(2.0))
        self.assert_results(paddle_api_function_call, paddle.to_tensor(-5.0))
        self.assert_results(paddle_api_function_call, paddle.to_tensor(0.0))

    def test_paddle_api_function_call_concat(self):
        a = paddle.to_tensor([[1, 2], [3, 4]])
        b = paddle.to_tensor([[5, 6], [7, 8]])
        self.assert_results(paddle_api_function_call_concat, a, b, 0)
        self.assert_results(paddle_api_function_call_concat, a, b, 1)

    def test_paddle_api_function_breakgraph_when_type_error(self):
        x = paddle.to_tensor([[1, 2], [3, 4]], dtype=paddle.float32)
        axis = paddle.to_tensor(1)
        self.assert_results(
            paddle_api_function_breakgraph_when_type_error, x, axis
        )


if __name__ == "__main__":
    unittest.main()
