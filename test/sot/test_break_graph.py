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
from test_case_base import TestCaseBase

import paddle
from paddle.jit.sot.utils.paddle_api_config import add_break_graph_apis


def ifelse_func(x, y):
    if x > 0:
        y = y + 1
    else:
        y = y + 2
    return y


class TestIfElse(TestCaseBase):
    def test_simple(self):
        x = paddle.to_tensor([1.0])
        y = paddle.to_tensor([2.0])
        self.assert_results(ifelse_func, x, y)


def multi_output(x: paddle.Tensor):
    m = x + 1
    if x > 0:
        return m
    else:
        return 2 * m


class TestBreakgraph(TestCaseBase):
    def test_simple(self):
        x = paddle.to_tensor(2)
        self.assert_results(multi_output, x)
        x = paddle.to_tensor(-2)
        self.assert_results(multi_output, x)


def print_break_graph(x, y):
    z = x + y
    print(x, z)
    out = y * z * 2
    return out


class TestPrint(TestCaseBase):
    def test_simple(self):
        x = paddle.to_tensor(2)
        y = paddle.to_tensor(3)
        self.assert_results(print_break_graph, x, y)


def to_tensor_break_graph(x, y):
    z = x + y
    out = y * paddle.to_tensor(2) * z
    return out


class TestToTensor(TestCaseBase):
    def test_simple(self):
        add_break_graph_apis([paddle.to_tensor])
        x = paddle.to_tensor(2)
        y = paddle.to_tensor(3)
        self.assert_results(to_tensor_break_graph, x, y)


def tensor_clear_gradient(x):
    x = paddle.to_tensor(x)
    x.clear_gradient()
    return x


class TestBreakGraphInResumeFn(TestCaseBase):
    def test_simple(self):
        x = paddle.to_tensor(2)
        self.assert_results(tensor_clear_gradient, x)


def inner_fn(a, b, c, d):
    return a + b * c - d


def multi_stack_args(a, b, c):
    out = inner_fn(a, b, c, paddle.to_tensor(4))
    return out


class TestMultiStackArgs(TestCaseBase):
    def test_simple(self):
        a = paddle.to_tensor(1)
        b = paddle.to_tensor(2)
        c = paddle.to_tensor(3)
        self.assert_results(multi_stack_args, a, b, c)


def break_graph_in_call_method(x):
    out = paddle.nn.functional.relu(paddle.to_tensor([4.0]))
    return x + out


def numpy_break_graph():
    a = paddle.to_tensor([1, 2])
    b = np.sum(a.numpy())
    print(b)
    return b


class TestBreakGraphInCallMethod(TestCaseBase):
    def test_simple(self):
        x = paddle.to_tensor([1.0])
        break_graph_in_call_method(x)
        x = paddle.to_tensor([2.0])
        break_graph_in_call_method(x)

        x = paddle.to_tensor([3.0])
        self.assert_results(break_graph_in_call_method, x)

    def test_numpy(self):
        self.assert_results(numpy_break_graph)


def test_break_graph_repeat(x):
    out = paddle.to_tensor(
        paddle.to_tensor(paddle.to_tensor(paddle.to_tensor([1.0])))
    )
    return x + out


class TestBreakGraphRepeat(TestCaseBase):
    def test_simple(self):
        x = paddle.to_tensor([1.0])
        test_break_graph_repeat(x)
        x = paddle.to_tensor([2.0])
        test_break_graph_repeat(x)

        x = paddle.to_tensor([3.0])
        self.assert_results(test_break_graph_repeat, x)


def break_graph_resume_pass_null(x, y):
    return paddle.add(x, y[0:50] if y is not None else None)


class TestBreakGraphResumePassNull(TestCaseBase):
    def test_break_graph_resume_pass_null(self):
        x = paddle.rand([50, 50], dtype=paddle.float32)
        y = paddle.rand([100, 50], dtype=paddle.float32)
        self.assert_results(break_graph_resume_pass_null, x, y)


class MyLayer(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.head = paddle.nn.Linear(3, 10)

    def forward_features(self, x):
        paddle.jit.sot.psdb.breakgraph()
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return self.head(x)


class TestBreakGraphInLayer(TestCaseBase):
    def test_break_graph_in_layer(self):
        x = paddle.rand([2, 3], dtype=paddle.float32)
        net = MyLayer()
        self.assert_results(net.forward, x)


def dummy(*args):
    return None


def break_graph_call_generator_function(x):
    return dummy(y for y in x)


class TestBreakGraphCallGeneratorFunction(TestCaseBase):
    def test_break_graph_when_call_generator_function(self):
        x = paddle.rand([1], dtype=paddle.float32)
        y = paddle.rand([1], dtype=paddle.float32)
        self.assert_results(break_graph_call_generator_function, [x, y])


def unary_not_break_graph(x):
    return not x


class TestUnaryNot(TestCaseBase):
    def test_unary_not_break_graph(self):
        x = paddle.to_tensor(0)
        self.assert_results(unary_not_break_graph, x)


if __name__ == "__main__":
    unittest.main()
