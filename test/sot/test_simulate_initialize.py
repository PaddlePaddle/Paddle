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
from paddle import nn
from paddle.jit.sot import symbolic_translate
from paddle.jit.sot.utils import strict_mode_guard


class A:
    def __init__(self, vals):
        vals.append(1)


def foo(x, y):
    out = nn.Softmax()(paddle.to_tensor([x, y], dtype="float32"))
    return out


def foo2(x, y):
    t = nn.Softmax()
    out1 = t(paddle.to_tensor([x, y], dtype="float32"))
    out2 = t(paddle.to_tensor([x, y], dtype="float32"))
    return out1 + out2


def error_foo(x):
    t = nn.Linear(10, 10)
    return t(x)


class NopLayer(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.weight = None


def created_layer_reconstruct():
    x = paddle.to_tensor([1, 2], dtype="float32")
    weight = NopLayer().weight
    if weight is not None:
        x += 1
    return x


def bar(x):
    a = A(x)
    t = paddle.to_tensor(x)
    return t.mean()


class TestInit(TestCaseBase):
    def test_init_paddle_layer(self):
        self.assert_results(foo, 1, 2)
        self.assert_results(foo2, 1, 2)

    def test_init_python_object(self):
        sot_output = symbolic_translate(bar)([1.0, 2.0])
        dyn_output = bar([1.0, 2.0])
        self.assert_nest_match(sot_output, dyn_output)

    def test_error(self):
        def run():
            inputs = paddle.randn((10, 10))
            symbolic_translate(error_foo)(inputs)

        self.assertRaises(paddle.jit.sot.utils.exceptions.InnerError, run)

    @strict_mode_guard(False)
    def test_created_layer_reconstruct(self):
        self.assert_results(created_layer_reconstruct)


if __name__ == "__main__":
    unittest.main()
