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

import numpy as np
from dygraph_to_static_utils import (
    Dy2StTestBase,
)

import paddle
import paddle.nn.functional as F


def call_lambda_as_func(x):
    x = paddle.to_tensor(x)

    add_func = lambda x, y: x + y
    mean_func = lambda x: paddle.mean(x)

    y = add_func(x, 1)
    y = add_func(y, add_func(y, -1))
    out = mean_func(y)

    return out


def call_lambda_directly(x):
    x = paddle.to_tensor(x)

    y = (lambda x, y: x + y)(x, x)
    out = (lambda x: paddle.mean(x))(y)

    return out


def call_lambda_in_func(x):
    x = paddle.to_tensor(x)

    add_func = lambda x: x + 1

    y = paddle.mean((lambda x: F.relu(x))(x))
    out = add_func(y) if y > 1 and y < 2 else (lambda x: x**2)(y)

    return out


def call_lambda_with_if_expr(x):
    x = paddle.to_tensor(x)

    add_func = lambda x: x + 1

    y = paddle.mean(x)
    out = add_func(y) if y or y < 2 else (lambda x: x**2)(y)

    return out


def call_lambda_with_if_expr2(x):
    x = paddle.to_tensor(x)

    add_func = lambda x: x + 1

    y = paddle.mean(x)

    # NOTE: y is Variable, but z<2 is python bool value
    z = 0
    out = add_func(y) if y or z < 2 else (lambda x: x**2)(y)

    return out


class TestLambda(Dy2StTestBase):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.x = np.array([1, 3]).astype('float32')

    def run_static(self, func):
        return self.run_dygraph(func, to_static=True)

    def run_dygraph(self, func, to_static=False):
        x_v = paddle.to_tensor(self.x)
        if to_static:
            ret = paddle.jit.to_static(func)(x_v)
        else:
            ret = func(x_v)
        return ret.numpy()

    def test_call_lambda_as_func(self):
        fn = call_lambda_as_func
        self.assertTrue((self.run_dygraph(fn) == self.run_static(fn)).all())

    def test_call_lambda_directly(self):
        fn = call_lambda_directly
        self.assertTrue((self.run_dygraph(fn) == self.run_static(fn)).all())

    def test_call_lambda_in_func(self):
        fn = call_lambda_in_func
        self.assertTrue((self.run_dygraph(fn) == self.run_static(fn)).all())

    def test_call_lambda_with_if_expr(self):
        fn = call_lambda_with_if_expr
        self.assertTrue((self.run_dygraph(fn) == self.run_static(fn)).all())

    def test_call_lambda_with_if_expr2(self):
        fn = call_lambda_with_if_expr2
        self.assertTrue((self.run_dygraph(fn) == self.run_static(fn)).all())


if __name__ == '__main__':
    unittest.main()
