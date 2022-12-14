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

import paddle
import paddle.fluid as fluid
import paddle.nn.functional as F
from paddle.jit.api import declarative


def call_lambda_as_func(x):
    x = fluid.dygraph.to_variable(x)

    add_func = lambda x, y: x + y
    mean_func = lambda x: paddle.mean(x)

    y = add_func(x, 1)
    y = add_func(y, add_func(y, -1))
    out = mean_func(y)

    return out


def call_lambda_directly(x):
    x = fluid.dygraph.to_variable(x)

    y = (lambda x, y: x + y)(x, x)
    out = (lambda x: paddle.mean(x))(y)

    return out


def call_lambda_in_func(x):
    x = fluid.dygraph.to_variable(x)

    add_func = lambda x: x + 1

    y = paddle.mean((lambda x: F.relu(x))(x))
    out = add_func(y) if y > 1 and y < 2 else (lambda x: x**2)(y)

    return out


def call_lambda_with_ifExpr(x):
    x = fluid.dygraph.to_variable(x)

    add_func = lambda x: x + 1

    y = paddle.mean(x)
    out = add_func(y) if y or y < 2 else (lambda x: x**2)(y)

    return out


def call_lambda_with_ifExpr2(x):
    x = fluid.dygraph.to_variable(x)

    add_func = lambda x: x + 1

    y = paddle.mean(x)

    # NOTE: y is Variable, but z<2 is python bool value
    z = 0
    out = add_func(y) if y or z < 2 else (lambda x: x**2)(y)

    return out


class TestLambda(unittest.TestCase):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.x = np.array([1, 3]).astype('float32')
        self.place = (
            fluid.CUDAPlace(0)
            if fluid.is_compiled_with_cuda()
            else fluid.CPUPlace()
        )
        self.init_func()

    def init_func(self):
        self.dyfuncs = [
            call_lambda_as_func,
            call_lambda_directly,
            call_lambda_in_func,
            call_lambda_with_ifExpr,
            call_lambda_with_ifExpr2,
        ]

    def run_static(self, func):
        return self.run_dygraph(func, to_static=True)

    def run_dygraph(self, func, to_static=False):

        with fluid.dygraph.guard(self.place):
            x_v = fluid.dygraph.to_variable(self.x)
            if to_static:
                ret = declarative(func)(x_v)
            else:
                ret = func(x_v)
            return ret.numpy()

    def test_ast_to_func(self):
        for func in self.dyfuncs:
            self.assertTrue(
                (self.run_dygraph(func) == self.run_static(func)).all()
            )


if __name__ == '__main__':
    unittest.main()
