# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph import declarative
from paddle.fluid.dygraph import ProgramTranslator

SEED = 2020
np.random.seed(SEED)


@declarative
def test_return_base(x):
    x = fluid.dygraph.to_variable(x)
    return x


@declarative
def test_inside_func_base(x):
    x = fluid.dygraph.to_variable(x)

    def inner_func(x):
        return x

    return inner_func(x)


@declarative
def test_return_if(x):
    x = fluid.dygraph.to_variable(x)
    if x < 0:
        x -= 1
        return -x
    x += 3
    return x


@declarative
def test_return_if_else(x):
    x = fluid.dygraph.to_variable(x)
    if x > 0:
        x += 10086
        return x
        x -= 3  # useless statement to test our code can handle it.
    else:
        x += 6666
        return x
        x -= 8888  # useless statement to test our code can handle it.


@declarative
def test_return_in_while(x):
    x = fluid.dygraph.to_variable(x)
    i = fluid.layers.fill_constant(shape=[1], dtype='int32', value=0)
    while i < 10:
        i += 1
        if i > 5:
            x += 110
            return x
        x += i
    return x


@declarative
def test_return_in_for(x):
    x = fluid.dygraph.to_variable(x)
    for i in range(10):
        if i <= 4:
            x += 1
            continue
        else:
            return x + 10086
    return x - 1


class TestReturnBase(unittest.TestCase):
    def setUp(self):
        self.input = np.ones((1)).astype('int32')
        self.place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        self.init_dygraph_func()
        self.program_translator = ProgramTranslator()

    def init_dygraph_func(self):
        self.dygraph_func = test_return_base

    def run_dygraph_mode(self):
        self.program_translator.enable(False)
        with fluid.dygraph.guard():
            res = self.dygraph_func(self.input)
            return res.numpy()

    def run_static_mode(self):
        self.program_translator.enable(True)
        with fluid.dygraph.guard():
            res = self.dygraph_func(self.input)
            return res.numpy()

    def test_transformed_static_result(self):
        static_res = self.run_static_mode()
        dygraph_res = self.run_dygraph_mode()
        self.assertTrue(
            np.allclose(dygraph_res, static_res),
            msg='dygraph res is {}\nstatic_res is {}'.format(dygraph_res,
                                                             static_res))


class TestInsideFuncBase(TestReturnBase):
    def init_dygraph_func(self):
        self.dygraph_func = test_inside_func_base


class TestReturnIf(TestReturnBase):
    def init_dygraph_func(self):
        self.dygraph_func = test_return_if


class TestReturnIfElse(TestReturnBase):
    def init_dygraph_func(self):
        self.dygraph_func = test_return_if_else


class TestReturnInWhile(TestReturnBase):
    def init_dygraph_func(self):
        self.dygraph_func = test_return_in_while


class TestReturnInFor(TestReturnBase):
    def init_dygraph_func(self):
        self.dygraph_func = test_return_in_for


if __name__ == '__main__':
    unittest.main()
