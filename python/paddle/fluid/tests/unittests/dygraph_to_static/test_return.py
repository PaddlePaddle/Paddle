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
import paddle.fluid.core as core
from paddle.fluid.dygraph import declarative
from paddle.fluid.dygraph import ProgramTranslator

from ifelse_simple_func import dyfunc_with_if_else

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


@declarative
def test_recursive_return(x):
    x = fluid.dygraph.to_variable(x)
    return dyfunc_with_if_else(x)


@declarative
def test_return_different_length_if_body(x):
    x = fluid.dygraph.to_variable(x)
    y = x + 1
    if x > 0:
        # x = to_variable(np.ones(1)) so it will return here
        return x, y
    else:
        return x


@declarative
def test_return_different_length_else(x):
    x = fluid.dygraph.to_variable(x)
    y = x + 1
    if x < 0:
        return x, y
    else:
        # x = to_variable(np.ones(1)) so it will return here
        return x


@declarative
def test_no_return(x):
    x = fluid.dygraph.to_variable(x)
    y = x + 1


@declarative
def test_return_none(x):
    x = fluid.dygraph.to_variable(x)
    y = x + 1
    if x > 0:
        # x = to_variable(np.ones(1)) so it will return here
        return None
    else:
        return x, y


@declarative
def test_return_no_variable(x):
    x = fluid.dygraph.to_variable(x)
    y = x + 1
    if x < 0:
        return x, y
    else:
        # x = to_variable(np.ones(1)) so it will return here
        return


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
            if isinstance(res, (tuple)):
                return tuple(r.numpy() for r in res)
            elif isinstance(res, core.VarBase):
                return res.numpy()
            return res

    def run_static_mode(self):
        self.program_translator.enable(True)
        with fluid.dygraph.guard():
            res = self.dygraph_func(self.input)
            if isinstance(res, tuple):
                return tuple(r.numpy() for r in res)
            elif isinstance(res, core.VarBase):
                return res.numpy()
            return res

    def test_transformed_static_result(self):
        dygraph_res = self.run_dygraph_mode()
        static_res = self.run_static_mode()
        if isinstance(dygraph_res, tuple):
            self.assertTrue(isinstance(static_res, tuple))
            self.assertEqual(len(dygraph_res), len(static_res))
            for i in range(len(dygraph_res)):
                self.assertTrue(
                    np.allclose(dygraph_res[i], static_res[i]),
                    msg='dygraph res is {}\nstatic_res is {}'.format(
                        dygraph_res[i], static_res[i]))

        elif isinstance(dygraph_res, np.ndarray):
            self.assertTrue(
                np.allclose(dygraph_res, static_res),
                msg='dygraph res is {}\nstatic_res is {}'.format(dygraph_res,
                                                                 static_res))
        else:
            self.assertEqual(dygraph_res, static_res)


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


class TestRecursiveReturn(TestReturnBase):
    def init_dygraph_func(self):
        self.input = self.input.astype(np.float32)
        self.dygraph_func = test_recursive_return


class TestReturnDifferentLengthIfBody(TestReturnBase):
    def init_dygraph_func(self):
        self.dygraph_func = test_return_different_length_if_body


class TestReturnDifferentLengthElse(TestReturnBase):
    def init_dygraph_func(self):
        self.dygraph_func = test_return_different_length_else


class TestNoReturn(TestReturnBase):
    def init_dygraph_func(self):
        self.dygraph_func = test_no_return


class TestReturnNone(TestReturnBase):
    def init_dygraph_func(self):
        self.dygraph_func = test_return_none


class TestReturnNoVariable(TestReturnBase):
    def init_dygraph_func(self):
        self.dygraph_func = test_return_no_variable


if __name__ == '__main__':
    unittest.main()
