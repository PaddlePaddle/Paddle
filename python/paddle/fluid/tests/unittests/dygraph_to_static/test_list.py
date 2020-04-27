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
from functools import partial

import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph.jit import declarative
from paddle.fluid.layers.utils import map_structure

SEED = 2020
np.random.seed(SEED)


# Situation 1: Test list append
@declarative
def test_list_append_without_control_flow(x):
    # Python list will not be transformed.
    x = fluid.dygraph.to_variable(x)
    a = []
    # It's a plain python control flow which won't be transformed
    if 2 > 1:
        a.append(x)
    return a


@declarative
def test_list_append_in_if(x):
    x = fluid.dygraph.to_variable(x)
    a = []
    if x.numpy()[0] > 0:
        a.append(x)
    else:
        a.append(
            fluid.layers.fill_constant(
                shape=[1, 2], value=9, dtype="int64"))
    return a


@declarative
def test_list_append_in_for_loop(x, iter_num):
    x = fluid.dygraph.to_variable(x)
    # Use `fill_constant` so that static analysis can analyze the type of iter_num is Tensor
    iter_num = fluid.layers.fill_constant(
        shape=[1], value=iter_num, dtype="int32"
    )  # TODO(liym27): Delete it if the type of parameter iter_num can be resolved
    a = []
    for i in range(iter_num):
        a.append(x)
    return a


@declarative
def test_list_append_in_for_loop_with_concat(x, iter_num):
    x = fluid.dygraph.to_variable(x)
    a = []
    # Use `fill_constant` so that static analysis can analyze the type of iter_num is Tensor
    iter_num = fluid.layers.fill_constant(
        shape=[1], value=iter_num, dtype="int32"
    )  # TODO(liym27): Delete it if the type of parameter iter_num can be resolved
    for i in range(iter_num):
        a.append(x)
    a = fluid.layers.concat(a, axis=0)
    return a


@declarative
def test_list_append_in_while_loop(x, iter_num):
    x = fluid.dygraph.to_variable(x)
    iter_num = fluid.layers.fill_constant(
        shape=[1], value=iter_num, dtype="int32")
    a = []
    i = 0
    while i < iter_num:
        a.append(x)
        i += 1
    return a


@declarative
def test_list_append_in_while_loop_with_stack(x, iter_num):
    x = fluid.dygraph.to_variable(x)
    iter_num = fluid.layers.fill_constant(
        shape=[1], value=iter_num, dtype="int32")
    a = []
    i = 0
    while i < iter_num.numpy()[0]:
        a.append(x)
        i += 1
    out = fluid.layers.stack(a, axis=1)
    return out


# Situation 2: Test list pop
@declarative
def test_list_pop_without_control_flow_1(x):
    x = fluid.dygraph.to_variable(x)
    a = []
    if 2 > 1:
        a.append(x)
    a.pop()
    return a


@declarative
def test_list_pop_without_control_flow_2(x):
    x = fluid.dygraph.to_variable(x)
    a = []
    if 2 > 1:
        a.append(x)
        a.append(x)
    last_tiem = a.pop()
    return last_tiem


@declarative
def test_list_pop_in_if(x):
    x = fluid.dygraph.to_variable(x)
    a = []
    if x.numpy()[0] > 0:
        a.append(x)
        a.append(
            fluid.layers.fill_constant(
                shape=[1, 2], value=9, dtype="int64"))
    else:
        a.append(x)
        a.append(
            fluid.layers.fill_constant(
                shape=[1, 2], value=9, dtype="int64"))
    item1 = a.pop()
    a.pop()
    return a, item1


@declarative
def test_list_pop_in_for_loop(x, iter_num):
    x = fluid.dygraph.to_variable(x)
    # Use `fill_constant` so that static analysis can analyze the type of iter_num is Tensor
    iter_num = fluid.layers.fill_constant(
        shape=[1], value=iter_num, dtype="int32"
    )  # TODO(liym27): Delete it if the type of parameter iter_num can be resolved

    a = []
    for i in range(iter_num):
        a.append(x)

    one = fluid.layers.ones(shape=[1], dtype="int32")
    for i in range(one.numpy()[0]):
        item = a.pop()

    return a, item


@declarative
def test_list_pop_in_while_loop(x, iter_num):
    x = fluid.dygraph.to_variable(x)
    iter_num = fluid.layers.fill_constant(
        shape=[1], value=iter_num, dtype="int32")
    a = []
    i = 0
    while i < iter_num:
        a.append(x)
        i += 1
        if i % 2 == 1:
            a.pop()
    return a


class TestListWithoutControlFlow(unittest.TestCase):
    def setUp(self):
        self.place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
        ) else fluid.CPUPlace()

        self.init_data()
        self.init_dygraph_func()

    def init_data(self):
        self.input = np.random.random((3)).astype('int32')

    def init_dygraph_func(self):
        self.all_dygraph_funcs = [
            test_list_append_without_control_flow,
            test_list_pop_without_control_flow_1,
            test_list_pop_without_control_flow_2,
        ]

    def varbase_to_numpy(self, res):
        if isinstance(res, (list, tuple)):
            res = map_structure(lambda x: x.numpy(), res)
        else:
            res = [res.numpy()]
        return res

    def run_dygraph_mode(self):
        with fluid.dygraph.guard():
            res = self.dygraph_func(self.input)
            return self.varbase_to_numpy(res)

    def run_static_mode(self):
        main_program = fluid.Program()
        with fluid.program_guard(main_program):
            res = self.dygraph_func(self.input)
            return self.varbase_to_numpy(res)

    def test_transformed_static_result(self):
        for dyfunc in self.all_dygraph_funcs:
            self.dygraph_func = dyfunc
            static_res_list = self.run_static_mode()
            dygraph_res_list = self.run_dygraph_mode()

            self.assertEqual(len(static_res_list), len(dygraph_res_list))
            for stat_res, dy_res in zip(static_res_list, dygraph_res_list):
                self.assertTrue(
                    np.allclose(stat_res, dy_res),
                    msg='dygraph_res is {}\nstatic_res is {}'.format(stat_res,
                                                                     dy_res))


class TestListInIf(TestListWithoutControlFlow):
    def init_dygraph_func(self):
        self.all_dygraph_funcs = [test_list_append_in_if, test_list_pop_in_if]


class TestListInWhileLoop(TestListWithoutControlFlow):
    def init_data(self):
        self.input = np.random.random((3)).astype('int32')
        self.iter_num = 3

    def init_dygraph_func(self):
        self.all_dygraph_funcs = [
            partial(
                test_list_append_in_while_loop, iter_num=self.iter_num),
            partial(
                test_list_pop_in_while_loop, iter_num=self.iter_num),
        ]


class TestListInWhileLoopWithStack(TestListInWhileLoop):
    def init_dygraph_func(self):
        self.all_dygraph_funcs = [
            partial(
                test_list_append_in_while_loop_with_stack,
                iter_num=self.iter_num)
        ]


class TestListInForLoop(TestListInWhileLoop):
    def init_dygraph_func(self):
        self.all_dygraph_funcs = [
            partial(
                test_list_append_in_for_loop, iter_num=self.iter_num),
            partial(
                test_list_pop_in_for_loop, iter_num=self.iter_num),
        ]


class TestListInForLoopWithConcat(TestListInWhileLoopWithStack):
    def init_dygraph_func(self):
        self.all_dygraph_funcs = [
            partial(
                test_list_append_in_for_loop_with_concat,
                iter_num=self.iter_num)
        ]


if __name__ == '__main__':
    unittest.main()
