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
from paddle.fluid.dygraph.jit import dygraph_to_static_graph

SEED = 2020
np.random.seed(SEED)


def test_list_without_control_flow(x):
    # Python list will not be transformed.
    x = fluid.dygraph.to_variable(x)
    a = []
    # It's a plain python control flow which won't be transformed
    if 2 > 1:
        a.append(x)
    return a


def test_list_in_if(x):
    x = fluid.dygraph.to_variable(x)
    a = []
    if x.numpy()[0] > 0:
        a.append(x)
    else:
        a.append(
            fluid.layers.fill_constant(
                shape=[1, 2], value=9, dtype="int64"))
    return a


def test_list_in_for_loop(x, iter_num):
    x = fluid.dygraph.to_variable(x)
    a = []
    for i in range(iter_num):
        a.append(x)
    return a


def test_list_in_for_loop_with_concat(x, iter_num):
    x = fluid.dygraph.to_variable(x)
    a = []
    for i in range(iter_num):
        a.append(x)
    out = fluid.layers.concat(a, axis=0)
    return out


def test_list_in_while_loop(x, iter_num):
    x = fluid.dygraph.to_variable(x)
    iter_num = fluid.layers.fill_constant(
        shape=[1], value=iter_num, dtype="int32")
    a = []
    i = 0
    # Note: `i < iter_num` can't be supported in dygraph mode now,
    # but PR22892 is fixing it https://github.com/PaddlePaddle/Paddle/pull/22892.
    # If PR22892 merged, change `i < iter_num.numpy()[0]` to `i < iter_num`.
    while i < iter_num.numpy()[0]:
        a.append(x)
        i += 1
    return a


def test_list_in_while_loop_with_stack(x, iter_num):
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


class TestListWithoutControlFlow(unittest.TestCase):
    def setUp(self):
        self.input = np.random.random((3)).astype('int32')
        self.place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        self.init_dygraph_func()

    def init_dygraph_func(self):
        self.dygraph_func = test_list_without_control_flow

    def run_dygraph_mode(self):
        with fluid.dygraph.guard():
            res = self.dygraph_func(self.input)
            if isinstance(res, (list, tuple)):
                res = res[0]
            return res.numpy()

    def run_static_mode(self):
        main_program = fluid.Program()
        with fluid.program_guard(main_program):
            tensor_list = dygraph_to_static_graph(self.dygraph_func)(self.input)
        exe = fluid.Executor(self.place)
        static_res = exe.run(main_program, fetch_list=tensor_list[0])

        return static_res[0]

    def test_transformed_static_result(self):
        static_res = self.run_static_mode()
        dygraph_res = self.run_dygraph_mode()
        self.assertTrue(
            np.allclose(dygraph_res, static_res),
            msg='dygraph res is {}\nstatic_res is {}'.format(dygraph_res,
                                                             static_res))


class TestListInIf(TestListWithoutControlFlow):
    def init_dygraph_func(self):
        self.dygraph_func = test_list_in_if

    def run_static_mode(self):
        main_program = fluid.Program()
        with fluid.program_guard(main_program):
            tensor_array = dygraph_to_static_graph(self.dygraph_func)(
                self.input)
            static_out = fluid.layers.array_read(
                tensor_array,
                i=fluid.layers.fill_constant(
                    shape=[1], value=0, dtype='int64'))
        exe = fluid.Executor(self.place)
        numpy_res = exe.run(main_program, fetch_list=static_out)
        return numpy_res[0]


class TestListInWhileLoop(TestListWithoutControlFlow):
    def setUp(self):
        self.iter_num = 3
        self.input = np.random.random((3)).astype('int32')
        self.place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        self.init_dygraph_func()

    def init_dygraph_func(self):
        self.dygraph_func = test_list_in_while_loop

    def run_dygraph_mode(self):
        with fluid.dygraph.guard():
            var_res = self.dygraph_func(self.input, self.iter_num)
            numpy_res = [ele.numpy() for ele in var_res]
            return numpy_res

    def run_static_mode(self):
        main_program = fluid.Program()
        with fluid.program_guard(main_program):
            tensor_array = dygraph_to_static_graph(self.dygraph_func)(
                self.input, self.iter_num)
            static_outs = []
            for i in range(self.iter_num):
                static_outs.append(
                    fluid.layers.array_read(
                        tensor_array,
                        i=fluid.layers.fill_constant(
                            shape=[1], value=i, dtype='int64')))

        exe = fluid.Executor(self.place)
        numpy_res = exe.run(main_program, fetch_list=static_outs)
        return numpy_res


class TestListInWhileLoopWithStack(TestListInWhileLoop):
    def init_dygraph_func(self):
        self.dygraph_func = test_list_in_while_loop_with_stack

    def run_dygraph_mode(self):
        with fluid.dygraph.guard():
            var_res = self.dygraph_func(self.input, self.iter_num)
            numpy_res = var_res.numpy()
            return numpy_res

    def run_static_mode(self):
        main_program = fluid.Program()
        with fluid.program_guard(main_program):
            out_var = dygraph_to_static_graph(self.dygraph_func)(self.input,
                                                                 self.iter_num)
        exe = fluid.Executor(self.place)
        numpy_res = exe.run(main_program, fetch_list=out_var)
        return numpy_res[0]


class TestListInForLoop(TestListInWhileLoop):
    def init_dygraph_func(self):
        self.dygraph_func = test_list_in_for_loop


class TestListInForLoopWithConcat(TestListInWhileLoopWithStack):
    def init_dygraph_func(self):
        self.dygraph_func = test_list_in_for_loop_with_concat


if __name__ == '__main__':
    unittest.main()
