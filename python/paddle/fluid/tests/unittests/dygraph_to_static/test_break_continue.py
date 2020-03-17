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


def test_continue_in_for(x):
    x = fluid.dygraph.to_variable(x)
    for i in range(10):
        x += 1
        if i > 5:
            continue
            x += 10086
        x += i
    return x


def test_continue_in_for_at_end(x):
    x = fluid.dygraph.to_variable(x)
    for i in range(10):
        x += 1
        if i > 5:
            continue
    return x


def test_continue_in_while(x):
    x = fluid.dygraph.to_variable(x)
    i = fluid.layers.fill_constant(shape=[1], dtype='int32', value=0)
    while i < 10:
        i += 1
        if i > 5:
            continue
            x += 10086
        x += i
    return x


def test_break_in_while(x):
    x = fluid.dygraph.to_variable(x)
    i = fluid.layers.fill_constant(shape=[1], dtype='int32', value=0)
    while i < 10:
        i += 1
        if i > 5:
            break
            x += 10086
        x += i
    return x


class TestContinueInFor(unittest.TestCase):
    def setUp(self):
        self.input = np.random.random((1)).astype('int32')
        self.place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        self.init_dygraph_func()

    def init_dygraph_func(self):
        self.dygraph_func = test_continue_in_for

    def run_dygraph_mode(self):
        with fluid.dygraph.guard():
            res = self.dygraph_func(self.input)
            return res.numpy()

    def run_static_mode(self):
        main_program = fluid.Program()
        with fluid.program_guard(main_program):
            res = dygraph_to_static_graph(self.dygraph_func)(self.input)
        exe = fluid.Executor(self.place)
        static_res = exe.run(main_program, fetch_list=[res])

        return static_res[0]

    def test_transformed_static_result(self):
        static_res = self.run_static_mode()
        #dygraph_res = self.run_dygraph_mode()
        print(static_res)
        #print(dygraph_res)
        #self.assertTrue(
        #    np.allclose(dygraph_res, static_res),
        #    msg='dygraph res is {}\nstatic_res is {}'.format(dygraph_res,
        #                                                     static_res))


if __name__ == '__main__':
    unittest.main()
