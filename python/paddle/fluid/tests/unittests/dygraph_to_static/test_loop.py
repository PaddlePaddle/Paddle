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

from __future__ import print_function

import gast
import inspect
import numpy as np
import paddle.fluid as fluid
import unittest

from paddle.fluid.dygraph.jit import dygraph_to_static_graph
from paddle.fluid.dygraph.dygraph_to_static.loop_transformer import NameVisitor

SEED = 2020
np.random.seed(SEED)


def while_loop_dyfunc(x):
    i = fluid.dygraph.to_variable(x)
    while x < 10:
        i = i + x
        x = x + 1
    return i


def for_loop_dyfunc(max_len):
    for i in range(max_len):
        ret = fluid.layers.zeros(shape=[1], dtype='float32')
        fluid.layers.increment(ret, value=2.0, in_place=True)
    return ret


class TestNameVisitor(unittest.TestCase):
    def setUp(self):
        self.loop_funcs = [while_loop_dyfunc, for_loop_dyfunc]
        self.loop_var_names = [set(["i", "x"]), set(["i", "ret", "max_len"])]
        self.create_var_names = [set(), set(["ret"])]

    def test_loop_vars(self):
        for i in range(len(self.loop_funcs)):
            func = self.loop_funcs[i]
            test_func = inspect.getsource(func)
            gast_root = gast.parse(test_func)
            name_visitor = NameVisitor(gast_root)
            for node in gast.walk(gast_root):
                if isinstance(node, (gast.While, gast.For)):
                    loop_var_names, create_var_names = name_visitor.get_loop_var_names(
                        node)
                    self.assertEqual(loop_var_names, self.loop_var_names[i])
                    self.assertEqual(create_var_names, self.create_var_names[i])


class TestTransformWhileLoop(unittest.TestCase):
    def setUp(self):
        self.place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        self.x = np.zeros(shape=(1), dtype=np.int32)

    def _run_static(self):
        main_program = fluid.Program()
        with fluid.program_guard(main_program):
            x_var = fluid.layers.assign(self.x)
            static_func = dygraph_to_static_graph(while_loop_dyfunc)

            out = static_func(x_var)
            exe = fluid.Executor(self.place)
            ret = exe.run(main_program, fetch_list=out)
        return ret

    def _run_dygraph(self):
        with fluid.dygraph.guard(self.place):
            ret = while_loop_dyfunc(fluid.dygraph.to_variable(self.x))
            return ret.numpy()

    def test_ast_to_func(self):
        static_numpy = self._run_static()
        self.assertTrue(
            np.allclose(
                np.full(
                    shape=(1), fill_value=45, dtype=np.int32), static_numpy))

        # Enable next lines after Paddle dygraph supports while x < 10 
        #
        # self._run_dygraph()
        # self.assertTrue(np.allclose(self._run_dygraph(), self._run_static()))


class TestTransformForLoop(unittest.TestCase):
    def setUp(self):
        self.place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        self.len = 100

    def _run_static(self):
        main_program = fluid.Program()
        with fluid.program_guard(main_program):
            static_func = dygraph_to_static_graph(for_loop_dyfunc)
            out = static_func(self.len)
            exe = fluid.Executor(self.place)
            ret = exe.run(main_program, fetch_list=out)
        return ret

    def _run_dygraph(self):
        with fluid.dygraph.guard(self.place):
            ret = for_loop_dyfunc(self.len)
            return ret.numpy()

    def test_ast_to_func(self):
        static_numpy = self._run_static()
        self.assertTrue(
            np.allclose(
                np.full(
                    shape=(1), fill_value=2, dtype=np.int32), static_numpy))
        self._run_dygraph()
        self.assertTrue(np.allclose(self._run_dygraph(), self._run_static()))


if __name__ == '__main__':
    unittest.main()
