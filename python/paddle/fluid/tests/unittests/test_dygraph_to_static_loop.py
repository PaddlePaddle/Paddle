#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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


class TestNameVisitor(unittest.TestCase):
    def test_loop_vars(self):
        test_func = inspect.getsource(while_loop_dyfunc)
        gast_root = gast.parse(test_func)
        name_visitor = NameVisitor(gast_root)
        for node in gast.walk(gast_root):
            if isinstance(node, gast.While):
                loop_var_names, create_var_names = name_visitor.get_loop_var_names(
                    node)
                self.assertEqual(loop_var_names, set(["i", "x"]))
                self.assertEqual(create_var_names, set())


class TestTransformWhile(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
