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

<<<<<<< HEAD
import inspect
import textwrap
import unittest

import numpy as np
from ifelse_simple_func import (
    dyfunc_with_if_else,
    dyfunc_with_if_else2,
    nested_if_else,
)

import paddle
import paddle.fluid as fluid
import paddle.nn.functional as F
from paddle.jit.dy2static.utils import ast_to_func
from paddle.utils import gast
=======
from __future__ import print_function

import unittest
import textwrap
from paddle.utils import gast
import inspect
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.dygraph_to_static.utils import ast_to_func

from ifelse_simple_func import dyfunc_with_if_else, dyfunc_with_if_else2, nested_if_else
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


class TestAST2Func(unittest.TestCase):
    """
    TestCase for the transformation from ast.AST into python callable function.
    """

    def _ast2func(self, func):
        source = inspect.getsource(func)
        source = textwrap.dedent(source)
        ast_root = gast.parse(source)
        transformed_func, _ = ast_to_func(ast_root, func)
        return transformed_func

    def test_ast2func(self):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def func(x, y):
            return x + y

        x, y = 10, 20
        self.assertEqual(func(x, y), self._ast2func(func)(x, y))

    def test_ast2func_dygraph(self):
        funcs = [dyfunc_with_if_else, dyfunc_with_if_else2, nested_if_else]
        x_data = np.random.random([10, 16]).astype('float32')
        for func in funcs:
            with fluid.dygraph.guard():
                x_v = fluid.dygraph.to_variable(x_data)
                true_ret = func(x_v).numpy()
                test_ret = self._ast2func(func)(x_v).numpy()
                self.assertTrue((true_ret == test_ret).all())

    def test_ast2func_static(self):
<<<<<<< HEAD
        def func(x):
            y = F.relu(x)
=======

        def func(x):
            y = fluid.layers.relu(x)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            loss = paddle.mean(y)
            return loss

        x_data = np.random.random([10, 16]).astype('float32')
        main_program = fluid.Program()
        with fluid.program_guard(main_program):
            x_v = fluid.layers.assign(x_data)
            true_ret = func(x_v)
            test_ret = self._ast2func(func)(x_v)
            exe = fluid.Executor(fluid.CPUPlace())
            ret = exe.run(main_program, fetch_list=[true_ret, test_ret])
            self.assertTrue((ret[0] == ret[1]).all())

    def test_ast2func_error(self):
        with self.assertRaises(Exception) as e:
            self.assertRaises(TypeError, ast_to_func("x = a + b", 'foo'))
<<<<<<< HEAD
        self.assertTrue(
            "Type of ast_root should be gast.AST or ast.AST" in str(e.exception)
        )
=======
        self.assertTrue("Type of ast_root should be gast.AST or ast.AST" in str(
            e.exception))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


if __name__ == '__main__':
    unittest.main()
