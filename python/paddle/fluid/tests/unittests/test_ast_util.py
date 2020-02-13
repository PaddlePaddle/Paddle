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

import unittest
import textwrap
import ast
import inspect
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph.jit import dygraph_to_static_output
from paddle.fluid.dygraph.dygraph_to_static.ast_utils import get_name_ids, ast_to_func


class TestGetNameIds(unittest.TestCase):
    """
    Test for parsing the ast.Name list from the ast.Nodes
    """

    def setUp(self):
        self.source = """
          def test_fn(x):
            return x+1
        """
        self.all_name_ids = {}

    def test_get_name_ids(self):
        source = textwrap.dedent(self.source)
        root = ast.parse(source)
        all_name_ids = get_name_ids([root])
        self.assertDictEqual(
            self.transfer_dict(self.all_name_ids),
            self.transfer_dict(all_name_ids))

    def transfer_dict(self, name_ids_dict):
        new_dict = {}
        for name, ctxs in name_ids_dict.items():
            new_dict[name] = [type(ctx) for ctx in ctxs]
        return new_dict


class TestGetNameIds2(TestGetNameIds):
    def setUp(self):
        self.source = """
          def test_fn(x, y):
            a = 1
            x = y + a
            if x > y:
               z = x * x
               z = z + a
            else:
               z = y * y
            return z
        """
        self.all_name_ids = {
            'x': [ast.Store(), ast.Load(), ast.Load(), ast.Load()],
            'a': [ast.Store(), ast.Load(), ast.Load()],
            'y': [ast.Load(), ast.Load(), ast.Load(), ast.Load()],
            'z': [ast.Store(), ast.Load(), ast.Store(), ast.Store()]
        }


class TestGetNameIds3(TestGetNameIds):
    def setUp(self):
        self.source = """
          def test_fn(x, y):
            z = 1
            if x > y:
               z = x * x
               z = z + y
            return z
        """
        self.all_name_ids = {
            'x': [ast.Load(), ast.Load(), ast.Load()],
            'y': [ast.Load(), ast.Load()],
            'z': [ast.Store(), ast.Store(), ast.Load(), ast.Store()]
        }


def dyfunc_with_if_else(x_v):
    if fluid.layers.mean(x_v).numpy()[0] > 5:
        x_v = x_v - 1
    else:
        x_v = x_v + 1
    return x_v


def dyfunc_with_if_else2(x):
    i, j = 0, 0
    if fluid.layers.reduce_mean(x).numpy()[0] > x.numpy()[i][j]:
        y = fluid.layers.relu(x)
    else:
        x_pow = fluid.layers.pow(x, 2)
        y = fluid.layers.tanh(x_pow)
    return y


class TestAST2Func(unittest.TestCase):
    """
    TestCase for the transformation from ast.AST into python callable function.
    """

    def _ast2func(self, func):
        # source = inspect.getsource(func)
        # source = textwrap.dedent(source)
        # ast_root = ast.parse(source)
        transformed_func = dygraph_to_static_output(func)
        return transformed_func

    def test_ast2func(self):
        def func(x, y):
            return x + y

        x, y = 10, 20
        self.assertEqual(func(x, y), self._ast2func(func)(x, y))

    def test_ast2func_dygraph(self):
        func = dyfunc_with_if_else
        x_data = np.random.random([10, 16]).astype('float32')
        with fluid.dygraph.guard():
            x_v = fluid.dygraph.to_variable(x_data)
            true_ret = func(x_v).numpy()
            test_ret = self._ast2func(func)(x_v).numpy()
            self.assertTrue((true_ret == test_ret).all())

    def test_ast2func_static(self):
        def func(x):
            y = fluid.layers.relu(x)
            loss = fluid.layers.mean(y)
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


class TestDygraphIfElse(unittest.TestCase):
    """
    TestCase for the transformation from control flow `if/else`
    dependent on tensor in Dygraph into Static `fluid.layers.cond`.
    """

    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = dyfunc_with_if_else

    def _run_static(self):
        main_program = fluid.Program()
        with fluid.program_guard(main_program):
            x_v = fluid.layers.assign(self.x)
            # Transform into static graph
            out = dygraph_to_static_output(self.dyfunc)(x_v)
            exe = fluid.Executor(fluid.CPUPlace())
            ret = exe.run(main_program, fetch_list=out)
            return ret

    def _run_dygraph(self):
        with fluid.dygraph.guard():
            x_v = fluid.dygraph.to_variable(self.x)
            ret = self.dyfunc(x_v)
            return ret.numpy()

    def test_ast_to_func(self):
        self.assertTrue((self._run_dygraph() == self._run_static()).all())


class TestDygraphIfElse2(TestDygraphIfElse):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = dyfunc_with_if_else2


if __name__ == '__main__':
    unittest.main()
