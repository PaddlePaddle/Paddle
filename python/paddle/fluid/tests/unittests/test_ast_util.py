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
import gast
import inspect
import numpy as np
import paddle.fluid as fluid
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
        self.all_name_ids = {'x': [gast.Param()]}

    def test_get_name_ids(self):
        source = textwrap.dedent(self.source)
        root = gast.parse(source)
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
            'x': [
                gast.Param(), gast.Store(), gast.Load(), gast.Load(),
                gast.Load()
            ],
            'a': [gast.Store(), gast.Load(), gast.Load()],
            'y':
            [gast.Param(), gast.Load(), gast.Load(), gast.Load(), gast.Load()],
            'z': [gast.Store(), gast.Load(), gast.Store(), gast.Store()]
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
            'x': [gast.Param(), gast.Load(), gast.Load(), gast.Load()],
            'y': [gast.Param(), gast.Load(), gast.Load()],
            'z': [gast.Store(), gast.Store(), gast.Load(), gast.Store()]
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
        source = inspect.getsource(func)
        source = textwrap.dedent(source)
        ast_root = gast.parse(source)
        transformed_func, _ = ast_to_func(ast_root, func.__name__)
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

    def test_ast2func_error(self):
        with self.assertRaises(Exception) as e:
            self.assertRaises(TypeError, ast_to_func("x = a + b", 'foo'))
        self.assertTrue("Type of ast_root should be gast.AST or ast.AST" in
                        str(e.exception))


if __name__ == '__main__':
    unittest.main()
