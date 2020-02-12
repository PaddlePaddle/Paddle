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
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph.jit import dygraph_to_static_output
from paddle.fluid.dygraph.dygraph_to_static.ast_utils import get_name_ids


class TestGetNameIds(unittest.TestCase):
    def setUp(self):
        self.source = """
          def test_fn(x):
            return x + 1
        """
        self.all_name_ids = {'x': [ast.Load()]}

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
            'z':
            [ast.Store(), ast.Load(), ast.Store(), ast.Store(), ast.Load()]
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
            'z':
            [ast.Store(), ast.Store(), ast.Load(), ast.Store(), ast.Load()]
        }


@dygraph_to_static_output
def dyfunc_with_ifElse(x_v):
    if x_v.numpy()[0] > 5:
        x_v = x_v - 1
    else:
        x_v = x_v + 1
    return x_v


class TestAST2Func(unittest.TestCase):
    def setUp(self):
        pass

    def test_run(self):
        x = np.array([5]).astype('float32')
        main_program = fluid.Program()
        with fluid.program_guard(main_program):
            x_v = fluid.layers.assign(x)
            out = dyfunc_with_ifElse(x_v)
            exe = fluid.Executor(fluid.CPUPlace())
            ret = exe.run(main_program, fetch_list=out)
            self.assertEqual(ret, [6.])


if __name__ == '__main__':
    unittest.TestCase()
