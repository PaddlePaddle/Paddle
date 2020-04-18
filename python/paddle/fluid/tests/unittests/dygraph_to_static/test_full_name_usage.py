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

import numpy as np
import paddle.fluid as fluid
import unittest


@fluid.dygraph.declarative
def dygraph_decorated_func(x):
    x = fluid.dygraph.to_variable(x)
    if fluid.layers.mean(x) > 0:
        x_v = x - 1
    else:
        x_v = x + 1
    return x_v


@fluid.dygraph.jit.declarative
def jit_decorated_func(x):
    x = fluid.dygraph.to_variable(x)
    if fluid.layers.mean(x) > 0:
        x_v = x - 1
    else:
        x_v = x + 1
    return x_v


class TestFullNameDecorator(unittest.TestCase):
    def test_run_success(self):
        x = np.ones([1, 2])
        answer = np.zeros([1, 2])
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            self.assertTrue(
                np.allclose(dygraph_decorated_func(x).numpy(), answer))
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            self.assertTrue(np.allclose(jit_decorated_func(x).numpy(), answer))


class TestImportProgramTranslator(unittest.TestCase):
    def test_diff_pkg_same_cls(self):
        dygraph_prog_trans = fluid.dygraph.ProgramTranslator()
        dy_to_stat_prog_trans = fluid.dygraph.dygraph_to_static.ProgramTranslator(
        )
        full_pkg_prog_trans = fluid.dygraph.dygraph_to_static.program_translator.ProgramTranslator(
        )
        self.assertEqual(dygraph_prog_trans, dy_to_stat_prog_trans)
        self.assertEqual(dygraph_prog_trans, full_pkg_prog_trans)


if __name__ == '__main__':
    unittest.main()
