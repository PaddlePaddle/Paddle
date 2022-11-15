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

import numpy as np
import paddle
import paddle.fluid as fluid
import unittest
from paddle.fluid.dygraph import declarative


@fluid.dygraph.declarative
def dygraph_decorated_func(x):
    x = fluid.dygraph.to_variable(x)
    if paddle.mean(x) > 0:
        x_v = x - 1
    else:
        x_v = x + 1
    return x_v


@fluid.dygraph.declarative
def jit_decorated_func(x):
    x = fluid.dygraph.to_variable(x)
    if paddle.mean(x) > 0:
        x_v = x - 1
    else:
        x_v = x + 1
    return x_v


@fluid.dygraph.declarative
def decorated_call_decorated(x):
    return jit_decorated_func(x)


class DoubleDecorated:
    @classmethod
    @declarative
    def double_decorated_func1(self, x):
        return dygraph_decorated_func(x)

    @classmethod
    @fluid.dygraph.declarative
    def double_decorated_func2(self, x):
        return jit_decorated_func(x)


class TestFullNameDecorator(unittest.TestCase):
    def test_run_success(self):
        x = np.ones([1, 2]).astype("float32")
        answer = np.zeros([1, 2]).astype("float32")
        with fluid.dygraph.guard():
            np.testing.assert_allclose(
                dygraph_decorated_func(x).numpy(), answer, rtol=1e-05
            )
            np.testing.assert_allclose(
                jit_decorated_func(x).numpy(), answer, rtol=1e-05
            )
            np.testing.assert_allclose(
                decorated_call_decorated(x).numpy(), answer, rtol=1e-05
            )
            with self.assertRaises(NotImplementedError):
                DoubleDecorated().double_decorated_func1(x)
            with self.assertRaises(NotImplementedError):
                DoubleDecorated().double_decorated_func2(x)


class TestImportProgramTranslator(unittest.TestCase):
    def test_diff_pkg_same_cls(self):
        dygraph_prog_trans = fluid.dygraph.ProgramTranslator()
        dy_to_stat_prog_trans = (
            fluid.dygraph.dygraph_to_static.ProgramTranslator()
        )
        full_pkg_prog_trans = (
            fluid.dygraph.dygraph_to_static.program_translator.ProgramTranslator()
        )
        self.assertEqual(dygraph_prog_trans, dy_to_stat_prog_trans)
        self.assertEqual(dygraph_prog_trans, full_pkg_prog_trans)


if __name__ == '__main__':
    unittest.main()
