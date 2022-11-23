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

import numpy
import unittest

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.dygraph_to_static import ProgramTranslator
from paddle.fluid.dygraph.jit import declarative


@paddle.jit.to_static
def dyfunc_assert_variable(x):
    x_v = fluid.dygraph.to_variable(x)
    assert x_v


@declarative
def dyfunc_assert_non_variable(x=True):
    assert x


class TestAssertVariable(unittest.TestCase):

    def _run(self, func, x, with_exception, to_static):
        ProgramTranslator().enable(to_static)
        if with_exception:
            with self.assertRaises(BaseException):
                with fluid.dygraph.guard():
                    func(x)
        else:
            with fluid.dygraph.guard():
                func(x)

    def _run_dy_static(self, func, x, with_exception):
        self._run(func, x, with_exception, True)
        self._run(func, x, with_exception, False)

    def test_non_variable(self):
        self._run_dy_static(dyfunc_assert_non_variable,
                            x=False,
                            with_exception=True)
        self._run_dy_static(dyfunc_assert_non_variable,
                            x=True,
                            with_exception=False)

    def test_bool_variable(self):
        self._run_dy_static(dyfunc_assert_variable,
                            x=numpy.array([False]),
                            with_exception=True)
        self._run_dy_static(dyfunc_assert_variable,
                            x=numpy.array([True]),
                            with_exception=False)

    def test_int_variable(self):
        self._run_dy_static(dyfunc_assert_variable,
                            x=numpy.array([0]),
                            with_exception=True)
        self._run_dy_static(dyfunc_assert_variable,
                            x=numpy.array([1]),
                            with_exception=False)


if __name__ == '__main__':
    unittest.main()
