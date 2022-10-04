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
"""Tests for logical operators of Dynamic-to-Static.
Only test simple cases here. The complex test samples like nested ifelse
or nested loop have been covered in file test_ifelse.py and test_loop.py"""

import unittest

from paddle.utils import gast
import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import ProgramTranslator
from paddle.fluid.dygraph.dygraph_to_static.logical_transformer import cmpop_node_to_str

program_translator = ProgramTranslator()

SEED = 2020
np.random.seed(22)


@paddle.jit.to_static
def test_logical_not(x):
    x = paddle.to_tensor(x)
    if not x:
        x = x - 1
    else:
        x = x + 1

    if x != 10:
        x = x - 1
    else:
        x = x + 1

    y = 0
    if not y:
        x = x + 4

    if y != 3:
        x = x + 2
    return x


@paddle.jit.to_static
def test_logical_not_2(x):
    x = paddle.to_tensor(x)

    y = None
    if y is not None and not y:
        x = x + 4

    if y != 3:
        x = x + 2
    return x


@paddle.jit.to_static
def test_logical_and(x):
    x = paddle.to_tensor(x)

    if x < 10 and x > 1:
        x = x - 1
    else:
        x = x + 1

    y = 3
    if y < 10 and y > 1:
        x = x - 2
    else:
        x = x + 2

    return x


@paddle.jit.to_static
def test_logical_and_2(x):
    x = paddle.to_tensor(x)

    a = None
    # NOTE(liym27):
    # because `a is not None` is False, then `a > 1` won't be run,
    # which means `convert_logical_and(a is not None, a > 1)` should not
    # run a>1.
    if a is not None and a > 1:
        x = x - 1
    else:
        x = x + 1

    b = 3

    if b is not None and b > 1:
        x = x - 1
    else:
        x = x + 1

    return x


@paddle.jit.to_static
def test_logical_or(x):
    x = paddle.to_tensor(x)

    if x < 10 or x > 1:
        x = x - 1
    else:
        x = x + 1

    a = 10
    if a > 3 or a < 1:
        x = x - 1
    else:
        x = x + 1

    return x


@paddle.jit.to_static
def test_logical_or_2(x):
    x = paddle.to_tensor(x)

    a = None
    if x > 1 or a is None or a > 1:
        x = x - 1
    else:
        x = x + 1
    return x


@paddle.jit.to_static
def test_logical_not_and_or(x):
    x = paddle.to_tensor(x)

    a = 1
    if x < 10 and (a < 4 or a > 0) or a < -1 or not x > -1:
        x = x - 1
    else:
        x = x + 1
    return x


@paddle.jit.to_static
def test_shape_equal(x):
    x = paddle.to_tensor(x)
    y = paddle.zeros([1, 2, 3])
    if x.shape == y.shape:
        return y
    else:
        return paddle.ones([1, 2, 3])


@paddle.jit.to_static
def test_shape_not_equal(x):
    x = paddle.to_tensor(x)
    y = paddle.zeros([1, 2, 3])
    if x.shape != y.shape:
        return y
    else:
        return paddle.ones([1, 2, 3])


class TestLogicalBase(unittest.TestCase):

    def setUp(self):
        self.input = np.array([3]).astype('int32')
        self.place = paddle.CUDAPlace(
            0) if fluid.is_compiled_with_cuda() else paddle.CPUPlace()
        self._set_test_func()

    def _set_test_func(self):
        raise NotImplementedError(
            "Method 'set_test_func' should be implemented.")

    def _run(self, to_static):
        program_translator.enable(to_static)
        with fluid.dygraph.guard(self.place):
            result = self.dygraph_func(self.input)
            return result.numpy()

    def _run_dygraph(self):
        return self._run(to_static=False)

    def _run_static(self):
        return self._run(to_static=True)


class TestLogicalNot(TestLogicalBase):

    def _set_test_func(self):
        self.dygraph_func = test_logical_not

    def test_transformed_result(self):
        dygraph_res = self._run_dygraph()
        static_res = self._run_static()
        np.testing.assert_allclose(
            dygraph_res,
            static_res,
            rtol=1e-05,
            err_msg='dygraph result is {}\nstatic_result is {}'.format(
                dygraph_res, static_res))


class TestLogicalNot2(TestLogicalBase):

    def _set_test_func(self):
        self.dygraph_func = test_logical_not_2

    def test_transformed_result(self):
        dygraph_res = self._run_dygraph()
        static_res = self._run_static()
        np.testing.assert_allclose(
            dygraph_res,
            static_res,
            rtol=1e-05,
            err_msg='dygraph result is {}\nstatic_result is {}'.format(
                dygraph_res, static_res))


class TestLogicalAnd(TestLogicalNot):

    def _set_test_func(self):
        self.dygraph_func = test_logical_and


class TestLogicalAnd2(TestLogicalNot):

    def _set_test_func(self):
        self.dygraph_func = test_logical_and_2


class TestLogicalOr(TestLogicalNot):

    def _set_test_func(self):
        self.dygraph_func = test_logical_or


class TestLogicalOr2(TestLogicalNot):

    def _set_test_func(self):
        self.dygraph_func = test_logical_or_2


class TestLogicalNotAndOr(TestLogicalNot):

    def _set_test_func(self):
        self.dygraph_func = test_logical_not_and_or


class TestShapeEqual(TestLogicalNot):

    def _set_test_func(self):
        self.input = np.ones([1, 2, 3]).astype('float32')
        self.dygraph_func = test_shape_equal


class TestShapeNotEqual(TestLogicalNot):

    def _set_test_func(self):
        self.input = np.ones([1, 2, 3]).astype('float32')
        self.dygraph_func = test_shape_not_equal


class TestCmpopNodeToStr(unittest.TestCase):

    def test_exception(self):
        with self.assertRaises(KeyError):
            cmpop_node_to_str(gast.Or())

    def test_expected_result(self):
        self.assertEqual(cmpop_node_to_str(gast.Eq()), "==")
        self.assertEqual(cmpop_node_to_str(gast.NotEq()), "!=")
        self.assertEqual(cmpop_node_to_str(gast.Lt()), "<")
        self.assertEqual(cmpop_node_to_str(gast.LtE()), "<=")
        self.assertEqual(cmpop_node_to_str(gast.Gt()), ">")
        self.assertEqual(cmpop_node_to_str(gast.GtE()), ">=")
        self.assertEqual(cmpop_node_to_str(gast.Is()), "is")
        self.assertEqual(cmpop_node_to_str(gast.IsNot()), "is not")
        self.assertEqual(cmpop_node_to_str(gast.In()), "in")
        self.assertEqual(cmpop_node_to_str(gast.NotIn()), "not in")


if __name__ == '__main__':
    unittest.main()
