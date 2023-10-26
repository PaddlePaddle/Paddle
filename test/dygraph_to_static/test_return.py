# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import numpy as np
from dygraph_to_static_util import ast_only_test, dy2static_unittest
from ifelse_simple_func import dyfunc_with_if_else

import paddle
from paddle import base
from paddle.base import core
from paddle.jit import to_static
from paddle.jit.dy2static.utils import Dygraph2StaticException

SEED = 2020
np.random.seed(SEED)


@to_static(full_graph=True)
def test_return_base(x):
    x = base.dygraph.to_variable(x)
    return x


@to_static(full_graph=True)
def test_inside_func_base(x):
    x = base.dygraph.to_variable(x)

    def inner_func(x):
        return x

    return inner_func(x)


@to_static(full_graph=True)
def test_return_if(x):
    x = base.dygraph.to_variable(x)
    if x < 0:
        x -= 1
        return -x
    x += 3
    return x


@to_static(full_graph=True)
def test_return_if_else(x):
    x = base.dygraph.to_variable(x)
    if x > 0:
        x += 10086
        return x
        x -= 3  # useless statement to test our code can handle it.
    else:
        x += 6666
        return x
        x -= 8888  # useless statement to test our code can handle it.


@to_static(full_graph=True)
def test_return_in_while(x):
    x = base.dygraph.to_variable(x)
    i = paddle.tensor.fill_constant(shape=[1], dtype='int32', value=0)
    while i < 10:
        i += 1
        if i > 5:
            x += 110
            return x
        x += i
    return x


@to_static(full_graph=True)
def test_return_in_for(x):
    x = base.dygraph.to_variable(x)
    for i in range(10):
        if i <= 4:
            x += 1
            continue
        else:
            return x + 10086
    return x - 1


@to_static(full_graph=True)
def test_recursive_return(x):
    x = base.dygraph.to_variable(x)
    return dyfunc_with_if_else(x)


@to_static(full_graph=True)
def test_return_different_length_if_body(x):
    x = base.dygraph.to_variable(x)
    y = x + 1
    if x > 0:
        # x = to_variable(np.ones(1)) so it will return here
        return x, y
    else:
        return x


@to_static(full_graph=True)
def test_return_different_length_else(x):
    x = base.dygraph.to_variable(x)
    y = x + 1
    if x < 0:
        return x, y
    else:
        # x = to_variable(np.ones(1)) so it will return here
        return x


@to_static(full_graph=True)
def test_no_return(x):
    x = base.dygraph.to_variable(x)
    y = x + 1


@to_static(full_graph=True)
def test_return_none(x):
    x = base.dygraph.to_variable(x)
    y = x + 1
    if x > 0:
        # x = to_variable(np.ones(1)) so it will return here
        return None
    else:
        return x, y


@to_static(full_graph=True)
def test_return_no_variable(x):
    x = base.dygraph.to_variable(x)
    y = x + 1
    if x < 0:
        return x, y
    else:
        # x = to_variable(np.ones(1)) so it will return here
        return


@to_static(full_graph=True)
def test_return_list_one_value(x):
    x = base.dygraph.to_variable(x)
    x += 1
    return [x]


@to_static(full_graph=True)
def test_return_list_many_values(x):
    x = base.dygraph.to_variable(x)
    x += 1
    y = x * 2
    z = x * x
    return [x, y, z]


@to_static(full_graph=True)
def test_return_tuple_one_value(x):
    x = base.dygraph.to_variable(x)
    x += 1
    return (x,)


@to_static(full_graph=True)
def test_return_tuple_many_values(x):
    x = base.dygraph.to_variable(x)
    x += 1
    y = x * 2
    z = x * x
    return (x, y, z)


def inner_func(x):
    a = 2
    if a < 0:
        y = x + 1
        return y
    y = x * 2
    return y


@to_static(full_graph=True)
def test_return_without_paddle_cond(x):
    # y shape is [10]
    y = paddle.ones([10])

    # the shape of inner_func(y) should be [10], not [1]
    y = inner_func(y)
    y = paddle.reshape(y, [2, 5])
    return y


def two_value(x):
    return x * 2, x + 1


def diff_return_hepler(x):
    if False:
        y = x + 1
        z = x - 1
        return y, z
    else:
        return two_value(x)


@to_static(full_graph=True)
def test_diff_return(x):
    x = paddle.to_tensor(x)
    y, z = diff_return_hepler(x)
    if y.shape[0] > 1:
        y = y + 1
    return y, z


@to_static(full_graph=True)
def test_return_if_else_2(x):
    rr = 0
    if True:
        rr = 1
        return 1
    else:
        a = 0


@to_static(full_graph=True)
def test_return_in_while_2(x):
    while True:
        a = 12
        return 12
    return 10


@to_static(full_graph=True)
def test_return_in_for_2(x):
    a = 12
    for i in range(10):
        return 12
    return 10


@to_static(full_graph=True)
def test_return_nested(x):
    def func():
        rr = 0
        if True:
            rr = 1
            return 1
            rr = 2
        else:
            a = 0
            return 4
        return 3

    return func()


@dy2static_unittest
class TestReturnBase(unittest.TestCase):
    def setUp(self):
        self.input = np.ones(1).astype('int32')
        self.place = (
            base.CUDAPlace(0)
            if base.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        self.init_dygraph_func()

    def init_dygraph_func(self):
        self.dygraph_func = test_return_base

    def _run(self, to_static=False):
        paddle.jit.enable_to_static(to_static)
        with base.dygraph.guard():
            res = self.dygraph_func(self.input)
            if isinstance(res, (tuple, list)):
                return tuple(r.numpy() for r in res)
            elif isinstance(res, core.eager.Tensor):
                return res.numpy()
            return res

    def _test_value_impl(self):
        dygraph_res = self._run(to_static=False)
        static_res = self._run(to_static=True)
        if isinstance(dygraph_res, tuple):
            self.assertTrue(isinstance(static_res, tuple))
            self.assertEqual(len(dygraph_res), len(static_res))
            for i in range(len(dygraph_res)):
                np.testing.assert_allclose(
                    dygraph_res[i], static_res[i], rtol=1e-05
                )
        elif isinstance(dygraph_res, np.ndarray):
            np.testing.assert_allclose(dygraph_res, static_res, rtol=1e-05)
        else:
            self.assertEqual(dygraph_res, static_res)

    def test_transformed_static_result(self):
        if hasattr(self, "error"):
            with self.assertRaisesRegex(Dygraph2StaticException, self.error):
                self._test_value_impl()
        else:
            self._test_value_impl()


class TestInsideFuncBase(TestReturnBase):
    def init_dygraph_func(self):
        self.dygraph_func = test_inside_func_base


class TestReturnIf(TestReturnBase):
    def init_dygraph_func(self):
        self.dygraph_func = test_return_if


class TestReturnOnlyIf(TestReturnBase):
    def init_dygraph_func(self):
        self.dygraph_func = test_return_if_else_2


class TestReturnInFor(TestReturnBase):
    def init_dygraph_func(self):
        self.dygraph_func = test_return_in_for


class TestReturnInWhile(TestReturnBase):
    def init_dygraph_func(self):
        self.dygraph_func = test_return_in_while


class TestReturnIfDiff(TestReturnBase):
    def init_dygraph_func(self):
        self.dygraph_func = test_diff_return


class TestReturnIfElse(TestReturnBase):
    def init_dygraph_func(self):
        self.dygraph_func = test_return_if_else


class TestReturnInWhile2(TestReturnBase):
    def init_dygraph_func(self):
        self.dygraph_func = test_return_in_while_2
        self.error = "Found return statement in While or For body and loop"

    @ast_only_test
    def test_transformed_static_result(self):
        super().test_transformed_static_result()


class TestReturnInFor2(TestReturnBase):
    def init_dygraph_func(self):
        self.dygraph_func = test_return_in_for_2
        self.error = "Found return statement in While or For body and loop"

    @ast_only_test
    def test_transformed_static_result(self):
        super().test_transformed_static_result()


class TestRecursiveReturn(TestReturnBase):
    def init_dygraph_func(self):
        self.input = self.input.astype(np.float32)
        self.dygraph_func = test_recursive_return


class TestReturnDifferentLengthIfBody(TestReturnBase):
    def init_dygraph_func(self):
        self.dygraph_func = test_return_different_length_if_body
        self.error = "Your if/else have different number of return value."

    @ast_only_test
    def test_transformed_static_result(self):
        super().test_transformed_static_result()


class TestReturnDifferentLengthElse(TestReturnBase):
    def init_dygraph_func(self):
        self.dygraph_func = test_return_different_length_else
        self.error = "Your if/else have different number of return value."

    @ast_only_test
    def test_transformed_static_result(self):
        super().test_transformed_static_result()


class TestNoReturn(TestReturnBase):
    def init_dygraph_func(self):
        self.dygraph_func = test_no_return


class TestReturnNone(TestReturnBase):
    def init_dygraph_func(self):
        self.dygraph_func = test_return_none
        self.error = "Your if/else have different number of return value."

    @ast_only_test
    def test_transformed_static_result(self):
        super().test_transformed_static_result()


class TestReturnNoVariable(TestReturnBase):
    def init_dygraph_func(self):
        self.dygraph_func = test_return_no_variable
        self.error = "Your if/else have different number of return value."

    @ast_only_test
    def test_transformed_static_result(self):
        super().test_transformed_static_result()


class TestReturnListOneValue(TestReturnBase):
    def init_dygraph_func(self):
        self.dygraph_func = test_return_list_one_value


class TestReturnListManyValue(TestReturnBase):
    def init_dygraph_func(self):
        self.dygraph_func = test_return_list_many_values


class TestReturnTupleOneValue(TestReturnBase):
    def init_dygraph_func(self):
        self.dygraph_func = test_return_tuple_one_value


class TestReturnTupleManyValue(TestReturnBase):
    def init_dygraph_func(self):
        self.dygraph_func = test_return_tuple_many_values


class TestReturnNested(TestReturnBase):
    def init_dygraph_func(self):
        self.dygraph_func = test_return_nested


class TestReturnSpecial(TestReturnBase):
    def init_dygraph_func(self):
        self.dygraph_func = test_return_without_paddle_cond


if __name__ == '__main__':
    unittest.main()
