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
import unittest

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.dygraph_to_static import ProgramTranslator

program_translator = ProgramTranslator()


# 0. for in range var.numpy()[0]
@paddle.jit.to_static
def for_in_range(x):
    z = fluid.layers.fill_constant([1], 'int32', 0)
    x = fluid.dygraph.to_variable(x)
    for i in range(x.numpy()[0]):
        z = z + i
    return z


# 1. for iter list 
@paddle.jit.to_static
def for_iter_list(x_array):
    z = fluid.layers.fill_constant([1], 'int32', 0)
    for x in x_array:
        z = z + x
    return z


# 2. for enumerate list
@paddle.jit.to_static
def for_enumerate_list(x_array):
    z = fluid.layers.fill_constant([1], 'int32', 0)
    for i, x in enumerate(x_array):
        z = z + x + i
    return z


# 3. for iter var.numpy()
@paddle.jit.to_static
def for_iter_var_numpy(x_array):
    z = fluid.layers.fill_constant([1], 'int32', 0)
    x_array = fluid.dygraph.to_variable(x_array)
    for x in x_array.numpy():
        z = z + x
    return z


# 4. for enumerate var.numpy()
@paddle.jit.to_static
def for_enumerate_var_numpy(x_array):
    y = fluid.layers.fill_constant([1], 'int32', 0)
    z = fluid.layers.fill_constant([1], 'int32', 0)
    x_array = fluid.dygraph.to_variable(x_array)
    for i, x in enumerate(x_array.numpy()):
        y = y + i
        z = z + x
    return y, z


# 5. for enumerate var.numpy() with start
@paddle.jit.to_static
def for_enumerate_var_numpy_with_start(x_array):
    y = fluid.layers.fill_constant([1], 'int32', 0)
    z = fluid.layers.fill_constant([1], 'int32', 0)
    x_array = fluid.dygraph.to_variable(x_array)
    for i, x in enumerate(x_array.numpy(), 1):
        y = y + i
        z = z + x
    return y, z


# 6. for in range with break
@paddle.jit.to_static
def for_in_range_with_break(x):
    z = fluid.layers.fill_constant([1], 'int32', 0)
    x = fluid.dygraph.to_variable(x)
    for i in range(x.numpy()[0]):
        z = z + i
        if i > 2:
            break
    return z


# 7. for enumerate var.numpy() with break
@paddle.jit.to_static
def for_enumerate_var_numpy_with_break(x_array):
    y = fluid.layers.fill_constant([1], 'int32', 0)
    z = fluid.layers.fill_constant([1], 'int32', 0)
    x_array = fluid.dygraph.to_variable(x_array)
    for i, x in enumerate(x_array.numpy()):
        y = y + i
        z = z + x
        if i > 2:
            break
    return y, z


# 8. for enumerate var.numpy() with continue
@paddle.jit.to_static
def for_enumerate_var_numpy_with_continue(x_array):
    y = fluid.layers.fill_constant([1], 'int32', 0)
    z = fluid.layers.fill_constant([1], 'int32', 0)
    x_array = fluid.dygraph.to_variable(x_array)
    for i, x in enumerate(x_array.numpy()):
        y = y + i
        if i > 2:
            continue
        z = z + x
    return y, z


# 9. for enumerate var.numpy() with start & break
@paddle.jit.to_static
def for_enumerate_var_numpy_with_start_break(x_array):
    y = fluid.layers.fill_constant([1], 'int32', 0)
    z = fluid.layers.fill_constant([1], 'int32', 0)
    x_array = fluid.dygraph.to_variable(x_array)
    for i, x in enumerate(x_array.numpy(), 1):
        y = y + i
        z = z + x
        if i > 2:
            break
    return y, z


# 10. for enumerate var.numpy() with start & continue
@paddle.jit.to_static
def for_enumerate_var_numpy_with_start_continue(x_array):
    y = fluid.layers.fill_constant([1], 'int32', 0)
    z = fluid.layers.fill_constant([1], 'int32', 0)
    x_array = fluid.dygraph.to_variable(x_array)
    for i, x in enumerate(x_array.numpy(), 1):
        y = y + i
        if i > 2:
            continue
        z = z + x
    return y, z


# 11. for iter var
@paddle.jit.to_static
def for_iter_var(x_array):
    z = fluid.layers.fill_constant([1], 'int32', 0)
    x_array = fluid.dygraph.to_variable(x_array)

    for x in x_array:
        z = z + x
    return z


# 12. for enumerate var
@paddle.jit.to_static
def for_enumerate_var(x_array):
    y = fluid.layers.fill_constant([1], 'int32', 0)
    z = fluid.layers.fill_constant([1], 'int32', 0)
    x_array = fluid.dygraph.to_variable(x_array)
    for i, x in enumerate(x_array):
        y = y + i
        z = z + x
    return y, z


# 13. for iter list[var]
@paddle.jit.to_static
def for_iter_var_list(x):
    # 1. prepare data, ref test_list.py
    x = fluid.dygraph.to_variable(x)
    iter_num = fluid.layers.fill_constant(shape=[1], value=5, dtype="int32")
    a = []
    for i in range(iter_num):
        a.append(x + i)
    # 2. iter list[var]
    y = fluid.layers.fill_constant([1], 'int32', 0)
    for x in a:
        y = y + x
    return y


# 14. for enumerate list[var]
@paddle.jit.to_static
def for_enumerate_var_list(x):
    # 1. prepare data, ref test_list.py
    x = fluid.dygraph.to_variable(x)
    iter_num = fluid.layers.fill_constant(shape=[1], value=5, dtype="int32")
    a = []
    for i in range(iter_num):
        a.append(x + i)
    # 2. iter list[var]
    y = fluid.layers.fill_constant([1], 'int32', 0)
    z = fluid.layers.fill_constant([1], 'int32', 0)
    for i, x in enumerate(a):
        y = y + i
        z = z + x
    return y, z


# 15. for enumerate list[var] with a nested for range
@paddle.jit.to_static
def for_enumerate_var_with_nested_range(x_array):
    x = fluid.layers.fill_constant([1], 'int32', 0)
    x_array = fluid.dygraph.to_variable(x_array)
    for i, num in enumerate(x_array):
        for idx in range(num):
            x = x + num
    return x


# 16. for iter var[idx]
@paddle.jit.to_static
def for_iter_var_idx(x_array):
    z = fluid.layers.fill_constant([1], 'int32', 0)
    x_array = fluid.dygraph.to_variable(x_array)

    for x in x_array[0:]:
        z = z + x
    return z


# 17. for a,b,c in z: (a, b, c) is a tuple
@paddle.jit.to_static
def for_tuple_as_iter_var(x_array):
    x = paddle.to_tensor(x_array)
    z = paddle.to_tensor(np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]))

    a_result = paddle.zeros([3])
    b_result = paddle.zeros([3])
    c_result = paddle.zeros([3])

    for a, b, c in z:
        a_result += a
        b_result += b
        c_result += c

    return a_result, b_result, c_result


# 18. for t in enumerate(collection): t is tuple of (idx, element)
@paddle.jit.to_static
def for_tuple_as_enumerate_iter(x_array):
    x = paddle.to_tensor(x_array)
    x_list = [x, x, x]

    a_result = paddle.zeros([5])

    for t in enumerate(x_list):
        a_result += t[1]

    return a_result


# 19. for i, (a, b, c, d, e) in enumerate(collection): (a, b, c, d, e) is a tuple
@paddle.jit.to_static
def for_tuple_as_enumerate_value(x_array):
    x = paddle.to_tensor(x_array)
    x_list = [x, x, x]

    a_result = paddle.zeros([1])
    b_result = paddle.zeros([1])
    c_result = paddle.zeros([1])
    d_result = paddle.zeros([1])
    e_result = paddle.zeros([1])

    for i, (a, b, c, d, e) in enumerate(x_list):
        a_result += a
        b_result += b
        c_result += c
        d_result += d
        e_result += e

    return a_result


# 20. test for function in a class
class ForwardContainsForLayer(paddle.nn.Layer):
    def __init__(self):
        super(ForwardContainsForLayer, self).__init__()
        self.high = 5
        self.low = 3

    @paddle.jit.to_static
    def forward(self, x):
        # just for test case, x is useless in this method
        y = paddle.zeros([10, 2, 3])
        z = []
        for i in range(self.high - self.low):
            z.append(y[i].clone())
        return z


# 21. for original list 
@paddle.jit.to_static
def for_original_list():
    z = fluid.layers.fill_constant([1], 'int32', 0)
    for x in [1, 2, 3]:
        z = z + x
    return z


# 22. for original tuple
@paddle.jit.to_static
def for_original_tuple():
    z = fluid.layers.fill_constant([1], 'int32', 0)
    for x in (1, 2, 3):
        z = z + x
    return z


class TestTransformBase(unittest.TestCase):
    def setUp(self):
        self.place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        self.set_input()
        self.set_test_func()

    def set_input(self):
        self.input = [1, 2, 3]

    def set_test_func(self):
        raise NotImplementedError(
            "For Enumerate test should implement set_test_func")

    def _run(self, to_static):
        program_translator.enable(to_static)
        with fluid.dygraph.guard():
            return self.dygraph_func(self.input)

    def get_dygraph_output(self):
        return self._run(to_static=False)

    def get_static_output(self):
        return self._run(to_static=True)


class TestTransform(TestTransformBase):
    def transformed_result_compare(self):
        dy_outs = self.get_dygraph_output()
        if not isinstance(dy_outs, (tuple, list)):
            dy_outs = (dy_outs, )

        st_outs = self.get_static_output()
        if not isinstance(st_outs, (tuple, list)):
            st_outs = (st_outs, )

        for x, y in zip(dy_outs, st_outs):
            self.assertTrue(np.allclose(x.numpy(), y.numpy()))


class TestTransformForOriginalList(TestTransform):
    def _run(self, to_static):
        program_translator.enable(to_static)
        with fluid.dygraph.guard():
            return self.dygraph_func()


class TestTransformError(TestTransformBase):
    def transformed_error(self, etype):
        with self.assertRaises(etype):
            dy_out = self.get_dygraph_output()
            st_out = self.get_static_output()


class TestForInRange(TestTransform):
    def set_input(self):
        self.input = np.array([5])

    def set_test_func(self):
        self.dygraph_func = for_in_range

    def test_transformed_result_compare(self):
        self.transformed_result_compare()


class TestForIterList(TestTransform):
    def set_test_func(self):
        self.dygraph_func = for_iter_list

    def test_transformed_result_compare(self):
        self.transformed_result_compare()


class TestForEnumerateSimple(TestForIterList):
    def set_test_func(self):
        self.dygraph_func = for_enumerate_list


class TestForInRangeWithBreak(TestForInRange):
    def set_test_func(self):
        self.dygraph_func = for_in_range_with_break


class TestForIterVarNumpy(TestTransform):
    def set_input(self):
        self.input = np.array([1, 2, 3, 4, 5])

    def set_test_func(self):
        self.dygraph_func = for_iter_var_numpy

    def test_transformed_result_compare(self):
        self.transformed_result_compare()


class TestForEnumerateVarNumpy(TestForIterVarNumpy):
    def set_test_func(self):
        self.dygraph_func = for_enumerate_var_numpy


class TestForEnumerateVarNumpyWithStart(TestForIterVarNumpy):
    def set_test_func(self):
        self.dygraph_func = for_enumerate_var_numpy_with_start


class TestForEnumerateVarNumpyWithBreak(TestForIterVarNumpy):
    def set_test_func(self):
        self.dygraph_func = for_enumerate_var_numpy_with_break


class TestForEnumerateVarNumpyWithContinue(TestForIterVarNumpy):
    def set_test_func(self):
        self.dygraph_func = for_enumerate_var_numpy_with_continue


class TestForEnumerateVarNumpyWithStartAndBreak(TestForIterVarNumpy):
    def set_test_func(self):
        self.dygraph_func = for_enumerate_var_numpy_with_start_break


class TestForEnumerateVarNumpyWithStartAndContinue(TestForIterVarNumpy):
    def set_test_func(self):
        self.dygraph_func = for_enumerate_var_numpy_with_start_continue


class TestForIterVar(TestForIterVarNumpy):
    def set_test_func(self):
        self.dygraph_func = for_iter_var


class TestForIterVarIdx(TestForIterVarNumpy):
    def set_test_func(self):
        self.dygraph_func = for_iter_var_idx


class TestForEnumerateVar(TestForIterVarNumpy):
    def set_test_func(self):
        self.dygraph_func = for_enumerate_var


class TestForEnumerateVarWithNestedRange(TestForIterVarNumpy):
    def set_test_func(self):
        self.dygraph_func = for_enumerate_var_with_nested_range


class TestForIterVarList(TestForInRange):
    def set_test_func(self):
        self.dygraph_func = for_iter_var_list


class TestForEnumerateVarList(TestForInRange):
    def set_test_func(self):
        self.dygraph_func = for_enumerate_var_list


class TestForTupleAsIterVar(TestForIterVarNumpy):
    def set_test_func(self):
        self.dygraph_func = for_tuple_as_iter_var


class TestForTupleAsEnumerateIter(TestForIterVarNumpy):
    def set_test_func(self):
        self.dygraph_func = for_tuple_as_enumerate_iter


class TestForTupleAsEnumerateValue(TestForIterVarNumpy):
    def set_test_func(self):
        self.dygraph_func = for_tuple_as_enumerate_value


class TestForwardContainsForLayer(TestForIterVarNumpy):
    def set_test_func(self):
        self.dygraph_func = ForwardContainsForLayer()


class TestForOriginalList(TestTransformForOriginalList):
    def set_test_func(self):
        self.dygraph_func = for_original_list

    def test_transformed_result_compare(self):
        self.transformed_result_compare()


class TestForOriginalTuple(TestTransformForOriginalList):
    def set_test_func(self):
        self.dygraph_func = for_original_tuple

    def test_transformed_result_compare(self):
        self.transformed_result_compare()


if __name__ == '__main__':
    unittest.main()
