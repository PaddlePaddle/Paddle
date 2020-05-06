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

import paddle.fluid as fluid
from paddle.fluid.dygraph.jit import declarative


# 0. for in range with var case
@declarative
def dygraph_for_in_range(x):
    z = fluid.layers.fill_constant([1], 'int32', 0)
    x = fluid.dygraph.to_variable(x)
    for i in range(x.numpy()[0]):
        z = z + i
    return z


# 1. for iter list 
@declarative
def dygraph_for_iter_list(x_array):
    z = fluid.layers.fill_constant([1], 'int32', 0)
    for x in x_array:
        z = z + x
    return z


# 2. for enumerate list
@declarative
def dygraph_for_enumerate_list(x_array):
    z = fluid.layers.fill_constant([1], 'int32', 0)
    for i, x in enumerate(x_array):
        z = z + x + i
    return z


# 3. for iter variable
@declarative
def dygraph_for_iter_var(x_array):
    z = fluid.layers.fill_constant([1], 'int32', 0)
    x_array = fluid.dygraph.to_variable(x_array)
    for x in x_array.numpy():
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

    def get_dygraph_output(self):
        with fluid.dygraph.guard():
            return self.dygraph_func(self.input)

    def get_static_output(self):
        with fluid.program_guard(fluid.Program()):
            return self.dygraph_func(self.input)


class TestTransform(TestTransformBase):
    def transformed_result_compare(self):
        dy_out = self.get_dygraph_output()
        print("dy out: %d" % dy_out)
        st_out = self.get_static_output()
        print("st out: %d" % st_out)
        self.assertTrue(np.allclose(dy_out.numpy(), st_out.numpy()))


# class TestTransformError(TestTransformBase):
#     def transformed_error(self, etype):
#         with self.assertRaises(etype):
#             dy_out = self.get_dygraph_output()
#             st_out = self.get_static_output()


class TestForInRange(TestTransform):
    def set_input(self):
        self.input = np.array([5])

    def set_test_func(self):
        self.dygraph_func = dygraph_for_in_range

    def test_transformed_result_compare(self):
        self.transformed_result_compare()


# class TestForIterList(TestTransform):
#     def set_test_func(self):
#         self.dygraph_func = dygraph_for_iter_list

#     def test_transformed_result_compare(self):
#         self.transformed_result_compare()

# class TestForEnumerateSimple(TestForIterList):
#     def set_test_func(self):
#         self.dygraph_func = dygraph_for_enumerate_list


class TestForIterVariable(TestTransform):
    def set_input(self):
        self.input = np.array([1, 2, 3, 4, 5])

    def set_test_func(self):
        self.dygraph_func = dygraph_for_iter_var

    def test_transformed_result_compare(self):
        self.transformed_result_compare()


if __name__ == '__main__':
    unittest.main()
