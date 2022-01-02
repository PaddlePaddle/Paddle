#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import numpy as np
import unittest
import sys
sys.path.append("..")
from op_test import OpTest
import paddle
import paddle.fluid as fluid
from paddle.fluid.core import ops

paddle.enable_static()
SEED = 2021


class TestElementwiseDiv(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "elementwise_div"
        self.place = paddle.NPUPlace(0)

        self.init_dtype()
        np.random.seed(SEED)
        x = np.random.uniform(1, 2, [110, 1]).astype(self.dtype)
        y = np.random.uniform(1, 2, [110, 2]).astype(self.dtype)
        out = np.divide(x, y)

        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(x),
            'Y': OpTest.np_dtype_to_fluid_dtype(y)
        }
        self.attrs = {}
        self.outputs = {'Out': out}

    def set_npu(self):
        self.__class__.use_npu = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
        # if True:
        #     return
        
        self.check_grad_with_place(
            self.place,
            ['X', 'Y'],
            'Out',
            max_relative_error=0.009, )

    def test_check_grad_ingore_x(self):
        # if True:
        #     return
        
        self.check_grad_with_place(
            self.place,
            ['Y'],
            'Out',
            max_relative_error=0.009,
            no_grad_set=set("X"), )

    def test_check_grad_ingore_y(self):
        # if True:
        #     return
        
        self.check_grad_with_place(
            self.place, ['X'], 'Out', no_grad_set=set("Y"))


@unittest.skipIf(False, "tmp")
class TestElementwiseDiv2(TestElementwiseDiv):
    def setUp(self):
        self.set_npu()
        self.op_type = "elementwise_div"
        self.place = paddle.NPUPlace(0)

        self.init_dtype()
        np.random.seed(SEED)
        x = np.random.uniform(1, 2, [110, 2]).astype(self.dtype)
        y = np.random.uniform(1, 2, [110, 1]).astype(self.dtype)
        out = np.divide(x, y)

        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(x),
            'Y': OpTest.np_dtype_to_fluid_dtype(y)
        }
        self.attrs = {}
        self.outputs = {'Out': out}

    def test_check_grad_ingore_x(self):
        self.check_grad_with_place(
            self.place,
            ['Y'],
            'Out',
            max_relative_error=0.009,
            no_grad_set=set("X"), )

    def test_check_grad_ingore_y(self):
        self.check_grad_with_place(
            self.place,
            ['X'],
            'Out',
            max_relative_error=0.007,
            no_grad_set=set("Y"))


@unittest.skipIf(False, "tmp")
class TestElementwiseDiv3(TestElementwiseDiv):
    def setUp(self):
        self.set_npu()
        self.op_type = "elementwise_div"
        self.place = paddle.NPUPlace(0)

        self.init_dtype()
        np.random.seed(SEED)
        x = np.random.uniform(1, 2, [8, 1, 38, 38]).astype(self.dtype)
        y = np.random.uniform(1, 2, [8, 512, 38, 38]).astype(self.dtype)
        out = np.divide(x, y)

        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(x),
            'Y': OpTest.np_dtype_to_fluid_dtype(y)
        }
        self.attrs = {}
        self.outputs = {'Out': out}

    def test_check_grad_ingore_x(self):
        self.check_grad_with_place(
            self.place,
            ['Y'],
            'Out',
            max_relative_error=0.009,
            no_grad_set=set("X"), )

    def test_check_grad_ingore_y(self):
        self.check_grad_with_place(
            self.place,
            ['X'],
            'Out',
            max_relative_error=0.007,
            no_grad_set=set("Y"))


@unittest.skipIf(False, "tmp")
class TestElementwiseDiv4(TestElementwiseDiv):
    def setUp(self):
        self.set_npu()
        self.op_type = "elementwise_div"
        self.place = paddle.NPUPlace(0)

        self.init_dtype()
        np.random.seed(SEED)
        x = np.random.uniform(1, 2, [8, 512, 38, 38]).astype(self.dtype)
        y = np.random.uniform(1, 2, [8, 1, 38, 38]).astype(self.dtype)
        out = np.divide(x, y)

        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(x),
            'Y': OpTest.np_dtype_to_fluid_dtype(y)
        }
        self.attrs = {}
        self.outputs = {'Out': out}

    def test_check_grad_ingore_x(self):
        self.check_grad_with_place(
            self.place,
            ['Y'],
            'Out',
            max_relative_error=0.009,
            no_grad_set=set("X"), )

    def test_check_grad_ingore_y(self):
        self.check_grad_with_place(
            self.place,
            ['X'],
            'Out',
            max_relative_error=0.007,
            no_grad_set=set("Y"))


if __name__ == '__main__':
    unittest.main()
