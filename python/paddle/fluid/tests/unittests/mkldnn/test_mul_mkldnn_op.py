#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np
import paddle
import paddle.fluid.core as core
from paddle.fluid.tests.unittests.op_test import OpTest

class TestMulOneDNNOp(OpTest):
    def setUp(self):
        self.op_type = "mul"
        self.attrs = {'use_mkldnn': True}
        self.init_dtype_type()
        self.init_shapes_and_attrs()

        self.inputs = {
            'X': np.random.random(self.x_shape).astype(self.dtype),
            'Y': np.random.random(self.y_shape).astype(self.dtype)
        }

        output = np.dot(np.reshape(self.inputs['X'], self.np_x_shape), np.reshape(self.inputs['Y'], self.np_y_shape))
        self.outputs = {'Out': np.reshape(output, self.out_shape)}

    def init_shapes_and_attrs(self):
        self.x_shape = (20, 5)
        self.y_shape = (5, 21)

        self.np_x_shape = (20, 5)
        self.np_y_shape = (5, 21)

        self.out_shape = (20, 21)

    def init_dtype_type(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(core.CPUPlace())

    # def test_check_grad_normal(self):
    #     self.check_grad(['X', 'Y'], 'Out')

    # def test_check_grad_ingore_x(self):
    #     self.check_grad(
    #         ['Y'], 'Out', max_relative_error=0.5, no_grad_set=set("X"))

    # def test_check_grad_ingore_y(self):
    #     self.check_grad(
    #         ['X'], 'Out', max_relative_error=0.5, no_grad_set=set('Y'))

class TestMulXNumColDims2OneDNNOp(TestMulOneDNNOp):
    def init_shapes_and_attrs(self):
        self.x_shape = (6, 7, 5)
        self.y_shape = (5, 21)

        self.np_x_shape = (42, 5)
        self.np_y_shape = (5, 21)

        self.out_shape = (6, 7, 21)

        self.attrs["x_num_col_dims"] = 2

class TestMulYNumColDims2OneDNNOp(TestMulOneDNNOp):
    def init_shapes_and_attrs(self):
        self.x_shape = (20, 6)
        self.y_shape = (2, 3, 21)

        self.np_x_shape = (20, 6)
        self.np_y_shape = (6, 21)

        self.out_shape = (20, 21)

        self.attrs["y_num_col_dims"] = 2

class TestMulYAndXNumColDims2OneDNNOp(TestMulOneDNNOp):
    def init_shapes_and_attrs(self):
        self.x_shape = (10, 5, 6)
        self.y_shape = (2, 3, 21)

        self.np_x_shape = (50, 6)
        self.np_y_shape = (6, 21)

        self.out_shape = (10, 5, 21)

        self.attrs["x_num_col_dims"] = 2
        self.attrs["y_num_col_dims"] = 2

if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
