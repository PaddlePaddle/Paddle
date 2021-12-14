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
from op_test import OpTest, skip_check_grad_ci
import paddle
import paddle.fluid as fluid
from paddle.framework import core
from paddle.fluid.dygraph.base import switch_to_static_graph

paddle.enable_static()

@skip_check_grad_ci("123")
class TestPutAlongAxisOp(OpTest):
    def setUp(self):
        self.init_data()
        self.reduce_op = "assign"
        self.dtype = 'float64'
        self.op_type = "put_along_axis"
        self.inputs = {
            'Input': self.xnp,
            'Index': self.index,
            'Value': self.value
            }
        self.attrs = {
            'Axis': self.axis,
            'Reduce': self.reduce_op
            }
        np.put_along_axis(self.xnp, self.index, self.value, self.axis)
        self.target = self.xnp # numpy put_along_axis is an inplace opearion.
        print('self.target', self.target)
        self.outputs = {'Result': self.target}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        pass
        #self.check_grad(['Input'], 'Result')

    def init_data(self):
        self.x_type = "float64"
        self.xnp = np.array([[1, 2], [3, 4]]).astype(self.x_type)
        self.value_type = "float64"
        self.value = np.array([[99, 99],[99, 99]]).astype(self.value_type)
        print(">>>self.xnp>>>", self.xnp)
        self.index_type = "int32"
        self.index =np.array([[0, 0], [1, 0]]).astype(self.index_type)
        self.axis = 1
        self.axis_type = "int64"

@skip_check_grad_ci("123")
class TestCase1(TestPutAlongAxisOp):
    # test reduce add op
    def init_data(self):
        self.reduce_op = 'add'
        self.x_type = "float64"
        self.xnp = np.array([[1, 2], [3, 4]]).astype(self.x_type)
        self.value_type = "float64"
        self.value = np.array([[99, 99],[99, 99]]).astype(self.value_type)
        print(">>>self.xnp>>>", self.xnp)
        self.index_type = "int32"
        self.index =np.array([[0, 0], [1, 0]]).astype(self.index_type)
        self.axis = 1
        self.axis_type = "int64"

@skip_check_grad_ci("123")
class TestCase2(TestPutAlongAxisOp):
    # test reduce add op
    def init_data(self):
        self.reduce_op = 'mul'
        self.x_type = "float64"
        self.xnp = np.array([[1, 2], [3, 4]]).astype(self.x_type)
        self.value_type = "float64"
        self.value = np.array([[99, 99],[99, 99]]).astype(self.value_type)
        print(">>>self.xnp>>>", self.xnp)
        self.index_type = "int32"
        self.index =np.array([[0, 0], [1, 0]]).astype(self.index_type)
        self.axis = 1
        self.axis_type = "int64"

# class TestCase2(TestTakeAlongAxisOp):
#     def init_data(self):
#         self.x_shape = (10, 20)
#         self.xnp = np.random.random(self.x_shape)
#         self.x_type = "float64"
#         self.index = [[1,2]]
#         self.index_type = "int32"
#         self.axis = 1
#         self.axis_type = "int64"

if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
