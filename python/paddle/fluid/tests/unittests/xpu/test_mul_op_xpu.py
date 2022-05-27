#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import sys
sys.path.append("..")
from op_test_xpu import XPUOpTest
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard
import time

paddle.enable_static()

from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper


class TestMulOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            # The input type of mul_op must be Variable.
            x1 = fluid.create_lod_tensor(
                np.array([[-1]]), [[1]], fluid.XPUPlace(0))
            x2 = fluid.create_lod_tensor(
                np.array([[-1]]), [[1]], fluid.XPUPlace(0))
            self.assertRaises(TypeError, fluid.layers.mul, x1, x2)
            # The input dtype of mul_op must be float32.
            x3 = fluid.layers.data(name='x3', shape=[4], dtype="int32")
            x4 = fluid.layers.data(name='x4', shape=[4], dtype="int32")
            self.assertRaises(TypeError, fluid.layers.mul, x3, x4)


class XPUTestMulOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'mul'
        self.use_dynamic_create_class = False

    class TestXPUMulOp1(XPUOpTest):
        def setUp(self):
            self.op_type = "mul"
            self.dtype = self.in_type
            self.inputs = {
                'X': np.random.random((3, 4, 2, 9)).astype(self.in_type_str),
                'Y': np.random.random((3, 6, 1, 2, 3)).astype(self.in_type_str)
            }
            self.attrs = {
                'x_num_col_dims': 2,
                'y_num_col_dims': 2,
            }
            result = np.dot(self.inputs['X'].reshape(3 * 4, 2 * 9),
                            self.inputs['Y'].reshape(3 * 6, 1 * 2 * 3))
            result = result.reshape(3, 4, 1, 2, 3)
            self.outputs = {'Out': result}

        def test_check_output(self):
            paddle.enable_static()
            place = paddle.XPUPlace(0)
            self.check_output_with_place(place, atol=0.01)

        def test_check_grad_normal(self):
            place = paddle.XPUPlace(0)
            paddle.enable_static()
            self.check_grad_with_place(
                place, ['X', 'Y'], 'Out', max_relative_error=0.1)

        def test_check_grad_ingore_x(self):
            place = paddle.XPUPlace(0)
            paddle.enable_static()
            self.check_grad_with_place(
                place, ['Y'],
                'Out',
                max_relative_error=0.1,
                no_grad_set=set("X"))

        def test_check_grad_ignore_y(self):
            place = paddle.XPUPlace(0)
            paddle.enable_static()
            self.check_grad_with_place(
                place, ['X'],
                'Out',
                max_relative_error=0.1,
                no_grad_set=set('Y'))

    class TestXPUMulOp2(XPUOpTest):
        def setUp(self):
            self.op_type = "mul"
            self.use_xpu = True
            self.dtype = self.in_type
            self.inputs = {
                'X': np.random.random((20, 5)).astype(self.in_type_str),
                'Y': np.random.random((5, 21)).astype(self.in_type_str)
            }
            self.outputs = {'Out': np.dot(self.inputs['X'], self.inputs['Y'])}

        def test_check_output(self):
            place = paddle.XPUPlace(0)
            paddle.enable_static()
            self.check_output_with_place(place, atol=0.01)

        def test_check_grad_normal(self):
            place = paddle.XPUPlace(0)
            paddle.enable_static()
            self.check_grad_with_place(
                place, ['X', 'Y'], 'Out', max_relative_error=0.1)

        def test_check_grad_ingore_x(self):
            place = paddle.XPUPlace(0)
            paddle.enable_static()
            self.check_grad_with_place(
                place, ['Y'],
                'Out',
                max_relative_error=0.1,
                no_grad_set=set("X"))

        def test_check_grad_ingore_y(self):
            place = paddle.XPUPlace(0)
            paddle.enable_static()
            self.check_grad_with_place(
                place, ['X'],
                'Out',
                max_relative_error=0.1,
                no_grad_set=set('Y'))


support_types = get_xpu_op_support_types('mul')
for stype in support_types:
    create_test_class(globals(), XPUTestMulOp, stype)

if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
