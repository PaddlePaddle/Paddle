#  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import sys
sys.path.append("..")
import paddle
from op_test import OpTest, skip_check_grad_ci
from op_test_xpu import XPUOpTest
import unittest
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper

paddle.enable_static()


class XPUTestElementwiseSubOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'elementwise_sub'
        self.use_dynamic_create_class = False

    class TestElementwiseOp(XPUOpTest):
        def setUp(self):
            self.op_type = "elementwise_sub"
            self.use_xpu = True
            self.dtype = self.in_type
            self.init_input_output()

        def init_input_output(self):
            self.inputs = {
                'X': np.random.uniform(0.1, 1, [2, 3, 4, 5]).astype(self.dtype),
                'Y': np.random.uniform(0.1, 1, [2, 3, 4, 5]).astype(self.dtype)
            }
            self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}

        def test_check_output(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_output_with_place(place, atol=1e-3)

        def test_check_grad_normal(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_grad_with_place(place, ['X', 'Y'], 'Out')

        def test_check_grad_ingore_x(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_grad_with_place(
                    place, ['Y'],
                    'Out',
                    max_relative_error=0.005,
                    no_grad_set=set("X"))

        def test_check_grad_ingore_y(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_grad_with_place(
                    place, ['X'],
                    'Out',
                    max_relative_error=0.005,
                    no_grad_set=set('Y'))

    @skip_check_grad_ci(
        reason="[skip shape check] Use y_shape(1) to test broadcast.")
    class TestElementwiseSubOp_scalar(TestElementwiseOp):
        def init_input_output(self):
            self.inputs = {
                'X': np.random.rand(10, 3, 4).astype(self.dtype),
                'Y': np.random.rand(1).astype(self.dtype)
            }
            self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}

    class TestElementwiseSubOp_Vector(TestElementwiseOp):
        def init_input_output(self):
            self.inputs = {
                'X': np.random.random((100, )).astype(self.dtype),
                'Y': np.random.random((100, )).astype(self.dtype)
            }
            self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}

    class TestElementwiseSubOp_broadcast_0(TestElementwiseOp):
        def init_input_output(self):
            self.inputs = {
                'X': np.random.rand(100, 3, 2).astype(self.dtype),
                'Y': np.random.rand(100).astype(self.dtype)
            }

            self.attrs = {'axis': 0}
            self.outputs = {
                'Out': self.inputs['X'] - self.inputs['Y'].reshape(100, 1, 1)
            }

    class TestElementwiseSubOp_broadcast_1(TestElementwiseOp):
        def init_input_output(self):
            self.inputs = {
                'X': np.random.rand(2, 100, 3).astype(self.dtype),
                'Y': np.random.rand(100).astype(self.dtype)
            }

            self.attrs = {'axis': 1}
            self.outputs = {
                'Out': self.inputs['X'] - self.inputs['Y'].reshape(1, 100, 1)
            }

    class TestElementwiseSubOp_broadcast_2(TestElementwiseOp):
        def init_input_output(self):
            self.inputs = {
                'X': np.random.rand(2, 3, 100).astype(self.dtype),
                'Y': np.random.rand(100).astype(self.dtype)
            }

            self.outputs = {
                'Out': self.inputs['X'] - self.inputs['Y'].reshape(1, 1, 100)
            }

    class TestElementwiseSubOp_broadcast_3(TestElementwiseOp):
        def init_input_output(self):
            self.inputs = {
                'X': np.random.rand(2, 10, 12, 3).astype(self.dtype),
                'Y': np.random.rand(10, 12).astype(self.dtype)
            }

            self.attrs = {'axis': 1}
            self.outputs = {
                'Out': self.inputs['X'] - self.inputs['Y'].reshape(1, 10, 12, 1)
            }

    class TestElementwiseSubOp_broadcast_4(TestElementwiseOp):
        def init_input_output(self):
            self.inputs = {
                'X': np.random.rand(2, 5, 3, 12).astype(self.dtype),
                'Y': np.random.rand(2, 5, 1, 12).astype(self.dtype)
            }
            self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}

    class TestElementwiseSubOp_commonuse_1(TestElementwiseOp):
        def init_input_output(self):
            self.inputs = {
                'X': np.random.rand(2, 3, 100).astype(self.dtype),
                'Y': np.random.rand(1, 1, 100).astype(self.dtype)
            }
            self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}

    class TestElementwiseSubOp_commonuse_2(TestElementwiseOp):
        def init_input_output(self):
            self.inputs = {
                'X': np.random.rand(10, 3, 1, 4).astype(self.dtype),
                'Y': np.random.rand(10, 1, 12, 1).astype(self.dtype)
            }
            self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}

    class TestElementwiseSubOp_xsize_lessthan_ysize(TestElementwiseOp):
        def init_input_output(self):
            self.inputs = {
                'X': np.random.rand(10, 12).astype(self.dtype),
                'Y': np.random.rand(2, 3, 10, 12).astype(self.dtype)
            }

            self.attrs = {'axis': 2}

            self.outputs = {
                'Out': self.inputs['X'].reshape(1, 1, 10, 12) - self.inputs['Y']
            }


support_types = get_xpu_op_support_types('elementwise_sub')
for stype in support_types:
    create_test_class(globals(), XPUTestElementwiseSubOp, stype)

if __name__ == '__main__':
    unittest.main()
