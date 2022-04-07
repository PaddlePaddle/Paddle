#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import sys
sys.path.append("..")
import unittest
import numpy as np
from op_test import OpTest, skip_check_grad_ci
from op_test_xpu import XPUOpTest
import paddle
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper
paddle.enable_static()


class XPUTestElementwiseMaxOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'elementwise_max'
        self.use_dynamic_create_class = False

    class TestElementwiseOp(XPUOpTest):
        def setUp(self):
            self.use_xpu = True
            self.op_type = "elementwise_max"
            self.dtype = self.in_type
            self.init_input_output()
            # If x and y have the same value, the max() is not differentiable.
            # So we generate test data by the following method
            # to avoid them being too close to each other.

        def init_input_output(self):
            x = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
            sgn = np.random.choice([-1, 1], [13, 17]).astype(self.dtype)
            y = x + sgn * np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
            self.inputs = {'X': x, 'Y': y}
            self.outputs = {
                'Out': np.maximum(self.inputs['X'], self.inputs['Y'])
            }

        def test_check_output(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_output_with_place(place)

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
                    max_relative_error=0.006,
                    no_grad_set=set("X"))

        def test_check_grad_ingore_y(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_grad_with_place(
                    place, ['X'],
                    'Out',
                    max_relative_error=0.006,
                    no_grad_set=set('Y'))

    @skip_check_grad_ci(
        reason="[skip shape check] Use y_shape(1) to test broadcast.")
    class TestElementwiseMaxOp_scalar(TestElementwiseOp):
        def init_input_output(self):
            x = np.random.random_integers(-5, 5, [2, 3, 20]).astype(self.dtype)
            y = np.array([0.5]).astype(self.dtype)
            self.inputs = {'X': x, 'Y': y}
            self.outputs = {
                'Out': np.maximum(self.inputs['X'], self.inputs['Y'])
            }

    class TestElementwiseMaxOp_Vector(TestElementwiseOp):
        def init_input_output(self):
            x = np.random.random((100, )).astype(self.dtype)
            sgn = np.random.choice([-1, 1], (100, )).astype(self.dtype)
            y = x + sgn * np.random.uniform(0.1, 1, (100, )).astype(self.dtype)
            self.inputs = {'X': x, 'Y': y}
            self.outputs = {
                'Out': np.maximum(self.inputs['X'], self.inputs['Y'])
            }

    class TestElementwiseMaxOp_broadcast_0(TestElementwiseOp):
        def init_input_output(self):
            x = np.random.uniform(0.5, 1, (100, 5, 2)).astype(self.dtype)
            sgn = np.random.choice([-1, 1], (100, )).astype(self.dtype)
            y = x[:, 0, 0] + sgn * \
                np.random.uniform(1, 2, (100, )).astype(self.dtype)
            self.inputs = {'X': x, 'Y': y}

            self.attrs = {'axis': 0}
            self.outputs = {
                'Out': np.maximum(self.inputs['X'],
                                  self.inputs['Y'].reshape(100, 1, 1))
            }

    class TestElementwiseMaxOp_broadcast_1(TestElementwiseOp):
        def init_input_output(self):
            x = np.random.uniform(0.5, 1, (2, 100, 3)).astype(self.dtype)
            sgn = np.random.choice([-1, 1], (100, )).astype(self.dtype)
            y = x[0, :, 0] + sgn * \
                np.random.uniform(1, 2, (100, )).astype(self.dtype)
            self.inputs = {'X': x, 'Y': y}

            self.attrs = {'axis': 1}
            self.outputs = {
                'Out': np.maximum(self.inputs['X'],
                                  self.inputs['Y'].reshape(1, 100, 1))
            }

    class TestElementwiseMaxOp_broadcast_2(TestElementwiseOp):
        def init_input_output(self):
            x = np.random.uniform(0.5, 1, (1, 3, 100)).astype(self.dtype)
            sgn = np.random.choice([-1, 1], (100, )).astype(self.dtype)
            y = x[0, 0, :] + sgn * \
                np.random.uniform(1, 2, (100, )).astype(self.dtype)
            self.inputs = {'X': x, 'Y': y}

            self.outputs = {
                'Out': np.maximum(self.inputs['X'],
                                  self.inputs['Y'].reshape(1, 1, 100))
            }

    class TestElementwiseMaxOp_broadcast_3(TestElementwiseOp):
        def init_input_output(self):
            x = np.random.uniform(0.5, 1, (2, 50, 2, 1)).astype(self.dtype)
            sgn = np.random.choice([-1, 1], (50, 2)).astype(self.dtype)
            y = x[0, :, :, 0] + sgn * \
                np.random.uniform(1, 2, (50, 2)).astype(self.dtype)
            self.inputs = {'X': x, 'Y': y}

            self.attrs = {'axis': 1}
            self.outputs = {
                'Out': np.maximum(self.inputs['X'],
                                  self.inputs['Y'].reshape(1, 50, 2, 1))
            }

    class TestElementwiseMaxOp_broadcast_4(TestElementwiseOp):
        def init_input_output(self):
            x = np.random.uniform(0.5, 1, (2, 3, 4, 5)).astype(self.dtype)
            sgn = np.random.choice([-1, 1], (2, 3, 1, 5)).astype(self.dtype)
            y = x + sgn * \
                np.random.uniform(1, 2, (2, 3, 1, 5)).astype(self.dtype)
            self.inputs = {'X': x, 'Y': y}

            self.outputs = {
                'Out': np.maximum(self.inputs['X'], self.inputs['Y'])
            }


support_types = get_xpu_op_support_types('elementwise_max')
for stype in support_types:
    create_test_class(globals(), XPUTestElementwiseMaxOp, stype)

if __name__ == '__main__':
    unittest.main()
