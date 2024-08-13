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

import unittest

import numpy as np
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test import OpTest, convert_float_to_uint16
from op_test_xpu import XPUOpTest

import paddle


class XPUTestElementwisePowOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'elementwise_pow'
        self.use_dynamic_create_class = False

    class TestElementwisePowOp(XPUOpTest):
        def setUp(self):
            self.op_type = "elementwise_pow"
            self.dtype = self.in_type
            self.compute_input_output()
            if self.dtype == np.uint16:
                # bfloat16 actually
                self.x = convert_float_to_uint16(self.tmp_x)
                self.y = convert_float_to_uint16(self.tmp_y)
            else:
                self.x = self.tmp_x.astype(self.dtype)
                self.y = self.tmp_y.astype(self.dtype)
            self.inputs = {
                'X': self.x,
                'Y': self.y,
            }
            self.outputs = {'Out': np.power(self.inputs['X'], self.inputs['Y'])}

        def compute_input_output(self):
            self.tmp_x = np.random.uniform(1, 2, [20, 5])
            self.tmp_y = np.random.uniform(1, 2, [20, 5])

        def test_check_output(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_output_with_place(place)

        def test_check_grad(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_grad_with_place(place, ['X', 'Y'], 'Out')

    class TestElementwisePowOp_big_shape_1(TestElementwisePowOp):
        def compute_input_output(self):
            self.tmp_x = np.random.uniform(1, 2, [10, 10])
            self.tmp_y = np.random.uniform(0.1, 1, [10, 10])

    class TestElementwisePowOp_big_shape_2(TestElementwisePowOp):
        def compute_input_output(self):
            self.tmp_x = np.random.uniform(1, 2, [10, 10])
            self.tmp_y = np.random.uniform(0.2, 2, [10, 10])

    class TestElementwisePowOp_scalar(TestElementwisePowOp):
        def compute_input_output(self):
            self.tmp_x = np.random.uniform(0.1, 1, [3, 3, 4])
            self.tmp_y = np.random.uniform(0.1, 1, [1])

    class TestElementwisePowOp_tensor(TestElementwisePowOp):
        def compute_input_output(self):
            self.tmp_x = np.random.uniform(0.1, 1, [100])
            self.tmp_y = np.random.uniform(1, 3, [100])

    class TestElementwisePowOp_broadcast_0(TestElementwisePowOp):
        def compute_input_output(self):
            self.tmp_x = np.random.uniform(0.1, 1, [2, 1, 100])
            self.tmp_y = np.random.uniform(0.1, 1, [100])

    class TestElementwisePowOp_broadcast_4(TestElementwisePowOp):
        def compute_input_output(self):
            self.tmp_x = np.random.uniform(0.1, 1, [2, 10, 3, 5])
            self.tmp_y = np.random.uniform(0.1, 1, [2, 10, 1, 5])

    class TestElementwisePowOpInt(OpTest):
        def setUp(self):
            self.op_type = "elementwise_pow"
            self.inputs = {
                'X': np.asarray([1, 3, 6]),
                'Y': np.asarray([1, 1, 1]),
            }
            self.outputs = {'Out': np.power(self.inputs['X'], self.inputs['Y'])}

        def test_check_output(self):
            self.check_output(check_dygraph=False)


support_types = get_xpu_op_support_types('elementwise_pow')
for stype in support_types:
    create_test_class(globals(), XPUTestElementwisePowOp, stype)

if __name__ == '__main__':
    unittest.main()
