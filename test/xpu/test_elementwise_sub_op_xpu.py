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
from op_test import (
    convert_float_to_uint16,
    skip_check_grad_ci,
)
from op_test_xpu import XPUOpTest

import paddle

paddle.enable_static()

INT_GROUP = [np.int32, np.int64]


class XPUTestElementwiseSubOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'elementwise_sub'
        self.use_dynamic_create_class = False

    class TestElementwiseOp(XPUOpTest):
        def setUp(self):
            self.op_type = "elementwise_sub"
            self.use_xpu = True
            self.dtype = self.in_type

            self.init_shape()
            self.init_input_output()

        def reshape_data(self, x, y):
            if len(x.shape) < len(y.shape):
                reshape_dims = [1 if i not in x.shape else i for i in y.shape]
                return np.reshape(x, reshape_dims)
            else:
                return x

        def gen_data_depend_on_dtype(self, shape):
            if self.dtype in INT_GROUP:
                return np.random.randint(1, 100, size=shape)
            else:
                return np.random.uniform(-1, 1, size=shape)

        def init_input_output(self):
            self.x = self.gen_data_depend_on_dtype(self.x_shape)
            self.y = self.gen_data_depend_on_dtype(self.y_shape)
            if self.dtype == np.uint16:
                tmp_x = self.reshape_data(self.x, self.y)
                tmp_y = self.reshape_data(self.y, self.x)
                tmp_out = tmp_x - tmp_y
                self.outputs = {'Out': convert_float_to_uint16(tmp_out)}
                self.x = convert_float_to_uint16(self.x)
                self.y = convert_float_to_uint16(self.y)
            else:
                tmp_x = self.reshape_data(self.x, self.y).astype(self.dtype)
                tmp_y = self.reshape_data(self.y, self.x).astype(self.dtype)
                self.outputs = {'Out': tmp_x - tmp_y}
            self.inputs = {
                'X': self.x.astype(self.dtype),
                'Y': self.y.astype(self.dtype),
            }

        def init_shape(self):
            self.x_shape = [2, 3, 4, 5]
            self.y_shape = [2, 3, 4, 5]

        def test_check_output(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_output_with_place(place, atol=1e-3)

        def test_check_grad_normal(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_grad_with_place(place, ['X', 'Y'], 'Out')

        def test_check_grad_ignore_x(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_grad_with_place(
                    place,
                    ['Y'],
                    'Out',
                    max_relative_error=0.005,
                    no_grad_set=set("X"),
                )

        def test_check_grad_ignore_y(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_grad_with_place(
                    place,
                    ['X'],
                    'Out',
                    max_relative_error=0.005,
                    no_grad_set=set('Y'),
                )

    class TestElementwiseSubOp_ZeroDim1(TestElementwiseOp):
        def init_shape(self):
            self.x_shape = []
            self.y_shape = []

    class TestElementwiseSubOp_ZeroDim2(TestElementwiseOp):
        def init_shape(self):
            self.x_shape = [13, 17]
            self.y_shape = []

    class TestElementwiseSubOp_ZeroDim3(TestElementwiseOp):
        def init_shape(self):
            self.x_shape = []
            self.y_shape = [13, 17]

    @skip_check_grad_ci(
        reason="[skip shape check] Use y_shape(1) to test broadcast."
    )
    class TestElementwiseSubOp_scalar(TestElementwiseOp):
        def init_shape(self):
            self.x_shape = [10, 3, 4]
            self.y_shape = [1]

    class TestElementwiseSubOp_Vector(TestElementwiseOp):
        def init_shape(self):
            self.x_shape = [100]
            self.y_shape = [100]

    class TestElementwiseSubOp_broadcast_0(TestElementwiseOp):
        def init_shape(self):
            self.x_shape = [100, 3, 2]
            self.y_shape = [100]
            self.attrs = {'axis': 0}

    class TestElementwiseSubOp_broadcast_1(TestElementwiseOp):
        def init_shape(self):
            self.x_shape = [2, 100, 3]
            self.y_shape = [100]
            self.attrs = {'axis': 1}

    class TestElementwiseSubOp_broadcast_2(TestElementwiseOp):
        def init_shape(self):
            self.x_shape = [2, 3, 100]
            self.y_shape = [100]

    class TestElementwiseSubOp_broadcast_3(TestElementwiseOp):
        def init_shape(self):
            self.x_shape = [2, 10, 12, 3]
            self.y_shape = [10, 12]
            self.attrs = {'axis': 1}

    class TestElementwiseSubOp_broadcast_4(TestElementwiseOp):
        def init_shape(self):
            self.x_shape = [2, 5, 3, 12]
            self.y_shape = [2, 5, 1, 12]

    class TestElementwiseSubOp_commonuse_1(TestElementwiseOp):
        def init_shape(self):
            self.x_shape = [2, 3, 100]
            self.y_shape = [1, 1, 100]

    class TestElementwiseSubOp_commonuse_2(TestElementwiseOp):
        def init_shape(self):
            self.x_shape = [10, 3, 1, 4]
            self.y_shape = [10, 1, 12, 1]

    class TestElementwiseSubOp_xsize_lessthan_ysize(TestElementwiseOp):
        def init_shape(self):
            self.x_shape = [10, 12]
            self.y_shape = [2, 3, 10, 12]
            self.attrs = {'axis': 2}


support_types = get_xpu_op_support_types('elementwise_sub')
for stype in support_types:
    create_test_class(globals(), XPUTestElementwiseSubOp, stype)

if __name__ == '__main__':
    unittest.main()
