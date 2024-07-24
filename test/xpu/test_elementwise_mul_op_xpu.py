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
    check_run_big_shape_test,
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


class XPUTestElementwiseMulOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'elementwise_mul'
        self.use_dynamic_create_class = False

    class ElementwiseMulOp(XPUOpTest):
        def init_kernel_type(self):
            self.use_mkldnn = False

        def setUp(self):
            self.op_type = 'elementwise_mul'
            self.use_xpu = True
            self.cal_x = None
            self.cal_y = None
            self.dtype = self.in_type
            self.axis = -1
            self.init_data()
            self.gen_output()
            self.init_input_output()
            self.init_kernel_type()
            self.init_axis()

        def gen_output(self):
            if self.cal_x is None:
                self.cal_x = self.x
            if self.cal_y is None:
                self.cal_y = self.y
            if self.dtype == np.uint16:
                self.out = np.multiply(self.cal_x, self.cal_y)
            else:
                self.out = np.multiply(
                    self.cal_x.astype(self.dtype), self.cal_y.astype(self.dtype)
                )

        def gen_data_depend_on_dtype(self, shape):
            if self.dtype == np.int32 or self.dtype == np.int64:
                return np.random.randint(1, 100, size=shape)
            else:
                return np.random.uniform(0.1, 1, size=shape)

        def test_check_output(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_output_with_place(place, check_dygraph=False)

        def test_check_grad_normal(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_grad_with_place(
                    place,
                    ['X', 'Y'],
                    'Out',
                    check_dygraph=False,
                )

        def test_check_grad_ignore_x(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_grad_with_place(
                    place,
                    ['Y'],
                    'Out',
                    no_grad_set=set("X"),
                    check_dygraph=False,
                )

        def test_check_grad_ignore_y(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_grad_with_place(
                    place,
                    ['X'],
                    'Out',
                    no_grad_set=set('Y'),
                    check_dygraph=False,
                )

        def init_data(self):
            self.x = self.gen_data_depend_on_dtype([13, 17])
            self.y = self.gen_data_depend_on_dtype([13, 17])

        def init_input_output(self):
            if self.dtype == np.uint16:
                self.x = convert_float_to_uint16(self.x)
                self.y = convert_float_to_uint16(self.y)
            else:
                self.x = self.x.astype(self.dtype)
                self.y = self.y.astype(self.dtype)

            self.inputs = {
                'X': self.x,
                'Y': self.y,
            }
            self.outputs = {'Out': self.out}
            self.attrs = {'axis': self.axis, 'use_mkldnn': self.use_mkldnn}

        def init_axis(self):
            pass

    class TestElementwiseMulOp_ZeroDim1(ElementwiseMulOp):
        def init_data(self):
            self.x = self.gen_data_depend_on_dtype([])
            self.y = self.gen_data_depend_on_dtype([])

    class TestElementwiseMulOp_ZeroDim2(ElementwiseMulOp):
        def init_data(self):
            self.x = self.gen_data_depend_on_dtype([13, 17])
            self.y = self.gen_data_depend_on_dtype([])

    class TestElementwiseMulOp_ZeroDim3(ElementwiseMulOp):
        def init_data(self):
            self.x = self.gen_data_depend_on_dtype([])
            self.y = self.gen_data_depend_on_dtype([13, 17])

    @skip_check_grad_ci(
        reason="[skip shape check] Use y_shape(1) to test broadcast."
    )
    class TestElementwiseMulOp_scalar(ElementwiseMulOp):
        def init_data(self):
            self.x = self.gen_data_depend_on_dtype([10, 3, 4])
            self.y = self.gen_data_depend_on_dtype([1])

    class TestElementwiseMulOp_Vector(ElementwiseMulOp):
        def init_data(self):
            self.x = self.gen_data_depend_on_dtype([100])
            self.y = self.gen_data_depend_on_dtype([100])

    class TestElementwiseMulOp_broadcast_0(ElementwiseMulOp):
        def init_data(self):
            self.x = self.gen_data_depend_on_dtype([100, 2, 3])
            self.y = self.gen_data_depend_on_dtype([100])
            self.cal_y = self.y.reshape(100, 1, 1)
            self.axis = 0

    class TestElementwiseMulOp_broadcast_1(ElementwiseMulOp):
        def init_data(self):
            self.x = self.gen_data_depend_on_dtype([2, 100, 3])
            self.y = self.gen_data_depend_on_dtype([100])
            self.cal_y = self.y.reshape(1, 100, 1)
            self.axis = 1

    class TestElementwiseMulOp_broadcast_2(ElementwiseMulOp):
        def init_data(self):
            self.x = self.gen_data_depend_on_dtype([2, 3, 100])
            self.y = self.gen_data_depend_on_dtype([100])
            self.cal_y = self.y.reshape(1, 1, 100)

    class TestElementwiseMulOp_broadcast_3(ElementwiseMulOp):
        def init_data(self):
            self.x = self.gen_data_depend_on_dtype([2, 10, 12, 3])
            self.y = self.gen_data_depend_on_dtype([10, 12])
            self.cal_y = self.y.reshape(1, 10, 12, 1)
            self.axis = 1

    class TestElementwiseMulOp_broadcast_4(ElementwiseMulOp):
        def init_data(self):
            self.x = self.gen_data_depend_on_dtype([10, 2, 11])
            self.y = self.gen_data_depend_on_dtype([10, 1, 11])

    class TestElementwiseMulOp_broadcast_5(ElementwiseMulOp):
        def init_data(self):
            self.x = self.gen_data_depend_on_dtype([10, 4, 2, 3])
            self.y = self.gen_data_depend_on_dtype([10, 4, 1, 3])

    class TestElementwiseMulOp_commonuse_1(ElementwiseMulOp):
        def init_data(self):
            self.x = self.gen_data_depend_on_dtype([2, 3, 100])
            self.y = self.gen_data_depend_on_dtype([1, 1, 100])

    class TestElementwiseMulOp_commonuse_2(ElementwiseMulOp):
        def init_data(self):
            self.x = self.gen_data_depend_on_dtype([30, 3, 1, 5])
            self.y = self.gen_data_depend_on_dtype([30, 1, 4, 1])

    class TestElementwiseMulOp_xsize_lessthan_ysize(ElementwiseMulOp):
        def init_data(self):
            self.x = self.gen_data_depend_on_dtype([10, 10])
            self.y = self.gen_data_depend_on_dtype([2, 2, 10, 10])
            self.cal_x = self.x.reshape(1, 1, 10, 10)
            self.axis = 2

    @check_run_big_shape_test()
    class TestElementwiseMulOpLargeShape1(ElementwiseMulOp):
        def init_data(self):
            self.x = self.gen_data_depend_on_dtype([8192, 1])
            self.y = self.gen_data_depend_on_dtype([1, 128])

    @check_run_big_shape_test()
    class TestElementwiseMulOpLargeShape2(ElementwiseMulOp):
        def init_data(self):
            self.x = self.gen_data_depend_on_dtype([1, 8192, 5, 128])
            self.y = self.gen_data_depend_on_dtype([1, 8192, 1, 128])

    @check_run_big_shape_test()
    class TestElementwiseMulOpLargeShape3(ElementwiseMulOp):
        def init_data(self):
            self.x = self.gen_data_depend_on_dtype([8192, 1728])
            self.y = self.gen_data_depend_on_dtype([8192, 1])
            self.cal_y = self.y.reshape([8192, 1])


support_types = get_xpu_op_support_types('elementwise_mul')
for stype in support_types:
    create_test_class(globals(), XPUTestElementwiseMulOp, stype)

if __name__ == '__main__':
    unittest.main()
