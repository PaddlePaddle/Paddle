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
from paddle import base

paddle.enable_static()

INT_GROUP = [np.int32, np.int64]


class XPUTestElementwiseDivOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'elementwise_div'
        self.use_dynamic_create_class = False

    class ElementwiseDivOp(XPUOpTest):
        def setUp(self):
            self.op_type = "elementwise_div"
            self.dtype = self.in_type
            self.init_dtype()
            self.use_xpu = True
            self.init_shape()
            self.init_input_output()
            """ Warning
            CPU gradient check error!
            'X': np.random.random((32,84)).astype("float32"),
            'Y': np.random.random((32,84)).astype("float32")
            """

        def gen_data_depend_on_dtype(self, shape):
            if self.dtype in INT_GROUP:
                return np.random.randint(1, 100, size=shape)
            else:
                return np.random.uniform(-1, 1, size=shape)

        def reshape_y_depend_on_x(self):
            if len(self.x_shape) <= len(self.y_shape) or self.y_shape == ():
                return self.y
            reshape_dims = [
                1 if i not in self.y_shape else i for i in self.x_shape
            ]
            return np.reshape(self.y, reshape_dims)

        def init_input_output(self):
            self.x = self.gen_data_depend_on_dtype(self.x_shape)
            self.y = self.gen_data_depend_on_dtype(self.y_shape)
            reshaped_y = self.reshape_y_depend_on_x()
            if self.dtype == np.uint16:
                self.outputs = {'Out': np.divide(self.x, reshaped_y)}
                self.inputs = {
                    'X': convert_float_to_uint16(self.x),
                    'Y': convert_float_to_uint16(self.y),
                }
            else:
                self.inputs = {
                    'X': self.x.astype(self.dtype),
                    'Y': self.y.astype(self.dtype),
                }
                reshaped_y.astype(self.dtype)
                self.outputs = {
                    'Out': (
                        self.inputs['X'] // reshaped_y
                        if self.dtype in INT_GROUP
                        else np.divide(self.inputs['X'], reshaped_y)
                    )
                }

        def test_check_output(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_output_with_place(place)

        def test_check_grad_normal(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_grad_with_place(
                    place, ['X', 'Y'], 'Out', max_relative_error=0.05
                )

        def test_check_grad_ignore_x(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_grad_with_place(
                    place,
                    ['Y'],
                    'Out',
                    max_relative_error=0.05,
                    no_grad_set=set("X"),
                )

        def test_check_grad_ignore_y(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_grad_with_place(
                    place,
                    ['X'],
                    'Out',
                    max_relative_error=0.05,
                    no_grad_set=set('Y'),
                )

        def init_dtype(self):
            pass

        def init_shape(self):
            self.x_shape = [13, 17]
            self.y_shape = [13, 17]

    class TestElementwiseDivOp_ZeroDim1(ElementwiseDivOp):
        def init_shape(self):
            self.x_shape = []
            self.y_shape = []

    class TestElementwiseDivOp_ZeroDim2(ElementwiseDivOp):
        def init_shape(self):
            self.x_shape = [13, 17]
            self.y_shape = []

    @skip_check_grad_ci(
        reason="[skip shape check] Use y_shape(1) to test broadcast."
    )
    class TestElementwiseDivOp_scalar(ElementwiseDivOp):
        def init_shape(self):
            self.x_shape = [20, 3, 4]
            self.y_shape = [1]

    class TestElementwiseDivOp_Vector(ElementwiseDivOp):
        def init_shape(self):
            self.x_shape = [100]
            self.y_shape = [100]

    class TestElementwiseDivOp_broadcast_0(ElementwiseDivOp):
        def init_shape(self):
            self.x_shape = [100, 3, 4]
            self.y_shape = [100]
            self.attrs = {'axis': 0}

    class TestElementwiseDivOp_broadcast_1(ElementwiseDivOp):
        def init_shape(self):
            self.x_shape = [2, 100, 4]
            self.y_shape = [100]
            self.attrs = {'axis': 1}

    class TestElementwiseDivOp_broadcast_2(ElementwiseDivOp):
        def init_shape(self):
            self.x_shape = [2, 3, 100]
            self.y_shape = [100]

    class TestElementwiseDivOp_broadcast_3(ElementwiseDivOp):
        def init_shape(self):
            self.x_shape = [2, 10, 12, 5]
            self.y_shape = [10, 12]
            self.attrs = {'axis': 1}

    class TestElementwiseDivOp_broadcast_4(ElementwiseDivOp):
        def init_shape(self):
            self.x_shape = [2, 3, 50]
            self.y_shape = [2, 1, 50]

    class TestElementwiseDivOp_broadcast_5(ElementwiseDivOp):
        def init_shape(self):
            self.x_shape = [2, 3, 4, 20]
            self.y_shape = [2, 3, 1, 20]

    class TestElementwiseDivOp_commonuse_1(ElementwiseDivOp):
        def init_shape(self):
            self.x_shape = [2, 3, 100]
            self.y_shape = [1, 1, 100]

    class TestElementwiseDivOp_commonuse_2(ElementwiseDivOp):
        def init_shape(self):
            self.x_shape = [30, 3, 1, 5]
            self.y_shape = [30, 1, 4, 1]

    class TestElementwiseDivOp_xsize_lessthan_ysize(ElementwiseDivOp):
        def init_shape(self):
            self.x_shape = [10, 12]
            self.y_shape = [2, 3, 10, 12]
            self.attrs = {'axis': 2}

    class TestElementwiseDivBroadcast(unittest.TestCase):
        def test_shape_with_batch_sizes(self):
            with base.program_guard(base.Program()):
                x_var = paddle.static.data(
                    name='x', dtype='float32', shape=[None, 3, None, None]
                )
                one = 2.0
                out = one / x_var
                exe = base.Executor(base.XPUPlace(0))
                x = np.random.uniform(0.1, 0.6, (1, 3, 32, 32)).astype(
                    'float32'
                )
                (out_result,) = exe.run(feed={'x': x}, fetch_list=[out])
                self.assertEqual((out_result == (2 / x)).all(), True)


support_types = get_xpu_op_support_types('elementwise_div')
for stype in support_types:
    create_test_class(globals(), XPUTestElementwiseDivOp, stype)

if __name__ == '__main__':
    unittest.main()
