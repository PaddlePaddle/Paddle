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
import sys
sys.path.append("..")
import unittest
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from op_test import OpTest, skip_check_grad_ci
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper

paddle.enable_static()


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
            self.init_input_output()
            """ Warning
            CPU gradient check error!
            'X': np.random.random((32,84)).astype("float32"),
            'Y': np.random.random((32,84)).astype("float32")
            """

        def init_input_output(self):
            self.inputs = {
                'X': np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype),
                'Y': np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
            }
            self.outputs = {
                'Out': np.divide(self.inputs['X'], self.inputs['Y'])
            }

        def test_check_output(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_output_with_place(place)

        def test_check_grad_normal(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_grad_with_place(
                    place, ['X', 'Y'], 'Out', max_relative_error=0.05)

        def test_check_grad_ingore_x(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_grad_with_place(
                    place, ['Y'],
                    'Out',
                    max_relative_error=0.05,
                    no_grad_set=set("X"))

        def test_check_grad_ingore_y(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_grad_with_place(
                    place, ['X'],
                    'Out',
                    max_relative_error=0.05,
                    no_grad_set=set('Y'))

        def init_dtype(self):
            pass

    @skip_check_grad_ci(
        reason="[skip shape check] Use y_shape(1) to test broadcast.")
    class TestElementwiseDivOp_scalar(ElementwiseDivOp):
        def init_input_output(self):
            self.inputs = {
                'X': np.random.uniform(0.1, 1, [20, 3, 4]).astype(self.dtype),
                'Y': np.random.uniform(0.1, 1, [1]).astype(self.dtype)
            }
            self.outputs = {'Out': self.inputs['X'] / self.inputs['Y']}

    class TestElementwiseDivOp_Vector(ElementwiseDivOp):
        def init_input_output(self):
            self.inputs = {
                'X': np.random.uniform(0.1, 1, [100]).astype(self.dtype),
                'Y': np.random.uniform(0.1, 1, [100]).astype(self.dtype)
            }
            self.outputs = {
                'Out': np.divide(self.inputs['X'], self.inputs['Y'])
            }

    class TestElementwiseDivOp_broadcast_0(ElementwiseDivOp):
        def init_input_output(self):
            self.inputs = {
                'X': np.random.uniform(0.1, 1, [100, 3, 4]).astype(self.dtype),
                'Y': np.random.uniform(0.1, 1, [100]).astype(self.dtype)
            }

            self.attrs = {'axis': 0}
            self.outputs = {
                'Out':
                np.divide(self.inputs['X'], self.inputs['Y'].reshape(100, 1, 1))
            }

    class TestElementwiseDivOp_broadcast_1(ElementwiseDivOp):
        def init_input_output(self):
            self.inputs = {
                'X': np.random.uniform(0.1, 1, [2, 100, 4]).astype(self.dtype),
                'Y': np.random.uniform(0.1, 1, [100]).astype(self.dtype)
            }

            self.attrs = {'axis': 1}
            self.outputs = {
                'Out':
                np.divide(self.inputs['X'], self.inputs['Y'].reshape(1, 100, 1))
            }

    class TestElementwiseDivOp_broadcast_2(ElementwiseDivOp):
        def init_input_output(self):
            self.inputs = {
                'X': np.random.uniform(0.1, 1, [2, 3, 100]).astype(self.dtype),
                'Y': np.random.uniform(0.1, 1, [100]).astype(self.dtype)
            }

            self.outputs = {
                'Out':
                np.divide(self.inputs['X'], self.inputs['Y'].reshape(1, 1, 100))
            }

    class TestElementwiseDivOp_broadcast_3(ElementwiseDivOp):
        def init_input_output(self):
            self.inputs = {
                'X':
                np.random.uniform(0.1, 1, [2, 10, 12, 5]).astype(self.dtype),
                'Y': np.random.uniform(0.1, 1, [10, 12]).astype(self.dtype)
            }

            self.attrs = {'axis': 1}
            self.outputs = {
                'Out': np.divide(self.inputs['X'],
                                 self.inputs['Y'].reshape(1, 10, 12, 1))
            }

    class TestElementwiseDivOp_broadcast_4(ElementwiseDivOp):
        def init_input_output(self):
            self.inputs = {
                'X': np.random.uniform(0.1, 1, [2, 3, 50]).astype(self.dtype),
                'Y': np.random.uniform(0.1, 1, [2, 1, 50]).astype(self.dtype)
            }
            self.outputs = {
                'Out': np.divide(self.inputs['X'], self.inputs['Y'])
            }

    class TestElementwiseDivOp_broadcast_5(ElementwiseDivOp):
        def init_input_output(self):
            self.inputs = {
                'X':
                np.random.uniform(0.1, 1, [2, 3, 4, 20]).astype(self.dtype),
                'Y': np.random.uniform(0.1, 1, [2, 3, 1, 20]).astype(self.dtype)
            }
            self.outputs = {
                'Out': np.divide(self.inputs['X'], self.inputs['Y'])
            }

    class TestElementwiseDivOp_commonuse_1(ElementwiseDivOp):
        def init_input_output(self):
            self.inputs = {
                'X': np.random.uniform(0.1, 1, [2, 3, 100]).astype(self.dtype),
                'Y': np.random.uniform(0.1, 1, [1, 1, 100]).astype(self.dtype),
            }
            self.outputs = {
                'Out': np.divide(self.inputs['X'], self.inputs['Y'])
            }

    class TestElementwiseDivOp_commonuse_2(ElementwiseDivOp):
        def init_input_output(self):
            self.inputs = {
                'X':
                np.random.uniform(0.1, 1, [30, 3, 1, 5]).astype(self.dtype),
                'Y':
                np.random.uniform(0.1, 1, [30, 1, 4, 1]).astype(self.dtype),
            }
            self.outputs = {
                'Out': np.divide(self.inputs['X'], self.inputs['Y'])
            }

    class TestElementwiseDivOp_xsize_lessthan_ysize(ElementwiseDivOp):
        def init_input_output(self):
            self.inputs = {
                'X': np.random.uniform(0.1, 1, [10, 12]).astype(self.dtype),
                'Y':
                np.random.uniform(0.1, 1, [2, 3, 10, 12]).astype(self.dtype),
            }

            self.attrs = {'axis': 2}

            self.outputs = {
                'Out': np.divide(self.inputs['X'], self.inputs['Y'])
            }

    class TestElementwiseDivBroadcast(unittest.TestCase):
        def test_shape_with_batch_sizes(self):
            with fluid.program_guard(fluid.Program()):
                x_var = fluid.data(
                    name='x', dtype='float32', shape=[None, 3, None, None])
                one = 2.
                out = one / x_var
                exe = fluid.Executor(fluid.XPUPlace(0))
                x = np.random.uniform(0.1, 0.6,
                                      (1, 3, 32, 32)).astype('float32')
                out_result, = exe.run(feed={'x': x}, fetch_list=[out])
                self.assertEqual((out_result == (2 / x)).all(), True)


support_types = get_xpu_op_support_types('elementwise_div')
for stype in support_types:
    create_test_class(globals(), XPUTestElementwiseDivOp, stype)

if __name__ == '__main__':
    unittest.main()
