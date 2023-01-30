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
<<<<<<< HEAD

import numpy as np
from op_test import OpTest, skip_check_grad_ci
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)

import paddle
=======
import numpy as np
from op_test import OpTest, skip_check_grad_ci
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard
import paddle
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

paddle.enable_static()


class XPUTestElementwiseMulOp(XPUOpTestWrapper):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def __init__(self):
        self.op_name = 'elementwise_mul'
        self.use_dynamic_create_class = False

    class ElementwiseMulOp(XPUOpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def init_kernel_type(self):
            self.use_mkldnn = False

        def setUp(self):
            self.op_type = 'elementwise_mul'
            self.use_xpu = True
            self.dtype = self.in_type
            self.axis = -1
            self.init_dtype()
            self.init_input_output()
            self.init_kernel_type()
            self.init_axis()

        def test_check_output(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_output_with_place(place)

        def test_check_grad_normal(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_grad_with_place(
<<<<<<< HEAD
                    place,
                    ['X', 'Y'],
                    'Out',
                    check_dygraph=(not self.use_mkldnn),
                )
=======
                    place, ['X', 'Y'],
                    'Out',
                    check_dygraph=(self.use_mkldnn == False))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        def test_check_grad_ingore_x(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_grad_with_place(
<<<<<<< HEAD
                    place,
                    ['Y'],
                    'Out',
                    no_grad_set=set("X"),
                    check_dygraph=(not self.use_mkldnn),
                )
=======
                    place, ['Y'],
                    'Out',
                    no_grad_set=set("X"),
                    check_dygraph=(self.use_mkldnn == False))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        def test_check_grad_ingore_y(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_grad_with_place(
<<<<<<< HEAD
                    place,
                    ['X'],
                    'Out',
                    no_grad_set=set('Y'),
                    check_dygraph=(not self.use_mkldnn),
                )
=======
                    place, ['X'],
                    'Out',
                    no_grad_set=set('Y'),
                    check_dygraph=(self.use_mkldnn == False))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        def init_input_output(self):
            self.x = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
            self.y = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
            self.out = np.multiply(self.x, self.y)
            self.inputs = {
                'X': OpTest.np_dtype_to_fluid_dtype(self.x),
<<<<<<< HEAD
                'Y': OpTest.np_dtype_to_fluid_dtype(self.y),
=======
                'Y': OpTest.np_dtype_to_fluid_dtype(self.y)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            }
            self.outputs = {'Out': self.out}
            self.attrs = {'axis': self.axis, 'use_mkldnn': self.use_mkldnn}

        def init_dtype(self):
            pass

        def init_axis(self):
            pass

<<<<<<< HEAD
    class TestElementwiseMulOp_ZeroDim1(ElementwiseMulOp):
        def init_input_output(self):
            self.inputs = {
                'X': np.random.uniform(-1, 1, []).astype(self.dtype),
                'Y': np.random.uniform(-1, 1, []).astype(self.dtype),
            }
            self.outputs = {'Out': self.inputs['X'] * self.inputs['Y']}

    class TestElementwiseMulOp_ZeroDim2(ElementwiseMulOp):
        def init_input_output(self):
            self.inputs = {
                'X': np.random.uniform(-1, 1, [13, 17]).astype(self.dtype),
                'Y': np.random.uniform(-1, 1, []).astype(self.dtype),
            }
            self.outputs = {'Out': self.inputs['X'] * self.inputs['Y']}

    class TestElementwiseMulOp_ZeroDim3(ElementwiseMulOp):
        def init_input_output(self):
            self.inputs = {
                'X': np.random.uniform(-1, 1, []).astype(self.dtype),
                'Y': np.random.uniform(-1, 1, [13, 17]).astype(self.dtype),
            }
            self.outputs = {'Out': self.inputs['X'] * self.inputs['Y']}

    @skip_check_grad_ci(
        reason="[skip shape check] Use y_shape(1) to test broadcast."
    )
    class TestElementwiseMulOp_scalar(ElementwiseMulOp):
        def init_input_output(self):
            self.inputs = {
                'X': np.random.rand(10, 3, 4).astype(self.dtype),
                'Y': np.random.rand(1).astype(self.dtype),
=======
    @skip_check_grad_ci(
        reason="[skip shape check] Use y_shape(1) to test broadcast.")
    class TestElementwiseMulOp_scalar(ElementwiseMulOp):

        def init_input_output(self):
            self.inputs = {
                'X': np.random.rand(10, 3, 4).astype(self.dtype),
                'Y': np.random.rand(1).astype(self.dtype)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            }
            self.outputs = {'Out': self.inputs['X'] * self.inputs['Y']}

    class TestElementwiseMulOp_Vector(ElementwiseMulOp):
<<<<<<< HEAD
        def init_input_output(self):
            self.inputs = {
                'X': np.random.random((100,)).astype(self.dtype),
                'Y': np.random.random((100,)).astype(self.dtype),
=======

        def init_input_output(self):
            self.inputs = {
                'X': np.random.random((100, )).astype(self.dtype),
                'Y': np.random.random((100, )).astype(self.dtype)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            }
            self.outputs = {
                'Out': np.multiply(self.inputs['X'], self.inputs['Y'])
            }

    class TestElementwiseMulOp_broadcast_0(ElementwiseMulOp):
<<<<<<< HEAD
        def init_input_output(self):
            self.inputs = {
                'X': np.random.rand(100, 2, 3).astype(self.dtype),
                'Y': np.random.rand(100).astype(self.dtype),
=======

        def init_input_output(self):
            self.inputs = {
                'X': np.random.rand(100, 2, 3).astype(self.dtype),
                'Y': np.random.rand(100).astype(self.dtype)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            }
            self.outputs = {
                'Out': self.inputs['X'] * self.inputs['Y'].reshape(100, 1, 1)
            }
            self.attrs = {'axis': 0}

    class TestElementwiseMulOp_broadcast_1(ElementwiseMulOp):
<<<<<<< HEAD
        def init_input_output(self):
            self.inputs = {
                'X': np.random.rand(2, 100, 3).astype(self.dtype),
                'Y': np.random.rand(100).astype(self.dtype),
=======

        def init_input_output(self):
            self.inputs = {
                'X': np.random.rand(2, 100, 3).astype(self.dtype),
                'Y': np.random.rand(100).astype(self.dtype)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            }

            self.attrs = {'axis': 1}
            self.outputs = {
                'Out': self.inputs['X'] * self.inputs['Y'].reshape(1, 100, 1)
            }

    class TestElementwiseMulOp_broadcast_2(ElementwiseMulOp):
<<<<<<< HEAD
        def init_input_output(self):
            self.inputs = {
                'X': np.random.rand(2, 3, 100).astype(self.dtype),
                'Y': np.random.rand(100).astype(self.dtype),
=======

        def init_input_output(self):
            self.inputs = {
                'X': np.random.rand(2, 3, 100).astype(self.dtype),
                'Y': np.random.rand(100).astype(self.dtype)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            }

            self.outputs = {
                'Out': self.inputs['X'] * self.inputs['Y'].reshape(1, 1, 100)
            }

    class TestElementwiseMulOp_broadcast_3(ElementwiseMulOp):
<<<<<<< HEAD
        def init_input_output(self):
            self.inputs = {
                'X': np.random.rand(2, 10, 12, 3).astype(self.dtype),
                'Y': np.random.rand(10, 12).astype(self.dtype),
=======

        def init_input_output(self):
            self.inputs = {
                'X': np.random.rand(2, 10, 12, 3).astype(self.dtype),
                'Y': np.random.rand(10, 12).astype(self.dtype)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            }

            self.attrs = {'axis': 1}
            self.outputs = {
<<<<<<< HEAD
                'Out': self.inputs['X'] * self.inputs['Y'].reshape(1, 10, 12, 1)
            }

    class TestElementwiseMulOp_broadcast_4(ElementwiseMulOp):
        def init_input_output(self):
            self.inputs = {
                'X': np.random.rand(10, 2, 11).astype(self.dtype),
                'Y': np.random.rand(10, 1, 11).astype(self.dtype),
=======
                'Out':
                self.inputs['X'] * self.inputs['Y'].reshape(1, 10, 12, 1)
            }

    class TestElementwiseMulOp_broadcast_4(ElementwiseMulOp):

        def init_input_output(self):
            self.inputs = {
                'X': np.random.rand(10, 2, 11).astype(self.dtype),
                'Y': np.random.rand(10, 1, 11).astype(self.dtype)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            }
            self.outputs = {'Out': self.inputs['X'] * self.inputs['Y']}

    class TestElementwiseMulOp_broadcast_5(ElementwiseMulOp):
<<<<<<< HEAD
        def init_input_output(self):
            self.inputs = {
                'X': np.random.rand(10, 4, 2, 3).astype(self.dtype),
                'Y': np.random.rand(10, 4, 1, 3).astype(self.dtype),
=======

        def init_input_output(self):
            self.inputs = {
                'X': np.random.rand(10, 4, 2, 3).astype(self.dtype),
                'Y': np.random.rand(10, 4, 1, 3).astype(self.dtype)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            }
            self.outputs = {'Out': self.inputs['X'] * self.inputs['Y']}

    class TestElementwiseMulOp_commonuse_1(ElementwiseMulOp):
<<<<<<< HEAD
        def init_input_output(self):
            self.inputs = {
                'X': np.random.rand(2, 3, 100).astype(self.dtype),
                'Y': np.random.rand(1, 1, 100).astype(self.dtype),
=======

        def init_input_output(self):
            self.inputs = {
                'X': np.random.rand(2, 3, 100).astype(self.dtype),
                'Y': np.random.rand(1, 1, 100).astype(self.dtype)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            }
            self.outputs = {'Out': self.inputs['X'] * self.inputs['Y']}

    class TestElementwiseMulOp_commonuse_2(ElementwiseMulOp):
<<<<<<< HEAD
        def init_input_output(self):
            self.inputs = {
                'X': np.random.rand(30, 3, 1, 5).astype(self.dtype),
                'Y': np.random.rand(30, 1, 4, 1).astype(self.dtype),
=======

        def init_input_output(self):
            self.inputs = {
                'X': np.random.rand(30, 3, 1, 5).astype(self.dtype),
                'Y': np.random.rand(30, 1, 4, 1).astype(self.dtype)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            }
            self.outputs = {'Out': self.inputs['X'] * self.inputs['Y']}

    class TestElementwiseMulOp_xsize_lessthan_ysize(ElementwiseMulOp):
<<<<<<< HEAD
        def init_input_output(self):
            self.inputs = {
                'X': np.random.rand(10, 10).astype(self.dtype),
                'Y': np.random.rand(2, 2, 10, 10).astype(self.dtype),
=======

        def init_input_output(self):
            self.inputs = {
                'X': np.random.rand(10, 10).astype(self.dtype),
                'Y': np.random.rand(2, 2, 10, 10).astype(self.dtype)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            }

            self.attrs = {'axis': 2}

            self.outputs = {
                'Out': self.inputs['X'].reshape(1, 1, 10, 10) * self.inputs['Y']
            }

<<<<<<< HEAD
=======
    class TestElementwiseMulOpError(unittest.TestCase):

        def test_errors(self):
            with program_guard(Program(), Program()):
                # the input of elementwise_mul must be Variable.
                x1 = fluid.create_lod_tensor(np.array([-1, 3, 5, 5]),
                                             [[1, 1, 1, 1]], fluid.XPUPlace(0))
                y1 = fluid.create_lod_tensor(np.array([-1, 3, 5, 5]),
                                             [[1, 1, 1, 1]], fluid.XPUPlace(0))
                self.assertRaises(TypeError, fluid.layers.elementwise_mul, x1,
                                  y1)

                # the input dtype of elementwise_mul must be float32
                x2 = fluid.layers.data(name='x2',
                                       shape=[3, 4, 5, 6],
                                       dtype="uint8")
                y2 = fluid.layers.data(name='y2',
                                       shape=[3, 4, 5, 6],
                                       dtype="uint8")
                self.assertRaises(TypeError, fluid.layers.elementwise_mul, x2,
                                  y2)

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

support_types = get_xpu_op_support_types('elementwise_mul')
for stype in support_types:
    create_test_class(globals(), XPUTestElementwiseMulOp, stype)

if __name__ == '__main__':
    unittest.main()
