#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

<<<<<<< HEAD
=======
from __future__ import print_function

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
import numpy as np
import unittest
import sys

sys.path.append("..")
from op_test import OpTest, skip_check_grad_ci
import paddle
import paddle.fluid as fluid

paddle.enable_static()


class ElementwiseMulOp(OpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_npu(self):
        self.__class__.use_npu = True
        self.place = paddle.NPUPlace(0)

    def setUp(self):
        self.set_npu()
        self.op_type = "elementwise_mul"
        self.dtype = np.float32
        self.axis = -1
        self.init_dtype()
        self.init_input_output()
        self.init_axis()

        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(self.x),
<<<<<<< HEAD
            'Y': OpTest.np_dtype_to_fluid_dtype(self.y),
=======
            'Y': OpTest.np_dtype_to_fluid_dtype(self.y)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }
        self.outputs = {'Out': self.out}
        self.attrs = {'axis': self.axis}

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
        self.check_grad_with_place(self.place, ['X', 'Y'], 'Out')

    def test_check_grad_ingore_x(self):
<<<<<<< HEAD
        self.check_grad_with_place(
            self.place, ['Y'], 'Out', no_grad_set=set("X")
        )

    def test_check_grad_ingore_y(self):
        self.check_grad_with_place(
            self.place, ['X'], 'Out', no_grad_set=set('Y')
        )
=======
        self.check_grad_with_place(self.place, ['Y'],
                                   'Out',
                                   no_grad_set=set("X"))

    def test_check_grad_ingore_y(self):
        self.check_grad_with_place(self.place, ['X'],
                                   'Out',
                                   no_grad_set=set('Y'))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.out = np.multiply(self.x, self.y)

    def init_dtype(self):
        pass

    def init_axis(self):
        pass


@skip_check_grad_ci(
<<<<<<< HEAD
    reason="[skip shape check] Use y_shape(1) to test broadcast."
)
class TestElementwiseMulOp_scalar(ElementwiseMulOp):
=======
    reason="[skip shape check] Use y_shape(1) to test broadcast.")
class TestElementwiseMulOp_scalar(ElementwiseMulOp):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.set_npu()
        self.op_type = "elementwise_mul"
        self.inputs = {
            'X': np.random.rand(10, 3, 4).astype(np.float32),
<<<<<<< HEAD
            'Y': np.random.rand(1).astype(np.float32),
=======
            'Y': np.random.rand(1).astype(np.float32)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }
        self.outputs = {'Out': self.inputs['X'] * self.inputs['Y']}


class TestElementwiseMulOp_Vector(ElementwiseMulOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.set_npu()
        self.op_type = "elementwise_mul"
        self.inputs = {
<<<<<<< HEAD
            'X': np.random.random((100,)).astype("float32"),
            'Y': np.random.random((100,)).astype("float32"),
=======
            'X': np.random.random((100, )).astype("float32"),
            'Y': np.random.random((100, )).astype("float32")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }
        self.outputs = {'Out': np.multiply(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseMulOp_broadcast_0(ElementwiseMulOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_input_output(self):
        self.x = np.random.rand(100, 2, 3).astype(self.dtype)
        self.y = np.random.rand(100).astype(self.dtype)
        self.out = self.x * self.y.reshape(100, 1, 1)

    def init_axis(self):
        self.axis = 0


class TestElementwiseMulOp_broadcast_1(ElementwiseMulOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.set_npu()
        self.op_type = "elementwise_mul"
        self.inputs = {
            'X': np.random.rand(2, 100, 3).astype(np.float32),
<<<<<<< HEAD
            'Y': np.random.rand(100).astype(np.float32),
=======
            'Y': np.random.rand(100).astype(np.float32)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }

        self.attrs = {'axis': 1}
        self.outputs = {
            'Out': self.inputs['X'] * self.inputs['Y'].reshape(1, 100, 1)
        }


class TestElementwiseMulOp_broadcast_2(ElementwiseMulOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.set_npu()
        self.op_type = "elementwise_mul"
        self.inputs = {
            'X': np.random.rand(2, 3, 100).astype(np.float32),
<<<<<<< HEAD
            'Y': np.random.rand(100).astype(np.float32),
=======
            'Y': np.random.rand(100).astype(np.float32)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }

        self.outputs = {
            'Out': self.inputs['X'] * self.inputs['Y'].reshape(1, 1, 100)
        }


class TestElementwiseMulOp_broadcast_3(ElementwiseMulOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.set_npu()
        self.op_type = "elementwise_mul"
        self.inputs = {
            'X': np.random.rand(2, 10, 12, 3).astype(np.float32),
<<<<<<< HEAD
            'Y': np.random.rand(10, 12).astype(np.float32),
=======
            'Y': np.random.rand(10, 12).astype(np.float32)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }

        self.attrs = {'axis': 1}
        self.outputs = {
            'Out': self.inputs['X'] * self.inputs['Y'].reshape(1, 10, 12, 1)
        }


class TestElementwiseMulOp_broadcast_4(ElementwiseMulOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.set_npu()
        self.op_type = "elementwise_mul"
        self.inputs = {
            'X': np.random.rand(10, 2, 11).astype(np.float32),
<<<<<<< HEAD
            'Y': np.random.rand(10, 1, 11).astype(np.float32),
=======
            'Y': np.random.rand(10, 1, 11).astype(np.float32)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }
        self.outputs = {'Out': self.inputs['X'] * self.inputs['Y']}


class TestElementwiseMulOp_broadcast_5(ElementwiseMulOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.set_npu()
        self.op_type = "elementwise_mul"
        self.inputs = {
            'X': np.random.rand(10, 4, 2, 3).astype(np.float32),
<<<<<<< HEAD
            'Y': np.random.rand(10, 4, 1, 3).astype(np.float32),
=======
            'Y': np.random.rand(10, 4, 1, 3).astype(np.float32)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }
        self.outputs = {'Out': self.inputs['X'] * self.inputs['Y']}


<<<<<<< HEAD
@unittest.skipIf(
    not paddle.is_compiled_with_npu(), "paddle is not compiled with NPU"
)
class TestElementwiseMulOpFp16(ElementwiseMulOp):
=======
@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "paddle is not compiled with NPU")
class TestElementwiseMulOpFp16(ElementwiseMulOp):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_dtype(self):
        self.dtype = np.float16


class TestElementwiseMulOp_commonuse_1(ElementwiseMulOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.set_npu()
        self.op_type = "elementwise_mul"
        self.inputs = {
            'X': np.random.rand(2, 3, 100).astype(np.float32),
<<<<<<< HEAD
            'Y': np.random.rand(1, 1, 100).astype(np.float32),
=======
            'Y': np.random.rand(1, 1, 100).astype(np.float32)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }
        self.outputs = {'Out': self.inputs['X'] * self.inputs['Y']}


class TestElementwiseMulOp_commonuse_2(ElementwiseMulOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.set_npu()
        self.op_type = "elementwise_mul"
        self.inputs = {
            'X': np.random.rand(30, 3, 1, 5).astype(np.float32),
<<<<<<< HEAD
            'Y': np.random.rand(30, 1, 4, 1).astype(np.float32),
=======
            'Y': np.random.rand(30, 1, 4, 1).astype(np.float32)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }
        self.outputs = {'Out': self.inputs['X'] * self.inputs['Y']}


class TestElementwiseMulOp_xsize_lessthan_ysize(ElementwiseMulOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.set_npu()
        self.op_type = "elementwise_mul"
        self.inputs = {
            'X': np.random.rand(10, 10).astype(np.float32),
<<<<<<< HEAD
            'Y': np.random.rand(2, 2, 10, 10).astype(np.float32),
=======
            'Y': np.random.rand(2, 2, 10, 10).astype(np.float32)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }

        self.attrs = {'axis': 2}

        self.outputs = {
            'Out': self.inputs['X'].reshape(1, 1, 10, 10) * self.inputs['Y']
        }


if __name__ == '__main__':
    unittest.main()
