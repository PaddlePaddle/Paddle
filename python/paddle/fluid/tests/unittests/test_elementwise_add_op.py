#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function
import unittest
import numpy as np
import paddle.fluid.core as core
from op_test import OpTest
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard


class TestElementwiseAddOp(OpTest):
    def init_kernel_type(self):
        self.use_mkldnn = False

    def setUp(self):
        self.op_type = "elementwise_add"
        self.dtype = np.float32
        self.axis = -1
        self.init_dtype()
        self.init_input_output()
        self.init_kernel_type()
        self.init_axis()

        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(self.x),
            'Y': OpTest.np_dtype_to_fluid_dtype(self.y)
        }
        self.attrs = {'axis': self.axis, 'use_mkldnn': self.use_mkldnn}
        self.outputs = {'Out': self.out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X', 'Y'], 'Out', max_relative_error=0.005)

    def test_check_grad_ingore_x(self):
        if self.dtype == np.float16:
            return
        self.check_grad(
            ['Y'], 'Out', max_relative_error=0.005, no_grad_set=set("X"))

    def test_check_grad_ingore_y(self):
        if self.dtype == np.float16:
            return
        self.check_grad(
            ['X'], 'Out', max_relative_error=0.005, no_grad_set=set('Y'))

    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.out = np.add(self.x, self.y)

    def init_dtype(self):
        pass

    def init_axis(self):
        pass


class TestFP16ElementwiseAddOp(TestElementwiseAddOp):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(place, atol=1e-3)


class TestElementwiseAddOp_scalar(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 4).astype(self.dtype)
        self.y = np.random.rand(1).astype(self.dtype)
        self.out = self.x + self.y


class TestFP16ElementwiseAddOp_scalar(TestFP16ElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 4).astype(self.dtype)
        self.y = np.random.rand(1).astype(self.dtype)
        self.out = self.x + self.y


class TestElementwiseAddOp_scalar2(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 4).astype(self.dtype)
        self.y = np.random.rand(1, 1).astype(self.dtype)
        self.out = self.x + self.y


class TestFP16ElementwiseAddOp_scalar2(TestFP16ElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 4).astype(self.dtype)
        self.y = np.random.rand(1, 1).astype(self.dtype)
        self.out = self.x + self.y


class TestElementwiseAddOp_Vector(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.random((32, )).astype(self.dtype)
        self.y = np.random.random((32, )).astype(self.dtype)
        self.out = np.add(self.x, self.y)


class TestFP16ElementwiseAddOp_Vector(TestFP16ElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.random((32, )).astype(self.dtype)
        self.y = np.random.random((32, )).astype(self.dtype)
        self.out = np.add(self.x, self.y)


class TestElementwiseAddOp_broadcast_0(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 4).astype(self.dtype)
        self.y = np.random.rand(2).astype(self.dtype)
        self.out = self.x + self.y.reshape(2, 1, 1)

    def init_axis(self):
        self.axis = 0


class TestFP16ElementwiseAddOp_broadcast_0(TestFP16ElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 4).astype(self.dtype)
        self.y = np.random.rand(2).astype(self.dtype)
        self.out = self.x + self.y.reshape(2, 1, 1)

    def init_axis(self):
        self.axis = 0


class TestElementwiseAddOp_broadcast_1(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 4).astype(self.dtype)
        self.y = np.random.rand(3).astype(self.dtype)
        self.out = self.x + self.y.reshape(1, 3, 1)

    def init_axis(self):
        self.axis = 1


class TestFP16ElementwiseAddOp_broadcast_1(TestFP16ElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 4).astype(self.dtype)
        self.y = np.random.rand(3).astype(self.dtype)
        self.out = self.x + self.y.reshape(1, 3, 1)

    def init_axis(self):
        self.axis = 1


class TestElementwiseAddOp_broadcast_2(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 4).astype(self.dtype)
        self.y = np.random.rand(4).astype(self.dtype)
        self.out = self.x + self.y.reshape(1, 1, 4)


class TestFP16ElementwiseAddOp_broadcast_2(TestFP16ElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 4).astype(self.dtype)
        self.y = np.random.rand(4).astype(self.dtype)
        self.out = self.x + self.y.reshape(1, 1, 4)


class TestElementwiseAddOp_broadcast_3(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 4, 5).astype(self.dtype)
        self.y = np.random.rand(3, 4).astype(self.dtype)
        self.out = self.x + self.y.reshape(1, 3, 4, 1)

    def init_axis(self):
        self.axis = 1


class TestFP16ElementwiseAddOp_broadcast_3(TestFP16ElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 4, 5).astype(self.dtype)
        self.y = np.random.rand(3, 4).astype(self.dtype)
        self.out = self.x + self.y.reshape(1, 3, 4, 1)

    def init_axis(self):
        self.axis = 1


class TestElementwiseAddOp_broadcast_4(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 4, 5).astype(self.dtype)
        self.y = np.random.rand(2, 1).astype(self.dtype)
        self.out = self.x + self.y.reshape(2, 1, 1, 1)

    def init_axis(self):
        self.axis = 0


class TestFP16ElementwiseAddOp_broadcast_4(TestFP16ElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 4, 5).astype(self.dtype)
        self.y = np.random.rand(2, 1).astype(self.dtype)
        self.out = self.x + self.y.reshape(2, 1, 1, 1)

    def init_axis(self):
        self.axis = 0


class TestElementwiseAddOp_broadcast_5(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 4).astype(self.dtype)
        self.y = np.random.rand(2, 1, 4).astype(self.dtype)
        self.out = self.x + self.y


class TestFP16ElementwiseAddOp_broadcast_5(TestFP16ElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 4).astype(self.dtype)
        self.y = np.random.rand(2, 1, 4).astype(self.dtype)
        self.out = self.x + self.y


class TestElementwiseAddOp_broadcast_6(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 4, 5).astype(self.dtype)
        self.y = np.random.rand(2, 3, 1, 5).astype(self.dtype)
        self.out = self.x + self.y


class TestFP16ElementwiseAddOp_broadcast_6(TestFP16ElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 4, 5).astype(self.dtype)
        self.y = np.random.rand(2, 3, 1, 5).astype(self.dtype)
        self.out = self.x + self.y


class TestElementwiseAddOp_rowwise_add_0(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 4).astype(self.dtype)
        self.y = np.random.rand(3, 4).astype(self.dtype)
        self.out = self.x + self.y.reshape(1, 3, 4)

    def init_axis(self):
        self.axis = 1


class TestFP16ElementwiseAddOp_rowwise_add_0(TestFP16ElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 4).astype(self.dtype)
        self.y = np.random.rand(3, 4).astype(self.dtype)
        self.out = self.x + self.y.reshape(1, 3, 4)

    def init_axis(self):
        self.axis = 1


class TestElementwiseAddOp_rowwise_add_1(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 1).astype(self.dtype)
        self.y = np.random.rand(1).astype(self.dtype)
        self.out = self.x + self.y.reshape(1, 1)

    def init_axis(self):
        self.axis = 1


class TestFP16ElementwiseAddOp_rowwise_add_1(TestFP16ElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 1).astype(self.dtype)
        self.y = np.random.rand(1).astype(self.dtype)
        self.out = self.x + self.y.reshape(1, 1)

    def init_axis(self):
        self.axis = 1


class TestElementwiseAddOp_channelwise_add(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(3, 20, 20).astype(self.dtype)
        self.y = np.random.rand(3, 1, 1).astype(self.dtype)
        self.out = self.x + self.y

    def init_axis(self):
        self.axis = -1


class TestFP16ElementwiseAddOp_channelwise_add(TestFP16ElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(3, 10, 20).astype(self.dtype)
        self.y = np.random.rand(3, 1, 1).astype(self.dtype)
        self.out = self.x + self.y

    def init_axis(self):
        self.axis = -1


class TestElementwiseAddOpError(OpTest):
    def test_errors(self):
        with program_guard(Program(), Program()):
            # the input of elementwise_add must be Variable.
            x1 = fluid.create_lod_tensor(
                np.array([-1, 3, 5, 5]), [[1, 1, 1, 1]], fluid.CPUPlace())
            y1 = fluid.create_lod_tensor(
                np.array([-1, 3, 5, 5]), [[1, 1, 1, 1]], fluid.CPUPlace())
            self.assertRaises(TypeError, fluid.layers.elementwise_add, x1, y1)

            # the input dtype of elementwise_add must be float16 or float32 or float64 or int32 or int64
            # float16 only can be set on GPU place
            x2 = fluid.layers.data(name='x2', shape=[3, 4, 5, 6], dtype="uint8")
            y2 = fluid.layers.data(name='y2', shape=[3, 4, 5, 6], dtype="uint8")
            self.assertRaises(TypeError, fluid.layers.elementwise_add, x2, y2)


if __name__ == '__main__':
    unittest.main()
