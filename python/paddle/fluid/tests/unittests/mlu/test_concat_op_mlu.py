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

from __future__ import print_function

import numpy as np
import unittest
import sys
sys.path.append("..")
from op_test import OpTest, skip_check_grad_ci
import paddle
import paddle.fluid as fluid

paddle.enable_static()
SEED = 2021


class TestConcatOp(OpTest):
    def setUp(self):
        self.set_mlu()
        self.op_type = "concat"
        self.place = paddle.device.MLUPlace(0)
        self.init_dtype()
        self.init_test_data()

        self.inputs = {'X': [('x0', self.x0), ('x1', self.x1), ('x2', self.x2)]}
        self.attrs = {'axis': self.axis}
        if self.axis < 0:
            self.actual_axis = self.axis + len(self.x0.shape)
            self.actual_axis = self.actual_axis if self.actual_axis > 0 else 0
        else:
            self.actual_axis = self.axis

        self.outputs = {
            'Out': np.concatenate(
                (self.x0, self.x1, self.x2), axis=self.actual_axis)
        }

    def set_mlu(self):
        self.__class__.use_mlu = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ['x0', 'x2'], 'Out')
        self.check_grad_with_place(self.place, ['x1'], 'Out')
        self.check_grad_with_place(self.place, ['x2'], 'Out')

    def init_test_data(self):
        self.x0 = np.random.random((1, 4, 50)).astype(self.dtype)
        self.x1 = np.random.random((2, 4, 50)).astype(self.dtype)
        self.x2 = np.random.random((3, 4, 50)).astype(self.dtype)
        self.axis = 0


class TestConcatOp2(TestConcatOp):
    def init_test_data(self):
        self.x0 = np.random.random((2, 3, 4, 5)).astype(self.dtype)
        self.x1 = np.random.random((2, 3, 4, 5)).astype(self.dtype)
        self.x2 = np.random.random((2, 3, 4, 5)).astype(self.dtype)
        self.axis = 1


@skip_check_grad_ci(
    reason="The function 'check_grad' for large inputs is too slow.")
class TestConcatOp3(TestConcatOp):
    def init_test_data(self):
        self.x0 = np.random.random((1, 256, 170, 256)).astype(self.dtype)
        self.x1 = np.random.random((1, 128, 170, 256)).astype(self.dtype)
        self.x2 = np.random.random((1, 128, 170, 256)).astype(self.dtype)
        self.axis = 1

    def test_check_grad(self):
        pass


@skip_check_grad_ci(
    reason="This test will meet fetch error when there is a null grad. The detailed information is in PR#17015."
)
class TestConcatOp4(TestConcatOp):
    def init_test_data(self):
        self.x0 = np.random.random((2, 3, 4, 5)).astype(self.dtype)
        self.x1 = np.random.random((2, 3, 4, 5)).astype(self.dtype)
        self.x2 = np.random.random((0, 3, 4, 5)).astype(self.dtype)
        self.axis = 0

    def test_check_grad(self):
        pass


class TestConcatOp5(TestConcatOp):
    def init_test_data(self):
        self.x0 = np.random.random((5, 1, 4, 5)).astype(self.dtype)
        self.x1 = np.random.random((5, 2, 4, 5)).astype(self.dtype)
        self.x2 = np.random.random((5, 3, 4, 5)).astype(self.dtype)
        self.axis = -3


#----------------Concat Fp16----------------
def create_test_fp16(parent):
    class TestConcatFp16(parent):
        def init_dtype(self):
            self.dtype = np.float16

    cls_name = "{0}_{1}".format(parent.__name__, "Fp16")
    TestConcatFp16.__name__ = cls_name
    globals()[cls_name] = TestConcatFp16


create_test_fp16(TestConcatOp)
create_test_fp16(TestConcatOp2)
create_test_fp16(TestConcatOp3)
create_test_fp16(TestConcatOp4)
create_test_fp16(TestConcatOp5)


#----------------Concat Int64----------------
def create_test_int64(parent):
    class TestConcatInt64(parent):
        def init_dtype(self):
            self.dtype = np.int64

        def test_check_grad(self):
            pass

    cls_name = "{0}_{1}".format(parent.__name__, "Int64")
    TestConcatInt64.__name__ = cls_name
    globals()[cls_name] = TestConcatInt64


create_test_int64(TestConcatOp)
create_test_int64(TestConcatOp2)
create_test_int64(TestConcatOp3)
create_test_int64(TestConcatOp4)
create_test_int64(TestConcatOp5)


#----------------Concat Int32----------------
def create_test_int32(parent):
    class TestConcatInt32(parent):
        def init_dtype(self):
            self.dtype = np.int32

        def test_check_grad(self):
            pass

    cls_name = "{0}_{1}".format(parent.__name__, "Int32")
    TestConcatInt32.__name__ = cls_name
    globals()[cls_name] = TestConcatInt32


create_test_int32(TestConcatOp)
create_test_int32(TestConcatOp2)
create_test_int32(TestConcatOp3)
create_test_int32(TestConcatOp4)
create_test_int32(TestConcatOp5)


#----------------Concat AxisTensor----------------
def create_test_AxisTensor(parent):
    class TestConcatAxisTensor(parent):
        def setUp(self):
            self.op_type = "concat"
            self.init_dtype()
            self.init_test_data()

            self.inputs = {
                'X': [('x0', self.x0), ('x1', self.x1), ('x2', self.x2)],
                'AxisTensor': np.array([self.axis]).astype("int32")
            }
            self.attrs = {}

            if self.axis < 0:
                self.actual_axis = self.axis + len(self.x0.shape)
                self.actual_axis = self.actual_axis if self.actual_axis > 0 else 0
            else:
                self.actual_axis = self.axis

            self.outputs = {
                'Out': np.concatenate(
                    (self.x0, self.x1, self.x2), axis=self.actual_axis)
            }

            self.place = paddle.device.MLUPlace(0)
            self.__class__.use_mlu = True

        def init_test_data(self):
            self.x0 = np.random.random((1, 4, 50)).astype(self.dtype)
            self.x1 = np.random.random((2, 4, 50)).astype(self.dtype)
            self.x2 = np.random.random((3, 4, 50)).astype(self.dtype)
            self.axis = 0

        def init_dtype(self):
            self.dtype = np.float32

    cls_name = "{0}_{1}".format(parent.__name__, "AxisTensor")
    TestConcatAxisTensor.__name__ = cls_name
    globals()[cls_name] = TestConcatAxisTensor


create_test_AxisTensor(TestConcatOp)
create_test_AxisTensor(TestConcatOp2)
create_test_AxisTensor(TestConcatOp3)
create_test_AxisTensor(TestConcatOp4)
create_test_AxisTensor(TestConcatOp5)

if __name__ == '__main__':
    unittest.main()
