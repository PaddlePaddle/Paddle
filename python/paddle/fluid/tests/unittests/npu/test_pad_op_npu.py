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

import unittest
import numpy as np
import sys

sys.path.append("..")
from op_test import OpTest
import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard

paddle.enable_static()


class TestPadOp(OpTest):
    def setUp(self):
        self.op_type = "pad"
        self.set_npu()
        self.init_dtype()
        self.initTestCase()

        self.inputs = {
            'X': np.random.random(self.shape).astype(self.dtype),
        }
        self.attrs = {}
        self.attrs['paddings'] = np.array(self.paddings).flatten()
        self.attrs['pad_value'] = self.pad_value
        self.outputs = {
            'Out': np.pad(
                self.inputs['X'],
                self.paddings,
                mode='constant',
                constant_values=self.pad_value,
            )
        }

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
        if self.dtype == np.float16:
            self.check_grad_with_place(
                self.place, ['X'], 'Out', max_relative_error=0.6
            )
        else:
            self.check_grad_with_place(self.place, ['X'], 'Out')

    def set_npu(self):
        self.__class__.use_npu = True
        self.place = paddle.NPUPlace(0)

    def init_dtype(self):
        self.dtype = np.float32

    def initTestCase(self):
        self.shape = (16, 16)
        self.paddings = [(1, 1), (2, 3)]
        self.pad_value = 0.0


class TestCase1(TestPadOp):
    def initTestCase(self):
        self.shape = (2, 3, 4, 5)
        self.paddings = [(0, 1), (2, 3), (2, 1), (1, 1)]
        self.pad_value = 0.0


class TestCase2(TestPadOp):
    def initTestCase(self):
        self.shape = (5, 5, 5)
        self.paddings = [(0, 0), (0, 0), (1, 2)]
        self.pad_value = 0.0


class TestCase3(TestPadOp):
    def initTestCase(self):
        self.shape = 100
        self.paddings = [(0, 1)]
        self.pad_value = 0.0


# ----------------Pad Fp16----------------


def create_test_fp16(parent):
    class TestPadFp16(parent):
        def init_dtype(self):
            self.dtype = np.float16

    cls_name = "{0}_{1}".format(parent.__name__, "Fp16")
    TestPadFp16.__name__ = cls_name
    globals()[cls_name] = TestPadFp16


create_test_fp16(TestPadOp)
create_test_fp16(TestCase1)
create_test_fp16(TestCase2)
create_test_fp16(TestCase3)


class TestPadOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            input_data = np.random.random((2, 2)).astype("float32")

            def test_Variable():
                paddle.nn.functional.pad(x=input_data, pad=[1, 1, 1, 1])

            self.assertRaises(TypeError, test_Variable)

            data = fluid.data(name='data', shape=[4], dtype='float16')
            paddle.nn.functional.pad(x=data, pad=[0, 1])


if __name__ == '__main__':
    unittest.main()
