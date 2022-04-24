# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import numpy as np
from op_test import OpTest
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid import Program, program_guard


class BaseTestCase(OpTest):
    def initTestCase(self):
        self.op_type = 'arg_min'
        self.dims = (3, 4, 5)
        self.dtype = 'float32'
        self.axis = 0

    def setUp(self):
        self.initTestCase()
        self.x = (1000 * np.random.random(self.dims)).astype(self.dtype)
        self.inputs = {'X': self.x}
        self.attrs = {'axis': self.axis}
        if self.op_type == "arg_min":
            self.outputs = {'Out': np.argmin(self.x, axis=self.axis)}
        else:
            self.outputs = {'Out': np.argmax(self.x, axis=self.axis)}

    def test_check_output(self):
        self.check_output()


class TestCase0(BaseTestCase):
    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (3, 4, 5)
        self.dtype = 'float32'
        self.axis = 0


class TestCase1(BaseTestCase):
    def initTestCase(self):
        self.op_type = 'arg_min'
        self.dims = (3, 4)
        self.dtype = 'float64'
        self.axis = 1


class TestCase2(BaseTestCase):
    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (3, 4)
        self.dtype = 'int64'
        self.axis = 0


@unittest.skipIf(not paddle.is_compiled_with_cuda(),
                 "FP16 test runs only on GPU")
class TestCase0FP16(BaseTestCase):
    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (3, 4, 5)
        self.dtype = np.float16
        self.axis = 0


@unittest.skipIf(not paddle.is_compiled_with_cuda(),
                 "FP16 test runs only on GPU")
class TestCase1FP16(BaseTestCase):
    def initTestCase(self):
        self.op_type = 'arg_min'
        self.dims = (3, 4)
        self.dtype = np.float16
        self.axis = 1


class TestCase2_1(BaseTestCase):
    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (3, 4)
        self.dtype = 'int64'
        self.axis = -1


class TestCase3(BaseTestCase):
    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (3, )
        self.dtype = 'int64'
        self.axis = 0


class TestCase4(BaseTestCase):
    def initTestCase(self):
        self.op_type = 'arg_min'
        self.dims = (1, )
        self.dtype = 'int32'
        self.axis = 0


class TestCase3_(BaseTestCase):
    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (3, )
        self.axis = 0


class BaseTestComplex1_1(OpTest):
    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (4, 5, 6)
        self.dtype = 'int32'
        self.axis = 2

    def setUp(self):
        self.initTestCase()
        self.x = (np.random.random(self.dims)).astype(self.dtype)
        self.inputs = {'X': self.x}
        self.attrs = {'axis': self.axis}
        self.attrs = {'dtype': int(core.VarDesc.VarType.INT32)}
        if self.op_type == "arg_min":
            self.outputs = {
                'Out': np.argmin(
                    self.x, axis=self.axis).asdtype("int32")
            }
        else:
            self.outputs = {
                'Out': np.argmax(
                    self.x, axis=self.axis).asdtype("int32")
            }


class BaseTestComplex1_2(OpTest):
    def initTestCase(self):
        self.op_type = 'arg_min'
        self.dims = (4, 5, 6)
        self.dtype = 'int32'
        self.axis = 2

    def setUp(self):
        self.initTestCase()
        self.x = (np.random.random(self.dims)).astype(self.dtype)
        self.inputs = {'X': self.x}
        self.attrs = {'axis': self.axis}
        self.attrs = {'dtype': int(core.VarDesc.VarType.INT32)}
        if self.op_type == "arg_min":
            self.outputs = {
                'Out': np.argmin(
                    self.x, axis=self.axis).asdtype("int32")
            }
        else:
            self.outputs = {
                'Out': np.argmax(
                    self.x, axis=self.axis).asdtype("int32")
            }


class BaseTestComplex2_1(OpTest):
    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (4, 5, 6)
        self.dtype = 'int32'
        self.axis = 2

    def setUp(self):
        self.initTestCase()
        self.x = (np.random.random(self.dims)).astype(self.dtype)
        self.inputs = {'X': self.x}
        self.attrs = {'axis': self.axis}
        self.attrs = {'dtype': int(core.VarDesc.VarType.INT32)}
        self.attrs = {'keep_dims': True}
        if self.op_type == "arg_min":
            self.outputs = {
                'Out': np.argmin(
                    self.x, axis=self.axis).asdtype("int32").reshape(4, 5, 1)
            }
        else:
            self.outputs = {
                'Out': np.argmax(
                    self.x, axis=self.axis).asdtype("int32").reshape(4, 5, 1)
            }


class BaseTestComplex2_2(OpTest):
    def initTestCase(self):
        self.op_type = 'arg_min'
        self.dims = (4, 5, 6)
        self.dtype = 'int32'
        self.axis = 2

    def setUp(self):
        self.initTestCase()
        self.x = (np.random.random(self.dims)).astype(self.dtype)
        self.inputs = {'X': self.x}
        self.attrs = {'axis': self.axis}
        self.attrs = {'dtype': int(core.VarDesc.VarType.INT32)}
        self.attrs = {'keep_dims': True}
        if self.op_type == "arg_min":
            self.outputs = {
                'Out': np.argmin(
                    self.x, axis=self.axis).asdtype("int32").reshape(4, 5, 1)
            }
        else:
            self.outputs = {
                'Out': np.argmax(
                    self.x, axis=self.axis).asdtype("int32").reshape(4, 5, 1)
            }


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
