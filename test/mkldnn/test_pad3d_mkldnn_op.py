#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle.fluid.tests.unittests.eager_op_test import (
    OpTest,
    convert_float_to_uint16,
)


class TestPad3dOneDNNOp(OpTest):
    def setUp(self):
        paddle.enable_static()
        self.value = 0.0
        self.initTestCase()
        self.dtype = self.get_dtype()
        self.op_type = "pad3d"
        self.python_api = paddle.nn.functional.pad
        self._cpu_only = True
        self.use_onednn = True
        self.inputs = {
            'X': np.random.uniform(-1.0, 1.0, self.shape).astype("float32")
            if self.dtype == np.uint16
            else np.random.uniform(-1.0, 1.0, self.shape).astype(self.dtype)
        }
        self.attrs = {}
        if self.variable_paddings:
            self.attrs['paddings'] = []
            self.inputs['Paddings'] = (
                np.array(self.paddings).flatten().astype("int32")
            )
        else:
            self.attrs['paddings'] = (
                np.array(self.paddings).flatten().astype("int32")
            )
        self.attrs['value'] = self.value
        self.attrs['mode'] = self.mode
        self.attrs['data_format'] = self.data_format
        if self.data_format == "NCDHW":
            paddings = [
                (0, 0),
                (0, 0),
                (self.paddings[4], self.paddings[5]),
                (self.paddings[2], self.paddings[3]),
                (self.paddings[0], self.paddings[1]),
            ]
        else:
            paddings = [
                (0, 0),
                (self.paddings[4], self.paddings[5]),
                (self.paddings[2], self.paddings[3]),
                (self.paddings[0], self.paddings[1]),
                (0, 0),
            ]
        if self.mode == "constant":
            out = np.pad(
                self.inputs['X'],
                paddings,
                mode=self.mode,
                constant_values=self.value,
            )
        elif self.mode == "reflect":
            out = np.pad(self.inputs['X'], paddings, mode=self.mode)
        elif self.mode == "replicate":
            out = np.pad(self.inputs['X'], paddings, mode="edge")
        elif self.mode == "circular":
            out = np.pad(self.inputs['X'], paddings, mode="wrap")
        self.outputs = {'Out': out}

        if self.dtype == np.uint16:
            self.inputs['X'] = convert_float_to_uint16(self.inputs['X'])
            self.outputs['Out'] = convert_float_to_uint16(self.outputs['Out'])

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X'], 'Out')

    def get_dtype(self):
        return np.float64

    def initTestCase(self):
        self.shape = (2, 3, 4, 5, 6)
        self.paddings = [0, 0, 0, 0, 0, 0]
        self.mode = "constant"
        self.data_format = "NCDHW"
        self.pad_value = 0.0
        self.variable_paddings = False


class TestCase1(TestPad3dOneDNNOp):
    def initTestCase(self):
        self.shape = (2, 3, 4, 5, 6)
        self.paddings = [0, 1, 2, 3, 4, 5]
        self.mode = "constant"
        self.data_format = "NCDHW"
        self.value = 1.0
        self.variable_paddings = False


class TestCase2(TestPad3dOneDNNOp):
    def initTestCase(self):
        self.shape = (2, 3, 4, 5, 6)
        self.paddings = [1, 1, 1, 1, 1, 1]
        self.mode = "constant"
        self.data_format = "NDHWC"
        self.value = 1.0
        self.variable_paddings = False


class TestCase3(TestPad3dOneDNNOp):
    def initTestCase(self):
        self.shape = (2, 3, 4, 5, 6)
        self.paddings = [0, 1, 1, 0, 2, 3]
        self.mode = "reflect"
        self.data_format = "NCDHW"
        self.variable_paddings = False


class TestCase4(TestPad3dOneDNNOp):
    def initTestCase(self):
        self.shape = (4, 4, 4, 4, 4)
        self.paddings = [0, 1, 2, 1, 2, 3]
        self.mode = "reflect"
        self.data_format = "NDHWC"
        self.variable_paddings = False


class TestCase5(TestPad3dOneDNNOp):
    def initTestCase(self):
        self.shape = (2, 3, 4, 5, 6)
        self.paddings = [0, 1, 2, 3, 2, 1]
        self.mode = "replicate"
        self.data_format = "NCDHW"
        self.variable_paddings = False


class TestCase6(TestPad3dOneDNNOp):
    def initTestCase(self):
        self.shape = (4, 4, 4, 4, 4)
        self.paddings = [5, 4, 2, 1, 2, 3]
        self.mode = "replicate"
        self.data_format = "NDHWC"
        self.variable_paddings = False


class TestCase7(TestPad3dOneDNNOp):
    def initTestCase(self):
        self.shape = (2, 3, 4, 5, 6)
        self.paddings = [0, 1, 2, 3, 2, 1]
        self.mode = "circular"
        self.data_format = "NCDHW"
        self.variable_paddings = False


class TestCase8(TestPad3dOneDNNOp):
    def initTestCase(self):
        self.shape = (4, 4, 4, 4, 4)
        self.paddings = [0, 1, 2, 1, 2, 3]
        self.mode = "circular"
        self.data_format = "NDHWC"
        self.variable_paddings = False


class TestCase9(TestPad3dOneDNNOp):
    def initTestCase(self):
        self.shape = (2, 3, 4, 5, 6)
        self.paddings = [0, 1, 2, 3, 4, 5]
        self.mode = "constant"
        self.data_format = "NCDHW"
        self.value = 1.0
        self.variable_paddings = True


class TestCase10(TestPad3dOneDNNOp):
    def initTestCase(self):
        self.shape = (2, 3, 4, 5, 6)
        self.paddings = [0, 1, 2, 3, 4, 5]
        self.mode = "constant"
        self.data_format = "NDHWC"
        self.value = 1.0
        self.variable_paddings = True


def create_test_class(parent):
    class TestBf16Case(parent):
        def init_data_type(self):
            self.dtype = np.uint16

    TestBf16Case.__name__ = "{}_{}".format(parent.__name__, "BF16")
    globals()[TestBf16Case.__name__] = TestBf16Case


create_test_class(TestPad3dOneDNNOp)
create_test_class(TestCase1)
create_test_class(TestCase2)
create_test_class(TestCase3)
create_test_class(TestCase4)
create_test_class(TestCase5)
create_test_class(TestCase6)
create_test_class(TestCase7)
create_test_class(TestCase8)
create_test_class(TestCase9)

if __name__ == "__main__":
    from paddle import enable_static

    enable_static()
    unittest.main()
