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
from paddle.fluid import core
from paddle.fluid.tests.unittests.eager_op_test import (
    OpTest,
    convert_float_to_uint16,
)


class TestPad3dFp16(OpTest):
    def setUp(self):
        paddle.enable_static()
        self.value = 0.0
        self.initTestCase()
        self.dtype = self.get_dtype()
        self.op_type = "pad3d"
        self.python_api = paddle.nn.functional.pad
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
        self.attrs['use_mkldnn'] = True
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

    def get_dtype(self):
        return np.float16

    def test_check_output(self):
        self.check_output(atol=1e-3)

    def test_check_grad_normal(self):
        self.check_grad(['X'], 'Out', max_relative_error=1.5e-3)

    def initTestCase(self):
        self.shape = (2, 3, 4, 5, 6)
        self.paddings = [0, 0, 0, 0, 0, 0]
        self.mode = "constant"
        self.data_format = "NCDHW"
        self.pad_value = 0.0
        self.variable_paddings = False


class TestCase1(TestPad3dFp16):
    def initTestCase(self):
        self.shape = (2, 3, 4, 5, 6)
        self.paddings = [0, 1, 2, 3, 4, 5]
        self.mode = "constant"
        self.data_format = "NCDHW"
        self.value = 1.0
        self.variable_paddings = False


class TestCase2(TestPad3dFp16):
    def initTestCase(self):
        self.shape = (2, 3, 4, 5, 6)
        self.paddings = [1, 1, 1, 1, 1, 1]
        self.mode = "constant"
        self.data_format = "NDHWC"
        self.value = 1.0
        self.variable_paddings = False


# ----------------Pad3d Bf16----------------


def create_test_bf16(parent):
    class TestPad3dBf16(parent):
        def get_dtype(self):
            return np.uint16

        def test_check_output(self):
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, atol=1e-2)

        def test_check_grad_normal(self):
            place = core.CUDAPlace(0)
            self.check_grad_with_place(
                place, ['X'], 'Out', max_relative_error=1e-2
            )

    cls_name = "{}_{}".format(parent.__name__, "BF16OP")
    TestPad3dBf16.__name__ = cls_name
    globals()[cls_name] = TestPad3dBf16


create_test_bf16(TestCase1)
create_test_bf16(TestCase2)

if __name__ == '__main__':
    unittest.main()
