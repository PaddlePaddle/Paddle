# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import sys

sys.path.append('../../../../legacy_test/')
import unittest

import numpy as np
from op_test import OpTest, convert_float_to_uint16

import paddle
from paddle.base import core


class TestPowOp(OpTest):
    def setUp(self):
        self.op_type = "pow"
        self.python_api = paddle.pow
        self.public_python_api = paddle.pow
        self.prim_op_type = "prim"
        self.dtype = self.get_dtype()
        self.init_test_data()
        self.if_enable_cinn()
        self.inputs = {'X': self.x}
        self.attrs = {'factor': self.factor}

        self.outputs = {'Out': np.power(self.x, self.factor)}

    def get_dtype(self):
        return "float64"

    def test_check_output(self):
        if self.dtype == np.uint16:
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, check_pir=True)
        else:
            self.check_output(check_pir=True)

    def test_check_grad(self):
        if self.dtype == np.uint16:
            place = core.CUDAPlace(0)
            self.check_grad_with_place(
                place,
                ['X'],
                'Out',
                check_prim=True,
                check_pir=True,
            )
        else:
            self.check_grad(
                ['X'],
                'Out',
                check_prim=True,
                check_pir=True,
            )

    def init_test_data(self):
        if self.dtype == np.uint16:
            x = np.random.random((5, 1, 4, 5)).astype(np.float32)
            # x = np.array([4,5,6]).astype(np.float32)
            self.x = convert_float_to_uint16(x)
        else:
            self.x = np.random.random((5, 1, 4, 5)).astype(self.dtype)
            # self.x = np.array([4,5,6]).astype(self.dtype)
        self.factor = 2

    def if_enable_cinn(self):
        pass


if __name__ == '__main__':
    unittest.main()
