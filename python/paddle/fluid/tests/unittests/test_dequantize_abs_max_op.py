#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import math
from op_test import OpTest


def quantize_max_abs(x, max_range):
    scale = np.max(np.abs(x).flatten())
    y = np.round(x / scale * max_range)
    return y, scale


def dequantize_max_abs(x, scale, max_range):
    y = (scale / max_range) * x
    return y


class TestDequantizeMaxAbsOp(OpTest):

    def set_args(self):
        self.num_bits = 8
        self.max_range = math.pow(2, self.num_bits - 1) - 1
        self.data_type = "int8"

    def setUp(self):
        self.set_args()
        self.op_type = "dequantize_abs_max"
        x = np.random.randn(31, 65).astype(self.data_type)
        yq, scale = quantize_max_abs(x, self.max_range)
        ydq = dequantize_max_abs(yq, scale, self.max_range)

        self.inputs = {
            'X': np.array(yq).astype(self.data_type),
            'Scale': np.array(scale).astype('float32')
        }
        self.attrs = {'max_range': self.max_range}
        self.outputs = {'Out': ydq}

    def test_check_output(self):
        self.check_output()


class TestDequantizeMaxAbsOp5Bits(TestDequantizeMaxAbsOp):

    def set_args(self):
        self.num_bits = 5
        self.max_range = math.pow(2, self.num_bits - 1) - 1
        self.data_type = "int8"


class TestDequantizeMaxAbsOpInt16(TestDequantizeMaxAbsOp):

    def set_args(self):
        self.num_bits = 16
        self.max_range = math.pow(2, self.num_bits - 1) - 1
        self.data_type = "int16"


if __name__ == "__main__":
    unittest.main()
