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


def quantize_max_abs(x, num_bits):
    range = math.pow(2, num_bits) - 1
    scale = np.max(np.abs(x).flatten())
    y = np.round(x / scale * range)
    return y, scale


def dequantize_max_abs(x, num_bits, scale):
    range = math.pow(2, num_bits) - 1
    y = (scale / range) * x
    return y


class TestFakeDequantizeMaxAbsOp(OpTest):
    def set_args(self):
        self.num_bits = 8

    def setUp(self):
        self.set_args()
        self.op_type = "fake_dequantize_max_abs"
        x = np.random.randn(31, 65).astype("float32")
        yq, scale = quantize_max_abs(x, self.num_bits)
        print 'scale ', scale
        ydq = dequantize_max_abs(yq, self.num_bits, scale)

        self.inputs = {'X': yq}
        self.attrs = {'num_bits': self.num_bits, 'scale': float(scale)}
        self.outputs = {'Out': ydq}

    def test_check_output(self):
        self.check_output()


class TestFakeDequantizeMaxAbsOp5Bits(OpTest):
    def set_args(self):
        self.num_bits = 5


if __name__ == "__main__":
    unittest.main()
