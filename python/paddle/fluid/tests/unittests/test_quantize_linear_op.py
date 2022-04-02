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

from __future__ import print_function

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


def channel_wise_quantize_max_abs(x, quant_bit=8, quant_axis=0):
    assert quant_axis in [0, 1], "The quant_axis should be 0 or 1."
    scales = []
    y = x.copy()
    max_range = math.pow(2, quant_bit - 1) - 1
    if quant_axis == 0:
        for i in range(x.shape[0]):
            scale = np.max(np.abs(x[i])).astype("float32")
            scales.append(scale)
            y[i] = np.round(x[i] * max_range / scale)
    elif quant_axis == 1:
        for i in range(x.shape[1]):
            scale = np.max(np.abs(x[:, i])).astype("float32")
            scales.append(scale)
            y[:, i] = np.round(x[:, i] * max_range / scale)
    return y, scales


def channel_wise_dequantize_max_abs(x,
                                    scales,
                                    quant_bits,
                                    quant_axis,
                                    activation_scale=None):
    assert quant_axis in [0, 1], "The quant_axis should be 0 or 1."

    if isinstance(quant_bits, list):
        max_range = math.pow(2, quant_bits[0] - 1) - 1
    else:
        max_range = math.pow(2, quant_bits - 1) - 1
    y = x.copy()
    if quant_axis == 0:
        for i in range(x.shape[0]):
            y[i] = x[i] * scales[i] / max_range
    elif quant_axis == 1:
        for i in range(x.shape[1]):
            y[:, i] = x[:, i] * scales[i] / max_range

    if activation_scale is not None:
        y = y * activation_scale / (math.pow(2, quant_bits[1] - 1) - 1)
    return y


class TestChannelWiseDequantizeOp(OpTest):
    def set_args(self):
        self.bit_length = 8
        self.data_type = "float32"
        self.quant_axis = 0
        self.zero_point = 0.

    def setUp(self):
        self.set_args()
        self.op_type = "dequantize_linear"
        x = np.random.randn(4, 3, 64, 64).astype(self.data_type)
        yq, scale = channel_wise_quantize_max_abs(x, self.bit_length,
                                                  self.quant_axis)
        ydq = channel_wise_dequantize_max_abs(yq, scale, self.bit_length,
                                              self.quant_axis)

        self.inputs = {
            'X': yq,
            'Scale': np.array(scale).astype(self.data_type),
            'ZeroPoint': self.zero_point
        }
        self.attrs = {
            'bit_length': self.bit_length,
            'quant_axis': self.quant_axis
        }
        self.outputs = {'Y': ydq}

    def test_check_output(self):
        self.check_output()


class TestChannelWiseDequantizeOp1(TestChannelWiseDequantizeOp):
    def set_args(self):
        self.bit_length = 8
        self.data_type = "float32"
        self.quant_axis = 1
        self.zero_point = 0.


class TestDequantizeOp(OpTest):
    def set_args(self):
        self.bit_length = 8
        self.quant_axis = -1
        self.max_range = math.pow(2, self.bit_length - 1) - 1
        self.data_type = "float32"
        self.zero_point = 0.

    def setUp(self):
        self.set_args()
        self.op_type = "dequantize_linear"
        x = np.random.randn(31, 65).astype(self.data_type)
        yq, scale = quantize_max_abs(x, self.max_range)
        ydq = dequantize_max_abs(yq, scale, self.max_range)

        self.inputs = {
            'X': yq,
            'Scale': np.array(scale).astype(self.data_type),
            'ZeroPoint': self.zero_point
        }
        self.attrs = {
            'bit_length': self.bit_length,
            'quant_axis': self.quant_axis
        }
        self.outputs = {'Y': ydq}

    def test_check_output(self):
        self.check_output()


class TestDequantizeOpDouble(TestDequantizeOp):
    def set_args(self):
        self.bit_length = 8
        self.max_range = math.pow(2, self.bit_length - 1) - 1
        self.data_type = "float64"
        self.zero_point = 0.
        self.quant_axis = -1


class TestFakeDequantizeMaxAbsOp5Bits(TestDequantizeOp):
    def set_args(self):
        self.bit_length = 5
        self.max_range = math.pow(2, self.bit_length - 1) - 1
        self.data_type = "float32"
        self.zero_point = 0.
        self.quant_axis = -1


class TestChannelWisequantizeOp(OpTest):
    def set_args(self):
        self.bit_length = 8
        self.data_type = "float32"
        self.quant_axis = 0
        self.zero_point = 0.

    def setUp(self):
        self.set_args()
        self.op_type = "quantize_linear"
        x = np.random.randn(4, 3, 64, 64).astype(self.data_type)
        yq, scale = channel_wise_quantize_max_abs(x, self.bit_length,
                                                  self.quant_axis)

        self.inputs = {
            'X': x,
            'Scale': np.array(scale).astype(self.data_type),
            'ZeroPoint': self.zero_point
        }
        self.attrs = {
            'bit_length': self.bit_length,
            'quant_axis': self.quant_axis
        }
        self.outputs = {'Y': yq}

    def test_check_output(self):
        self.check_output()


class TestChannelWisequantizeOp1(TestChannelWisequantizeOp):
    def set_args(self):
        self.bit_length = 8
        self.data_type = "float32"
        self.quant_axis = 1
        self.zero_point = 0.


class TestquantizeOp(OpTest):
    def set_args(self):
        self.bit_length = 8
        self.quant_axis = -1
        self.max_range = math.pow(2, self.bit_length - 1) - 1
        self.data_type = "float32"
        self.zero_point = 0.

    def setUp(self):
        self.set_args()
        self.op_type = "dequantize_linear"
        x = np.random.randn(31, 65).astype(self.data_type)
        yq, scale = quantize_max_abs(x, self.max_range)

        self.inputs = {
            'X': x,
            'Scale': np.array(scale).astype(self.data_type),
            'ZeroPoint': self.zero_point
        }
        self.attrs = {
            'bit_length': self.bit_length,
            'quant_axis': self.quant_axis
        }
        self.outputs = {'Y': yq}

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
