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


class TestFakeChannelWiseDequantizeMaxAbsOpTwoScales(OpTest):
    def set_args(self):
        self.quant_bits = [8, 8]
        self.data_type = "float32"
        self.activation_scale = 0.7861

    def setUp(self):
        self.set_args()
        self.op_type = "fake_channel_wise_dequantize_max_abs"
        x = np.random.randn(4, 3, 64, 64).astype(self.data_type)
        yq, scales = channel_wise_quantize_max_abs(x, self.quant_bits[0], 1)
        ydq = channel_wise_dequantize_max_abs(yq, scales, self.quant_bits, 1,
                                              self.activation_scale)

        self.inputs = {
            'X': yq,
            'Scales': [("scales0", np.array(scales).astype(self.data_type)),
                       ("scales1", np.array(
                           [self.activation_scale]).astype(self.data_type))]
        }
        self.attrs = {'quant_bits': self.quant_bits}
        self.outputs = {'Out': ydq}

    def test_check_output(self):
        self.check_output()


class TestFakeChannelWiseDequantizeMaxAbsOpOneScale(OpTest):
    def set_args(self):
        self.quant_bits = [8]
        self.data_type = "float32"
        self.quant_axis = 0

    def setUp(self):
        self.set_args()
        self.op_type = "fake_channel_wise_dequantize_max_abs"
        x = np.random.randn(4, 3, 64, 64).astype(self.data_type)
        yq, scales = channel_wise_quantize_max_abs(x, self.quant_bits[0],
                                                   self.quant_axis)
        ydq = channel_wise_dequantize_max_abs(yq, scales, self.quant_bits,
                                              self.quant_axis)

        self.inputs = {
            'X': yq,
            'Scales': [("scales0", np.array(scales).astype(self.data_type))]
        }
        self.attrs = {
            'quant_bits': self.quant_bits,
            'quant_axis': self.quant_axis
        }
        self.outputs = {'Out': ydq}

    def test_check_output(self):
        self.check_output()


class TestFakeChannelWiseDequantizeMaxAbsOpOneScale1(
        TestFakeChannelWiseDequantizeMaxAbsOpOneScale):
    def set_args(self):
        self.quant_bits = [8]
        self.data_type = "float32"
        self.quant_axis = 1


class TestFakeDequantizeMaxAbsOp(OpTest):
    def set_args(self):
        self.num_bits = 8
        self.max_range = math.pow(2, self.num_bits - 1) - 1
        self.data_type = "float32"

    def setUp(self):
        self.set_args()
        self.op_type = "fake_dequantize_max_abs"
        x = np.random.randn(31, 65).astype(self.data_type)
        yq, scale = quantize_max_abs(x, self.max_range)
        ydq = dequantize_max_abs(yq, scale, self.max_range)

        self.inputs = {'X': yq, 'Scale': np.array(scale).astype(self.data_type)}
        self.attrs = {'max_range': self.max_range}
        self.outputs = {'Out': ydq}

    def test_check_output(self):
        self.check_output()


class TestFakeDequantizeMaxAbsOpDouble(TestFakeDequantizeMaxAbsOp):
    def set_args(self):
        self.num_bits = 8
        self.max_range = math.pow(2, self.num_bits - 1) - 1
        self.data_type = "float64"


class TestFakeDequantizeMaxAbsOp5Bits(TestFakeDequantizeMaxAbsOp):
    def set_args(self):
        self.num_bits = 5
        self.max_range = math.pow(2, self.num_bits - 1) - 1
        self.data_type = "float32"


if __name__ == "__main__":
    import paddle
    paddle.disable_static()
    unittest.main()
