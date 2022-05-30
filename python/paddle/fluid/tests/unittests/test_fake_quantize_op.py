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
import math
import numpy as np
import math
from op_test import OpTest
import paddle.fluid.core as core


# numpy.round has different behavior in comparision to c++ round function
# so we use round_c instead of numpy.round to align the output data
def round_c_single_element(x):
    dtype = type(x)
    if x >= 0:
        return dtype(np.floor(x + 0.5))
    else:
        return dtype(np.ceil(x - 0.5))


round_c = np.vectorize(round_c_single_element)


class TestFakeQuantizeOp(OpTest):
    def setUp(self):
        self.set_dtype()
        self.op_type = "fake_quantize_abs_max"
        self.attrs = {'bit_length': 8}
        self.inputs = {'X': np.random.random((124, 240)).astype(self.dtype), }
        scale = np.max(np.abs(self.inputs['X'])).astype(self.dtype)
        self.outputs = {
            'Out': round_c(self.inputs['X'] / scale * (
                (1 << (self.attrs['bit_length'] - 1)) - 1)),
            'OutScale': np.array(scale).astype(self.dtype),
        }

    def set_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output()


class TestFakeQuantizeOpFloat16(TestFakeQuantizeOp):
    def set_dtype(self):
        self.dtype = np.float16


class TestFakeQuantizeOp1(OpTest):
    def setUp(self):
        self.op_type = "fake_quantize_abs_max"
        self.attrs = {'bit_length': 8}
        self.inputs = {'X': np.zeros((10, 10)).astype("float32"), }
        scale = np.max(np.abs(self.inputs['X'])).astype("float32")
        inv_scale = 1.0 / (scale + 1e-6) if scale < 1e-30 else 1.0 / scale
        self.outputs = {
            'Out': np.round(self.inputs['X'] * inv_scale * (
                (1 << (self.attrs['bit_length'] - 1)) - 1)),
            'OutScale': np.array(scale).astype("float32"),
        }

    def test_check_output(self):
        self.check_output()


class TestFakeQuantizeOp2(OpTest):
    def setUp(self):
        self.op_type = "fake_quantize_abs_max"
        self.attrs = {'bit_length': 8}
        self.inputs = {'X': np.full((10, 10), 1e-40).astype("float32"), }
        scale = np.max(np.abs(self.inputs['X'])).astype("float32")
        inv_scale = 1.0 / (scale + 1e-6) if scale < 1e-30 else 1.0 / scale
        self.outputs = {
            'Out': np.round(self.inputs['X'] * inv_scale * (
                (1 << (self.attrs['bit_length'] - 1)) - 1)),
            'OutScale': np.array(scale).astype("float32"),
        }

    def test_check_output(self):
        self.check_output()


class TestFakeChannelWiseQuantizeOp(OpTest):
    def setUp(self):
        self.set_dtype()
        self.set_arg()
        assert self.quant_axis in [0, 1], "quant_axis should be 0 or 1."

        self.op_type = "fake_channel_wise_quantize_abs_max"
        self.attrs = {'bit_length': 8, 'quant_axis': self.quant_axis}

        scales = []
        outputs = self.inputs['X'].copy()
        bnt = (1 << (self.attrs['bit_length'] - 1)) - 1
        if self.quant_axis == 0:
            for i in range(self.inputs['X'].shape[0]):
                scale_v = np.max(np.abs(self.inputs['X'][i])).astype(self.dtype)
                scales.append(scale_v)
                outputs[i] = round_c(
                    self.dtype(bnt) * (self.dtype(1.0) / scale_v) * outputs[i])
        elif self.quant_axis == 1:
            for i in range(self.inputs['X'].shape[1]):
                scale_v = np.max(np.abs(self.inputs['X'][:, i])).astype(
                    self.dtype)
                scales.append(scale_v)
                outputs[:, i] = round_c(
                    self.dtype(bnt) * (self.dtype(1.0) / scale_v) *
                    outputs[:, i])

        self.outputs = {
            'Out': outputs,
            'OutScale': np.array(scales).astype(self.dtype),
        }

    def set_arg(self):
        self.quant_axis = 0
        self.inputs = {
            'X': np.random.random((20, 15, 6, 6)).astype(self.dtype),
        }

    def set_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output()


class TestFakeChannelWiseQuantizeOpFloat16(TestFakeChannelWiseQuantizeOp):
    def set_dtype(self):
        self.dtype = np.float16


class TestFakeChannelWiseQuantizeOp1(TestFakeChannelWiseQuantizeOp):
    def set_quant_axis(self):
        self.quant_axis = 1
        self.inputs = {
            'X': np.random.random((15, 20, 5, 5)).astype(self.dtype),
        }


class TestFakeChannelWiseQuantizeOp1Float16(TestFakeChannelWiseQuantizeOp1):
    def set_dtype(self):
        self.dtype = np.float16


class TestFakeChannelWiseQuantizeOp2(TestFakeChannelWiseQuantizeOp):
    def set_quant_axis(self):
        self.quant_axis = 0
        self.inputs = {'X': np.random.random((30, 15)).astype(self.dtype), }


class TestFakeChannelWiseQuantizeOp3(TestFakeChannelWiseQuantizeOp):
    def set_quant_axis(self):
        self.quant_axis = 1
        self.inputs = {'X': np.random.random((30, 15)).astype(self.dtype), }


class TestFakeQuantizeRangeAbsMaxOp(OpTest):
    def setUp(self):
        self.set_dtype()
        self.op_type = "fake_quantize_range_abs_max"
        self.attrs = {
            'bit_length': int(5),
            'window_size': int(1),
            'is_test': False
        }
        x = (np.random.random((8, 16, 7, 7)) - 0.5) * 10
        x = x.astype(self.dtype)
        self.inputs = {
            'X': x,
            'Iter': np.zeros(1).astype("int64"),
            'InScale': np.zeros(1).astype(self.dtype)
        }
        scale = np.max(np.abs(self.inputs['X'])).astype(self.dtype)

        out_scales = np.zeros(self.attrs['window_size']).astype(self.dtype)
        out_scales[0] = scale
        self.outputs = {
            'Out': round_c(
                self.dtype((1 << (self.attrs['bit_length'] - 1)) - 1) *
                (self.dtype(1.0) / scale) * self.inputs['X']),
            'OutScale': scale,
            'OutScales': out_scales,
        }

    def set_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output()


class TestFakeQuantizeRangeAbsMaxOpFloat16(TestFakeQuantizeRangeAbsMaxOp):
    def set_dtype(self):
        self.dtype = np.float16


class TestMovingAverageAbsMaxScaleOp(OpTest):
    def setUp(self):
        self.op_type = "moving_average_abs_max_scale"
        self.attrs = {'moving_rate': float(0.9), 'is_test': False}
        accum = np.zeros(1).astype("float32")
        accum[0] = 1
        state = np.zeros(1).astype("float32")
        state[0] = 1
        x = np.random.random((8, 16, 7, 7)).astype("float32")
        self.inputs = {
            'X': x,
            'InAccum': accum,
            'InState': state,
        }

        out = x
        out_accum = np.zeros(1).astype("float32")
        out_state = np.zeros(1).astype("float32")
        out_scale = np.zeros(1).astype("float32")
        out_accum[0] = self.attrs['moving_rate'] * accum[0] + np.max(
            np.abs(self.inputs['X'])).astype("float32")
        out_state[0] = self.attrs['moving_rate'] * state[0] + 1
        out_scale = out_accum / out_state
        self.outputs = {
            'Out': out,
            'OutAccum': out_accum,
            'OutState': out_state,
            'OutScale': out_scale,
        }

    def test_check_output(self):
        self.check_output()


class TestFakeQuantizeRangeAbsMaxOp2(OpTest):
    def setUp(self):
        self.set_dtype()
        self.op_type = "fake_quantize_range_abs_max"
        self.attrs = {
            'bit_length': int(8),
            'window_size': int(1),
            'is_test': True
        }
        x = (np.random.random((8, 16, 7, 7)) - 0.5) * 10
        x = x.astype(self.dtype)
        scale = np.array([np.max(np.abs(x)).astype(self.dtype) - 1.0])
        out_scales = np.zeros(self.attrs['window_size']).astype(self.dtype)
        out_scales[0] = scale.astype(self.dtype)
        self.inputs = {
            'X': x,
            'Iter': np.zeros(1).astype("int64"),
            'InScale': scale.astype(self.dtype)
        }
        xs = np.clip(x, -scale, scale).astype(self.dtype)
        qs = round_c(
            self.dtype(
                self.dtype((1 << (self.attrs['bit_length'] - 1)) - 1) * (
                    self.dtype(1.0) / scale) * xs))
        self.outputs = {
            'Out': qs,
            'OutScale': scale.astype(self.dtype),
            'OutScales': out_scales,
        }

    def set_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output(no_check_set=set(['OutScale', 'OutScales']))


class TestFakeQuantizeRangeAbsMaxOp2Float16(TestFakeQuantizeRangeAbsMaxOp2):
    def set_dtype(self):
        self.dtype = np.float16


class TestMovingOpBase(OpTest):
    def setUp(self):
        self.set_dtype()
        self.init_type()
        self.attrs = {
            'bit_length': int(5),
            'moving_rate': float(0.9),
            'is_test': False
        }
        accum = np.zeros(1).astype(self.dtype)
        accum[0] = 1
        state = np.zeros(1).astype(self.dtype)
        state[0] = self.dtype(1.0)
        scale = np.zeros(1).astype(self.dtype)
        scale[0] = 0.001
        self.inputs = {
            'X': np.random.random((8, 16, 7, 7)).astype(self.dtype),
            'InScale': scale,
            'InAccum': accum,
            'InState': state,
        }

        out_accum = np.zeros(1).astype(self.dtype)
        out_state = np.zeros(1).astype(self.dtype)
        out_scale = np.zeros(1).astype(self.dtype)
        out_accum[0] = self.dtype(self.attrs['moving_rate']) * self.dtype(accum[
            0]) + np.max(np.abs(self.inputs['X'])).astype(self.dtype)
        out_state[0] = self.dtype(self.attrs['moving_rate']) * self.dtype(state[
            0]) + self.dtype(1.0)
        out_scale = self.dtype(self.dtype(out_accum) / self.dtype(out_state))
        out_data = self.calc_output(out_scale)
        self.outputs = {
            'Out': out_data,
            'OutAccum': out_accum,
            'OutState': out_state,
            'OutScale': out_scale,
        }

    def set_dtype(self):
        self.dtype = np.float32

    def init_type(self):
        self.op_type = "fake_quantize_moving_average_abs_max"

    def calc_output(self, out_scale):
        return round_c(self.inputs['X'] / out_scale * (
            (1 << (self.attrs['bit_length'] - 1)) - 1))

    def test_check_output(self):
        self.check_output()


class TestMovingOpBaseFloat16(TestMovingOpBase):
    def set_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output(atol=1e-2)


class TestFakeQuantDequantMovingOp(TestMovingOpBase):
    def init_type(self):
        self.op_type = "fake_quantize_dequantize_moving_average_abs_max"

    def calc_output(self, out_scale):
        range_v = (1 << (self.attrs['bit_length'] - 1)) - 1
        return np.round(self.inputs['X'] / out_scale *
                        range_v) * out_scale / range_v

    def test_check_grad(self):
        x = self.inputs["X"]
        gradient = [np.ones(x.shape) / np.product(x.shape)]
        self.check_grad(["X"], "Out", user_defined_grads=gradient)


class TestFakeQuantDequantAbsOp(OpTest):
    def setUp(self):
        self.op_type = "fake_quantize_dequantize_abs_max"
        self.attrs = {'bit_length': 8}
        self.inputs = {'X': np.random.random((124, 240)).astype("float32"), }
        scale = np.max(np.abs(self.inputs['X'])).astype("float32")
        out_data = self.calc_output(scale)
        self.outputs = {
            'Out': out_data,
            'OutScale': np.array(scale).astype("float32"),
        }

    def calc_output(self, scale):
        range_v = (1 << (self.attrs['bit_length'] - 1)) - 1
        return np.round(self.inputs['X'] / scale * range_v) * scale / range_v

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        x = self.inputs["X"]
        gradient = [np.ones(x.shape) / np.product(x.shape)]
        self.check_grad(["X"], "Out", user_defined_grads=gradient)


class TestChannelWiseFakeQuantDequantOp(OpTest):
    def setUp(self):
        self.set_arg()
        assert self.quant_axis in [0, 1], "quant_axis should be 0 or 1."

        self.op_type = "fake_channel_wise_quantize_dequantize_abs_max"
        self.attrs = {'bit_length': 8, 'quant_axis': self.quant_axis}

        scales = []
        outputs = self.inputs['X'].copy()
        range_v = (1 << (self.attrs['bit_length'] - 1)) - 1
        if self.quant_axis == 0:
            for i in range(self.inputs['X'].shape[0]):
                scale_v = np.max(np.abs(self.inputs['X'][i])).astype("float32")
                scales.append(scale_v)
                outputs[i] = np.round(outputs[i] * range_v /
                                      scale_v) * scale_v / range_v
        elif self.quant_axis == 1:
            for i in range(self.inputs['X'].shape[1]):
                scale_v = np.max(np.abs(self.inputs['X'][:, i])).astype(
                    "float32")
                scales.append(scale_v)
                outputs[:, i] = np.round(outputs[:, i] * range_v /
                                         scale_v) * scale_v / range_v

        self.outputs = {
            'Out': outputs,
            'OutScale': np.array(scales).astype("float32"),
        }

    def set_arg(self):
        self.quant_axis = 0
        self.inputs = {
            'X': np.random.random((3, 4, 64, 64)).astype("float32"),
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        x = self.inputs["X"]
        gradient = [np.ones(x.shape) / np.product(x.shape)]
        self.check_grad(["X"], "Out", user_defined_grads=gradient)


class TestChannelWiseFakeQuantDequantOp1(TestChannelWiseFakeQuantDequantOp):
    def set_arg(self):
        self.quant_axis = 1
        self.inputs = {
            'X': np.random.random((15, 20, 5, 5)).astype("float32"),
        }


class TestChannelWiseFakeQuantDequantOp2(TestChannelWiseFakeQuantDequantOp):
    def set_arg(self):
        self.quant_axis = 0
        self.inputs = {'X': np.random.random((30, 15)).astype("float32"), }


class TestChannelWiseFakeQuantDequantOp3(TestChannelWiseFakeQuantDequantOp):
    def set_arg(self):
        self.quant_axis = 1
        self.inputs = {'X': np.random.random((30, 15)).astype("float32"), }


def quantize_max_abs(x, max_range):
    scale = np.max(np.abs(x).flatten())
    y = np.round(x / scale * max_range)
    return y, scale


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


class TestChannelWiseQuantizeOp(OpTest):
    def set_args(self):
        self.bit_length = 8
        self.data_type = "float32"
        self.quant_axis = 0

    def setUp(self):
        self.set_args()
        self.op_type = "quantize_linear"
        x = np.random.randn(4, 3, 64, 64).astype(self.data_type)
        yq, scale = channel_wise_quantize_max_abs(x, self.bit_length,
                                                  self.quant_axis)
        scale = np.array(scale).astype(self.data_type)
        zero_point = np.zeros(scale.shape, dtype="int32")

        self.inputs = {'X': x, 'Scale': scale, 'ZeroPoint': zero_point}
        self.attrs = {
            'bit_length': self.bit_length,
            'quant_axis': self.quant_axis
        }
        self.outputs = {'Y': yq}

    def test_check_output(self):
        self.check_output()


class TestChannelWiseQuantizeOp1(TestChannelWiseQuantizeOp):
    def set_args(self):
        self.bit_length = 8
        self.data_type = "float32"
        self.quant_axis = 1


class TestChannelWiseQuantizeOpTrain(OpTest):
    def set_args(self):
        self.bit_length = 8
        self.data_type = "float32"
        self.quant_axis = 0
        self.is_test = False

    def setUp(self):
        self.set_args()
        self.op_type = "quantize_linear"
        x = np.random.randn(4, 3, 64, 64).astype(self.data_type)
        yq, scale = channel_wise_quantize_max_abs(x, self.bit_length,
                                                  self.quant_axis)
        scale = np.array(scale).astype(self.data_type)
        zero_point = np.zeros(scale.shape, dtype="int32")

        self.inputs = {'X': x, 'Scale': scale, 'ZeroPoint': zero_point}
        self.attrs = {
            'bit_length': self.bit_length,
            'quant_axis': self.quant_axis,
            'is_test': self.is_test
        }
        self.outputs = {'Y': yq, 'OutScale': scale}

    def test_check_output(self):
        self.check_output()


class TestquantizeOp(OpTest):
    def set_args(self):
        self.bit_length = 8
        self.quant_axis = -1
        self.max_range = math.pow(2, self.bit_length - 1) - 1
        self.data_type = "float32"

    def setUp(self):
        self.set_args()
        self.op_type = "quantize_linear"
        x = np.random.randn(31, 65).astype(self.data_type)
        yq, scale = quantize_max_abs(x, self.max_range)
        scale = np.array(scale).astype(self.data_type)
        zero_point = np.zeros(scale.shape, dtype="int32")

        self.inputs = {'X': x, 'Scale': scale, 'ZeroPoint': zero_point}
        self.attrs = {
            'bit_length': self.bit_length,
            'quant_axis': self.quant_axis,
        }
        self.outputs = {'Y': yq}

    def test_check_output(self):
        self.check_output()


class TestquantizeOpTrain(TestquantizeOp):
    def set_args(self):
        self.bit_length = 8
        self.quant_axis = -1
        self.max_range = math.pow(2, self.bit_length - 1) - 1
        self.data_type = "float32"
        self.is_test = False

    def setUp(self):
        self.set_args()
        self.op_type = "quantize_linear"
        x = np.random.randn(31, 65).astype(self.data_type)
        yq, scale = quantize_max_abs(x, self.max_range)
        scale = np.array(scale).astype(self.data_type)
        zero_point = np.zeros(scale.shape, dtype="int32")

        self.inputs = {'X': x, 'Scale': scale, 'ZeroPoint': zero_point}
        self.attrs = {
            'bit_length': self.bit_length,
            'quant_axis': self.quant_axis,
            'is_test': self.is_test
        }
        self.outputs = {'Y': yq, 'OutScale': scale}

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
