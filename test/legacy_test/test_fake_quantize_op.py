# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import itertools
import math
import unittest

import numpy as np
from op_test import OpTest

from paddle import _C_ops


def round_c_single_element(val):
    dtype = type(val)
    if val >= 0:
        return dtype(np.floor(val + 0.5))
    return dtype(np.ceil(val - 0.5))


# rounding to nearest ties away from zero
round_c = np.vectorize(round_c_single_element)


def get_compute_type(dtype):
    assert dtype in [np.float16, np.float32, np.float64]
    if dtype == np.float16:
        return np.float32
    return dtype


def fake_channel_wise_quantize_dequantize_abs_max_wrapper(
    x, bit_length=8, round_type=1, quant_axis=0
):
    return _C_ops.fake_channel_wise_quantize_dequantize_abs_max(
        x, bit_length, round_type, quant_axis
    )


def fake_quantize_dequantize_moving_average_abs_max_wrapper(
    x,
    in_scale,
    in_accum,
    in_state,
    moving_rate=0.9,
    bit_length=8,
    is_test=False,
    round_type=1,
):
    return _C_ops.fake_quantize_dequantize_moving_average_abs_max(
        x,
        in_scale,
        in_accum,
        in_state,
        moving_rate,
        bit_length,
        is_test,
        round_type,
    )


def fake_quantize_dequantize_abs_max_wrapper(x, bit_length=8, round_type=1):
    return _C_ops.fake_quantize_dequantize_abs_max(x, bit_length, round_type)


class TestFakeQuantizeAbsMaxOp(OpTest):
    def setUp(self):
        self.op_type = 'fake_quantize_abs_max'
        self.attrs = {'bit_length': 8}

    def _fake_quantize_abs_max(
        self, dtype, input_shape, distribution, round_type='TiesAwayFromZero'
    ):
        input_data = distribution(input_shape).astype(dtype)
        compute_type = get_compute_type(dtype)
        scale = np.max(np.abs(input_data)).flatten()
        bnt = (1 << (self.attrs['bit_length'] - 1)) - 1
        inv_scale = 1.0 / (scale + 1e-6) if scale < 1e-30 else 1.0 / scale
        if round_type == 'TiesToEven':
            round_out = np.round(
                input_data.astype(compute_type) * inv_scale * bnt
            )
            output_data = np.clip(round_out, -bnt - 1, bnt)
            self.attrs['round_type'] = 0
        else:
            output_data = round_c(
                input_data.astype(compute_type) * inv_scale * bnt
            )
            self.attrs['round_type'] = 1
        self.inputs = {'X': input_data}
        self.outputs = {'Out': output_data, 'OutScale': scale}
        self.dtype = dtype
        self.check_output(check_dygraph=False)

    def test_fake_quantize_abs_max(self):
        self._fake_quantize_abs_max(np.float32, (124, 240), np.random.random)

    def test_fake_quantize_abs_max_round1(self):
        self._fake_quantize_abs_max(
            np.float32, (124, 240), np.random.random, round_type='TiesToEven'
        )

    def test_fake_quantize_abs_max_float16(self):
        self._fake_quantize_abs_max(np.float16, (124, 240), np.random.random)

    def test_fake_quantize_abs_max_underflow(self):
        self._fake_quantize_abs_max(np.float32, (10, 10), np.zeros)

    def test_fake_quantize_abs_max_underflow2(self):
        self._fake_quantize_abs_max(
            np.float32, (10, 10), lambda shape: np.full(shape, 1e-40)
        )


class TestFakeChannelWiseQuantizeAbsMaxOp(OpTest):
    def setUp(self):
        self.op_type = 'fake_channel_wise_quantize_abs_max'
        self.attrs = {'bit_length': 8}

    def _fake_channel_wise_quantize_abs_max(
        self,
        dtype,
        input_shape,
        quant_axis,
        distribution,
        round_type='TiesToEven',
    ):
        assert quant_axis in [0, 1], 'quant_axis should be 0 or 1.'
        input_data = distribution(input_shape).astype(dtype)
        compute_type = get_compute_type(dtype)
        bnt = (1 << (self.attrs['bit_length'] - 1)) - 1
        compute_axis = tuple(
            i for i in range(len(input_shape)) if i != quant_axis
        )
        scale_broadcast = np.amax(input_data, axis=compute_axis, keepdims=True)
        if round_type == 'TiesToEven':
            round_out = np.round(
                input_data.astype(compute_type) / scale_broadcast * bnt
            )
            output_data = np.clip(round_out, -bnt - 1, bnt)
            self.attrs['round_type'] = 0
        else:
            output_data = round_c(
                bnt * input_data.astype(compute_type) / scale_broadcast
            )
            self.attrs['round_type'] = 1
        if quant_axis == 1:
            scale_broadcast = np.transpose(scale_broadcast, (1, *compute_axis))
        scale = scale_broadcast.reshape(input_shape[quant_axis], -1)[:, 0]
        self.inputs = {'X': input_data}
        self.outputs = {'Out': output_data, 'OutScale': scale}
        self.dtype = dtype
        self.attrs['quant_axis'] = quant_axis
        self.check_output(check_dygraph=False)

    def test_fake_channel_wise_quantize_abs_max(self):
        dtype_options = [np.float32, np.float16]
        input_shape_quant_axis_options = [
            [(20, 15, 6, 6), 0],
            [(20, 15, 6, 6), 1],
            [(30, 30), 0],
            [(30, 30), 1],
        ]
        round_type_options = ['TiesToEven', 'TiesAwayFromZero']
        for dtype, input_shape_quant_axis, round_type in itertools.product(
            dtype_options, input_shape_quant_axis_options, round_type_options
        ):
            input_shape, quant_axis = input_shape_quant_axis
            with self.subTest(
                dtype=dtype,
                input_shape=input_shape,
                quant_axis=quant_axis,
                round_type=round_type,
            ):
                self._fake_channel_wise_quantize_abs_max(
                    dtype, input_shape, quant_axis, np.random.random, round_type
                )


class TestFakeChannelWiseQuantizeDequantizeAbsMaxOp(OpTest):
    def setUp(self):
        self.op_type = 'fake_channel_wise_quantize_dequantize_abs_max'
        self.attrs = {'bit_length': 8}

    def _fake_channel_wise_quantize_dequantize_abs_max(
        self,
        dtype,
        input_shape,
        quant_axis,
        distribution,
        round_type='TiesToEven',
    ):
        assert quant_axis in [0, 1], 'quant_axis should be 0 or 1.'
        input_data = distribution(input_shape).astype(dtype)
        compute_type = get_compute_type(dtype)
        bnt = (1 << (self.attrs['bit_length'] - 1)) - 1
        output_data = input_data.copy().astype(compute_type)
        compute_axis = tuple(
            i for i in range(len(input_shape)) if i != quant_axis
        )
        scale_broadcast = np.amax(input_data, axis=compute_axis, keepdims=True)
        if round_type == 'TiesToEven':
            round_out = np.round(bnt * output_data / scale_broadcast)
            output_data = (
                np.clip(round_out, -bnt - 1, bnt) * scale_broadcast / bnt
            )
            self.attrs['round_type'] = 0
        else:
            output_data = (
                round_c(bnt * output_data / scale_broadcast)
                * scale_broadcast
                / bnt
            )
            self.attrs['round_type'] = 1
        if quant_axis == 1:
            scale_broadcast = np.transpose(scale_broadcast, (1, *compute_axis))
        scale = scale_broadcast.reshape(input_shape[quant_axis], -1)[:, 0]
        self.python_api = fake_channel_wise_quantize_dequantize_abs_max_wrapper
        self.inputs = {'X': input_data}
        self.outputs = {'Out': output_data, 'OutScale': scale}
        self.dtype = dtype
        self.attrs['quant_axis'] = quant_axis
        self.check_output(check_dygraph=False, check_pir=True)
        gradient = [np.ones(input_data.shape) / np.prod(input_data.shape)]
        self.check_grad(['X'], 'Out', user_defined_grads=gradient)

    def test_channel_wise_fake_quant_dequant_abs_max(self):
        input_shape_quant_axis_options = [
            [(3, 4, 64, 64), 0],
            [(15, 20, 5, 5), 1],
            [(30, 15), 0],
            [(30, 15), 1],
        ]
        round_type_options = ['TiesToEven', 'TiesAwayFromZero']
        for input_shape_quant_axis, round_type in itertools.product(
            input_shape_quant_axis_options, round_type_options
        ):
            input_shape, quant_axis = input_shape_quant_axis
            with self.subTest(
                input_shape=input_shape,
                quant_axis=quant_axis,
                round_type=round_type,
            ):
                self._fake_channel_wise_quantize_dequantize_abs_max(
                    np.float32,
                    input_shape,
                    quant_axis,
                    np.random.random,
                    round_type=round_type,
                )


class TestFakeQuantizeRangeAbsMaxOp(OpTest):
    def setUp(self):
        self.op_type = 'fake_quantize_range_abs_max'
        self.attrs = {'bit_length': 5, 'window_size': 1}

    def _fake_quantize_range_abs_max(
        self,
        dtype,
        input_shape,
        distribution,
        is_test=False,
        round_type='TiesToEven',
    ):
        input_data = distribution(input_shape).astype(dtype)
        compute_type = get_compute_type(dtype)
        bnt = (1 << (self.attrs['bit_length'] - 1)) - 1
        in_scale = np.zeros(1).astype(dtype)
        out_scale = np.zeros(self.attrs['window_size']).astype(dtype)
        out_scale[0] = np.max(np.abs(input_data))
        if is_test:
            out_scale[0] = in_scale[0] = out_scale[0] - 1.0
        if round_type == 'TiesToEven':
            round_out = np.round(
                input_data.astype(compute_type) / out_scale[0] * bnt
            )
            self.attrs['round_type'] = 0
            output_data = np.clip(round_out, -bnt - 1, bnt)
        else:
            if is_test:
                clip_data = np.clip(input_data, -in_scale, in_scale)
            else:
                clip_data = input_data
            output_data = round_c(
                clip_data.astype(compute_type) / out_scale[0] * bnt
            )
            self.attrs['round_type'] = 1
        self.inputs = {
            'X': input_data,
            'Iter': np.zeros(1).astype(np.int64),
            'InScale': in_scale,
        }
        self.outputs = {
            'Out': output_data,
            'OutScale': np.array([], dtype) if is_test else out_scale,
            'OutScales': np.array([], dtype) if is_test else out_scale,
        }
        self.dtype = dtype
        self.attrs['is_test'] = is_test
        self.check_output(check_dygraph=False)

    def test_fake_quantize_range_abs_max(self):
        dtype_options = [np.float16, np.float32]
        is_test_options = [False, True]
        round_type_options = ['TiesToEven', 'TiesAwayFromZero']
        for dtype, is_test, round_type in itertools.product(
            dtype_options, is_test_options, round_type_options
        ):
            self.attrs['bit_length'] = 8 if is_test else 5
            with self.subTest(
                dtype=dtype, is_test=is_test, round_type=round_type
            ):
                self._fake_quantize_range_abs_max(
                    dtype,
                    (8, 16, 6, 6),
                    lambda shape: (np.random.random(shape) - 0.4) * 10,
                    is_test=is_test,
                    round_type=round_type,
                )


class TestMovingAverageAbsMaxScaleOp(OpTest):
    def setUp(self):
        self.op_type = 'moving_average_abs_max_scale'
        self.attrs = {'moving_rate': 0.9, 'is_test': False}

    def _moving_average_abs_max_scale(self, dtype, input_shape, distribution):
        input_data = distribution(input_shape).astype(dtype)
        in_accum = np.ones(1).astype(dtype)
        in_state = np.ones(1).astype(dtype)
        out_accum = self.attrs['moving_rate'] * in_accum + np.max(
            np.abs(input_data)
        )
        out_state = self.attrs['moving_rate'] * in_state + 1.0
        out_scale = out_accum / out_state
        self.inputs = {
            'X': input_data,
            'InAccum': in_accum,
            'InState': in_state,
        }
        self.outputs = {
            'Out': input_data,
            'OutAccum': out_accum,
            'OutState': out_state,
            'OutScale': out_scale,
        }
        self.dtype = dtype
        self.check_output(check_dygraph=False)

    def test_moving_average_abs_max(self):
        self._moving_average_abs_max_scale(
            np.float32, (8, 16, 7, 7), np.random.random
        )


class TestFakeQuantizeMovingAverageAbsMaxOp(OpTest):
    def setUp(self):
        self.op_type = 'fake_quantize_moving_average_abs_max'
        self.attrs = {'bit_length': 5, 'moving_rate': 0.9, 'is_test': False}
        self.python_api = (
            fake_quantize_dequantize_moving_average_abs_max_wrapper
        )

    def _fake_quantize_moving_average_abs_max(
        self,
        dtype,
        input_shape,
        distribution,
        dequantize=False,
        with_gradient=False,
        round_type='TiesAwayFromZero',
    ):
        input_data = distribution(input_shape).astype(dtype)
        compute_type = get_compute_type(dtype)
        bnt = (1 << (self.attrs['bit_length'] - 1)) - 1
        in_accum = np.ones(1).astype(dtype)
        in_state = np.ones(1).astype(dtype)
        in_scale = np.array([0.001]).astype(dtype)
        out_accum = self.attrs['moving_rate'] * in_accum + np.max(
            np.abs(input_data)
        )
        out_state = self.attrs['moving_rate'] * in_state + 1.0
        out_scale = out_accum / out_state
        if round_type == 'TiesToEven':
            round_out = np.round(
                input_data.astype(compute_type) / out_scale * bnt
            )
            quant_data = np.clip(round_out, -bnt - 1, bnt)
            self.attrs['round_type'] = 0
        else:
            quant_data = round_c(
                input_data.astype(compute_type) / out_scale * bnt
            )
            self.attrs['round_type'] = 1
        if dequantize:
            output_data = (quant_data * out_scale / bnt).astype(dtype)
            self.op_type = 'fake_quantize_dequantize_moving_average_abs_max'
        else:
            output_data = quant_data.astype(dtype)
        self.inputs = {
            'X': input_data,
            'InScale': in_scale,
            'InAccum': in_accum,
            'InState': in_state,
        }
        self.outputs = {
            'Out': output_data,
            'OutAccum': out_accum,
            'OutState': out_state,
            'OutScale': out_scale,
        }
        self.dtype = dtype
        self.check_output(check_dygraph=False)
        if with_gradient:
            gradient = [np.ones(input_data.shape) / np.prod(input_data.shape)]
            self.check_grad(['X'], 'Out', user_defined_grads=gradient)

    def test_fake_quantize_moving_average_abs_max(self):
        self._fake_quantize_moving_average_abs_max(
            np.float32, (8, 16, 7, 7), np.random.random
        )

    def test_fake_quantize_moving_average_abs_max_float16(self):
        self._fake_quantize_moving_average_abs_max(
            np.float16, (8, 16, 7, 7), np.random.random
        )

    def test_fake_quantize_moving_average_abs_max_round1(self):
        self._fake_quantize_moving_average_abs_max(
            np.float32, (8, 16, 7, 7), np.random.random, round_type='TiesToEven'
        )

    def test_fake_quantize_dequantize_moving_average_abs_max(self):
        self._fake_quantize_moving_average_abs_max(
            np.float32,
            (8, 16, 7, 7),
            np.random.random,
            dequantize=True,
            with_gradient=True,
        )


class TestFakeQuantizeDequantizeAbsMaxOp(OpTest):
    def setUp(self):
        self.op_type = 'fake_quantize_dequantize_abs_max'
        self.attrs = {'bit_length': 8}
        self.python_api = fake_quantize_dequantize_abs_max_wrapper

    def _fake_quantize_dequantize_abs_max(
        self, dtype, input_shape, distribution, round_type='TiesAwayFromZero'
    ):
        input_data = distribution(input_shape).astype(dtype)
        scale = np.max(np.abs(input_data)).flatten().astype(dtype)
        bnt = (1 << (self.attrs['bit_length'] - 1)) - 1
        if round_type == 'TiesToEven':
            round_out = np.round(input_data / scale * bnt)
            output_data = np.clip(round_out, -bnt - 1, bnt) * scale / bnt
            self.attrs['round_type'] = 0
        else:
            output_data = round_c(input_data / scale * bnt) * scale / bnt
            self.attrs['round_type'] = 1
        self.inputs = {'X': input_data}
        self.outputs = {
            'Out': output_data,
            'OutScale': np.array(scale).astype(dtype),
        }
        self.dtype = dtype
        self.check_output(check_dygraph=False)
        gradient = [np.ones(input_data.shape) / np.prod(input_data.shape)]
        self.check_grad(['X'], 'Out', user_defined_grads=gradient)

    def test_fake_quantize_dequantize_abs_max(self):
        self._fake_quantize_dequantize_abs_max(
            np.float32, (124, 240), np.random.random
        )

    def test_fake_quantize_dequantize_abs_max_round1(self):
        self._fake_quantize_dequantize_abs_max(
            np.float32, (124, 240), np.random.random, round_type='TiesToEven'
        )


class TestChannelWiseFakeQuantizeDequantizeAbsMaxOp(OpTest):
    def setUp(self):
        self.op_type = 'fake_channel_wise_quantize_dequantize_abs_max'
        self.attrs = {'bit_length': 8}

    def _fake_channel_wise_quantize_dequantize_abs_max(
        self,
        dtype,
        input_shape,
        quant_axis,
        distribution,
        round_type='TiesToEven',
    ):
        assert quant_axis in [0, 1], 'quant_axis should be 0 or 1.'
        input_data = distribution(input_shape).astype(dtype)
        compute_type = get_compute_type(dtype)
        bnt = (1 << (self.attrs['bit_length'] - 1)) - 1
        output_data = input_data.copy().astype(compute_type)
        compute_axis = tuple(
            i for i in range(len(input_shape)) if i != quant_axis
        )
        scale_broadcast = np.amax(input_data, axis=compute_axis, keepdims=True)
        if round_type == 'TiesToEven':
            round_out = np.round(bnt * output_data / scale_broadcast)
            output_data = (
                np.clip(round_out, -bnt - 1, bnt) * scale_broadcast / bnt
            )
            self.attrs['round_type'] = 0
        else:
            output_data = (
                round_c(bnt * output_data / scale_broadcast)
                * scale_broadcast
                / bnt
            )
            self.attrs['round_type'] = 1
        if quant_axis == 1:
            scale_broadcast = np.transpose(scale_broadcast, (1, *compute_axis))
        scale = scale_broadcast.reshape(input_shape[quant_axis], -1)[:, 0]
        self.python_api = fake_channel_wise_quantize_dequantize_abs_max_wrapper
        self.inputs = {'X': input_data}
        self.outputs = {'Out': output_data, 'OutScale': scale}
        self.dtype = dtype
        self.attrs['quant_axis'] = quant_axis
        self.check_output(check_dygraph=False)
        gradient = [np.ones(input_data.shape) / np.prod(input_data.shape)]
        self.check_grad(['X'], 'Out', user_defined_grads=gradient)

    def test_channel_wise_fake_quant_dequant_abs_max(self):
        input_shape_quant_axis_options = [
            [(3, 4, 64, 64), 0],
            [(15, 20, 5, 5), 1],
            [(30, 15), 0],
            [(30, 15), 1],
        ]
        round_type_options = ['TiesToEven', 'TiesAwayFromZero']
        for input_shape_quant_axis, round_type in itertools.product(
            input_shape_quant_axis_options, round_type_options
        ):
            input_shape, quant_axis = input_shape_quant_axis
            with self.subTest(
                input_shape=input_shape,
                quant_axis=quant_axis,
                round_type=round_type,
            ):
                self._fake_channel_wise_quantize_dequantize_abs_max(
                    np.float32,
                    input_shape,
                    quant_axis,
                    np.random.random,
                    round_type=round_type,
                )


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
        yq, scale = channel_wise_quantize_max_abs(
            x, self.bit_length, self.quant_axis
        )
        scale = np.array(scale).astype(self.data_type)
        zero_point = np.zeros(scale.shape, dtype="int32")

        self.inputs = {'X': x, 'Scale': scale, 'ZeroPoint': zero_point}
        self.attrs = {
            'bit_length': self.bit_length,
            'quant_axis': self.quant_axis,
        }
        self.outputs = {'Y': yq}

    def test_check_output(self):
        self.check_output(check_dygraph=False)


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
        yq, scale = channel_wise_quantize_max_abs(
            x, self.bit_length, self.quant_axis
        )
        scale = np.array(scale).astype(self.data_type)
        zero_point = np.zeros(scale.shape, dtype="int32")

        self.inputs = {'X': x, 'Scale': scale, 'ZeroPoint': zero_point}
        self.attrs = {
            'bit_length': self.bit_length,
            'quant_axis': self.quant_axis,
            'is_test': self.is_test,
        }
        self.outputs = {'Y': yq, 'OutScale': scale}

    def test_check_output(self):
        self.check_output(check_dygraph=False)


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
        self.check_output(check_dygraph=False)


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
        self.attrs = {
            'bit_length': self.bit_length,
            'quant_axis': self.quant_axis,
            'moving_rate': 0.9,
            'is_test': self.is_test,
        }

        x = np.random.randn(31, 65).astype(self.data_type)
        scale = np.array([0.001]).astype(self.data_type)
        zero_point = np.zeros(scale.shape, dtype="int32")
        in_accum = np.ones(1).astype(self.data_type)
        in_state = np.ones(1).astype(self.data_type)
        out_accum = self.attrs['moving_rate'] * in_accum + np.max(np.abs(x))
        out_state = self.attrs['moving_rate'] * in_state + 1.0
        out_scale = out_accum / out_state

        round_out = np.round(x / out_scale * self.max_range)
        quant_data = np.clip(round_out, -self.max_range - 1, self.max_range)

        self.inputs = {
            'X': x,
            'Scale': scale,
            'ZeroPoint': zero_point,
            'InAccum': in_accum,
            'InState': in_state,
        }
        self.outputs = {
            'Y': quant_data,
            'OutScale': out_scale,
            'OutAccum': out_accum,
            'OutState': out_state,
        }

    def test_check_output(self):
        self.check_output(check_dygraph=False)


if __name__ == '__main__':
    unittest.main()
