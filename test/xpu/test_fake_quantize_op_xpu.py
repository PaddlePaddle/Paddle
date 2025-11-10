#   Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test import convert_float_to_uint16
from op_test_xpu import XPUOpTest

import paddle
from paddle import _C_ops
from paddle.base import core


def round_c_single_element(val):
    dtype = type(val)
    if val >= 0:
        return dtype(np.floor(val + 0.5))
    return dtype(np.ceil(val - 0.5))


# rounding to nearest ties away from zero
round_c = np.vectorize(round_c_single_element)


def fake_quantize_dequantize_abs_max_wrapper(x, bit_length=8, round_type=1):
    return _C_ops.fake_quantize_dequantize_abs_max(x, bit_length, round_type)


class XPUTestFakeQuantizeDequantizeAbsMaxOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'fake_quantize_dequantize_abs_max'
        self.use_dynamic_create_class = False

    class TestFakeQuantizeDequantizeAbsMaxOp(XPUOpTest):
        def setUp(self):
            self.init_dtype()
            self.set_xpu()
            self.op_type = "fake_quantize_dequantize_abs_max"
            self.python_api = fake_quantize_dequantize_abs_max_wrapper
            self.place = paddle.XPUPlace(0)
            self.inputs = {}
            self.init_distribution()
            self.init_shape()
            self.init_attrs()
            self.init_data()

            scale = np.max(np.abs(self.input_data)).flatten().astype(self.dtype)
            bnt = (1 << (self.attrs['bit_length'] - 1)) - 1
            if self.round_type == 'TiesToEven':
                round_out = np.round(self.input_data / scale * bnt)
                output_data = np.clip(round_out, -bnt - 1, bnt) * scale / bnt
                self.attrs['round_type'] = 0
            else:
                output_data = (
                    round_c(self.input_data / scale * bnt) * scale / bnt
                )
                self.attrs['round_type'] = 1
            self.inputs = {'X': self.input_data}
            self.outputs = {
                'Out': output_data.astype(self.dtype),
                'OutScale': np.array(scale).astype(self.dtype),
            }

        def set_xpu(self):
            self.__class__.use_xpu = True
            self.__class__.no_need_check_grad = False
            self.__class__.op_type = self.dtype

        def init_distribution(self):
            self.distribution = np.random.random

        def init_shape(self):
            self.shape = (124, 240)

        def init_attrs(self):
            self.round_type = 'TiesAwayFromZero'
            self.attrs = {'bit_length': 8}

        def init_data(self):
            if self.dtype == np.uint16:
                self.input_data = self.distribution(self.shape).astype(
                    'float32'
                )
                self.input_data = convert_float_to_uint16(self.input_data)
            else:
                self.input_data = self.distribution(self.shape).astype(
                    self.dtype
                )

        def init_dtype(self):
            self.dtype = self.in_type

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            if core.is_compiled_with_xpu():
                gradient = [
                    np.ones(self.input_data.shape)
                    / np.prod(self.input_data.shape)
                ]
                self.check_grad(['X'], 'Out', user_defined_grads=gradient)

    # TODO support round_type == 0
    # class TestFakeQuantizeDequantizeAbsMaxOp1(TestFakeQuantizeDequantizeAbsMaxOp):
    #     def init_param(self):
    #         self.round_type = 'TiesToEven'
    #         self.bit_length = 8


support_types = get_xpu_op_support_types('fake_quantize_dequantize_abs_max')
for stype in support_types:
    create_test_class(globals(), XPUTestFakeQuantizeDequantizeAbsMaxOp, stype)


def get_compute_type(dtype):
    assert dtype in [
        np.float16,
        np.float32,
        np.uint16,
    ]  # uint16 is for bfloat16
    if dtype == np.float16 or dtype == np.uint16:
        return np.float32
    return dtype


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


class XPUTestFakeQuantizeDequantizeMovingAverageAbsMaxOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'fake_quantize_dequantize_moving_average_abs_max'
        self.use_dynamic_create_class = False

    class TestFakeQuantizeDequantizeMovingAverageAbsMaxOp(XPUOpTest):
        def setUp(self):
            self.init_dtype()
            self.set_xpu()
            self.op_type = "fake_quantize_dequantize_moving_average_abs_max"
            self.python_api = (
                fake_quantize_dequantize_moving_average_abs_max_wrapper
            )
            self.place = paddle.XPUPlace(0)
            self.inputs = {}
            self.init_distribution()
            self.init_shape()
            self.init_attrs()
            self.init_data()

            compute_type = get_compute_type(self.dtype)
            bnt = (1 << (self.attrs['bit_length'] - 1)) - 1
            in_accum = np.ones(1).astype(self.dtype)
            in_state = np.ones(1).astype(self.dtype)
            in_scale = np.array([0.001]).astype(self.dtype)
            out_accum = self.attrs['moving_rate'] * in_accum + np.max(
                np.abs(self.input_data)
            )
            out_state = self.attrs['moving_rate'] * in_state + 1.0
            out_scale = out_accum / out_state
            if self.round_type == 'TiesToEven':
                round_out = np.round(
                    self.input_data.astype(compute_type) / out_scale * bnt
                )
                quant_data = np.clip(round_out, -bnt - 1, bnt)
                self.attrs['round_type'] = 0
            else:
                quant_data = round_c(
                    self.input_data.astype(compute_type) / out_scale * bnt
                )
                self.attrs['round_type'] = 1
            output_data = (quant_data * out_scale / bnt).astype(self.dtype)
            self.inputs = {
                'X': self.input_data,
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

        def set_xpu(self):
            self.__class__.use_xpu = True
            self.__class__.no_need_check_grad = False
            self.__class__.op_type = self.dtype

        def init_distribution(self):
            self.distribution = np.random.random

        def init_shape(self):
            self.shape = (8, 16, 7, 7)

        def init_attrs(self):
            self.round_type = 'TiesAwayFromZero'
            self.attrs = {
                'bit_length': 8,
                'moving_rate': 0.9,
                'is_test': False,
            }

        def init_data(self):
            if self.dtype == np.uint16:
                self.input_data = self.distribution(self.shape).astype(
                    'float32'
                )
                self.input_data = convert_float_to_uint16(self.input_data)
            else:
                self.input_data = self.distribution(self.shape).astype(
                    self.dtype
                )

        def init_dtype(self):
            self.dtype = self.in_type

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            if core.is_compiled_with_xpu():
                gradient = [
                    np.ones(self.input_data.shape)
                    / np.prod(self.input_data.shape)
                ]
                self.check_grad(['X'], 'Out', user_defined_grads=gradient)

    # TODO support round_type == 0
    # class TestFakeQuantizeDequantizeMovingAverageAbsMaxOp1(TestFakeQuantizeDequantizeMovingAverageAbsMaxOp):
    #     def init_param(self):
    #         self.round_type = 'TiesToEven'
    #         self.bit_length = 8


support_types = get_xpu_op_support_types(
    'fake_quantize_dequantize_moving_average_abs_max'
)
for stype in support_types:
    create_test_class(
        globals(), XPUTestFakeQuantizeDequantizeMovingAverageAbsMaxOp, stype
    )


class XPUTestMovingAverageAbsMaxScaleOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'moving_average_abs_max_scale'
        self.use_dynamic_create_class = False

    class TestMovingAverageAbsMaxScaleOp(XPUOpTest):
        def setUp(self):
            self.init_dtype()
            self.set_xpu()
            self.op_type = "moving_average_abs_max_scale"
            self.place = paddle.XPUPlace(0)
            self.inputs = {}
            self.init_distribution()
            self.init_shape()
            self.init_attrs()
            self.init_data()

            in_accum = np.ones(1).astype(self.dtype)
            in_state = np.ones(1).astype(self.dtype)
            out_accum = self.attrs['moving_rate'] * in_accum + np.max(
                np.abs(self.input_data)
            )
            out_state = self.attrs['moving_rate'] * in_state + 1.0
            out_scale = out_accum / out_state
            self.inputs = {
                'X': self.input_data,
                'InAccum': in_accum,
                'InState': in_state,
            }
            self.outputs = {
                'Out': self.input_data,
                'OutAccum': out_accum,
                'OutState': out_state,
                'OutScale': out_scale,
            }

        def set_xpu(self):
            self.__class__.use_xpu = True
            self.__class__.no_need_check_grad = False
            self.__class__.op_type = self.dtype

        def init_distribution(self):
            self.distribution = np.random.random

        def init_shape(self):
            self.shape = (8, 16, 7, 7)

        def init_attrs(self):
            self.attrs = {'moving_rate': 0.9, 'is_test': False}

        def init_data(self):
            if self.dtype == np.uint16:
                self.input_data = self.distribution(self.shape).astype(
                    'float32'
                )
                self.input_data = convert_float_to_uint16(self.input_data)
            else:
                self.input_data = self.distribution(self.shape).astype(
                    self.dtype
                )

        def init_dtype(self):
            self.dtype = self.in_type

        def test_check_output(self):
            self.check_output_with_place(self.place)


support_types = get_xpu_op_support_types('moving_average_abs_max_scale')
for stype in support_types:
    create_test_class(globals(), XPUTestMovingAverageAbsMaxScaleOp, stype)

if __name__ == '__main__':
    unittest.main()
