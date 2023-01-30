"""Define some layers used to export quantization model."""
# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle import _legacy_C_ops as _C_ops
from paddle.fluid.framework import in_dygraph_mode
from paddle.nn import Layer

from .base_quanter import BaseQuanter


class LinearQuanterDequanter(Layer):
    def __init__(self, quanter, dequanter):
        super(LinearQuanterDequanter, self).__init__()
        self._quanter = quanter
        self._dequanter = dequanter

    def forward(self, input):
        return self._dequanter(self._quanter(input))

    @staticmethod
    def from_quanter(quanter: BaseQuanter):
        return LinearQuanterDequanter(
            LinearQuanter.from_quanter(quanter),
            LinearDequanter.from_quanter(quanter),
        )


class LinearQuanter(Layer):
    def __init__(self, scales, zero_point=None, quant_axis=None, bit_length=8):
        super(LinearQuanter, self).__init__()
        self._scales = paddle.to_tensor(scales)
        self._zero_point = (
            paddle.zeros([1], dtype="float32")
            if zero_point is None
            else paddle.to_tensor(zero_point)
        )
        self._quant_axis = quant_axis
        self._bit_length = bit_length

    def forward(self, input):
        if in_dygraph_mode():
            return _C_ops.quantize_linear(
                input,
                self._scales,
                self._zero_point,
                "quant_axis",
                self._quant_axis,
                "bit_length",
                self._bit_length,
            )
        else:
            out = self._helper.create_variable_for_type_inference(input.dtype)
            self._helper.append_op(
                type='quantize_linear',
                inputs={
                    'X': input,
                    'Scale': self._scales,
                    'ZeroPoint': self._zero_point,
                },
                outputs={'Y': out},
                attrs={
                    'quant_axis': self._quant_axis,
                    'bit_length': self._bit_length,
                },
            )
            return out

    @staticmethod
    def from_quanter(quanter: BaseQuanter):

        return LinearQuanter(
            quanter.scales(),
            zero_point=quanter.zero_points(),
            quant_axis=quanter.quant_axis(),
            bit_length=quanter.bit_length(),
        )


class LinearDequanter(Layer):
    def __init__(self, scales, zero_point=None, quant_axis=None, bit_length=8):
        super(LinearDequanter, self).__init__()
        self._scales = paddle.to_tensor(scales)
        self._zero_point = (
            paddle.zeros([1], dtype="float32")
            if zero_point is None
            else paddle.to_tensor(zero_point)
        )
        self._quant_axis = quant_axis
        self._bit_length = bit_length

    def forward(self, input):
        if in_dygraph_mode():
            return _C_ops.dequantize_linear(
                input,
                self._scales,
                self._zero_point,
                "quant_axis",
                self._quant_axis,
                "bit_length",
                self._bit_length,
            )
        else:
            out = self._helper.create_variable_for_type_inference(input.dtype)
            self._helper.append_op(
                type='dequantize_linear',
                inputs={
                    'X': input,
                    'Scale': self._scales,
                    'ZeroPoint': self._zero_point,
                },
                outputs={'Y': out},
                attrs={
                    'quant_axis': self._quant_axis,
                    'bit_length': self._bit_length,
                },
            )
            return out

    @staticmethod
    def from_quanter(quanter: BaseQuanter):
        return LinearDequanter(
            quanter.scales(),
            zero_point=quanter.zero_points(),
            quant_axis=quanter.quant_axis(),
            bit_length=quanter.bit_length(),
        )
