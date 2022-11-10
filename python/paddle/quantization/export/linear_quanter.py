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

from paddle.nn import Layer
from paddle import _C_ops
from ..observer import BaseObserver
from ..quanter import BaseQuanter

__all__ = ["LinearQuanter", "LinearDequanter", "LinearQuanterDequanter"]


class LinearQuanterDequanter(Layer):
    def __init__(self, quanter, dequanter):
        super(LinearQuanterDequanter, self).__init__()
        self._quanter = quanter
        self._dequanter = dequanter

    def forward(self, input):
        return self._dequanter(self._quanter(input))

    @staticmethod
    def from_observer(observer: BaseObserver):
        return LinearQuanterDequanter(
            LinearQuanter.from_observer(observer),
            LinearDequanter.from_observer(observer),
        )

    @staticmethod
    def from_quanter(quanter: BaseQuanter):
        return LinearQuanterDequanter(
            LinearQuanter.from_quanter(quanter),
            LinearDequanter.from_quanter(quanter),
        )


class LinearQuanter(Layer):
    def __init__(self, scales, zero_point=None, quant_axis=None, bit_length=8):
        super(LinearQuanter, self).__init__()
        self._scales = scales
        self._zero_point = zero_point
        self._quant_axis = None
        self._bit_length = bit_length

    def forward(self, input):
        return _C_ops.quant_linear(
            input,
            self._scales,
            self._zero_point,
            "quant_axis",
            self._quant_axis,
            "bit_length",
            self._bit_length,
        )

    @staticmethod
    def from_observer(observer: BaseObserver):
        return LinearQuanter(
            observer.scales(),
            zero_point=observer.zero_points(),
            quant_axis=observer.quant_axis(),
            bit_length=observer.bit_length(),
        )

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
        self._scales = scales
        self._zero_point = zero_point
        self._quant_axis = None
        self._bit_length = bit_length

    def forward(self, input):
        return _C_ops.dequant_linear(
            input,
            self._scales,
            self._zero_point,
            "quant_axis",
            self._quant_axis,
            "bit_length",
            self._bit_length,
        )

    @staticmethod
    def from_observer(observer: BaseObserver):
        return LinearDequanter(
            observer.scales(),
            zero_point=observer.zero_points(),
            quant_axis=observer.quant_axis(),
            bit_length=observer.bit_length(),
        )

    @staticmethod
    def from_quanter(quanter: BaseQuanter):
        return LinearDequanter(
            quanter.scales(),
            zero_point=quanter.zero_points(),
            quant_axis=quanter.quant_axis(),
            bit_length=quanter.bit_length(),
        )
