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

import paddle
from paddle.nn import Layer
from paddle.framework import ParamAttr
from paddle.utils import unique_name
from paddle.nn.initializer import Constant
import paddle.nn.functional as fn
from .quanter import BaseQuanter
from ..factory import QuanterFactory, ObserverFactory
from ..observers import AbsmaxObserver

__all__ = ["SoftRoundingFakeQuanter"]

DEFAULT_GAMMA = -0.1
DEFAULT_ZETA = 1.1


class SoftRoundingFakeQuanter(QuanterFactory):
    def __init__(
        sel,
        observer: ObserverFactory = AbsmaxObserver(),
        quant_axis_bits=8,
        gamma=DEFAULT_GAMMA,
        zeta=DEFAULT_ZETA,
        warmup=100,
    ):
        args = locals()
        args.pop("self")
        args.pop("__class__")
        super(FakeQuanterWithAbsMaxObserver, self).__init__(**args)

    def get_class(self):
        return SoftRoundingFakeQuanterLayer


class SoftRoundingFakeQuanterLayer(BaseQuanter):
    def __init__(
        self,
        layer: Layer,
        observer: ObserverFactory = None,
        quant_bits=8,
        gamma=DEFAULT_GAMMA,
        zeta=DEFAULT_ZETA,
        warmup=100,
    ):
        super(SoftRoundingFakeQuanterLayer, self).__init__()
        self._layer = layer
        self._alpha = None
        self._observer = observer.instance()
        self._quant_bits = quant_bits
        self._bnt = (1 << (self._quant_bits - 1)) - 1
        self._gamma = gamma
        self._zeta = zeta
        self._warmup = warmup
        self._current_batch_id = 0

    def create_alpha(self, input):
        alpha_prefix = "_".join([self._layer.full_name, "_alpha"])
        # TODO: Init the alpha by quantization scales and weights
        alpha_attr = ParamAttr(
            name=unique_name.generate(alpha_prefix),
            initializer=Constant(0.001),
            trainable=True,
        )
        return self.create_parameter(
            shape=input.shape, attr=alpha_attr, dtype=input.dtype
        )

    def _quant(self, x, scales):
        s = scales / self._bnt
        quant_x = paddle.floor(x / s)
        return quant_x

    def _dequant(self, x, scales):
        s = scales / self._bnt
        dequant_x = s * x
        return dequant_x

    def forward(self, input):
        self._current_batch_id += 1
        if self._current_batch_id < self._warmup:
            return self._observer(input)

        if self._alpha is None:
            self._alpha = self.create_alpha(input)

        h_v = paddle.clip(
            fn.sigmoid(self._alpha) * (self._zeta - self._gamma) + self._gamma,
            0,
            1,
        )

        quantized_input = self._quant(input, self._observer.scales())
        clip_input = paddle.clip(quantized_input + h_v, -self._bnt, self._bnt)
        dequant_input = self._dequant(clip_input, self._observer.scales())

        return dequant_input

    def bit_length(self):
        return self._observer.bit_length()

    def quant_axis(self):
        return self._observer.quant_axis()

    def scales(self):
        return self._observer.scales()

    def zero_points(self):
        return self._observer.zero_points()
