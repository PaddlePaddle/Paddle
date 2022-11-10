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
import paddle.nn.functional as fn
from ..quanter import BaseQuanter
from ..factory import QuanterFactory, ObserverFactory
from ..observers import AbsmaxObserver

__all__ = ["QDropFakeQuanter"]


class QDropFakeQuanter(QuanterFactory):
    def __init__(
        sel,
        observer: ObserverFactory = AbsmaxObserver(),
        quant_bits=8,
        drop_prob=0.5,
        warmup=100,
    ):
        args = locals()
        args.pop("self")
        args.pop("__class__")
        super(FakeQuanterWithAbsMaxObserver, self).__init__(**args)

    def get_class(self):
        return QDropFakeQuanterLayer


class QDropFakeQuanterLayer(BaseQuanter):
    def __init__(
        self,
        layer: Layer,
        observer: ObserverFactory = None,
        quant_bits=8,
        drop_prob=0.5,
        warmup=100,
    ):
        super(QDropFakeQuanterLayer, self).__init__()
        self._layer = layer
        self._observer = observer.instance()
        self._quant_bits = quant_bits
        self._bnt = (1 << (self._quant_bits - 1)) - 1
        self._warmup = warmup
        self._current_batch_id = 0
        self._drop_prob = drop_prob

    def _quant_dequant(self, input, scales):
        scales = scales / self._bnt
        return paddle.round(input / scales) * scales

    def forward(self, input):
        self._current_batch_id += 1
        if self._current_batch_id < self._warmup:
            return self._observer(input)

        dequantized_tensor = self._quant_dequant(input, self._observer.scales())
        noise = input - dequantized_tensor
        noise = fn.dropout(noise, p=self._drop_prob)
        return input - noise

    def bit_length(self):
        return self._observer.bit_length()

    def quant_axis(self):
        return self._observer.quant_axis()

    def scales(self):
        return self._observer.scales()

    def zero_points(self):
        return self._observer.zero_points()
