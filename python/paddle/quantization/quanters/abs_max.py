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
from paddle import _legacy_C_ops
from paddle.fluid.framework import _varbase_creator
from paddle.framework import ParamAttr
from paddle.nn.initializer import Constant
from paddle.utils import unique_name

from ..base_quanter import BaseQuanter
from ..factory import quanter

__all__ = []


@quanter("FakeQuanterWithAbsMaxObserver")
class FakeQuanterWithAbsMaxObserverLayer(BaseQuanter):
    r"""
    FakeQuantMovingAverageAbsMax layer does the moving_average_abs_max quant and then dequant.
    Its computational formula is described as below:

    :math:`scale = (moving\_rate*accum+max(abs(x)))/(moving\_rate*state+1)`
    :math:`range = 2^{bit\_length - 1} - 1`
    :math:`Out = round(X / scale * range) * scale / range`

    Examples:
       .. code-block:: python

            from paddle.quantization import QuantConfig
            from paddle.quantization.quanters import FakeQuanterWithAbsMaxObserver
            quanter = FakeQuanterWithAbsMaxObserver(name="test", moving_rate=0.99)
            q_config = QuantConfig(activation=quanter, weight=quanter)
    """

    def __init__(
        self,
        layer,
        name=None,
        moving_rate=0.9,
        quant_bits=8,
        dtype='float32',
        reduce_type=None,
    ):
        super(FakeQuanterWithAbsMaxObserverLayer, self).__init__()
        self._moving_rate = moving_rate
        self._quant_bits = quant_bits
        self._reduce_type = reduce_type
        scale_prefix = (
            "{}.scale".format(name) if name else 'quant_dequant.scale'
        )
        scale_attr = ParamAttr(
            name=unique_name.generate(scale_prefix),
            initializer=Constant(0.001),
            trainable=False,
        )
        self._scale = self.create_parameter(
            shape=[1], attr=scale_attr, dtype=dtype
        )
        self._scale.stop_gradient = True

        state_prefix = (
            "{}.state".format(name) if name else 'quant_dequant.state'
        )
        state_attr = ParamAttr(
            name=unique_name.generate(state_prefix),
            initializer=Constant(1),
            trainable=False,
        )
        self._state = self.create_parameter(
            shape=[1], attr=state_attr, dtype=dtype
        )
        self._state.stop_gradient = True

        accum_prefix = (
            "{}.accum".format(name) if name else 'quant_dequant.accum'
        )
        accum_attr = ParamAttr(
            name=unique_name.generate(accum_prefix),
            initializer=Constant(1),
            trainable=False,
        )
        self._accum = self.create_parameter(
            shape=[1], attr=accum_attr, dtype=dtype
        )
        self._accum.stop_gradient = True

    def forward(self, input):
        attrs = (
            'moving_rate',
            self._moving_rate,
            'bit_length',
            self._quant_bits,
            'is_test',
            not self.training,
        )
        quant_out = _varbase_creator(
            type=input.type,
            name="{}.quantized.dequantized".format(input.name),
            shape=input.shape,
            dtype=input.dtype,
            persistable=False,
        )
        if self._reduce_type == "max":
            paddle.distributed.all_reduce(
                self._scale, op=paddle.distributed.ReduceOp.MAX
            )

        state = self._state if self.training else None
        accum = self._accum if self.training else None

        (
            out,
            _,
            _,
            _,
        ) = _legacy_C_ops.fake_quantize_dequantize_moving_average_abs_max(
            input,
            self._scale,
            accum,
            state,
            quant_out,
            self._scale,
            state,
            accum,
            *attrs
        )

        return out

    def bit_length(self):
        return self.bits

    def quant_axis(self):
        return None

    def scales(self):
        return self._scale

    def zero_points(self):
        return None
