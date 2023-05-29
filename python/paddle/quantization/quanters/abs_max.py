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
from paddle.fluid.data_feeder import check_variable_and_dtype
from paddle.fluid.framework import _create_tensor
from paddle.framework import ParamAttr, core
from paddle.nn.initializer import Constant
from paddle.utils import unique_name

from ..base_quanter import BaseQuanter
from ..factory import QuanterFactory


class FakeQuanterWithAbsMaxObserver(QuanterFactory):
    r"""
    Compute quantization parameters and simulate quantization.

    It collects maximum absolute values of target tensor with moving average.
    The average value will be used as quantization scale to quantize and
    dequantize the tensor.

    And it is symmetric uniform quantization which means the zero point is always 0.

    The computational formula of moving average is described as below:

    .. math::
            state = rate * state + 1
            accum = rate * accum + max(abs(x))
            scale = accum / state

    Where:

    - :math:`x` is the input tensor.
    - :math:`state` and :math:`accum` are zero-initialized accumulators.
    - :math:`rate` is moving average rate.
    - :math:`scale` is quantization scale

    And the computational formula of simulate quantization is:

    .. math::
            range = 2^{bit\_length - 1} - 1
            out = round(x / scale * range) * scale / range

    Where:

    - :math:`{bit\_length}` is the length of bits.
    - :math:`x` is the input tensor and :math:`out` is the output of simulate quantization.

    Args:
        moving_rate(float, optional): The rate of moving average.
        bit_length(int, optional): Number of bits to represent an quantized integer in binary.
        dtype(str, optional): The data type of input tensor.
        name (str, optional): This parameter is used by developers to print debugging information. \
            For details, please refer to :ref:`api_guide_Name`. Default is None.

    Examples:
       .. code-block:: python

            from paddle.quantization import QuantConfig
            from paddle.quantization.quanters import FakeQuanterWithAbsMaxObserver
            quanter = FakeQuanterWithAbsMaxObserver(moving_rate=0.99)
            q_config = QuantConfig(activation=quanter, weight=quanter)
    """

    def __init__(
        self,
        moving_rate=0.9,
        bit_length=8,
        dtype='float32',
        name=None,
    ):
        super().__init__(
            name=name,
            moving_rate=moving_rate,
            bit_length=bit_length,
            dtype=dtype,
        )

    def _get_class(self):
        return FakeQuanterWithAbsMaxObserverLayer


class FakeQuanterWithAbsMaxObserverLayer(BaseQuanter):
    def __init__(
        self,
        layer,
        name=None,
        moving_rate=0.9,
        bit_length=8,
        dtype='float32',
    ):
        super().__init__()
        self._moving_rate = moving_rate
        self._bit_length = bit_length
        scale_prefix = f"{name}.scale" if name else 'quant_dequant.scale'
        scale_attr = ParamAttr(
            name=unique_name.generate(scale_prefix),
            initializer=Constant(0.001),
            trainable=False,
        )
        self._scale = self.create_parameter(
            shape=[1], attr=scale_attr, dtype=dtype
        )
        self._scale.stop_gradient = True

        state_prefix = f"{name}.state" if name else 'quant_dequant.state'
        state_attr = ParamAttr(
            name=unique_name.generate(state_prefix),
            initializer=Constant(1),
            trainable=False,
        )
        self._state = self.create_parameter(
            shape=[1], attr=state_attr, dtype=dtype
        )
        self._state.stop_gradient = True

        accum_prefix = f"{name}.accum" if name else 'quant_dequant.accum'
        accum_attr = ParamAttr(
            name=unique_name.generate(accum_prefix),
            initializer=Constant(1),
            trainable=False,
        )
        self._accum = self.create_parameter(
            shape=[1], attr=accum_attr, dtype=dtype
        )
        self._accum.stop_gradient = True

    def dynamic_forward(self, input):
        attrs = (
            'moving_rate',
            self._moving_rate,
            'bit_length',
            self._bit_length,
            'is_test',
            not self.training,
        )
        quant_out = _create_tensor(
            type=input.type,
            name=f"{input.name}.quantized.dequantized",
            shape=input.shape,
            dtype=input.dtype,
            persistable=False,
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
            *attrs,
        )

        return out

    def static_forward(self, input):
        check_variable_and_dtype(
            input, 'input', ['float32'], "FakeQuantMovingAverageAbsMax"
        )
        attrs = {
            'moving_rate': self._moving_rate,
            'bit_length': self._bit_length,
            'is_test': not self.training,
        }
        inputs = {"X": [input], "InScale": [self._scale]}
        quant_out = self._helper.create_variable(
            name=f"{input.name}.quantized.dequantized",
            dtype=input.dtype,
            type=core.VarDesc.VarType.LOD_TENSOR,
            persistable=False,
            stop_gradient=False,
        )
        outputs = {"Out": [quant_out], "OutScale": [self._scale]}

        if self.training:
            inputs['InState'] = [self._state]
            inputs['InAccum'] = [self._accum]
            outputs['OutState'] = [self._state]
            outputs['OutAccum'] = [self._accum]

        self._helper.append_op(
            type="fake_quantize_dequantize_moving_average_abs_max",
            inputs=inputs,
            outputs=outputs,
            attrs=attrs,
        )

        return quant_out

    def forward(self, input):
        if paddle.in_dynamic_mode():
            return self.dynamic_forward(input)
        else:
            return self.static_forward(input)

    def bit_length(self):
        return self._bit_length

    def quant_axis(self):
        return -1

    def scales(self):
        return self._scale

    def zero_points(self):
        return None
