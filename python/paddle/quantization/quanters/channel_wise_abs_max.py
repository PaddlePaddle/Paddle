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

from typing import Dict

import paddle
from paddle import _legacy_C_ops
from paddle.fluid.data_feeder import check_variable_and_dtype
from paddle.fluid.framework import _create_tensor
from paddle.framework import ParamAttr, core
from paddle.nn.initializer import Constant
from paddle.utils import unique_name

from ..base_quanter import BaseQuanter
from ..factory import QuanterFactory

CHANNEL_AXIS: Dict[type, int] = {
    paddle.nn.Conv2D: 0,
    paddle.nn.Linear: 1,
    paddle.distributed.fleet.meta_parallel.ColumnParallelLinear: 1,
    paddle.distributed.fleet.meta_parallel.RowParallelLinear: 1,
}


class FakeQuanterChannelWiseAbsMaxObserver(QuanterFactory):
    r"""
    Compute quantization parameters and simulate quantization.

    It collects per-channel maximum absolute values of target tensor.
    The average value will be used as quantization scale to quantize and
    dequantize the tensor.

    And it is symmetric uniform quantization which means the zero point is always 0.

    The computational formula of simulate quantization is:

    .. math::
            range = 2^{bit\_length - 1} - 1
            out = round(x / scale * range) * scale / range

    Where:

    - :math:`{bit\_length}` is the length of bits.
    - :math:`x` is the input tensor and :math:`out` is the output of simulate quantization.

    Args:
        bit_length(int, optional): Number of bits to represent an quantized integer in binary.
        dtype(str, optional): The data type of input tensor.
        name (str, optional): This parameter is used by developers to print debugging information. \
            For details, please refer to :ref:`api_guide_Name`. Default is None.

    Examples:
       .. code-block:: python

            from paddle.quantization import QuantConfig
            from paddle.quantization.quanters import FakeQuanterChannelWiseAbsMaxObserver
            quanter = FakeQuanterChannelWiseAbsMaxObserver()
            q_config = QuantConfig(activation=None, weight=quanter)
    """

    def __init__(
        self,
        bit_length=8,
        dtype='float32',
        name=None,
    ):
        super().__init__(
            bit_length=bit_length,
            dtype=dtype,
            name=name,
        )

    def _get_class(self):
        return FakeQuanterChannelWiseAbsMaxObserverLayer


class FakeQuanterChannelWiseAbsMaxObserverLayer(BaseQuanter):
    def __init__(
        self,
        layer,
        bit_length=8,
        dtype='float32',
        name=None,
    ):
        super().__init__()
        self._bit_length = bit_length
        try:
            self._quant_axis = CHANNEL_AXIS[type(layer)]
        except:
            for key in CHANNEL_AXIS.keys():
                if issubclass(type(layer), key):
                    self._quant_axis = CHANNEL_AXIS[key]
                    break
        self._channel_num = layer.weight.shape[self._quant_axis]
        self._reduce_type = 'max'

        scale_prefix = f"{name}.scale" if name else 'quant_dequant.scale'
        self._scale_name = unique_name.generate(scale_prefix)
        scale_attr = ParamAttr(
            name=self._scale_name,
            initializer=Constant(0.001),
            trainable=False,
        )
        self._scale = self.create_parameter(
            shape=[self._channel_num], attr=scale_attr, dtype=dtype
        )
        self._scale.stop_gradient = True

    def dynamic_forward(self, input):
        attrs = (
            'bit_length',
            self._quant_bits,
            'quant_axis',
            self._quant_axis,
        )
        quant_out = _create_tensor(
            type=input.type,
            name=f"{input.name}.quantized.dequantized",
            shape=input.shape,
            dtype=input.dtype,
            persistable=False,
        )

        out_scale = self._scale
        if self._reduce_type == "max":
            paddle.distributed.all_reduce(
                out_scale, op=paddle.distributed.ReduceOp.MAX
            )
        (
            out,
            _,
        ) = _legacy_C_ops.fake_channel_wise_quantize_dequantize_abs_max(
            input, quant_out, out_scale, *attrs
        )

        return out

    def static_forward(self, input):
        check_variable_and_dtype(
            input, 'input', ['float32'], "FakeQuantChannelWiseAbsMax"
        )
        attrs = {'bit_length': self._quant_bits, 'quant_axis': self._quant_axis}
        inputs = {"X": [input]}
        quant_out = self._helper.create_variable(
            name=f"{input.name}.quantized.dequantized",
            dtype=input.dtype,
            type=core.VarDesc.VarType.LOD_TENSOR,
            persistable=False,
            stop_gradient=False,
        )
        outputs = {"Out": [quant_out], "OutScale": [self._scale]}

        self._helper.append_op(
            type="fake_channel_wise_quantize_dequantize_abs_max",
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
        return self._quant_axis

    def scales(self):
        return self._scale

    def zero_points(self):
        return None
