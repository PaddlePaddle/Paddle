#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import logging

import paddle
from paddle import _C_ops, _legacy_C_ops, in_dynamic_mode
from paddle.base.data_feeder import check_variable_and_dtype
from paddle.base.framework import _create_tensor
from paddle.base.log_helper import get_logger
from paddle.framework import ParamAttr, core
from paddle.nn import functional as F
from paddle.nn.initializer import Constant
from paddle.nn.quant.lsq import FakeQuantActLSQPlus, FakeQuantWeightLSQPlus
from paddle.utils import unique_name

from ..layer.layers import Layer

__all__ = [
    'FakeQuantAbsMax',
    'FakeQuantMovingAverageAbsMax',
    'FakeQuantChannelWiseAbsMax',
    'QuantizedConv2D',
    'QuantizedConv2DTranspose',
    'QuantizedLinear',
    'MovingAverageAbsMaxScale',
    'MAOutputScaleLayer',
    'FakeQuantMAOutputScaleLayer',
    'QuantStub',
    'QuantizedRowParallelLinear',
    'QuantizedColumnParallelLinear',
    'QuantizedMatmul',
]

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s'
)


class FakeQuantAbsMax(Layer):
    r"""
    FakeQuantAbsMax layer does the abs_max quant and then dequant.
    Its computational formula is described as below:

    :math:`scale = max(abs(X))`
    :math:`range = 2^{bit\_length - 1} - 1`
    :math:`Out = round(X / scale * range) * scale / range`
    """

    def __init__(
        self,
        name=None,
        quant_bits=8,
        dtype='float32',
        quant_on_weight=False,
        reduce_type=None,
    ):
        super().__init__()
        self._quant_bits = quant_bits
        self._name = name
        self._reduce_type = reduce_type
        scale_prefix = f"{name}.scale" if name else 'quant_dequant.scale'
        self._scale_name = unique_name.generate(scale_prefix)
        if quant_on_weight:
            scale_attr = ParamAttr(
                name=self._scale_name,
                initializer=Constant(0.001),
                trainable=False,
            )
            self._scale = self.create_parameter(
                shape=[1], attr=scale_attr, dtype=self._dtype
            )
            self._scale.stop_gradient = True
        else:
            self._scale = None

    def forward(self, input):
        if in_dynamic_mode():
            attrs = ('bit_length', self._quant_bits)
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

            if not out_scale:
                out_scale = _create_tensor(
                    type=core.VarDesc.VarType.LOD_TENSOR,
                    name=self._scale_name,
                    shape=[1],
                    dtype=self._dtype,
                    persistable=False,
                )
                out_scale.stop_gradient = True
            (
                out1,
                out2,
            ) = _C_ops.fake_quantize_dequantize_abs_max(
                input, self._quant_bits, 1
            )
            _C_ops.assign_out_(out1, quant_out)
            _C_ops.assign_out_(out2, out_scale)
            return quant_out

        check_variable_and_dtype(input, 'input', ['float32'], "FakeQuantAbsMax")
        attrs = {'bit_length': self._quant_bits}
        inputs = {"X": [input]}
        quant_out = self._helper.create_variable(
            name=f"{input.name}.quantized.dequantized",
            dtype=input.dtype,
            type=core.VarDesc.VarType.LOD_TENSOR,
            persistable=False,
            stop_gradient=False,
        )
        out_scale = self._scale
        if not out_scale:
            out_scale = self._helper.create_variable(
                name=self._scale_name,
                dtype=self._dtype,
                type=core.VarDesc.VarType.LOD_TENSOR,
                persistable=False,
                stop_gradient=True,
            )
        outputs = {"Out": [quant_out], "OutScale": [out_scale]}

        self._helper.append_op(
            type="fake_quantize_dequantize_abs_max",
            inputs=inputs,
            outputs=outputs,
            attrs=attrs,
        )

        return quant_out


class FakeQuantMovingAverageAbsMax(Layer):
    r"""
    FakeQuantMovingAverageAbsMax layer does the moving_average_abs_max quant and then dequant.
    Its computational formula is described as below:

    :math:`scale = (moving\_rate*accum+max(abs(x)))/(moving\_rate*state+1)`
    :math:`range = 2^{bit\_length - 1} - 1`
    :math:`Out = round(X / scale * range) * scale / range`
    """

    def __init__(
        self,
        name=None,
        moving_rate=0.9,
        quant_bits=8,
        dtype='float32',
        reduce_type=None,
    ):
        super().__init__()
        self._moving_rate = moving_rate
        self._quant_bits = quant_bits
        self._reduce_type = reduce_type
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

    def forward(self, input):
        if in_dynamic_mode():
            attrs = (
                'moving_rate',
                self._moving_rate,
                'bit_length',
                self._quant_bits,
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
            if self._reduce_type == "max":
                paddle.distributed.all_reduce(
                    self._scale, op=paddle.distributed.ReduceOp.MAX
                )

            state = self._state if self.training else None
            accum = self._accum if self.training else None

            (
                out1,
                out2,
                out3,
                out4,
            ) = _C_ops.fake_quantize_dequantize_moving_average_abs_max(
                input,
                self._scale,
                accum,
                state,
                self._moving_rate,
                self._quant_bits,
                not self.training,
                1,
            )
            _C_ops.assign_out_(out1, quant_out)
            if out2._is_initialized():
                _C_ops.assign_out_(out2, self._scale)
            if state:
                _C_ops.assign_out_(out3, state)
            if accum:
                _C_ops.assign_out_(out4, accum)
            return quant_out

        check_variable_and_dtype(
            input, 'input', ['float32'], "FakeQuantMovingAverageAbsMax"
        )
        attrs = {
            'moving_rate': self._moving_rate,
            'bit_length': self._quant_bits,
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


class FakeQuantChannelWiseAbsMax(Layer):
    def __init__(
        self,
        name=None,
        channel_num=None,
        quant_bits=8,
        quant_axis=0,
        dtype='float32',
        quant_on_weight=False,
        reduce_type=None,
    ):
        assert (
            quant_on_weight
        ), "Channel_wise only can be used on weight quantization."
        super().__init__()
        self._quant_bits = quant_bits
        self._quant_axis = quant_axis
        self._dtype = dtype
        self._name = name
        self._channel_num = channel_num
        self._reduce_type = reduce_type
        scale_prefix = f"{name}.scale" if name else 'quant_dequant.scale'
        self._scale_name = unique_name.generate(scale_prefix)
        if quant_on_weight:
            scale_attr = ParamAttr(
                name=self._scale_name,
                initializer=Constant(0.0),
                trainable=False,
            )
            self._scale = self.create_parameter(
                shape=[self._channel_num], attr=scale_attr, dtype=self._dtype
            )
            self._scale.stop_gradient = True
        else:
            self._scale = None

    def forward(self, input):
        if in_dynamic_mode():
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
            if out_scale is None:
                out_scale = _create_tensor(
                    type=core.VarDesc.VarType.LOD_TENSOR,
                    name=self._scale_name,
                    shape=[self._channel_num],
                    dtype=self._dtype,
                    persistable=False,
                )
                out_scale.stop_gradient = True

            (
                out,
                scale,
            ) = _C_ops.fake_channel_wise_quantize_dequantize_abs_max(
                input, self._quant_bits, 1, self._quant_axis
            )
            _C_ops.assign_out_(out, quant_out)
            _C_ops.assign_out_(scale, out_scale)
            return quant_out

        check_variable_and_dtype(
            input, 'input', ['float32'], "FakeQuantChannelWiseAbsMax"
        )
        attrs = {
            'bit_length': self._quant_bits,
            'round_type': 1,
            'quant_axis': self._quant_axis,
        }
        inputs = {"X": [input]}
        quant_out = self._helper.create_variable(
            name=f"{input.name}.quantized.dequantized",
            dtype=input.dtype,
            type=core.VarDesc.VarType.LOD_TENSOR,
            persistable=False,
            stop_gradient=False,
        )
        out_scale = self._scale
        if not out_scale:
            out_scale = self._helper.create_variable(
                name=self._scale_name,
                dtype=self._dtype,
                type=core.VarDesc.VarType.LOD_TENSOR,
                persistable=False,
                stop_gradient=True,
            )
        outputs = {"Out": [quant_out], "OutScale": [out_scale]}

        self._helper.append_op(
            type="fake_channel_wise_quantize_dequantize_abs_max",
            inputs=inputs,
            outputs=outputs,
            attrs=attrs,
        )

        return quant_out


class MovingAverageAbsMaxScale(Layer):
    def __init__(
        self, name=None, moving_rate=0.9, dtype='float32', reduce_type=None
    ):
        r"""
        MovingAverageMaxScale layer is used to calculating the output quantization
        scale of Layer. Its computational formula is described as below:

        :math:`scale = (moving\_rate*accum+max(abs(x)))/(moving\_rate*state+1)`
        :math:`Out = X`
        """
        super().__init__()
        self._moving_rate = moving_rate
        self._reduce_type = reduce_type
        scale_prefix = f'{name}.scale' if name else 'outscale.scale'
        scale_name = unique_name.generate(scale_prefix)
        scale_attr = ParamAttr(
            name=scale_name, initializer=Constant(0), trainable=False
        )
        self._scale = self.create_parameter(
            shape=[1], attr=scale_attr, dtype=dtype
        )
        self._scale.stop_gradient = True

        state_prefix = f"{name}.state" if name else 'outscale.state'
        state_attr = ParamAttr(
            name=unique_name.generate(state_prefix),
            initializer=Constant(0),
            trainable=False,
        )
        self._state = self.create_parameter(
            shape=[1], attr=state_attr, dtype=dtype
        )
        self._state.stop_gradient = True

        accum_prefix = f"{name}.accum" if name else 'outscale.accum'
        accum_attr = ParamAttr(
            name=unique_name.generate(accum_prefix),
            initializer=Constant(0),
            trainable=False,
        )
        self._accum = self.create_parameter(
            shape=[1], attr=accum_attr, dtype=dtype
        )
        self._accum.stop_gradient = True

    def forward(self, input):
        if in_dynamic_mode():
            attrs = (
                'moving_rate',
                self._moving_rate,
                'is_test',
                not self.training,
            )

            quant_out = _create_tensor(
                type=input.type,
                name=f"{input.name}.tmp",
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

            out, _, _, _ = _legacy_C_ops.moving_average_abs_max_scale(
                input,
                accum,
                state,
                quant_out,
                self._scale,
                state,
                accum,
                *attrs,
            )
            return out

        check_variable_and_dtype(
            input, 'input', ['float32', 'float64'], 'MovingAverageAbsMaxScale'
        )

        attrs = {'moving_rate': self._moving_rate, 'is_test': not self.training}
        inputs = {"X": [input]}
        quant_out = self._helper.create_variable(
            name=f"{input.name}.tmp",
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
            type="moving_average_abs_max_scale",
            inputs=inputs,
            outputs=outputs,
            attrs=attrs,
        )

        return quant_out


QuantStub = MovingAverageAbsMaxScale


class QuantizedConv2D(Layer):
    """
    The computational logic of QuantizedConv2D is the same with Conv2D.
    The only difference is that its inputs are all fake quantized.
    """

    def __init__(
        self,
        layer,
        weight_bits=8,
        activation_bits=8,
        moving_rate=0.9,
        weight_quantize_type='abs_max',
        activation_quantize_type='abs_max',
        weight_pre_layer=None,
        act_pre_layer=None,
        weight_quant_layer=None,
        act_quant_layer=None,
    ):
        super().__init__()
        # For Conv2D
        self._groups = layer._groups
        self._stride = layer._stride
        self._padding = layer._padding
        self._padding_mode = layer._padding_mode
        if self._padding_mode != 'zeros':
            self._reversed_padding_repeated_twice = (
                layer._reversed_padding_repeated_twice
            )
        self._dilation = layer._dilation
        self._data_format = layer._data_format
        self.weight = layer.weight
        self.bias = layer.bias

        # For FakeQuant
        self._conv2d_quant_axis = 0
        if weight_quant_layer is not None:
            self._fake_quant_weight = weight_quant_layer()
        else:
            self._fake_quant_weight = _get_fake_quant_type(
                weight_quantize_type,
                name=self.weight.name,
                moving_rate=moving_rate,
                quant_bits=weight_bits,
                dtype=self._dtype,
                quant_on_weight=True,
                channel_num=self.weight.shape[self._conv2d_quant_axis],
                quant_axis=self._conv2d_quant_axis,
            )
        if act_quant_layer is not None:
            self._fake_quant_input = act_quant_layer()
        else:
            self._fake_quant_input = _get_fake_quant_type(
                activation_quantize_type,
                name=layer.full_name(),
                moving_rate=moving_rate,
                quant_bits=activation_bits,
                dtype=self._dtype,
                quant_on_weight=False,
            )

        self._act_preprocess = (
            act_pre_layer() if act_pre_layer is not None else None
        )
        self._weight_preprocess = (
            weight_pre_layer() if weight_pre_layer is not None else None
        )

    def forward(self, input):
        if self._act_preprocess is not None:
            input = self._act_preprocess(input)
        quant_input = self._fake_quant_input(input)

        weight = self.weight
        if self._weight_preprocess is not None:
            weight = self._weight_preprocess(self.weight)
        quant_weight = self._fake_quant_weight(weight)

        if self._padding_mode != 'zeros':
            quant_input = F.pad(
                quant_input,
                self._reversed_padding_repeated_twice,
                mode=self._padding_mode,
                data_format=self._data_format,
            )
            self._padding = 0

        return F.conv2d(
            quant_input,
            quant_weight,
            bias=self.bias,
            padding=self._padding,
            stride=self._stride,
            dilation=self._dilation,
            groups=self._groups,
            data_format=self._data_format,
        )


class QuantizedConv2DTranspose(Layer):
    """

    The computational logic of QuantizedConv2DTranspose is the same with Conv2DTranspose.
    The only difference is that its inputs are all fake quantized.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn as nn
            >>> from paddle.nn.quant.quant_layers import QuantizedConv2DTranspose

            >>> x_var = paddle.uniform((2, 4, 8, 8), dtype='float32', min=-1., max=1.)
            >>> conv = nn.Conv2DTranspose(4, 6, (3, 3))
            >>> conv_quantized = QuantizedConv2DTranspose(conv)
            >>> y_quantized = conv_quantized(x_var)
            >>> y_var = conv(x_var)
            >>> print(y_var.shape)
            [2, 6, 10, 10]
            >>> print(y_quantized.shape)
            [2, 6, 10, 10]

    """

    def __init__(
        self,
        layer,
        weight_bits=8,
        activation_bits=8,
        moving_rate=0.9,
        weight_quantize_type='abs_max',
        activation_quantize_type='abs_max',
        weight_pre_layer=None,
        act_pre_layer=None,
        weight_quant_layer=None,
        act_quant_layer=None,
    ):
        r"""
        Constructor.

        The arguments are the same as ImperativeQuantAware.
        """
        super().__init__()
        # For Conv2DTranspose
        self._groups = layer._groups
        self._stride = layer._stride
        self._padding = layer._padding
        self._output_padding = layer.output_padding
        self._dilation = layer._dilation
        self._data_format = layer._data_format
        self.weight = layer.weight
        self.bias = layer.bias
        # For FakeQuant
        self._conv2d_transpose_quant_axis = 1
        if weight_quant_layer is not None:
            self._fake_quant_weight = weight_quant_layer()
        else:
            self._fake_quant_weight = _get_fake_quant_type(
                weight_quantize_type,
                name=self.weight.name,
                moving_rate=moving_rate,
                quant_bits=weight_bits,
                dtype=self._dtype,
                quant_on_weight=True,
                channel_num=self.weight.shape[
                    self._conv2d_transpose_quant_axis
                ],
                quant_axis=self._conv2d_transpose_quant_axis,
            )
        if act_quant_layer is not None:
            self._fake_quant_input = act_quant_layer()
        else:
            self._fake_quant_input = _get_fake_quant_type(
                activation_quantize_type,
                name=layer.full_name(),
                moving_rate=moving_rate,
                quant_bits=activation_bits,
                dtype=self._dtype,
                quant_on_weight=False,
            )

        self._act_preprocess = (
            act_pre_layer() if act_pre_layer is not None else None
        )
        self._weight_preprocess = (
            weight_pre_layer() if weight_pre_layer is not None else None
        )

    def forward(self, input, output_size=None):
        if self._act_preprocess is not None:
            input = self._act_preprocess(input)
        quant_input = self._fake_quant_input(input)

        weight = self.weight
        if self._weight_preprocess is not None:
            weight = self._weight_preprocess(self.weight)
        quant_weight = self._fake_quant_weight(weight)

        if output_size is None:
            output_padding = self._output_padding
        else:
            output_padding = 0

        return F.conv2d_transpose(
            quant_input,
            quant_weight,
            bias=self.bias,
            padding=self._padding,
            output_padding=output_padding,
            stride=self._stride,
            dilation=self._dilation,
            groups=self._groups,
            output_size=output_size,
            data_format=self._data_format,
        )


class QuantizedLinear(Layer):
    """
    The computational logic of QuantizedLinear is the same with Linear.
    The only difference is that its inputs are all fake quantized.
    """

    def __init__(
        self,
        layer,
        weight_bits=8,
        activation_bits=8,
        moving_rate=0.9,
        weight_quantize_type='abs_max',
        activation_quantize_type='abs_max',
        weight_pre_layer=None,
        act_pre_layer=None,
        weight_quant_layer=None,
        act_quant_layer=None,
    ):
        super().__init__()
        # For Linear
        self.weight = layer.weight
        self.bias = layer.bias
        self.name = layer.name
        # For FakeQuant
        self._linear_quant_axis = 1

        if weight_quant_layer is not None:
            self._fake_quant_weight = weight_quant_layer()
        else:
            self._fake_quant_weight = _get_fake_quant_type(
                weight_quantize_type,
                name=self.weight.name,
                moving_rate=moving_rate,
                quant_bits=weight_bits,
                dtype=self._dtype,
                quant_on_weight=True,
                channel_num=self.weight.shape[self._linear_quant_axis],
                quant_axis=self._linear_quant_axis,
                quant_linear=True,
            )

        if act_quant_layer is not None:
            self._fake_quant_input = act_quant_layer()
        else:
            self._fake_quant_input = _get_fake_quant_type(
                activation_quantize_type,
                name=layer.full_name(),
                moving_rate=moving_rate,
                quant_bits=activation_bits,
                dtype=self._dtype,
                quant_on_weight=False,
            )

        self._act_preprocess = (
            act_pre_layer() if act_pre_layer is not None else None
        )
        self._weight_preprocess = (
            weight_pre_layer() if weight_pre_layer is not None else None
        )

    def forward(self, input):
        if self._act_preprocess is not None:
            input = self._act_preprocess(input)
        quant_input = self._fake_quant_input(input)

        weight = self.weight
        if self._weight_preprocess is not None:
            weight = self._weight_preprocess(self.weight)
        quant_weight = self._fake_quant_weight(weight)

        out = F.linear(
            x=quant_input, weight=quant_weight, bias=self.bias, name=self.name
        )
        return out


class QuantizedColumnParallelLinear(Layer):
    def __init__(
        self,
        layer,
        weight_bits=8,
        activation_bits=8,
        moving_rate=0.9,
        weight_quantize_type='abs_max',
        activation_quantize_type='abs_max',
        weight_pre_layer=None,
        act_pre_layer=None,
        weight_quant_layer=None,
        act_quant_layer=None,
    ):
        super().__init__()
        '''

        '''
        assert (
            weight_quant_layer is None
        ), "When quantizing ColumnParallelLinear, weight_quant_layer should be None."
        assert (
            act_quant_layer is None
        ), "When quantizing ColumnParallelLinear, act_quant_layer should be None."

        self.weight = layer.weight
        self.bias = layer.bias
        self.name = layer._name
        # For FakeQuant
        self._linear_quant_axis = 1

        self.is_mp = layer.is_mp
        self.model_parallel_group = layer.model_parallel_group
        self.gather_output = layer.gather_output

        self._fake_quant_weight = _get_fake_quant_type(
            weight_quantize_type,
            name=self.weight.name,
            moving_rate=moving_rate,
            quant_bits=weight_bits,
            dtype=self._dtype,
            quant_on_weight=True,
            channel_num=self.weight.shape[self._linear_quant_axis],
            quant_axis=self._linear_quant_axis,
            reduce_type='max'
            if paddle.distributed.get_world_size() > 1
            else None,
        )

        self._fake_quant_input = _get_fake_quant_type(
            activation_quantize_type,
            name=layer.full_name(),
            moving_rate=moving_rate,
            quant_bits=activation_bits,
            dtype=self._dtype,
            quant_on_weight=False,
            reduce_type=None,
        )

        self._act_preprocess = (
            act_pre_layer() if act_pre_layer is not None else None
        )
        self._weight_preprocess = (
            weight_pre_layer() if weight_pre_layer is not None else None
        )

    def forward(self, input):
        if self.is_mp:
            input_parallel = paddle.distributed.collective._c_identity(
                input, group=self.model_parallel_group
            )
        else:
            input_parallel = input

        if self._act_preprocess is not None:
            input_parallel = self._act_preprocess(input_parallel)
        quant_input = self._fake_quant_input(input_parallel)

        weight = self.weight
        if self._weight_preprocess is not None:
            weight = self._weight_preprocess(self.weight)
        quant_weight = self._fake_quant_weight(weight)

        output_parallel = F.linear(
            x=quant_input, weight=quant_weight, bias=self.bias, name=self.name
        )

        if self.gather_output and self.is_mp:
            output = paddle.distributed.collective._c_concat(
                output_parallel, group=self.model_parallel_group
            )
        else:
            output = output_parallel
        return output


class QuantizedRowParallelLinear(Layer):
    def __init__(
        self,
        layer,
        weight_bits=8,
        activation_bits=8,
        moving_rate=0.9,
        weight_quantize_type='abs_max',
        activation_quantize_type='abs_max',
        weight_pre_layer=None,
        act_pre_layer=None,
        weight_quant_layer=None,
        act_quant_layer=None,
    ):
        super().__init__()
        assert (
            weight_quant_layer is None
        ), "When quantizing RowParallelLinear, weight_quant_layer cannot defined by yourself."
        assert (
            act_quant_layer is None
        ), "When quantizing RowParallelLinear, act_quant_layer cannot defined by yourself."

        # For Linear
        self.weight = layer.weight
        self.bias = layer.bias
        self.name = layer._name
        # For FakeQuant
        self._linear_quant_axis = 1

        self.input_is_parallel = layer.input_is_parallel
        self.is_mp = layer.is_mp
        self.model_parallel_group = layer.model_parallel_group

        self._fake_quant_weight = _get_fake_quant_type(
            weight_quantize_type,
            name=self.weight.name,
            moving_rate=moving_rate,
            quant_bits=weight_bits,
            dtype=self._dtype,
            quant_on_weight=True,
            channel_num=self.weight.shape[self._linear_quant_axis],
            quant_axis=self._linear_quant_axis,
            reduce_type='max'
            if paddle.distributed.get_world_size() > 1
            else None,
        )

        self._fake_quant_input = _get_fake_quant_type(
            activation_quantize_type,
            name=layer.full_name(),
            moving_rate=moving_rate,
            quant_bits=activation_bits,
            dtype=self._dtype,
            quant_on_weight=False,
            reduce_type='max'
            if paddle.distributed.get_world_size() > 1
            else None,
        )

        self._act_preprocess = (
            act_pre_layer() if act_pre_layer is not None else None
        )
        self._weight_preprocess = (
            weight_pre_layer() if weight_pre_layer is not None else None
        )

    def forward(self, input):
        if self.input_is_parallel or (not self.is_mp):
            input_parallel = input
        else:
            # split last dim
            input_parallel = paddle.distributed.collective._c_split(
                input, group=self.model_parallel_group
            )

        if self._act_preprocess is not None:
            input_parallel = self._act_preprocess(input_parallel)
        quant_input = self._fake_quant_input(input_parallel)

        weight = self.weight
        if self._weight_preprocess is not None:
            weight = self._weight_preprocess(self.weight)
        quant_weight = self._fake_quant_weight(weight)

        output_parallel = F.linear(
            x=quant_input, weight=quant_weight, name=self.name
        )
        if self.is_mp:
            output_ = paddle.distributed.collective._mp_allreduce(
                output_parallel,
                group=self.model_parallel_group,
                use_calc_stream=True,
                use_model_parallel=True,
            )
        else:
            output_ = output_parallel
        output = output_ + self.bias if self.bias is not None else output_
        return output


class QuantizedMatmul(Layer):
    """
    The computational logic of QuantizedMatmul is the same with Matmul.
    The only difference is that its inputs are all fake quantized.
    """

    def __init__(
        self,
        layer=None,
        weight_bits=8,
        activation_bits=8,
        moving_rate=0.9,
        weight_quantize_type='abs_max',
        activation_quantize_type='abs_max',
        weight_pre_layer=None,
        act_pre_layer=None,
        weight_quant_layer=None,
        act_quant_layer=None,
    ):
        super().__init__()

        # For FakeQuant
        if act_quant_layer is not None:
            self._fake_quant_x = act_quant_layer()
            self._fake_quant_y = act_quant_layer()
        else:
            self._fake_quant_x = _get_fake_quant_type(
                activation_quantize_type,
                moving_rate=moving_rate,
                quant_bits=activation_bits,
                quant_on_weight=False,
            )
            self._fake_quant_y = _get_fake_quant_type(
                activation_quantize_type,
                moving_rate=moving_rate,
                quant_bits=activation_bits,
                quant_on_weight=False,
            )

        self._act_preprocess_x = (
            act_pre_layer() if act_pre_layer is not None else None
        )
        self._act_preprocess_y = (
            act_pre_layer() if act_pre_layer is not None else None
        )

    def forward(self, x, y, transpose_x=False, transpose_y=False, name=None):
        if self._act_preprocess_x is not None:
            x = self._act_preprocess_x(x)
        quant_x = self._fake_quant_x(x)

        if self._act_preprocess_y is not None:
            y = self._act_preprocess_y(y)
        quant_y = self._fake_quant_y(y)

        out = paddle.matmul(quant_x, quant_y, transpose_x, transpose_y, name)
        return out


class MAOutputScaleLayer(Layer):
    """
    Add MovingAverageMaxScale layer to the behind of the input layer.
    Calculate the scale (moving average abs max) for the output of the input layer.
    """

    def __init__(
        self,
        layer=None,
        moving_rate=0.9,
        name=None,
        dtype='float32',
        reduce_type=None,
    ):
        r"""
        Construct
        """
        super().__init__()
        self._layer = layer
        if name is None:
            name = layer.full_name()
        self._ma_output_scale = MovingAverageAbsMaxScale(
            name, moving_rate, dtype, reduce_type
        )

    def forward(self, *inputs, **kwargs):
        out = self._layer(*inputs, **kwargs)
        # TODO (jc): support the ops of several outputs
        if isinstance(out, (list, tuple, dict)):
            return out
        else:
            return self._ma_output_scale(out)


class FakeQuantMAOutputScaleLayer(Layer):
    """
    Add FakeQuantMovingAverageAbsMax layer to the behind of the input layer.
    """

    def __init__(
        self,
        layer,
        weight_bits=8,
        activation_bits=8,
        moving_rate=0.9,
        name=None,
        reduce_type=None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self._layer = layer
        self._fake_quant_output = _get_fake_quant_type(
            'moving_average_abs_max',
            name=layer.full_name() if name is None else name,
            moving_rate=moving_rate,
            quant_bits=activation_bits,
            dtype=self._dtype,
            quant_on_weight=False,
            reduce_type=reduce_type,
        )

    def forward(self, *inputs, **kwargs):
        out = self._layer(*inputs, **kwargs)
        # TODO (jc): support the ops of several outputs
        if (isinstance(out, (list, tuple))) and len(out) > 1:
            return out
        else:
            return self._fake_quant_output(out)


def _get_fake_quant_type(quant_type, **kwargs):
    call_args = {
        "name": kwargs.get("name", None),
        "quant_bits": kwargs.get("quant_bits", 8),
        "dtype": kwargs.get("dtype", "float32"),
        "reduce_type": kwargs.get("reduce_type", None),
    }

    if quant_type == 'abs_max':
        call_args["quant_on_weight"] = kwargs.get("quant_on_weight", False)
    elif quant_type == 'moving_average_abs_max':
        call_args["moving_rate"] = kwargs.get("moving_rate", 0.9)
    elif quant_type == 'channel_wise_abs_max':
        call_args["quant_on_weight"] = kwargs.get("quant_on_weight", False)
        call_args["channel_num"] = kwargs.get("channel_num", None)
        call_args["quant_axis"] = kwargs.get("quant_axis", 0)
        assert call_args["channel_num"] is not None, (
            "You need to input channel_num"
            "when you use channel_wise_abs_max strategy."
        )
    elif quant_type == 'lsq_weight':
        call_args["all_positive"] = kwargs.get("all_positive", False)
        call_args["per_channel"] = False
        call_args["channel_num"] = 1
        call_args["quant_linear"] = kwargs.get("quant_linear", False)
    elif quant_type == 'channel_wise_lsq_weight':
        quant_type = 'lsq_weight'
        call_args["all_positive"] = kwargs.get("all_positive", False)
        call_args["per_channel"] = True
        call_args["channel_num"] = kwargs.get("channel_num", None)
        call_args["quant_linear"] = kwargs.get("quant_linear", False)
        assert call_args["channel_num"] is not None, (
            "You need to input channel_num"
            "when you use channel_wise_abs_max strategy."
        )
    elif quant_type == 'lsq_act':
        call_args["all_positive"] = kwargs.get("all_positive", False)
        call_args["symmetric"] = kwargs.get("symmetric", True)
    fake_quant_map = {
        'abs_max': FakeQuantAbsMax,
        'moving_average_abs_max': FakeQuantMovingAverageAbsMax,
        'channel_wise_abs_max': FakeQuantChannelWiseAbsMax,
        'lsq_weight': FakeQuantWeightLSQPlus,
        'lsq_act': FakeQuantActLSQPlus,
    }

    return fake_quant_map[quant_type](**call_args)
