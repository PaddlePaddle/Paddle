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

from paddle.fluid.dygraph import layers
from paddle.fluid import core
from paddle.fluid import dygraph_utils
from paddle.fluid import unique_name
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.framework import _varbase_creator
from paddle.fluid.framework import in_dygraph_mode
from paddle.fluid.initializer import Constant
from paddle.fluid.data_feeder import check_variable_and_dtype
from paddle.nn import functional as F

__all__ = [
    'FakeQuantMovingAverage', 'FakeQuantAbsMax',
    'FakeChannelWiseQuantDequantAbsMax', 'QuantizedConv2D', 'QuantizedLinear',
    'QuantizedNoweightLayer', 'MovingAverageAbsMaxScale'
]


class FakeQuantMovingAverage(layers.Layer):
    r"""
    FakeQuantMovingAverage layer does the moving_average_abs_max quant and then dequant.
    Its computational formula is described as below:

    :math:`scale = (moving\_rate*accum+max(abs(x)))/(moving\_rate*state+1)`
    :math:`range = 2^{bit\_length - 1} - 1`
    :math:`Out = round(X / scale * range) * scale / range`
    """

    def __init__(self,
                 name=None,
                 moving_rate=0.9,
                 quant_bits=8,
                 dtype='float32'):
        super(FakeQuantMovingAverage, self).__init__()
        self._moving_rate = moving_rate
        self._quant_bits = quant_bits

        scale_prefix = "{}.scale".format(
            name) if name else 'quant_dequant.scale'
        scale_attr = ParamAttr(
            name=unique_name.generate(scale_prefix),
            initializer=Constant(0.001),
            trainable=False)
        self._scale = self.create_parameter(
            shape=[1], attr=scale_attr, dtype=dtype)
        self._scale.stop_gradient = True

        state_prefix = "{}.state".format(
            name) if name else 'quant_dequant.state'
        state_attr = ParamAttr(
            name=unique_name.generate(state_prefix),
            initializer=Constant(1),
            trainable=False)
        self._state = self.create_parameter(
            shape=[1], attr=state_attr, dtype=dtype)
        self._state.stop_gradient = True

        accum_prefix = "{}.accum".format(
            name) if name else 'quant_dequant.accum'
        accum_attr = ParamAttr(
            name=unique_name.generate(accum_prefix),
            initializer=Constant(1),
            trainable=False)
        self._accum = self.create_parameter(
            shape=[1], attr=accum_attr, dtype=dtype)
        self._accum.stop_gradient = True

    def forward(self, input):
        if in_dygraph_mode():
            attrs = ('moving_rate', self._moving_rate, 'bit_length',
                     self._quant_bits, 'is_test', not self.training)
            quant_out = _varbase_creator(
                type=input.type,
                name="{}.quantized.dequantized".format(input.name),
                shape=input.shape,
                dtype=input.dtype,
                persistable=False)
            state = self._state if self.training else None
            accum = self._accum if self.training else None

            out, _, _, _ = core.ops.fake_quantize_dequantize_moving_average_abs_max(
                input, self._scale, accum, state, quant_out, self._scale, state,
                accum, *attrs)
            return out

        check_variable_and_dtype(input, 'input', ['float32'],
                                 "FakeQuantMovingAverage")
        attrs = {
            'moving_rate': self._moving_rate,
            'bit_length': self._quant_bits,
            'is_test': not self.training
        }
        inputs = {"X": [input], "InScale": [self._scale]}
        quant_out = self._helper.create_variable(
            name="{}.quantized.dequantized".format(input.name),
            dtype=input.dtype,
            type=core.VarDesc.VarType.LOD_TENSOR,
            persistable=False,
            stop_gradient=False)
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
            attrs=attrs)

        return quant_out


class FakeQuantAbsMax(layers.Layer):
    r"""
    FakeQuantAbsMax layer does the abs_max quant and then dequant.
    Its computational formula is described as below:

    :math:`scale = max(abs(X))`
    :math:`range = 2^{bit\_length - 1} - 1`
    :math:`Out = round(X / scale * range) * scale / range`
    """

    def __init__(self,
                 name=None,
                 quant_bits=8,
                 dtype='float32',
                 quant_on_weight=False):
        super(FakeQuantAbsMax, self).__init__()
        self._quant_bits = quant_bits
        self._name = name
        scale_prefix = "{}.scale".format(
            name) if name else 'quant_dequant.scale'
        self._scale_name = unique_name.generate(scale_prefix)
        if quant_on_weight:
            scale_attr = ParamAttr(
                name=self._scale_name,
                initializer=Constant(0.0),
                trainable=False)
            self._scale = self.create_parameter(
                shape=[1], attr=scale_attr, dtype=self._dtype)
            self._scale.stop_gradient = True
        else:
            self._scale = None

    def forward(self, input):
        if in_dygraph_mode():
            attrs = ('bit_length', self._quant_bits)
            quant_out = _varbase_creator(
                type=input.type,
                name="{}.quantized.dequantized".format(input.name),
                shape=input.shape,
                dtype=input.dtype,
                persistable=False)
            out_scale = self._scale
            if not out_scale:
                out_scale = _varbase_creator(
                    type=core.VarDesc.VarType.LOD_TENSOR,
                    name=self._scale_name,
                    shape=[1],
                    dtype=self._dtype,
                    persistable=False)
                out_scale.stop_gradient = True
            out, _, = core.ops.fake_quantize_dequantize_abs_max(
                input, quant_out, out_scale, *attrs)
            return out

        check_variable_and_dtype(input, 'input', ['float32'], "FakeQuantAbsMax")
        attrs = {'bit_length': self._quant_bits}
        inputs = {"X": [input]}
        quant_out = self._helper.create_variable(
            name="{}.quantized.dequantized".format(input.name),
            dtype=input.dtype,
            type=core.VarDesc.VarType.LOD_TENSOR,
            persistable=False,
            stop_gradient=False)
        out_scale = self._scale
        if not out_scale:
            out_scale = self._helper.create_variable(
                name=self._scale_name,
                dtype=self._dtype,
                type=core.VarDesc.VarType.LOD_TENSOR,
                persistable=False,
                stop_gradient=True)
        outputs = {"Out": [quant_out], "OutScale": [out_scale]}

        self._helper.append_op(
            type="fake_quantize_dequantize_abs_max",
            inputs=inputs,
            outputs=outputs,
            attrs=attrs)

        return quant_out


class FakeChannelWiseQuantDequantAbsMax(layers.Layer):
    def __init__(self,
                 name=None,
                 channel_num=None,
                 quant_bits=8,
                 quant_axis=0,
                 dtype='float32',
                 quant_on_weight=False):
        assert quant_on_weight == True, "Channel_wise only can be used on weight quantization."
        super(FakeChannelWiseQuantDequantAbsMax, self).__init__()
        self._quant_bits = quant_bits
        self._quant_axis = quant_axis
        self._dtype = dtype
        self._name = name
        self._channel_num = channel_num
        scale_prefix = "{}.scale".format(
            name) if name else 'quant_dequant.scale'
        self._scale_name = unique_name.generate(scale_prefix)
        if quant_on_weight:
            scale_attr = ParamAttr(
                name=self._scale_name,
                initializer=Constant(0.0),
                trainable=False)
            self._scale = self.create_parameter(
                shape=[self._channel_num], attr=scale_attr, dtype=self._dtype)
            self._scale.stop_gradient = True
        else:
            self._scale = None

    def forward(self, input):
        if in_dygraph_mode():
            attrs = ('bit_length', self._quant_bits, 'quant_axis',
                     self._quant_axis)
            quant_out = _varbase_creator(
                type=input.type,
                name="{}.quantized.dequantized".format(input.name),
                shape=input.shape,
                dtype=input.dtype,
                persistable=False)

            out_scale = self._scale
            if out_scale is None:
                out_scale = _varbase_creator(
                    type=core.VarDesc.VarType.LOD_TENSOR,
                    name=self._scale_name,
                    shape=[self._channel_num],
                    dtype=self._dtype,
                    persistable=False)
                out_scale.stop_gradient = True

            out, _, = core.ops.fake_channel_wise_quantize_dequantize_abs_max(
                input, quant_out, out_scale, *attrs)
            return out

        check_variable_and_dtype(input, 'input', ['float32'],
                                 "FakeChannelWiseQuantDequantAbsMax")
        attrs = {'bit_length': self._quant_bits, 'quant_axis': self._quant_axis}
        inputs = {"X": [input]}
        quant_out = self._helper.create_variable(
            name="{}.quantized.dequantized".format(input.name),
            dtype=input.dtype,
            type=core.VarDesc.VarType.LOD_TENSOR,
            persistable=False,
            stop_gradient=False)
        out_scale = self._scale
        if not out_scale:
            out_scale = self._helper.create_variable(
                name=self._scale_name,
                dtype=self._dtype,
                type=core.VarDesc.VarType.LOD_TENSOR,
                persistable=False,
                stop_gradient=True)
        outputs = {"Out": [quant_out], "OutScale": [out_scale]}

        self._helper.append_op(
            type="fake_channel_wise_quantize_dequantize_abs_max",
            inputs=inputs,
            outputs=outputs,
            attrs=attrs)

        return quant_out


def _get_fake_quant_type(quant_type, **kwargs):
    call_args = {
        "name": kwargs.get("name", None),
        "quant_bits": kwargs.get("quant_bits", 8),
        "dtype": kwargs.get("dtype", "float32")
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
            "when you use channel_wise_abs_max strategy.")
    fake_quant_map = {
        'abs_max': FakeQuantAbsMax,
        'moving_average_abs_max': FakeQuantMovingAverage,
        'channel_wise_abs_max': FakeChannelWiseQuantDequantAbsMax
    }

    return fake_quant_map[quant_type](**call_args)


class QuantizedConv2D(layers.Layer):
    """
    The computational logic of QuantizedConv2D is the same with Conv2D.
    The only difference is that its inputs are all fake quantized.
    """

    def __init__(self,
                 layer,
                 weight_bits=8,
                 activation_bits=8,
                 moving_rate=0.9,
                 weight_quantize_type='abs_max',
                 activation_quantize_type='abs_max',
                 weight_pre_layer=None,
                 act_pre_layer=None,
                 weight_quant_layer=None,
                 act_quant_layer=None):
        super(QuantizedConv2D, self).__init__()
        # For Conv2D
        self._groups = getattr(layer, '_groups')
        self._stride = getattr(layer, '_stride')
        self._padding = getattr(layer, '_padding')
        self._padding_mode = getattr(layer, '_padding_mode')
        if self._padding_mode != 'zeros':
            self._reversed_padding_repeated_twice = getattr(
                layer, '_reversed_padding_repeated_twice')
        self._dilation = getattr(layer, '_dilation')
        self._data_format = getattr(layer, '_data_format')
        self.weight = getattr(layer, 'weight')
        self.bias = getattr(layer, 'bias')

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
                quant_axis=self._conv2d_quant_axis)
        if act_quant_layer is not None:
            self._fake_quant_input = act_quant_layer()
        else:
            self._fake_quant_input = _get_fake_quant_type(
                activation_quantize_type,
                name=layer.full_name(),
                moving_rate=moving_rate,
                quant_bits=activation_bits,
                dtype=self._dtype,
                quant_on_weight=False)

        self._act_preprocess = act_pre_layer(
        ) if act_pre_layer is not None else None
        self._weight_preprocess = weight_pre_layer(
        ) if weight_pre_layer is not None else None

    def forward(self, input):
        if self._act_preprocess is not None:
            input = self._act_preprocess(input)
        quant_input = self._fake_quant_input(input)

        weight = self.weight
        if self._weight_preprocess is not None:
            weight = self._weight_preprocess(self.weight)
        quant_weight = self._fake_quant_weight(weight)

        if self._padding_mode != 'zeros':
            quant_input = F.pad(quant_input,
                                self._reversed_padding_repeated_twice,
                                mode=self._padding_mode,
                                data_format=self._data_format)
            self._padding = 0

        return F.conv2d(
            quant_input,
            quant_weight,
            bias=self.bias,
            padding=self._padding,
            stride=self._stride,
            dilation=self._dilation,
            groups=self._groups,
            data_format=self._data_format)


class QuantizedLinear(layers.Layer):
    """
    The computational logic of QuantizedLinear is the same with Linear.
    The only difference is that its inputs are all fake quantized.
    """

    def __init__(self,
                 layer,
                 weight_bits=8,
                 activation_bits=8,
                 moving_rate=0.9,
                 weight_quantize_type='abs_max',
                 activation_quantize_type='abs_max',
                 weight_pre_layer=None,
                 act_pre_layer=None,
                 weight_quant_layer=None,
                 act_quant_layer=None):
        super(QuantizedLinear, self).__init__()
        # For Linear
        self.weight = getattr(layer, 'weight')
        self.bias = getattr(layer, 'bias')
        self.name = getattr(layer, 'name')
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
                quant_axis=self._linear_quant_axis)

        if act_quant_layer is not None:
            self._fake_quant_input = act_quant_layer()
        else:
            self._fake_quant_input = _get_fake_quant_type(
                activation_quantize_type,
                name=layer.full_name(),
                moving_rate=moving_rate,
                quant_bits=activation_bits,
                dtype=self._dtype,
                quant_on_weight=False)

        self._act_preprocess = act_pre_layer(
        ) if act_pre_layer is not None else None
        self._weight_preprocess = weight_pre_layer(
        ) if weight_pre_layer is not None else None

    def forward(self, input):
        if self._act_preprocess is not None:
            input = self._act_preprocess(input)
        quant_input = self._fake_quant_input(input)

        weight = self.weight
        if self._weight_preprocess is not None:
            weight = self._weight_preprocess(self.weight)
        quant_weight = self._fake_quant_weight(weight)

        out = F.linear(
            x=quant_input, weight=quant_weight, bias=self.bias, name=self.name)
        return out


class QuantizedNoweightLayer(layers.Layer):
    def __init__(self,
                 layer,
                 weight_bits=8,
                 activation_bits=8,
                 moving_rate=0.9,
                 *args,
                 **kwargs):

        super(QuantizedNoweightLayer, self).__init__()
        self._layer = layer
        self._fake_quant_input = _get_fake_quant_type(
            'moving_average_abs_max',
            name=layer.full_name(),
            moving_rate=moving_rate,
            quant_bits=activation_bits,
            dtype=self._dtype,
            quant_on_weight=False)

    def forward(self, input):
        quant_input = self._fake_quant_input(input)
        # TODO (jc): support ops that have several inputs
        if isinstance(input, list):
            assert len(input) == 1, \
                "The QuantizedNoweightLayer should only have one input."
        return self._layer.forward(quant_input)


class MovingAverageAbsMaxScale(layers.Layer):
    def __init__(self, name=None, moving_rate=0.9, dtype='float32'):
        r"""
        MovingAverageMaxScale layer is used to calculating the output quantization
        scale of Layer. Its computational formula is described as below:

        :math:`scale = (moving\_rate*accum+max(abs(x)))/(moving\_rate*state+1)`
        :math:`Out = X`
        """
        super(MovingAverageAbsMaxScale, self).__init__()
        self._moving_rate = moving_rate

        scale_prefix = '{}.scale'.format(name) if name else 'outscale.scale'
        scale_name = unique_name.generate(scale_prefix)
        scale_attr = ParamAttr(
            name=scale_name, initializer=Constant(1), trainable=False)
        self._scale = self.create_parameter(
            shape=[1], attr=scale_attr, dtype=dtype)
        self._scale.stop_gradient = True

        state_prefix = "{}.state".format(name) if name else 'outscale.state'
        state_attr = ParamAttr(
            name=unique_name.generate(state_prefix),
            initializer=Constant(1),
            trainable=False)
        self._state = self.create_parameter(
            shape=[1], attr=state_attr, dtype=dtype)
        self._state.stop_gradient = True

        accum_prefix = "{}.accum".format(name) if name else 'outscale.accum'
        accum_attr = ParamAttr(
            name=unique_name.generate(accum_prefix),
            initializer=Constant(1),
            trainable=False)
        self._accum = self.create_parameter(
            shape=[1], attr=accum_attr, dtype=dtype)
        self._accum.stop_gradient = True

    def forward(self, input):
        if in_dygraph_mode():
            attrs = ('moving_rate', self._moving_rate, 'is_test',
                     not self.training)
            state = self._state if self.training else None
            accum = self._accum if self.training else None
            quant_out = _varbase_creator(
                type=input.type,
                name="{}.tmp".format(input.name),
                shape=input.shape,
                dtype=input.dtype,
                persistable=False)

            out, _, _, _ = core.ops.moving_average_abs_max_scale(
                input, accum, state, quant_out, self._scale, state, accum,
                *attrs)
            return out

        check_variable_and_dtype(input, 'input', ['float32', 'float64'],
                                 'MovingAverageAbsMaxScale')

        attrs = {'moving_rate': self._moving_rate, 'is_test': not self.training}
        inputs = {"X": [input]}
        quant_out = self._helper.create_variable(
            name="{}.tmp".format(input.name),
            dtype=input.dtype,
            type=core.VarDesc.VarType.LOD_TENSOR,
            persistable=False,
            stop_gradient=False)
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
            attrs=attrs)

        return quant_out


class QuantizedOutputLayer(layers.Layer):
    def __init__(self, layer=None, moving_rate=0.9, dtype='float32'):
        r"""
        Add MovingAverageMaxScale layer to the behind of the input layer.
        """
        super(QuantizedOutputLayer, self).__init__()
        self._layer = layer
        self._moving_average_abs_max_scale = \
            MovingAverageAbsMaxScale(layer.full_name(), moving_rate, dtype)

    def forward(self, input):
        if isinstance(input, list):
            assert len(input) == 1, \
                "The QuantizedOutputLayer should only have one input."
        out = self._layer(input)
        return self._moving_average_abs_max_scale(out)
