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

from .....dygraph import layers
from ..... import core
from ..... import dygraph_utils
from ..... import unique_name
from .....param_attr import ParamAttr
from .....framework import _varbase_creator, in_dygraph_mode
from .....initializer import Constant
from .....data_feeder import check_variable_and_dtype

__all__ = ['FakeQuant', 'QuantizedConv2D', 'QuantizedLinear']


class FakeQuant(layers.Layer):
    def __init__(self,
                 name=None,
                 moving_rate=0.9,
                 quant_bits=8,
                 dtype='float32'):
        super(FakeQuant, self).__init__()
        self._moving_rate = moving_rate
        self._quant_bits = quant_bits

        scale_prefix = "{}.quant_dequant.scale".format(
            name) if name else 'quant_dequant.scale'
        scale_attr = ParamAttr(name=unique_name.generate(scale_prefix))
        self.scale = self.create_parameter(
            shape=[1],
            attr=scale_attr,
            dtype=dtype,
            is_bias=False,
            default_initializer=Constant(0.001))
        self.scale.stop_gradient = True

        state_prefix = "{}.quant_dequant.state".format(
            name) if name else 'quant_dequant.state'
        state_attr = ParamAttr(name=unique_name.generate(state_prefix))
        self.state = self.create_parameter(
            shape=[1],
            attr=state_attr,
            dtype=dtype,
            is_bias=False,
            default_initializer=Constant(1))
        self.state.stop_gradient = True

        accum_prefix = "{}.quant_dequant.accum".format(
            name) if name else 'quant_dequant.accum'
        accum_attr = ParamAttr(name=unique_name.generate(accum_prefix))
        self.accum = self.create_parameter(
            shape=[1],
            attr=accum_attr,
            dtype=dtype,
            is_bias=False,
            default_initializer=Constant(1))
        self.accum.stop_gradient = True

    def forward(self, input):
        if in_dygraph_mode():
            attrs = ('moving_rate', self._moving_rate, 'bit_length',
                     self._quant_bits, 'is_test', not self.training)
            quant_out = _varbase_creator(
                type=input.type,
                name="{}.quant_dequant".format(input.name),
                shape=input.shape,
                dtype=input.dtype,
                persistable=False)
            state = self.state if self.training else None
            accum = self.accum if self.training else None

            out, _, _, _ = core.ops.fake_quantize_dequantize_moving_average_abs_max(
                input, self.scale, accum, state, quant_out, self.scale, state,
                accum, *attrs)
            return out

        check_variable_and_dtype(input, 'input', ['float32', 'float64'],
                                 "FakeQuant")
        attrs = {
            'moving_rate': self._moving_rate,
            'bit_length': self._quant_bits,
            'is_test': not self.training
        }
        inputs = {"X": [input], "InScale": [self.scale]}
        quant_out = self._helper.create_variable(
            name="{}.quant_dequant".format(input.name),
            dtype=input.dtype,
            type=core.VarDesc.VarType.LOD_TENSOR,
            persistable=False,
            stop_gradient=False)
        outputs = {"Out": [quant_out], "OutScale": [self.scale]}

        if self.training:
            inputs['InState'] = [self.state]
            inputs['InAccum'] = [self.accum]
            outputs['OutState'] = [self.state]
            outputs['OutAccum'] = [self.accum]

        self._helper.append_op(
            type="fake_quantize_dequantize_moving_average_abs_max",
            inputs=inputs,
            outputs=outputs,
            attrs=attrs)

        return quant_out


class QuantizedConv2D(layers.Layer):
    def __init__(self, layer, weight_bits=8, activation_bits=8,
                 moving_rate=0.9):
        super(QuantizedConv2D, self).__init__()
        # For Conv2D
        self._groups = getattr(layer, '_groups')
        self._stride = getattr(layer, '_stride')
        self._padding = getattr(layer, '_padding')
        self._dilation = getattr(layer, '_dilation')
        self._act = getattr(layer, '_act')
        self._use_cudnn = getattr(layer, '_use_cudnn')
        self._dtype = getattr(layer, '_dtype')
        self._l_type = getattr(layer, '_l_type')
        self.weight = getattr(layer, 'weight')
        self.bias = getattr(layer, 'bias')
        # For FakeQuant
        self._fake_quant_input = FakeQuant(layer.full_name(), moving_rate,
                                           activation_bits, self._dtype)

        self._fake_quant_weight = FakeQuant(self.weight.name, moving_rate,
                                            weight_bits, self._dtype)

    def forward(self, input):
        quant_input = self._fake_quant_input(input)
        quant_weight = self._fake_quant_weight(self.weight)

        if in_dygraph_mode() and self._l_type == 'conv2d':
            attrs = ('strides', self._stride, 'paddings', self._padding,
                     'dilations', self._dilation, 'groups', self._groups
                     if self._groups else 1, 'use_cudnn', self._use_cudnn)
            pre_bias = core.ops.conv2d(quant_input, quant_weight, *attrs)

            pre_act = dygraph_utils._append_bias_in_dygraph(pre_bias, self.bias,
                                                            1)
            return dygraph_utils._append_activation_in_dygraph(pre_act,
                                                               self._act)
        check_variable_and_dtype(quant_input, 'input',
                                 ['float16', 'float32', 'float64'],
                                 'QuantizedConv2D')
        attrs = {
            'strides': self._stride,
            'paddings': self._padding,
            'dilations': self._dilation,
            'groups': self._groups if self._groups else 1,
            'use_cudnn': self._use_cudnn,
            'use_mkldnn': False,
        }
        pre_bias = self._helper.create_variable_for_type_inference(
            dtype=self._dtype)

        self._helper.append_op(
            type=self._l_type,
            inputs={
                'Input': quant_input,
                'Filter': quant_weight,
            },
            outputs={"Output": pre_bias},
            attrs=attrs)

        if self.bias is not None:
            pre_act = self._helper.create_variable_for_type_inference(
                dtype=self._dtype)
            self._helper.append_op(
                type='elementwise_add',
                inputs={'X': [pre_bias],
                        'Y': [self.bias]},
                outputs={'Out': [pre_act]},
                attrs={'axis': 1})
        else:
            pre_act = pre_bias

        return self._helper.append_activation(pre_act, act=self._act)


class QuantizedLinear(layers.Layer):
    def __init__(self, layer, weight_bits=8, activation_bits=8,
                 moving_rate=0.9):
        super(QuantizedLinear, self).__init__()
        # For Linear
        self._act = getattr(layer, '_act')
        self._dtype = getattr(layer, '_dtype')
        self.weight = getattr(layer, 'weight')
        self.bias = getattr(layer, 'bias')
        # For FakeQuant
        self._fake_quant_input = FakeQuant(layer.full_name(), moving_rate,
                                           activation_bits, self._dtype)

        self._fake_quant_weight = FakeQuant(self.weight.name, moving_rate,
                                            weight_bits, self._dtype)

    def forward(self, input):
        quant_input = self._fake_quant_input(input)
        quant_weight = self._fake_quant_weight(self.weight)
        if in_dygraph_mode():
            pre_bias = _varbase_creator(dtype=input.dtype)
            core.ops.matmul(quant_input, quant_weight, pre_bias, 'transpose_X',
                            False, 'transpose_Y', False, "alpha", 1)
            pre_act = dygraph_utils._append_bias_in_dygraph(
                pre_bias, self.bias, axis=len(input.shape) - 1)

            return dygraph_utils._append_activation_in_dygraph(pre_act,
                                                               self._act)

        check_variable_and_dtype(input, 'input',
                                 ['float16', 'float32', 'float64'],
                                 "QuantizedLinear")
        attrs = {
            "transpose_X": False,
            "transpose_Y": False,
            "alpha": 1,
        }
        inputs = {"X": [quant_input], "Y": [quant_weight]}
        mul_out = self._helper.create_variable_for_type_inference(self._dtype)

        self._helper.append_op(
            type="matmul",
            inputs=inputs,
            outputs={"Out": [mul_out]},
            attrs=attrs)
        if self.bias is not None:
            pre_activation = self._helper.create_variable_for_type_inference(
                dtype=self._dtype)
            self._helper.append_op(
                type='elementwise_add',
                inputs={'X': [mul_out],
                        'Y': [self.bias]},
                outputs={'Out': [pre_activation]},
                attrs={'axis': len(input.shape) - 1})
        else:
            pre_activation = mul_out
        return self._helper.append_activation(pre_activation, act=self._act)
