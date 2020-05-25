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
import sys
from ....dygraph import layers
from ....dygraph.nn import Conv2D, Linear
from .... import core
from .... import dygraph_utils
from .... import unique_name
from ....param_attr import ParamAttr
from ....framework import _varbase_creator, in_dygraph_mode
from ....initializer import Constant
from ....log_helper import get_logger
from ....data_feeder import check_variable_and_dtype

__all__ = ['quantize_qat']

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')
_quant_layers_map = {'Conv2D': Conv2D, 'Linear': Linear}


class FakeQuant(layers.Layer):
    def __init__(self,
                 name=None,
                 moving_rate=0.9,
                 quant_bits=8,
                 dtype='float32',
                 is_test=False):
        super(FakeQuant, self).__init__()
        self._moving_rate = moving_rate
        self._quant_bits = quant_bits
        self._is_test = is_test
        scale_attr = ParamAttr(name="{}.quant_dequant.scale".format(name)
                               if name else None)
        self.scale = self.create_parameter(
            shape=[1],
            attr=scale_attr,
            dtype=dtype,
            is_bias=False,
            default_initializer=Constant(0.001))

        if not self._is_test:
            state_attr = ParamAttr(
                name=unique_name.generate('quant_dequant.state'))
            self.state = self.create_parameter(
                shape=[1],
                attr=state_attr,
                dtype=dtype,
                is_bias=False,
                default_initializer=Constant(1))
            accum_attr = ParamAttr(
                name=unique_name.generate('quant_dequant.accum'))
            self.accum = self.create_parameter(
                shape=[1],
                attr=accum_attr,
                dtype=dtype,
                is_bias=False,
                default_initializer=Constant(1))
        else:
            self.state = None
            self.accum = None

    def forward(self, input):
        quant_out = _varbase_creator(
            type=input.type,
            name="{}.quant_dequant".format(input.name),
            shape=input.shape,
            dtype=input.dtype,
            persistable=False)

        attrs = ('moving_rate', self._moving_rate, 'bit_length',
                 self._quant_bits, 'is_test', self._is_test)

        out, _, _, _ = core.ops.fake_quantize_dequantize_moving_average_abs_max(
            input, self.scale, self.accum, self.state, quant_out, self.scale,
            self.state, self.accum, *attrs)
        return out


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
        is_test = not layer.training
        self._fake_quant_input = FakeQuant(layer.full_name(), moving_rate,
                                           activation_bits, self._dtype,
                                           is_test)

        self._fake_quant_weight = FakeQuant(self.weight.name, moving_rate,
                                            weight_bits, self._dtype, is_test)

    def forward(self, input):
        quant_input = self._fake_quant_input(input)
        quant_weight = self._fake_quant_input(self.weight)

        if self._l_type == 'conv2d':
            attrs = ('strides', self._stride, 'paddings', self._padding,
                     'dilations', self._dilation, 'groups', self._groups
                     if self._groups else 1, 'use_cudnn', self._use_cudnn)
            pre_bias = core.ops.conv2d(quant_input, quant_weight, *attrs)

            pre_act = dygraph_utils._append_bias_in_dygraph(pre_bias, self.bias,
                                                            1)
            return dygraph_utils._append_activation_in_dygraph(pre_act,
                                                               self._act)
        attrs = {
            'strides': self._stride,
            'paddings': self._padding,
            'dilations': self._dilation,
            'groups': self._groups if self._groups else 1,
            'use_cudnn': self._use_cudnn,
            'use_mkldnn': False,
        }

        check_variable_and_dtype(quant_input, 'input',
                                 ['float16', 'float32', 'float64'], 'Conv2D')
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
        is_test = not layer.training
        self._fake_quant_input = FakeQuant(layer.full_name(), moving_rate,
                                           activation_bits, self._dtype,
                                           is_test)

        self._fake_quant_weight = FakeQuant(self.weight.name, moving_rate,
                                            weight_bits, self._dtype, is_test)

    def forward(self, input):
        quant_input = self._fake_quant_input(input)
        quant_weight = self._fake_quant_input(self.weight)
        pre_bias = _varbase_creator(dtype=input.dtype)
        core.ops.matmul(quant_input, quant_weight, pre_bias, 'transpose_X',
                        False, 'transpose_Y', False, "alpha", 1)
        pre_act = dygraph_utils._append_bias_in_dygraph(
            pre_bias, self.bias, axis=len(input.shape) - 1)

        return dygraph_utils._append_activation_in_dygraph(pre_act, self._act)


def _get_quantized_counterpart(layer,
                               weight_bits=8,
                               activation_bits=8,
                               moving_rate=0.9):
    quant_layers = tuple(_quant_layers_map.values())
    quantized_counterpart = tuple('Quantized' + k
                                  for k in _quant_layers_map.keys())

    predicate = lambda value: isinstance(layer, value)
    index_generator = (i for i, v in enumerate(quant_layers) if predicate(v))

    try:
        index = next(index_generator)
    except StopIteration:
        _logger.fatal("Don't find any quantized layer for {}.".format(
            layer.full_name()))
        sys.exit(-1)

    module = __import__(__name__)
    quantized_layer = getattr(module, quantized_counterpart[index])(
        layer, weight_bits, activation_bits, moving_rate)
    return quantized_layer


def quantize_qat(model,
                 weight_bits=8,
                 activation_bits=8,
                 moving_rate=0.9,
                 quantizable_layer_type=['Conv2D', 'Linear']):
    quant_layers = tuple(_quant_layers_map[layer]
                         if layer in _quant_layers_map else layer
                         for layer in quantizable_layer_type)
    for layer in quant_layers:
        assert not isinstance(
            layer, str), "{} is unspported to be quantized.".format(layer)

    for name, layer in model.named_sublayers():
        if not isinstance(layer, quant_layers):
            continue

        scopes = name.split('.')
        target = scopes[-1]
        obj = model
        parent = model
        for i in range(len(scopes) - 1):
            obj = getattr(parent, scopes[i])
            parent = obj

        quant_layer = _get_quantized_counterpart(layer, weight_bits,
                                                 activation_bits, moving_rate)
        setattr(obj, target, quant_layer)
