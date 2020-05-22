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
import six
from ....dygraph import layers
from ....layer_helper import LayerHelper
from ....dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear
from ....dygraph.base import to_variable
from .... import core
from .... import framework
from .... import unique_name
from ....param_attr import ParamAttr
from ....framework import _varbase_creator
from ....initializer import Constant
from ....log_helper import get_logger

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

        out, out_scale, _, _ = core.ops.fake_quantize_dequantize_moving_average_abs_max(
            input, self.scale, self.accum, self.state, quant_out, self.scale,
            self.state, self.accum, *attrs)
        return out, out_scale


def quantize_qat(model,
                 weight_bits=8,
                 activation_bits=8,
                 moving_rate=0.9,
                 quantizable_layer_type=['Conv2D', 'Linear']):
    def _add_fake_quant(layer, input):
        _logger.info("------------------------------------------------{}".
                     format(layer.full_name()))
        is_test = not layer.training
        quant_inputs = []

        # add fake_quant for activation inputs
        for layer_in in input:
            _logger.info("***********************************************{}".
                         format(layer_in.name))
            # prevent pruning the fake_quant_grad op
            layer_in.stop_gradient = False
            dtype = 'float64' if layer_in.dtype == core.VarDesc.VarType.FP64 else 'float32'
            fake_quant_input = FakeQuant(layer_in.name, moving_rate,
                                         activation_bits, dtype, is_test)
            quant_input, _ = fake_quant_input(layer_in)
            quant_inputs.append(quant_input)

        # add fake_quant for weight inputs
        params = layer._parameters.items()
        for key, param in params:
            _logger.info("key-------------------------:{}".format(key))
            _logger.info("name-------------------------:{}".format(param.name))
            if key.find('weight') != -1:
                dtype = 'float64' if param.dtype == core.VarDesc.VarType.FP64 else 'float32'
                fake_quant_weight = FakeQuant(param.name, moving_rate,
                                              weight_bits, dtype, is_test)
                quant_param, _ = fake_quant_weight(param)
                layer._parameters[key] = quant_param

        return tuple(quant_inputs)

    quant_layers = tuple(_quant_layers_map[layer]
                         if layer in _quant_layers_map else layer
                         for layer in quantizable_layer_type)
    for layer in quant_layers:
        assert not isinstance(
            layer, str), "{} is unspported to be quantized.".format(layer)

    processed_layers = filter(lambda layer: isinstance(layer, quant_layers),
                              model.sublayers())
    for layer in processed_layers:
        layer.register_forward_pre_hook(_add_fake_quant)
