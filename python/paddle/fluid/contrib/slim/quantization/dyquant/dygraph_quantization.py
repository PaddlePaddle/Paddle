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
from .....dygraph.nn import Conv2D, Linear
from .....log_helper import get_logger
from .quant_nn import FakeQuant, QuantizedConv2D, QuantizedLinear

__all__ = ['quantize_qat']

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')
_quant_layers_map = {'Conv2D': Conv2D, 'Linear': Linear}


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

    module = sys.modules[__name__]
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
