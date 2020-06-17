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
import numpy as np
from paddle.fluid import dygraph
from paddle.fluid.dygraph.nn import Conv2D
from paddle.fluid.dygraph.nn import Linear
from paddle.fluid.log_helper import get_logger
from .quant_nn import QuantizedConv2D
from .quant_nn import QuantizedLinear

__all__ = ['DygraphQuantAware']

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')


class DygraphQuantAware(object):
    def __init__(self,
                 weight_bits=8,
                 activation_bits=8,
                 weight_quantize_type='abs_max',
                 activation_quantize_type='moving_average_abs_max',
                 moving_rate=0.9,
                 quantizable_layer_type=['Conv2D', 'Linear']):
        super(DygraphQuantAware, self).__init__()
        self._weight_bits = weight_bits
        self._activation_bits = activation_bits
        self._moving_rate = moving_rate

        quant_type = {'abs_max', 'moving_average_abs_max'}
        if activation_quantize_type not in quant_type:
            raise ValueError(
                "Unknown activation_quantize_type : '%s'. It can only be "
                "'abs_max' or 'moving_average_abs_max' now." %
                (str(activation_quantize_type)))
        if weight_quantize_type not in quant_type:
            raise ValueError(
                "Unknown weight_quantize_type: '%s'. It can only be "
                "'abs_max' or 'moving_average_abs_max' now." %
                (str(weight_quantize_type)))
        self._activation_quantize_type = activation_quantize_type
        self._weight_quantize_type = weight_quantize_type

        self._quant_layers_map = {'Conv2D': Conv2D, 'Linear': Linear}
        self._quantizable_layer_type = tuple(
            self._quant_layers_map[layer]
            if layer in self._quant_layers_map else layer
            for layer in quantizable_layer_type)
        for layer in self._quantizable_layer_type:
            assert not isinstance(
                layer, str), "{} is unspported to be quantized.".format(layer)
        self._translator = dygraph.ProgramTranslator()
        self._translator.enable_declarative = False

    def quantize(self, model):
        for name, layer in model.named_sublayers():
            if not isinstance(layer, self._quantizable_layer_type):
                continue

            scopes = name.split('.')
            target = scopes[-1]
            obj = model
            parent = model
            for i in range(len(scopes) - 1):
                obj = getattr(parent, scopes[i])
                parent = obj

            quant_layer = self._get_quantized_counterpart(layer)
            setattr(obj, target, quant_layer)

    def save_infer_quant_model(self,
                               dirname,
                               model,
                               input_shape,
                               input_dtype,
                               feed,
                               fetch,
                               append_batch_size=True):
        assert isinstance(
            input_shape, list), "The parameter `input_shape` shoubld be a list."
        assert isinstance(
            input_dtype, list), "The parameter `input_dtype` shoubld be a list."
        assert isinstance(feed, list), "The parameter `feed` shoubld be a list."
        assert isinstance(fetch,
                          list), "The parameter `fetch` shoubld be a list."
        assert len(input_shape) == len(
            input_dtype
        ), "The length of input_shape should be equal to  input_dtype's."
        assert len(input_dtype) == len(
            feed), "The length of input_shape should be equal to  feed's."

        with dygraph.guard():
            self._translator.enable_declarative = True
            model.eval()
            input_vars = []
            for shape, dtype in zip(input_shape, input_dtype):
                raw_data = np.random.random(shape)
                input_data = raw_data[np.newaxis, :].astype(
                    dtype) if append_batch_size else raw_data.astype(dtype)
                input_var = dygraph.to_variable(input_data)
                input_vars.append(input_var)
            model(*input_vars)
        self._translator.save_inference_model(dirname, feed, fetch)

    def _get_quantized_counterpart(self, layer):
        quant_layers = tuple(self._quant_layers_map.values())
        quantized_counterpart = tuple('Quantized' + k
                                      for k in self._quant_layers_map.keys())

        predicate = lambda value: isinstance(layer, value)
        index_generator = (i for i, v in enumerate(quant_layers)
                           if predicate(v))

        try:
            index = next(index_generator)
        except StopIteration:
            _logger.fatal("The layer {} is unsupported to be quantized.".format(
                layer.full_name()))
            sys.exit(-1)

        module = sys.modules[__name__]
        quantized_layer = getattr(module, quantized_counterpart[index])(
            layer, self._weight_bits, self._activation_bits, self._moving_rate,
            self._weight_quantize_type, self._activation_quantize_type)
        return quantized_layer
