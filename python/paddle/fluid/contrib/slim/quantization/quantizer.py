# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import os
import abc
from abc import abstractmethod
from ....framework import Program
from ..graph import ImitationGraph
from ..graph import save_inference_graph_model
from .quantization_performer import QuantizationPerformer

__all__ = ['Quantizer', 'StaticQuantizer', 'DynamicQuantizer']


class Quantizer(object):
    """
    Base class of all quantizers.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self._performer = None

    @abstractmethod
    def quantize(self, graph, place):
        assert self._performer is not None
        self._performer.quantize_transform(graph, place)

    @abstractmethod
    def convert_model(self,
                      graph,
                      place,
                      dirname=None,
                      target_device='mobile',
                      save_as_int8=True):
        def _convert_for_mobile(_graph):
            for op in list(_graph.all_ops()):
                if op.type == "fake_dequantize_max_abs":
                    op.desc.set_type("dequantize")
                if op.type == "fake_quantize_abs_max" or \
                        op.type == "fake_quantize_range_abs_max":
                    op.desc.set_type("quantize")

        def _convert_for_server(_graph):
            pass

        convert_funcs = {
            'mobile': _convert_for_mobile,
            'server': _convert_for_server
        }
        self._performer.freeze_graph(graph, place)
        if save_as_int8:
            self._performer.convert_to_int8(graph, place)
        if target_device in convert_funcs.keys():
            convert_funcs[target_device](graph)
        else:
            raise ValueError("The device is not supported!")
        if dirname is None:
            print("The save path is None, so the model is not saved!")
        else:
            feeds = graph.in_nodes.values()
            fetches = graph.out_nodes.values()
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            save_inference_graph_model(dirname, feeds, fetches, place, graph)
        return graph


class StaticQuantizer(Quantizer):
    """
    The scale of activation is calculated during training and used for inference.
    """

    def __init__(self, weight_bits=8, activation_bits=8, window_size=10000):
        super(StaticQuantizer, self).__init__()
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.window_size = window_size
        self._performer = QuantizationPerformer(
            weight_bits=weight_bits,
            activation_bits=activation_bits,
            activation_quantize_type='range_abs_max',
            weight_quantize_type='abs_max',
            window_size=window_size)

    def quantize(self, graph, place):
        super(StaticQuantizer, self).quantize(graph, place)

    def convert_model(self,
                      graph,
                      place,
                      dirname=None,
                      target_device='mobile',
                      save_as_int8=True):
        return super(StaticQuantizer, self).convert_model(
            graph, place, dirname, target_device, save_as_int8)


class DynamicQuantizer(Quantizer):
    """
    The scale of activation will be calculated on each mini-batch during inference.
    """

    def __init__(self, weight_bits=8, activation_bits=8):
        super(DynamicQuantizer, self).__init__()
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self._performer = QuantizationPerformer(
            weight_bits=weight_bits,
            activation_bits=activation_bits,
            activation_quantize_type='abs_max',
            weight_quantize_type='abs_max')

    def quantize(self, graph, place):
        super(DynamicQuantizer, self).quantize(graph, place)

    def convert_model(self,
                      graph,
                      place,
                      dirname=None,
                      target_device='mobile',
                      save_as_int8=True):
        return super(DynamicQuantizer, self).convert_model(
            graph, place, dirname, target_device, save_as_int8)
