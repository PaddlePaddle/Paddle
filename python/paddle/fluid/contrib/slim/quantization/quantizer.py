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
from ....contrib import GraphQuantizeTranspiler

__all__ = ['Quantizer', 'StaticQuantizer', 'DynamicQuantizer']


class Quantizer(object):
    """
    Base class of all quantizers.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self._transpiler = None

    @abstractmethod
    def quantize(self, graph, program_exe, scope):
        assert self._transpiler is not None
        need_inited = self._transpiler.training_transpile(graph)
        init_program = Program()
        for var, initializer in need_inited.iteritems():
            init_program.global_block()._clone_variable(var)
            initializer(var, init_program.global_block())
        program_exe.run(program=init_program, scope=scope)

    @abstractmethod
    def convert_model(self,
                      graph,
                      place,
                      scope,
                      feeds,
                      fetches,
                      dirname=None,
                      exe=None,
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
        test_graph = graph.clone(for_test=True).prune(feeds, fetches)
        self._transpiler.freeze_graph(test_graph, place, scope)
        if save_as_int8:
            self._transpiler.convert_to_int8(test_graph, place, scope)
        if target_device in convert_funcs.keys():
            convert_funcs[target_device](test_graph)
        else:
            raise ValueError("The device is not supported!")
        if dirname is None:
            print("The save path is None, so the model is not saved!")
        else:
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            save_inference_graph_model(
                dirname, [feed_var.name for feed_var in feeds],
                [fetch_var.name for fetch_var in fetches], exe, test_graph)
        return test_graph


class StaticQuantizer(Quantizer):
    """
    The scale of activation is calculated during training and used for inference.
    """

    def __init__(self, weight_bits=8, activation_bits=8, window_size=10000):
        super(StaticQuantizer, self).__init__()
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.window_size = window_size
        self._transpiler = GraphQuantizeTranspiler(
            weight_bits=weight_bits,
            activation_bits=activation_bits,
            activation_quantize_type='range_abs_max',
            weight_quantize_type='abs_max',
            window_size=window_size)

    def quantize(self, graph, program_exe, scope):
        super(StaticQuantizer, self).quantize(graph, program_exe, scope)

    def convert_model(self,
                      graph,
                      place,
                      scope,
                      feeds,
                      fetches,
                      dirname=None,
                      exe=None,
                      target_device='mobile',
                      save_as_int8=True):
        return super(StaticQuantizer, self).convert_model(
            graph, place, scope, feeds, fetches, dirname, exe, target_device,
            save_as_int8)


class DynamicQuantizer(Quantizer):
    """
    The scale of activation will be calculated on each mini-batch during inference.
    """

    def __init__(self, weight_bits=8, activation_bits=8):
        super(DynamicQuantizer, self).__init__()
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self._transpiler = GraphQuantizeTranspiler(
            weight_bits=weight_bits,
            activation_bits=activation_bits,
            activation_quantize_type='abs_max',
            weight_quantize_type='abs_max')

    def quantize(self, graph, program_exe, scope):
        super(DynamicQuantizer, self).quantize(graph, program_exe, scope)

    def convert_model(self,
                      graph,
                      place,
                      scope,
                      feeds,
                      fetches,
                      dirname=None,
                      exe=None,
                      target_device='mobile',
                      save_as_int8=True):
        return super(DynamicQuantizer, self).convert_model(
            graph, place, scope, feeds, fetches, dirname, exe, target_device,
            save_as_int8)
