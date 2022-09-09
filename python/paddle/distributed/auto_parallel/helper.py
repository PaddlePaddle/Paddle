# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from collections import defaultdict

import paddle

from paddle.nn import Layer
from paddle.jit import to_static, not_to_static
from paddle.distributed.utils import get_logger
from paddle.fluid.framework import Operator, Parameter, _non_static_mode
from paddle.fluid.framework import program_guard
from paddle.fluid.executor import global_scope
from paddle.fluid.dygraph.dygraph_to_static.program_translator import StaticFunction

from .utils import to_list
from .converter import Converter


class ProxyLayer(Layer):
    """
    ProxyLayer implements all logic for converting dygraph model into
    static Program IR. Meanwhile, it provides conviential interfaces for
    auto parallel to visit feed/fetch/loss/metric variables.
    """

    def __init__(self, layer, loss_func, metrics):
        super(ProxyLayer, self).__init__()
        # NOTE: All verify logics are finished in Engine.Prepare
        self.inner_layer = layer
        self.loss_func = loss_func
        self.metrics = metrics
        # train / eval / predict
        self.mode = None

        # generated program vars
        self._input_vars = defaultdict(list)
        self._label_vars = defaultdict(list)
        self._output_vars = defaultdict(list)
        self._loss_vars = defaultdict(list)
        self._metric_vars = defaultdict(list)

    def _train(self, inputs, labels):
        """
        Train process of inner_layer with forward/loss/metric logic.
        """
        # step 1. save feed variables of Program
        mode = 'train'
        self._input_vars[mode] = inputs
        self._label_vars[mode] = labels

        # step 2. call inner_layer.forward
        self._output_vars[mode] = self.inner_layer(*inputs)

        # step 3. calculate loss if needed
        new_inputs = self._prepare(self.output_vars, labels)
        self._loss_vars[mode] = self.call_loss(new_inputs)

        # step 4. calculate metrics if needed
        self._metric_vars[mode] = self.call_metrics(new_inputs)

    def _eval(self, inputs, labels):
        """
        Evaluate process of inner_layer with forward/loss/metric logic.
        """
        # TODO(dev): we can reuse codes with self._train after making
        # sure if they can.

        # step 1. save feed variables of Program
        mode = 'eval'
        self._input_vars[mode] = inputs
        self._label_vars[mode] = labels

        # step 2. call inner_layer.forward
        self._output_vars[mode] = self.inner_layer(*inputs)

        # step 3. calculate loss if needed
        new_inputs = self._prepare(self.output_vars, labels)
        self._loss_vars[mode] = self.call_loss(new_inputs)

        # step 4. calculate metrics if needed
        self._metric_vars[mode] = self.call_metrics(new_inputs)

    def _predict(self, inputs, labels):
        """
        Predict process of inner_layer with forward logic.
        """
        # step 1. save feed variables of Program
        mode = 'predict'
        self._input_vars[mode] = inputs
        self._label_vars[mode] = labels

        # step 2. call inner_layer.forward
        self._output_vars[mode] = self.inner_layer(*inputs)

    @not_to_static
    def _prepare(self, outputs, labels):
        """
        Concat outputs and labels as a single list

        NOTE(dev): We use @not_to_static to avoid AST Analysis.
        """
        return to_list(outputs) + to_list(labels)

    def call_loss(self, inputs):
        """
        Apply Loss Function on outputs and labels.

        Args:
            inputs: List[Variable]

        Returns: List[Variable]
        """
        res = []
        if self.loss_func is not None:
            res = self.loss_func(*inputs)
        return res

    def call_metrics(self, inputs):
        """
        Apply Metrics Function on outputs and labels.

        Args:
            inputs: List[Variable]

        Returns: List[Variable]
        """
        outs = []
        for metric in self.metrics:
            outs.extend(metric.compute(*inputs))

        return outs

    def set_mode(self, mode):
        self.mode = mode
        self.training = mode == 'train'

    def clone(self):
        return ProxyLayer(self.inner_layer, self.loss_func, self.metrics)

    @property
    def input_vars(self):
        return self._input_vars[self.mode]

    @property
    def label_vars(self):
        return self._label_vars[self.mode]

    @property
    def output_vars(self):
        return self._output_vars[self.mode]

    @property
    def loss_vars(self):
        return self._loss_vars[self.mode]

    @property
    def metric_vars(self):
        return self._metric_vars[self.mode]

    @property
    def startup_program(self):
        return self.inner_layer._startup_program()


class BuildInfo:

    def __init__(self):
        self.clear()

    def has_cache(self, mode, update=False):
        is_cache = self.states[mode]
        if update:
            self.cache(mode)
        return is_cache

    def cache(self, mode):
        self.states[mode] = True

    def clear(self):
        self.states = defaultdict(bool)


class ProgramHelper(object):
    """
    A Helper class for Engine to provides different Program IR according specified 'mode'.
    """

    def __init__(self, layer, loss_func, metrics, inputs_spec, labels_spec):
        # original model config information
        # TODO(Aurelius84): Implenet append_backward and optimizer in ProxyLayer
        # after distribute engine satisify basic condition.
        self.proxy_layer = ProxyLayer(layer, loss_func, metrics)
        self.inputs_spec = inputs_spec
        self.labels_spec = labels_spec

        self.build_info = BuildInfo()
        self._logger = get_logger(logging.INFO)
        self.lazy_init = False

    def reset(self):
        """
        Reset all state of current Object.
        """
        self.build_info.clear()
        self.proxy_layer = self.proxy_layer.clone()

    def build_program(self, mode):
        """
        Convert dygraph model into static Program IR.
        """
        assert mode in ['train', 'eval', 'predict']
        self.proxy_layer.set_mode(mode)
        # skip if we has already built program.
        if self.build_info.has_cache(mode, True):
            self._logger.info(
                "Already build program with mode = %s, use cached program." %
                mode)
            return

        self._logger.info("start to build program for mode = %s." % mode)
        input_spec = [self.inputs_spec, self.labels_spec]
        static_func = to_static(self.static_func(), input_spec=input_spec)

        func_name = '_' + mode
        setattr(self.proxy_layer, func_name, static_func)

        # NOTE(dev): Because @to_static is a Lazy mechanism, so we explicitly call this to trigger
        # generating Program IR immediately.
        getattr(self.proxy_layer, func_name).concrete_program

        self._build_startup_program()

    def _build_startup_program(self):
        """
        Create and Sync parameters into startup program.
        """
        if len(self.startup_program.global_block().ops) > 1:
            self.lazy_init = True
            return
        for param in self.concrete_program.parameters:
            Parameter(name=param.name,
                      desc=param,
                      type=param.type,
                      shape=param.shape,
                      dtype=param.dtype,
                      stop_gradient=param.stop_gradient,
                      block=self.startup_program.global_block())

    def apply_optimizer(self, optimizer):
        """
        Append backward and generate optimizer operations.
        """
        self._verify_optimizer(optimizer)
        self._logger.info("start to apply optimizer: %s ",
                          type(optimizer).__name__)
        # clear optimizer parameters
        original_params = optimizer._parameter_list
        optimizer._parameter_list = None
        with program_guard(self.main_program, self.startup_program):
            res = optimizer.minimize(self.loss_vars[0])

        # restore optimizer parameters
        optimizer._parameter_list = original_params
        return res

    def _verify_optimizer(self, optimizer):
        assert optimizer is not None
        assert hasattr(optimizer,
                       "minimize"), "Optimizer must have minimize() method."
        assert self.proxy_layer.mode == 'train', "Required mode == 'train', but received '%s'" % self.proxy_layer.mode
        assert len(
            self.loss_vars
        ) == 1, "Required len(loss_vars) == 1, but received len(loss_vars) = %s" % len(
            self.loss_vars)

    def to(self, mode):
        """
        Switch underly proxy layer mode into target mode.
        """
        assert mode in ['train', 'eval', 'predict']
        func = getattr(self.proxy_layer, '_' + mode)
        assert isinstance(
            func, StaticFunction), "Please call build_program(mode) firstly."
        self.proxy_layer.set_mode(mode)

    def static_func(self):
        """
        Return StaticFunction instance with underly target mode.
        """
        assert self.proxy_layer.mode in [
            'train', 'eval', 'predict'
        ], "Please call build_program(mode) firstly."
        func_name = '_' + self.proxy_layer.mode
        return getattr(self.proxy_layer, func_name)

    def init(self, main_program, place, dist_context):
        if self.lazy_init:
            return
        for param in self.concrete_program.parameters:
            # create var in scope and share parameters to scope
            if param.name not in main_program.global_block().vars:
                continue
            # get param_var's dist_attr
            var = main_program.global_block().vars[param.name]
            var_dist_attr = dist_context.get_tensor_dist_attr_for_program(var)
            dist_attr = {
                "dims_mapping": var_dist_attr.dims_mapping,
                "process_shape": var_dist_attr.process_mesh.topology,
                "process_group": var_dist_attr.process_mesh.processes
            }
            # slice param_value with dist_attr
            # share sliced_param_value with param_tensor in global_scope
            param_tensor = global_scope().var(param.name).get_tensor()
            sliced_param = Converter.slice_with_dist_attr(
                param.numpy(), dist_attr)
            param_tensor.set(sliced_param, place)

    @property
    def concrete_program(self):
        return self.static_func().concrete_program

    @property
    def main_program(self):
        return self.concrete_program.main_program

    @property
    def startup_program(self):
        try:
            return self.proxy_layer.startup_program
        except Exception as err:
            if isinstance(err, AssertionError):
                return self.concrete_program.startup_program
            raise err

    @property
    def input_vars(self):
        return to_list(self.proxy_layer.input_vars)

    @property
    def output_vars(self):
        return to_list(self.proxy_layer.output_vars)

    @property
    def label_vars(self):
        return to_list(self.proxy_layer.label_vars)

    @property
    def loss_vars(self):
        return to_list(self.proxy_layer.loss_vars)

    @property
    def metric_vars(self):
        return to_list(self.proxy_layer.metric_vars)
