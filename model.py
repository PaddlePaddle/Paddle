# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import

import inspect
import os
import pickle
from collections import OrderedDict

import numpy as np

from paddle import fluid
from paddle.fluid.framework import in_dygraph_mode, Variable
from paddle.fluid.executor import global_scope
from paddle.fluid.io import is_belong_to_optimizer
from paddle.fluid.dygraph.base import to_variable

__all__ = ['Model', 'Loss', 'CrossEntropy', 'Input']


def to_list(value):
    if value is None:
        return value
    if isinstance(value, (list, tuple)):
        return value
    return [value]


def to_numpy(var):
    assert isinstance(var, (Variable, fluid.core.VarBase)), "not a variable"
    if isinstance(var, fluid.core.VarBase):
        return var.numpy()
    t = global_scope().find_var(var.name).get_tensor()
    return np.array(t)


def extract_args(func):
    if hasattr(inspect, 'getfullargspec'):
        return inspect.getfullargspec(func)[0]
    else:
        return inspect.getargspec(func)[0]


class Input(fluid.dygraph.Layer):
    def __init__(self, shape=None, dtype=None, name=None):
        self.shape = shape
        self.dtype = dtype
        self.name = name

    def forward(self):
        return fluid.data(self.name, shape=self.shape, dtype=self.dtype)


class Loss(object):
    def __init__(self, average=True):
        super(Loss, self).__init__()
        self.average = average

    def forward(self, outputs, labels):
        raise NotImplementedError()

    def __call__(self, outputs, labels):
        labels = to_list(labels)
        if in_dygraph_mode():
            labels = [to_variable(l) for l in labels]
        losses = to_list(self.forward(to_list(outputs), labels))
        if self.average:
            losses = [fluid.layers.reduce_mean(l) for l in losses]
        else:
            losses = [fluid.layers.reduce_sum(l) for l in losses]
        return losses


class CrossEntropy(Loss):
    def __init__(self, average=True):
        super(CrossEntropy, self).__init__()

    def forward(self, outputs, labels):
        return [
            fluid.layers.cross_entropy(o, l) for o, l in zip(outputs, labels)
        ]


class StaticGraphAdapter(object):
    def __init__(self, model):
        super(StaticGraphAdapter, self).__init__()
        self.model = model
        # with `_build_once` gone, parameters are now created in `__init__`
        # so we need to keep track of the parameters already created
        self._startup_prog = fluid.default_startup_program()
        self._orig_prog = fluid.default_main_program()

        self._label_vars = {}  # label variables
        self._input_vars = {}  # label variables
        self._endpoints = {}
        self._loss_endpoint = None
        self._executor = None
        self._progs = {}
        self._compiled_progs = {}

    @property
    def mode(self):
        return self.model.mode

    @mode.setter
    def mode(self, value):
        self.model.mode = value

    def train(self, inputs, labels=None):
        assert self.model._optimizer, \
            "model not ready, please call `model.prepare()` first"
        self.mode = 'train'
        return self._run(inputs, labels)

    def eval(self, inputs, labels=None):
        self.mode = 'eval'
        return self._run(inputs, labels)

    def test(self, inputs):
        self.mode = 'test'
        return self._run(inputs, None)

    def parameters(self, *args, **kwargs):
        return None

    def save(self, path):
        def _save(state, path):
            if not state:
                return
            state = {
                k: to_numpy(v) if isinstance(v, Variable) else v
                for k, v in state.items()
            }
            with open(path, 'wb') as f:
                pickle.dump(state, f)

        base = os.path.basename(path)
        assert base != "", "path should be of 'dirname/filename' format"
        dir_name = os.path.dirname(path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)
        param_path = path + ".pdparams"
        _save(self.model.state_dict(), param_path)
        prog = self._progs.get('train', None)
        if prog is None or self.model._optimizer is None:
            return
        # XXX `optimizer.state_dict()` only work in dygraph mode
        optim_path = path + ".pdopt"
        optim = {
            p.name: p
            for p in filter(is_belong_to_optimizer, prog.list_vars())
        }
        if not optim:
            return

        _save(optim, optim_path)

    def load(self, path):
        def _load(path):
            if not os.path.exists(path):
                return
            with open(path, 'rb') as f:
                return pickle.load(f)

        param_path = path + ".pdparams"
        param_state = _load(param_path)
        assert param_state, "failed to load parameters, please check path"

        if self._executor is None:
            executor = fluid.Executor(fluid.CPUPlace())._default_executor
        else:
            executor = self._executor._default_executor

        fluid.core._create_loaded_parameter(
            list(self.model.state_dict().values()), global_scope(), executor)

        for key, var in self.model.state_dict().items():
            assert key in param_state, \
                "parameter [{}] is not found in model file [{}]".format(
                    key, param_path)
            self._set_var(var, param_state[key])

        # FIXME what if a different optimizer is used?
        if not self.model._optimizer:
            return
        optim_path = path + ".pdopt"
        optim_state = _load(optim_path)
        if optim_state is None:
            return

        self._load_optimizer(optim_state, executor)

    def _load_optimizer(self, state, executor):
        prog = self._progs.get('train', None)
        optim = list(filter(is_belong_to_optimizer, prog.list_vars()))
        if not optim:
            return

        fluid.core._create_loaded_parameter(optim, global_scope(), executor)

        converted_state = dict(state)
        for var in optim:
            if var.name in ["@LR_DECAY_COUNTER@", "global_step"]:
                # When using learning rate scheduler, dygraph would name the
                # global step var as "global_step" to save, while static-graph
                # would has a state var named as "@LR_DECAY_COUNTER@".
                # NOTE: dygraph saved global_step is 1 larger than that in
                # static-graph, since the time of global_step to increase is
                # different.
                state_val = (
                    np.array(converted_state.pop("global_step")) - 1
                ) if "global_step" in converted_state else converted_state.pop(
                    "@LR_DECAY_COUNTER@", None)
                if state_val is not None:
                    converted_state[var.name] = state_val
            elif var.name.startswith("learning_rate_"):
                # When using static learning rate, static-graph would make it
                # a persistable var named 'unique_name.generate("learning_rate")',
                # However, dygraph wouldn't save it.
                if var.name not in state: continue
            else:
                # moment and other accumulators
                if var.name not in converted_state:
                    # try to convert from dygraph name
                    opt_name = self.model._optimizer._name
                    opt_cls_name = self.model._optimizer.__class__.__name__
                    opt_unq_name = None
                    for name in self.model._optimizer._accumulators.keys():
                        accum_name = name if opt_name is None else name[len(
                            opt_name) + 1:]
                        for param_name, state_var in self.model._optimizer._accumulators[
                                name].items():
                            if opt_unq_name is None:
                                # can not infer out the exact unique(opt_name),
                                # thus try to extract rather than generate
                                for state_key in sorted(
                                        state.keys(),
                                        key=lambda x: len(x),
                                        reverse=True):
                                    prefix = param_name + "_" + (
                                        opt_cls_name if opt_name is None else
                                        opt_name) + "_"
                                    if state_key.startswith(prefix):
                                        prefix_offset = state_key[len(
                                            prefix):].find("_") + len(prefix)
                                        opt_unq_name = state_key[len(
                                            param_name + "_"):prefix_offset]
                                        # TODO: assert
                                        # assert opt_unq_name is None
                                    # gen(param.name + "_" + gen(opt_name) + "_" + accum_name)
                                    # always end with "_0" since the unique optimizer._name
                            dy_state_name = (param_name + "_" + opt_unq_name +
                                             "_" + accum_name + "_0")
                            converted_state[
                                state_var.name] = converted_state.pop(
                                    dy_state_name)

            assert var.name in converted_state, \
                "variable [{}] is not in optimizer state file".format(var.name)
            self._set_var(var, converted_state[var.name])

    def _set_var(self, var, ndarray):
        t = global_scope().find_var(var.name).get_tensor()
        p = t._place()
        if p.is_cpu_place():
            place = fluid.CPUPlace()
        elif p.is_cuda_pinned_place():
            place = fluid.CUDAPinnedPlace()
        else:
            p = fluid.core.Place()
            p.set_place(t._place())
            place = fluid.CUDAPlace(p.gpu_device_id())

        t.set(ndarray, place)

    def _run(self, inputs, labels=None):
        compiled_prog = self._compiled_progs.get(self.mode, None)
        assert compiled_prog, \
            "Model is not ready, please call `model.prepare()` first"

        inputs = to_list(inputs)
        if labels is not None:
            labels = to_list(labels)
        assert len(inputs) == len(self._input_vars[self.mode]), \
            "number of inputs" \
            + " does not match number of arguments of `forward` method"

        feed = {}
        input_names = [v.name for v in self._input_vars[self.mode]]
        for idx, n in enumerate(input_names):
            # train and test may take different arguments
            if inputs[idx] is not None:
                feed[n] = inputs[idx]
        if labels is not None:
            for idx, v in enumerate(self._label_vars[self.mode]):
                feed[v.name] = labels[idx]

        endpoints = self._endpoints[self.mode]
        fetch_list = endpoints['output'] + endpoints['loss']
        num_output = len(endpoints['output'])
        out = self._executor.run(compiled_prog,
                                 feed=feed,
                                 fetch_list=fetch_list)
        if self.mode == 'test':
            return out[:num_output]
        else:
            return out[:num_output], out[num_output:]

    def prepare(self):
        modes = ['train', 'eval', 'test']
        for mode in modes:
            self._make_program(mode)
            self._compile_and_initialize(self._progs[mode], mode)

    def _make_program(self, mode):
        prog = self._progs.get(mode, None)
        if prog is not None:
            return

        prog = self._orig_prog.clone()
        # NOTE: When defining learning rate scheduling in static-graph, ops to
        # increase the global step var and calculate learning rate would be
        # prepended into _orig_prog. test program maked by `_orig_prog.clone`
        # also would include these ops. Thus must prune these ops in test
        # program, otherwise the global step would be changed in test.
        if mode != 'train':
            for op in list(prog.global_block().ops):
                prog.global_block()._remove_op(0)
        if mode == 'train' and self.model._optimizer \
            and self.model._optimizer._learning_rate_map:
            # HACK workaround learning rate map issue
            lr_var = self.model._optimizer._learning_rate_map[self._orig_prog]
            self.model._optimizer._learning_rate_map[prog] = lr_var
        losses = []
        with fluid.program_guard(prog, self._startup_prog):
            if isinstance(self.model._inputs, dict):
                ins = [self.model._inputs[n] \
                    for n in extract_args(self.model.forward) if n != 'self']
            else:
                ins = self.model._inputs
            lbls = self.model._labels if self.model._labels else []
            inputs = [k.forward() for k in to_list(ins)]
            labels = [k.forward() for k in to_list(lbls)]
            outputs = to_list(self.model.forward(*inputs))
            if mode != 'test':
                if self.model._loss_function:
                    losses = self.model._loss_function(outputs, labels)
                if mode == 'train' and self.model._optimizer:
                    self._loss_endpoint = fluid.layers.sum(losses)
                    self.model._optimizer.minimize(self._loss_endpoint)
        if mode != 'train':  # clone again to put it in test mode
            prog = prog.clone(for_test=True)

        self._input_vars[mode] = inputs
        self._label_vars[mode] = labels
        self._progs[mode] = prog
        self._endpoints[mode] = {"output": outputs, "loss": losses}

    def _compile_and_initialize(self, prog, mode):
        compiled_prog = self._compiled_progs.get(mode, None)
        if compiled_prog is not None:
            return compiled_prog

        device = self.model._device
        device_ids = self.model._device_ids

        if device.lower() == 'gpu':
            places = fluid.cuda_places(device_ids)
        else:
            places = fluid.cpu_places(len(device_ids) if device_ids else None)

        # XXX *ALL WEIGHTS* should be initialized upon model construction
        # even if `forward()` may run different code path for different mode
        # therefore startup program only needs to run once
        if self._executor is None:
            self._executor = fluid.Executor(places[0])
            # XXX incremental initialization
            uninitialized = []
            for var_py in self._startup_prog.list_vars():
                var = fluid.global_scope().find_var(var_py.name)
                if var and var.get_tensor()._is_initialized():
                    continue
                uninitialized.append(var_py)
            if uninitialized:
                startup_prog = self._startup_prog._prune(uninitialized)
                self._executor.run(startup_prog)

        compiled_prog = fluid.CompiledProgram(prog)
        if len(places) > 1:
            loss_name = None
            if mode == 'train' and self._loss_endpoint is not None:
                loss_name = self._loss_endpoint.name
            compiled_prog = compiled_prog.with_data_parallel(
                loss_name=loss_name, places=places)
        self._compiled_progs[mode] = compiled_prog


class DynamicGraphAdapter(object):
    def __init__(self, model):
        super(DynamicGraphAdapter, self).__init__()
        self.model = model

    @property
    def mode(self):
        return self.model.mode

    @mode.setter
    def mode(self, value):
        self.model.mode = value

    # TODO multi device in dygraph mode not implemented at present time
    def train(self, inputs, labels=None):
        assert self.model._optimizer, \
            "model not ready, please call `model.prepare()` first"
        super(Model, self.model).train()
        self.mode = 'train'
        inputs = to_list(inputs)
        if labels is not None:
            labels = to_list(labels)
        outputs = self.model.forward(*[to_variable(x) for x in inputs])
        losses = self.model._loss_function(outputs, labels)
        final_loss = fluid.layers.sum(losses)
        final_loss.backward()
        self.model._optimizer.minimize(final_loss)
        self.model.clear_gradients()
        return [to_numpy(o) for o in to_list(outputs)], \
            [to_numpy(l) for l in losses]

    def eval(self, inputs, labels=None):
        super(Model, self.model).eval()
        self.mode = 'eval'
        inputs = to_list(inputs)
        if labels is not None:
            labels = to_list(labels)
        outputs = self.model.forward(*[to_variable(x) for x in inputs])

        if self.model._loss_function:
            losses = self.model._loss_function(outputs, labels)
        else:
            losses = []

        # To be consistent with static graph
        # return empty loss if loss_function is None
        return [to_numpy(o) for o in to_list(outputs)], \
            [to_numpy(l) for l in losses]

    def test(self, inputs):
        super(Model, self.model).eval()
        self.mode = 'test'
        inputs = [to_variable(x) for x in to_list(inputs)]
        outputs = self.model.forward(*inputs)
        return [to_numpy(o) for o in to_list(outputs)]

    def parameters(self, *args, **kwargs):
        return super(Model, self.model).parameters(*args, **kwargs)

    def save(self, path):
        params = self.model.state_dict()
        fluid.save_dygraph(params, path)
        if self.model._optimizer is None:
            return
        if self.model._optimizer.state_dict():
            optim = self.model._optimizer.state_dict()
            fluid.save_dygraph(optim, path)

    def load(self, path):
        params, optim = fluid.load_dygraph(path)
        self.model.set_dict(params)
        if self.model._optimizer is None or optim is None:
            return

        # If optimizer performs set_dict when state vars haven't been created,
        # which would happen when set_dict before minimize, the state would be
        # stored in optimizer._accumulators_holder and loaded lazily.
        # To contrive this when loading from static-graph saved states, extend
        # state dict to include keys named accoring to dygraph naming rules.
        # TODO: if len(self.model._optimizer._accumulators) > 0
        converted_state = dict(optim)
        opt_unq_name = self.model._optimizer._name
        opt_cls_name = self.model._optimizer.__class__.__name__
        opt_name = opt_unq_name[:opt_unq_name.rfind("_")]  # remove suffix idx
        param_names = [param.name for param in self.model.parameters()]
        for var_name, state_var in sorted(
                optim.items(), key=lambda x: len(x[0]), reverse=True):
            if var_name in ["@LR_DECAY_COUNTER@", "global_step"]:
                # NOTE: dygraph saved global_step is 1 larger than that in
                # static-graph, since the time of global_step to increase is
                # different.
                if var_name == "@LR_DECAY_COUNTER@":
                    converted_state["global_step"] = np.array(
                        converted_state.pop("@LR_DECAY_COUNTER@")) + 1
            else:
                # moment and other accumulators
                # extend state dict to include promising dygraph names
                for param_name in param_names:
                    if var_name.startswith(param_name + "_" + opt_name):
                        # when init optimizer with name
                        accum_name = var_name[len(param_name + "_" + opt_name +
                                                  "_"):]
                    elif var_name.startswith(param_name +
                                             "_") and opt_name == opt_cls_name:
                        # when init optimizer without name
                        accum_name = var_name[len(param_name + "_"):]
                    else:
                        continue
                    # remove suffix idx
                    accum_name = accum_name[:accum_name.rfind("_")]
                    # state names always end with "_0" in dygraph because of the
                    # unique optimizer._name
                    dy_state_name = (param_name + "_" + opt_unq_name + "_" +
                                     accum_name + "_0")
                    converted_state[dy_state_name] = state_var

        self.model._optimizer.set_dict(converted_state)


class Model(fluid.dygraph.Layer):
    """
    FIXME: add more comments and usage
    """

    def __init__(self):
        super(Model, self).__init__(self.__class__.__name__)
        self.mode = 'train'
        self._inputs = None
        self._labels = None
        self._loss_function = None
        self._loss_weights = None
        self._loss = None
        self._optimizer = None
        self._device = None
        self._device_ids = None
        self._optimizer = None
        if in_dygraph_mode():
            self._adapter = DynamicGraphAdapter(self)
        else:
            self._adapter = StaticGraphAdapter(self)

    def train(self, *args, **kwargs):
        return self._adapter.train(*args, **kwargs)

    def eval(self, *args, **kwargs):
        return self._adapter.eval(*args, **kwargs)

    def test(self, *args, **kwargs):
        return self._adapter.test(*args, **kwargs)

    def save(self, *args, **kwargs):
        return self._adapter.save(*args, **kwargs)

    def load(self, *args, **kwargs):
        return self._adapter.load(*args, **kwargs)

    def prepare(self,
                optimizer=None,
                loss_function=None,
                inputs=None,
                labels=None,
                device=None,
                device_ids=None):
        """
        FIXME: add comments
        Args:
            optimizer (Optimizer|None): optimizer must be set in training
                and should be a Optimizer instance. It can be None in eval
                and test mode.
            loss_function (Loss|None): loss function must be set in training
                and should be a Loss instance. It can be None when there is
                no loss.
            inputs (Input|list|dict|None): inputs, entry points of network,
                could be a Input layer, or lits of Input layers,
                or dict (name: Input), or None. For static graph,
                inputs must be set. For dynamic graph, it could be None.
            labels (Input|list|None): labels, entry points of network,
                could be a Input layer or lits of Input layers, or None.
                For static graph, if set loss_function in Model.prepare(), it
                must be set. Otherwise, it could be None.
            device (str|None): specify device type, 'CPU' or 'GPU'.
                If None, automatically select device according to
                installation package version.
            device_ids (list[int]|None): specify device index. If None,
                the available device will be obtained from the environment
                variable when the model is executed: If the GPU is used, the
                currently available device ID is obtained from the environment
                variable FLAGS_selected_gpus or CUDA_VISIBLE_DEVICES when the
                model is executed; CPU, when the model is executed,
                the currently available CPU number is obtained from the
                environment variable CPU_NUM. For example, export CPU_NUM=4,
                if the environment variable is not set, the executor will add
                the variable to the environment variable and set its value to 1.
                The default is None.
        """
        self._optimizer = optimizer
        if loss_function:
            if not isinstance(loss_function, Loss):
                raise TypeError(
                    "'loss_function' must be sub classes of 'Loss'")
        self._loss_function = loss_function
        if not in_dygraph_mode():
            if not isinstance(inputs, (list, dict, Input)):
                raise TypeError(
                    "'inputs' must be list or dict in static graph mode")
            if loss_function and not isinstance(labels, (list, Input)):
                raise TypeError("'labels' must be list in static graph mode")
        self._inputs = inputs
        self._labels = labels
        self._device = device
        if device is None:
            self._device = 'GPU' if fluid.is_compiled_with_cuda() else 'CPU'
        self._device_ids = device_ids
        if not in_dygraph_mode():
            self._adapter.prepare()

    def parameters(self, *args, **kwargs):
        return self._adapter.parameters(*args, **kwargs)
