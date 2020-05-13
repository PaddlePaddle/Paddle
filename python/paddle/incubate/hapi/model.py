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
from __future__ import division
from __future__ import print_function

import inspect
import os
import pickle
import numpy as np
import six
import warnings
from collections import Iterable

from paddle import fluid
from paddle.fluid.framework import in_dygraph_mode, Variable
from paddle.fluid.executor import global_scope
from paddle.fluid.io import is_belong_to_optimizer
from paddle.fluid.dygraph.base import to_variable
from paddle.fluid.dygraph.parallel import ParallelEnv
from paddle.fluid.layers.utils import flatten
from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy
from paddle.fluid.incubate.fleet.base import role_maker
from paddle.io import DataLoader, Dataset

from .loss import Loss
from .distributed import DistributedBatchSampler, _all_gather, prepare_distributed_context, _parallel_context_initialized
from .metrics import Metric
from .callbacks import config_callbacks
from .utils import to_list, to_numpy, flatten_list, restore_flatten_list

__all__ = [
    'Model',
    'Input',
    'set_device',
]


def set_device(device):
    """
    Args:
        device (str): specify device type, 'cpu' or 'gpu'.
        
    Returns:
        fluid.CUDAPlace or fluid.CPUPlace: Created GPU or CPU place.
    """

    assert isinstance(device, six.string_types) and device.lower() in ['cpu', 'gpu'], \
    "Expected device in ['cpu', 'gpu'], but got {}".format(device)

    place = fluid.CUDAPlace(ParallelEnv().dev_id) \
            if device.lower() == 'gpu' and fluid.is_compiled_with_cuda() \
                else fluid.CPUPlace()

    return place


class Input(fluid.dygraph.Layer):
    def __init__(self, shape=None, dtype=None, name=None):
        super(Input, self).__init__()
        self.shape = shape
        self.dtype = dtype
        self.name = name

    def forward(self):
        return fluid.data(self.name, shape=self.shape, dtype=self.dtype)


class StaticGraphAdapter(object):
    """
    Model traning/inference with a static graph.
    """

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

        self._merge_count = {
            'eval_total': 0,
            'test_total': 0,
            'eval_batch': 0,
            'test_batch': 0
        }

        self._nranks = ParallelEnv().nranks
        self._local_rank = ParallelEnv().local_rank

    @property
    def mode(self):
        return self.model.mode

    @mode.setter
    def mode(self, value):
        self.model.mode = value

    def train_batch(self, inputs, labels=None):
        assert self.model._optimizer, \
            "model not ready, please call `model.prepare()` first"
        self.mode = 'train'
        return self._run(inputs, labels)

    def eval_batch(self, inputs, labels=None):
        self.mode = 'eval'
        return self._run(inputs, labels)

    def test_batch(self, inputs):
        self.mode = 'test'
        return self._run(inputs, None)

    def parameters(self, *args, **kwargs):
        return super(Model, self.model).parameters(*args, **kwargs)

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

    def load(self, param_state_pairs, optim_state):
        if self._executor is None:
            executor = fluid.Executor(fluid.CPUPlace())._default_executor
        else:
            executor = self._executor._default_executor

        # restore parameter states
        fluid.core._create_loaded_parameter(
            [param for param, state in param_state_pairs],
            global_scope(), executor)
        for param, state in param_state_pairs:
            self._set_var(param, state)

        # restore optimizer states
        # FIXME what if a different optimizer is used?
        if not self.model._optimizer or not optim_state:
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
                if var.name not in state:
                    continue
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
                                        opt_cls_name
                                        if opt_name is None else opt_name) + "_"
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
        if self.mode == 'test':
            fetch_list = endpoints['output']
        else:
            metric_list, metric_splits = flatten_list(endpoints['metric'])
            fetch_list = endpoints['loss'] + metric_list
            num_loss = len(endpoints['loss'])

        # if fetch Variable is same as input Variable, do not fetch
        # from program, get it from input directly
        pruned_fetch_list = []
        pruned_fetch_idx_name_map = [""] * len(fetch_list)
        for i, fetch_var in enumerate(fetch_list):
            if fetch_var.name in feed.keys():
                pruned_fetch_idx_name_map[i] = fetch_var.name
            else:
                pruned_fetch_list.append(fetch_var)

        rets = self._executor.run(compiled_prog,
                                  feed=feed,
                                  fetch_list=pruned_fetch_list,
                                  return_numpy=False)

        # restore pruned fetch_list Variable from feeds
        for i, name in enumerate(pruned_fetch_idx_name_map):
            if len(name) > 0:
                rets.insert(i, feed[name])

        # LoDTensor cannot be fetch as numpy directly
        rets = [np.array(v) for v in rets]
        if self.mode == 'test':
            return rets[:]
        losses = rets[:num_loss]
        metric_states = restore_flatten_list(rets[num_loss:], metric_splits)
        metrics = []
        for metric, state in zip(self.model._metrics, metric_states):
            # cut off padding size
            if self.mode != 'train' and self.model._test_dataloader is not None \
                    and isinstance(self.model._test_dataloader, DataLoader) \
                    and self._nranks > 1:
                total_size = len(self.model._test_dataloader.dataset)
                # TODO: fixme if have better way to get batch size
                samples = state[0].shape[0]
                current_count = self._merge_count.get(self.mode + '_total', 0)
                if current_count + samples >= total_size:
                    state = [
                        s[:int(total_size - current_count), ...] for s in state
                    ]
                    self._merge_count[self.mode + '_total'] = 0
                    self._merge_count[self.mode + '_batch'] = int(total_size -
                                                                  current_count)
                else:
                    self._merge_count[self.mode + '_total'] += samples
                    self._merge_count[self.mode + '_batch'] = samples

            metrics.append(metric.update(*state))
        return (losses, metrics) if len(metrics) > 0 else losses

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
            new_lr_var = prog.global_block().vars[lr_var.name]
            self.model._optimizer._learning_rate_map[prog] = new_lr_var

        losses = []
        metrics = []
        with fluid.program_guard(prog, self._startup_prog):
            ins = self.model._inputs
            lbls = self.model._labels if self.model._labels else []
            inputs = [k.forward() for k in to_list(ins)]
            labels = [k.forward() for k in to_list(lbls)]
            self._label_vars[mode] = labels
            outputs = to_list(self.model.forward(*inputs))

            if mode != 'test' and self.model._loss_function:
                losses = self.model._loss_function(outputs, labels)

            if self._nranks > 1 and mode != 'train':
                outputs = [_all_gather(o, self._nranks) for o in outputs]
                if mode != 'test':
                    labels = [_all_gather(l, self._nranks) for l in labels]

            if mode != 'test':
                for metric in self.model._metrics:
                    metrics.append(
                        to_list(metric.add_metric_op(*(outputs + labels))))

            if mode == 'train' and self.model._optimizer:
                self._loss_endpoint = fluid.layers.sum(losses)
                if self._nranks > 1:
                    role = role_maker.PaddleCloudRoleMaker(is_collective=True)
                    fleet.init(role)
                    dist_strategy = DistributedStrategy()
                    dist_strategy.mode = "collective"
                    dist_strategy.collective_mode = "grad_allreduce"
                    self.model._optimizer = fleet.distributed_optimizer(
                        self.model._optimizer, strategy=dist_strategy)

                self.model._optimizer.minimize(self._loss_endpoint)

        if mode != 'train':  # clone again to put it in test mode
            prog = prog.clone(for_test=True)

        self._input_vars[mode] = inputs

        self._progs[mode] = prog
        self._endpoints[mode] = {
            "output": outputs,
            "loss": losses,
            "metric": metrics
        }

    def _compile_and_initialize(self, prog, mode):
        compiled_prog = self._compiled_progs.get(mode, None)
        if compiled_prog is not None:
            return compiled_prog

        assert self.model._place is not None, \
            "device is not set, please call `model.prepare()` first"

        place = self.model._place

        # XXX *ALL WEIGHTS* should be initialized upon model construction
        # even if `forward()` may run different code path for different mode
        # therefore startup program only needs to run once
        if self._executor is None:
            self._executor = fluid.Executor(place)
            # XXX incremental initialization
            uninitialized = []
            for var_py in self._startup_prog.list_vars():
                var = fluid.global_scope().find_var(var_py.name)
                if not var_py.name.startswith('nccl_id') and var and \
                        var.get_tensor()._is_initialized():
                    continue

                uninitialized.append(var_py)
            if uninitialized:
                startup_prog = self._startup_prog._prune(uninitialized)
                self._executor.run(startup_prog)

        if self._nranks < 2:
            compiled_prog = fluid.CompiledProgram(prog)
        else:
            compiled_prog = prog

        self._compiled_progs[mode] = compiled_prog


class DynamicGraphAdapter(object):
    def __init__(self, model):
        super(DynamicGraphAdapter, self).__init__()
        self.model = model
        self._nranks = ParallelEnv().nranks
        self._local_rank = ParallelEnv().local_rank
        self._merge_count = {
            'eval_total': 0,
            'test_total': 0,
            'eval_batch': 0,
            'test_batch': 0
        }

        if self._nranks > 1:
            stradegy = fluid.dygraph.parallel.ParallelStrategy()
            stradegy.nranks = ParallelEnv().nranks
            stradegy.local_rank = ParallelEnv().local_rank
            stradegy.trainer_endpoints = ParallelEnv().trainer_endpoints
            stradegy.current_endpoint = ParallelEnv().current_endpoint
            self.ddp_model = fluid.dygraph.parallel.DataParallel(self.model,
                                                                 stradegy)

    @property
    def mode(self):
        return self.model.mode

    @mode.setter
    def mode(self, value):
        self.model.mode = value

    # TODO multi device in dygraph mode not implemented at present time
    def train_batch(self, inputs, labels=None):
        assert self.model._optimizer, \
            "model not ready, please call `model.prepare()` first"
        super(Model, self.model).train()
        self.mode = 'train'
        inputs = to_list(inputs)
        if labels is not None:
            labels = [to_variable(l) for l in to_list(labels)]
        if self._nranks > 1:
            outputs = self.ddp_model.forward(* [to_variable(x) for x in inputs])
            losses = self.model._loss_function(outputs, labels)
            final_loss = fluid.layers.sum(losses)
            final_loss = self.ddp_model.scale_loss(final_loss)
            final_loss.backward()
            self.ddp_model.apply_collective_grads()
        else:
            outputs = self.model.forward(* [to_variable(x) for x in inputs])
            losses = self.model._loss_function(outputs, labels)
            final_loss = fluid.layers.sum(losses)
            final_loss.backward()

        self.model._optimizer.minimize(final_loss)
        self.model.clear_gradients()
        metrics = []
        for metric in self.model._metrics:
            metric_outs = metric.add_metric_op(*(to_list(outputs) + to_list(
                labels)))
            m = metric.update(* [to_numpy(m) for m in to_list(metric_outs)])
            metrics.append(m)

        return ([to_numpy(l) for l in losses], metrics) \
            if len(metrics) > 0 else [to_numpy(l) for l in losses]

    def eval_batch(self, inputs, labels=None):
        super(Model, self.model).eval()
        self.mode = 'eval'
        inputs = to_list(inputs)
        if labels is not None:
            labels = [to_variable(l) for l in to_list(labels)]
        outputs = self.model.forward(* [to_variable(x) for x in inputs])
        if self.model._loss_function:
            losses = self.model._loss_function(outputs, labels)
        else:
            losses = []
        if self._nranks > 1:
            outputs = [_all_gather(o, self._nranks) for o in to_list(outputs)]
            labels = [_all_gather(l, self._nranks) for l in labels]
        metrics = []
        for metric in self.model._metrics:
            # cut off padding value.
            if self.model._test_dataloader is not None and self._nranks > 1 \
                    and isinstance(self.model._test_dataloader, DataLoader):
                total_size = len(self.model._test_dataloader.dataset)
                samples = outputs[0].shape[0]
                current_count = self._merge_count.get(self.mode + '_total', 0)
                if current_count + samples >= total_size:
                    outputs = [
                        o[:int(total_size - current_count)] for o in outputs
                    ]
                    labels = [
                        l[:int(total_size - current_count)] for l in labels
                    ]
                    self._merge_count[self.mode + '_total'] = 0
                    self._merge_count[self.mode + '_batch'] = int(total_size -
                                                                  current_count)
                else:
                    self._merge_count[self.mode + '_total'] += samples
                    self._merge_count[self.mode + '_batch'] = samples

            metric_outs = metric.add_metric_op(*(to_list(outputs) + to_list(
                labels)))
            m = metric.update(* [to_numpy(m) for m in to_list(metric_outs)])
            metrics.append(m)

        # To be consistent with static graph
        # return empty loss if loss_function is None
        return ([to_numpy(l) for l in losses], metrics) \
            if len(metrics) > 0 else [to_numpy(l) for l in losses]

    def test_batch(self, inputs):
        super(Model, self.model).eval()
        self.mode = 'test'
        inputs = [to_variable(x) for x in to_list(inputs)]
        outputs = self.model.forward(*inputs)
        if self._nranks > 1 and isinstance(self.model._place, fluid.CUDAPlace):
            outputs = [_all_gather(o, self._nranks) for o in to_list(outputs)]

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

    def load(self, param_state_pairs, optim_state):
        # restore parameter states
        for param, state in param_state_pairs:
            param.set_value(state)

        # resotre optimizer states
        if not self.model._optimizer or not optim_state:
            return

        # If optimizer performs set_dict when state vars haven't been created,
        # which would happen when set_dict before minimize, the state would be
        # stored in optimizer._accumulators_holder and loaded lazily.
        # To contrive this when loading from static-graph saved states, extend
        # state dict to include keys named accoring to dygraph naming rules.
        # TODO: if len(self.model._optimizer._accumulators) > 0
        converted_state = dict(optim_state)
        opt_unq_name = self.model._optimizer._name
        if opt_unq_name is None:
            opt_unq_name = ''

        opt_cls_name = self.model._optimizer.__class__.__name__
        opt_name = opt_unq_name[:opt_unq_name.rfind("_")]  # remove suffix idx
        param_names = [param.name for param in self.model.parameters()]
        for var_name, state_var in sorted(
                optim_state.items(), key=lambda x: len(x[0]), reverse=True):
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
    An Model object is network with training and inference features.
    Dynamic graph and static graph are supported at the same time,
    switched by `fluid.enable_dygraph()`. The usage is as follows.
    But note, the switching between dynamic and static should be before
    instantiating a Model. The input description, i.e, hapi.Input,
    must be required for static graph.

    Usage:
        .. code-block:: python

        import numpy as np
        import paddle
        import paddle.fluid as fluid
        #import paddle.incubate.hapi as hapi
        from paddle.incubate.hapi import Model, Input, set_device
        from paddle.incubate.hapi.loss import CrossEntropy
        from paddle.incubate.hapi.dataset import MNIST

        class MyModel(Model):
            def __init__(self):
                super(MyModel, self).__init__()
                self._fc = fluid.dygraph.Linear(784, 10, act='softmax')
            def forward(self, x):
                y = self._fc(x)
                return y
        device = set_device('gpu')
        # if use static graph, do not set
        fluid.enable_dygraph(device)
        model = MyModel()
        optim = fluid.optimizer.SGD(learning_rate=1e-3,
            parameter_list=model.parameters())
        
        inputs = [Input([None, 784], 'float32', name='x')]
        labels = [Input([None, 1], 'int64', name='label')]
        
        mnist_data = MNIST(mode='train')
        model.prepare(optim,
                      CrossEntropy(average=True),
                      hapi.metrics.Accuracy(),
                      inputs,
                      labels,
                      device=device)
        model.fit(mnist_data, epochs=2, batch_size=32, verbose=1)
    """

    def __init__(self):
        super(Model, self).__init__(self.__class__.__name__)
        self.mode = 'train'
        self._inputs = None
        self._labels = None
        self._loss_function = None
        self._loss_weights = None
        self._optimizer = None
        self._device = None
        self._optimizer = None
        self._test_dataloader = None

        # init backend
        if fluid.in_dygraph_mode():
            self._adapter = DynamicGraphAdapter(self)
        else:
            self._adapter = StaticGraphAdapter(self)

    def train_batch(self, inputs, labels=None):
        """
        Run one training step on a batch of data.

        Args:
            inputs (list): A list of numpy.ndarray, each is a batch of
                input data.
            labels (list): A list of numpy.ndarray, each is a batch of
                input label. If has no labels, set None. Default is None.

        Returns:
            A list of scalar training loss if the model has no metrics,
            or a tuple (list of scalar loss, list of metrics) if the model
            set metrics.

        Examples:

            .. code-block:: python
            
              import numpy as np
              import paddle.fluid as fluid
              from paddle.incubate.hapi import Model, Input, set_device

              class MyModel(Model):
                  def __init__(self):
                      super(MyModel, self).__init__()
                      self._fc = Linear(784, 1, act='softmax')
                  def forward(self, x):
                      y = self._fc(x)
                      return y

              device = hapi.set_device('gpu')
              fluid.enable_dygraph(device)

              model = MyModel()
              optim = fluid.optimizer.SGD(learning_rate=1e-3,
                  parameter_list=model.parameters())

              inputs = [Input([None, 784], 'float32', name='x')]
              labels = [Input([None, 1], 'int64', name='label')]
              model.prepare(optim,
                            CrossEntropy(average=True),
                            inputs=inputs,
                            labels=labels,
                            device=device)
              data = np.random.random(size=(4,784)).astype(np.float32)
              label = np.random.randint(0, 10, size=(4, 1)).astype(np.int64)
              loss = model.train_batch([data], [label])
              print(loss)
        """
        return self._adapter.train_batch(inputs, labels)

    def eval_batch(self, inputs, labels=None):
        """
        Run one evaluating step on a batch of data.

        Args:
            inputs (list): A list of numpy.ndarray, each is a batch of
                input data.
            labels (list): A list of numpy.ndarray, each is a batch of
                input label. If has no labels, set None. Default is None.

        Returns:
            A list of scalar testing loss if the model has no metrics,
            or a tuple (list of scalar loss, list of metrics) if the model
            set metrics.

        Examples:

            .. code-block:: python
            
              import numpy as np
              import paddle.fluid as fluid
              from paddle.incubate.hapi import Model, Input, set_device

              class MyModel(Model):
                  def __init__(self):
                      super(MyModel, self).__init__()
                      self._fc = fluid.dygraph.Linear(784, 1, act='softmax')
                  def forward(self, x):
                      y = self._fc(x)
                      return y

              device = set_device('gpu')
              fluid.enable_dygraph(device)

              model = MyModel()
              optim = fluid.optimizer.SGD(learning_rate=1e-3,
                  parameter_list=model.parameters())

              inputs = [Input([None, 784], 'float32', name='x')]
              labels = [Input([None, 1], 'int64', name='label')]
              model.prepare(optim,
                            CrossEntropy(average=True),
                            inputs=inputs,
                            labels=labels,
                            device=device)
              data = np.random.random(size=(4,784)).astype(np.float32)
              label = np.random.randint(0, 10, size=(4, 1)).astype(np.int64)
              loss = model.eval_batch([data], [label])
              print(loss)
        """
        return self._adapter.eval_batch(inputs, labels)

    def test_batch(self, inputs):
        """
        Run one testing step on a batch of data.

        Args:
            inputs (list): A list of numpy.ndarray, each is a batch of
                input data.

        Returns:
            A list of numpy.ndarray of predictions, that is the outputs
            of Model forward.

        Examples:

            .. code-block:: python
            
              import numpy as np
              import paddle.fluid as fluid
              from paddle.incubate.hapi import Model, Input, set_device

              class MyModel(Model):
                  def __init__(self):
                      super(MyModel, self).__init__()
                      self._fc = fluid.dygraph.Linear(784, 1, act='softmax')
                  def forward(self, x):
                      y = self._fc(x)
                      return y

              device = set_device('gpu')
              fluid.enable_dygraph(device)

              model = MyModel()
              inputs = [Input([None, 784], 'float32', name='x')]
              model.prepare(inputs=inputs,
                            device=device)
              data = np.random.random(size=(4,784)).astype(np.float32)
              out = model.eval_batch([data])
              print(out)
        """
        return self._adapter.test_batch(inputs)

    def save(self, path):
        """
        This function saves parameters, optimizer infomation to path.

        The parameters contains all the trainable Variable, will save to
        a file with suffix ".pdparams".
        The optimizer information contains all the variable used by optimizer.
        For Adam optimizer, contains beta1, beta2, momentum etc. All the
        information will save to a file with suffix ".pdopt". (If the optimizer
        have no variable need to save (like SGD), the fill will not generated).

        This function will silently overwrite existing file
        at the target location.

        Args:
            path (str): The file prefix to save model. The format is
                'dirname/file_prefix' or 'file_prefix'. if empty str. A exception
                 will be raised.

        Returns:
            None

        Examples:

            .. code-block:: python
            
              import paddle.fluid as fluid
              from paddle.incubate.hapi import Model, set_device
              
              class MyModel(Model):
                  def __init__(self):
                      super(MyModel, self).__init__()
                      self._fc = fluid.dygraph.Linear(784, 1, act='softmax')
                  def forward(self, x):
                      y = self._fc(x)
                      return y
              
              device = set_device('cpu')
              fluid.enable_dygraph(device)
              model = MyModel()
              model.save('checkpoint/test')
        """
        if ParallelEnv().local_rank == 0:
            self._adapter.save(path)

    def load(self, path, skip_mismatch=False, reset_optimizer=False):
        """
        Load from files storing the model states and optimizer states. The file
        for optimizer states is not necessary if no need to restore the optimizer.

        NOTE: parameters are retrieved out from the file storing model states
        accoring to their structured names.

        For fine-tuning or transfer-learning models where some of the layers have
        changed, keep parameters needed to restore have same structured names in
        the pre-trained model and fine-tuning model.

        Args:
            path (str): The prefix of files storing the model states and
                optimizer states. The files would be `path.pdparams` and
                `path.pdopt` separately, and the latter is not necessary
                when no need to restore.
            skip_mismatch (bool): Whether to skip the loading of mismatch
                parameter or raise an error when mismatch happens (not found
                the parameter in file storing model states of or receives a
                mismatch shape).
            reset_optimizer (bool): If True, ignore the providing file storing
                optimizer states and initialize optimizer states from scratch.
                Otherwise, restore optimizer states from `path.pdopt` if
                a optimizer has been set to the model. Default False.

        Returns:
            None

        Examples:

            .. code-block:: python
            
              import paddle.fluid as fluid
              from paddle.incubate.hapi import Model, set_device
              
              class MyModel(Model):
                  def __init__(self):
                      super(MyModel, self).__init__()
                      self._fc = fluid.dygraph.Linear(784, 1, act='softmax')
                  def forward(self, x):
                      y = self._fc(x)
                      return y
              
              device = set_device('cpu')
              fluid.enable_dygraph(device)
              model = MyModel()
              model.load('checkpoint/test')
        """

        def _load_state_from_path(path):
            if not os.path.exists(path):
                return
            with open(path, 'rb') as f:
                return pickle.load(f) if six.PY2 else pickle.load(
                    f, encoding='latin1')

        def _check_match(key, param):
            state = param_state.get(key, None)
            if state is None:
                raise ValueError(
                    "{} is not found in the providing file.".format(key))
            if list(state.shape) != list(param.shape):
                raise ValueError(
                    "{} receives a shape {}, but the expected shape is {}.".
                    format(key, list(state.shape), list(param.shape)))
            return param, state

        def _strip_postfix(path):
            path, ext = os.path.splitext(path)
            assert ext in ['', '.pdparams', '.pdopt', '.pdmodel'], \
                    "Unknown postfix {} from weights".format(ext)
            return path

        path = _strip_postfix(path)
        param_state = _load_state_from_path(path + ".pdparams")
        assert param_state, "Failed to load parameters, please check path."

        matched_param_state = []
        for key, param in self.state_dict().items():
            try:
                match_res = _check_match(key, param)
            except ValueError as err:
                if skip_mismatch:
                    warnings.warn(
                        ("Skip loading for {}. ".format(key) + str(err)))
                    # reset optimizer when mismatch happens
                    reset_optimizer = True
                else:
                    raise err
            matched_param_state.append(match_res)

        optim_state = None if reset_optimizer else _load_state_from_path(
            path + ".pdopt")
        return self._adapter.load(matched_param_state, optim_state)

    def parameters(self, *args, **kwargs):
        """
        Returns a list of parameters of the model.

        Returns:
            A list of Parameter in static graph.
            A list of ParamBase in dynamic graph.

        Examples:

            .. code-block:: python

              from paddle.incubate.hapi.model import Model, Input, set_device
              class MyModel(Model):
                  def __init__(self):
                      super(MyModel, self).__init__()
                      self._fc = fluid.dygraph.Linear(20, 10, act='softmax')
                  def forward(self, x):
                      y = self._fc(x)
                      return y

              fluid.enable_dygraph()
              model = MyModel()
              params = model.parameters()
        """
        return self._adapter.parameters()

    def prepare(self,
                optimizer=None,
                loss_function=None,
                metrics=None,
                inputs=None,
                labels=None,
                device=None):
        """
        Configures the model before runing.

        Args:
            optimizer (Optimizer|None): Optimizer must be set in training
                and should be a Optimizer instance. It can be None in eval
                and test mode.
            loss_function (Loss|None): Loss function must be set in training
                and should be a Loss instance. It can be None when there is
                no loss.
            metrics (Metric|list of Metric|None): If metrics is set, all
                metrics will be calculated and output in train/eval mode.
            inputs (Input|list|dict|None): `inputs`, entry points of network,
                could be a Input layer, or lits of Input layers,
                or dict (name: Input), or None. For static graph,
                inputs must be set. For dynamic graph, it could be None.
            labels (Input|list|None): `labels`, entry points of network,
                could be a Input layer or lits of Input layers, or None.
                For static graph, if labels is required in loss_function,
                labels must be set. Otherwise, it could be None.
            device (str|fluid.CUDAPlace|fluid.CPUPlace|None): Specify device
                type, 'CPU', 'GPU', fluid.CUDAPlace or fluid.CPUPlace.
                If None, automatically select device according to
                installation package version.

        Returns:
            None
        """

        if isinstance(device, fluid.CUDAPlace) or \
            (isinstance(device, six.string_types) and device.lower() == 'gpu') \
            or (device is None and fluid.is_compiled_with_cuda()):
            if isinstance(device, fluid.CUDAPlace):
                self._place = device
            else:
                self._place = fluid.CUDAPlace(ParallelEnv().dev_id) \
                    if ParallelEnv().nranks > 1 else fluid.CUDAPlace(0)

            global _parallel_context_initialized
            if ParallelEnv().nranks > 1 and not _parallel_context_initialized:
                if fluid.in_dygraph_mode():
                    main_prog_seed = fluid.default_main_program().random_seed
                    startup_prog_seed = fluid.default_startup_program(
                    ).random_seed
                    fluid.disable_dygraph()
                    fluid.enable_dygraph(self._place)
                    # enable_dygraph would create and switch to a new program,
                    # thus also copy seed to the new program
                    fluid.default_main_program().random_seed = main_prog_seed
                    fluid.default_startup_program(
                    ).random_seed = startup_prog_seed
                    fluid.dygraph.parallel.prepare_context()
                else:
                    prepare_distributed_context(self._place)

                _parallel_context_initialized = True
        elif isinstance(device, fluid.CPUPlace):
            self._place = device
        elif (isinstance(device, six.string_types) and device.lower() == 'cpu') \
            or (device is None):
            self._place = fluid.CPUPlace()
        else:
            raise ValueError(
                "Expected device in ('gpu', 'cpu', fluid.CUDAPlace, fluid.CPUPlace, None), \
                but got {}".format(device))

        self._optimizer = optimizer
        if loss_function:
            if not isinstance(loss_function, Loss):
                raise TypeError("'loss_function' must be sub classes of 'Loss'")
        self._loss_function = loss_function
        if not in_dygraph_mode():
            if not isinstance(inputs, (list, dict, Input)):
                raise TypeError(
                    "'inputs' must be list or dict in static graph mode")

        metrics = metrics or []
        for metric in to_list(metrics):
            assert isinstance(metric, Metric), \
                "{} is not sub class of Metric".format(
                    metric.__class__.__name__)
        self._metrics = to_list(metrics)

        self._inputs = to_list(inputs) if not isinstance(inputs, dict) else [
            inputs[n] for n in extract_args(self.forward) if n != 'self'
        ]
        self._labels = to_list(labels)

        if not in_dygraph_mode():
            self._adapter.prepare()

    def fit(
            self,
            train_data=None,
            eval_data=None,
            batch_size=1,
            epochs=1,
            eval_freq=1,
            log_freq=10,
            save_dir=None,
            save_freq=1,
            verbose=2,
            drop_last=False,
            shuffle=True,
            num_workers=0,
            callbacks=None, ):
        """
        Trains the model for a fixed number of epochs. If `eval_data` is set,
        evaluation will be done at the end of each epoch.

        Args:
            train_data (Dataset|DataLoader): An iterable data loader is used for 
                train. An instance of paddle paddle.io.Dataset or 
                paddle.io.Dataloader is recomended. Default: None.
            eval_data (Dataset|DataLoader): An iterable data loader is used for
                evaluation at the end of epoch. If None, will not do evaluation. 
                An instance of paddle.io.Dataset or paddle.io.Dataloader 
                is recomended. Default: None.
            batch_size (int): Integer number. The batch size of train_data
                and eval_data. When train_data and eval_data are both the
                instance of Dataloader, this parameter will be ignored.
                Default: 1.
            epochs (int): Integer number. The number of epochs to train
                the model. Default: 1.
            eval_freq (int): The frequency, in number of epochs, an evalutation
                is performed. Default: 1.
            log_freq (int): The frequency, in number of steps, the training logs
                are printed. Default: 10.
            save_dir(str|None): The directory to save checkpoint during training.
                If None, will not save checkpoint. Default: None.
            save_freq (int): The frequency, in number of epochs, to save
                checkpoint. Default: 1.
            verbose (int): The verbosity mode, should be 0, 1, or 2. 0 = silent,
                1 = progress bar, 2 = one line per epoch. Default: 2.
            drop_last (bool): Whether drop the last incomplete batch of
                train_data when dataset size is not divisible by the batch size.
                When train_data is an instance of Dataloader, this parameter
                will be ignored. Default: False.
            shuffle (bool): Whther to shuffle train_data. When train_data is
                an instance of Dataloader, this parameter will be ignored.
                Default: True.
            num_workers (int): The number of subprocess to load data, 0 for no
                subprocess used and loading data in main process.
                When train_data and eval_data are both the instance of
                Dataloader, this parameter will be ignored. Default: 0.
            callbacks (Callback|None): A list of `Callback` instances to apply
                during training. If None, `ProgBarLogger` and `ModelCheckpoint`
                are automatically inserted. Default: None.

        Returns:
            None

        Examples:
            1. An example use Dataset and set btch size, shuffle in fit.
               How to make a batch is done internally.

            .. code-block:: python

              from paddle.incubate.hapi.model import Model, Input, set_device
              from paddle.incubate.hapi.loss import CrossEntropy
              from paddle.incubate.hapi.metrics import Accuracy
              from paddle.incubate.hapi.datasets import MNIST
              from paddle.incubate.hapi.vision.models import LeNet

              dynamic = True
              device = set_device(FLAGS.device)
              fluid.enable_dygraph(device) if dynamic else None
           
              train_dataset = MNIST(mode='train')
              val_dataset = MNIST(mode='test')
           
              inputs = [Input([None, 1, 28, 28], 'float32', name='image')]
              labels = [Input([None, 1], 'int64', name='label')]
           
              model = LeNet()
              optim = fluid.optimizer.Adam(
                  learning_rate=0.001, parameter_list=model.parameters())
              model.prepare(
                  optim,
                  CrossEntropy(),
                  Accuracy(topk=(1, 2)),
                  inputs=inputs,
                  labels=labels,
                  device=device)
              model.fit(train_dataset,
                        val_dataset,
                        epochs=2,
                        batch_size=64,
                        save_dir='mnist_checkpoint')

            2. An example use DataLoader, batch size and shuffle is set in
               DataLoader.

            .. code-block:: python

              from paddle.incubate.hapi.model import Model, Input, set_device
              from paddle.incubate.hapi.loss import CrossEntropy
              from paddle.incubate.hapi.metrics import Accuracy
              from paddle.incubate.hapi.datasets import MNIST
              from paddle.incubate.hapi.vision.models import LeNet

              dynamic = True
              device = set_device(FLAGS.device)
              fluid.enable_dygraph(device) if dynamic else None
           
              train_dataset = MNIST(mode='train')
              train_loader = fluid.io.DataLoader(train_dataset,
                  places=device, batch_size=64)
              val_dataset = MNIST(mode='test')
              val_loader = fluid.io.DataLoader(val_dataset,
                  places=device, batch_size=64)
           
              inputs = [Input([None, 1, 28, 28], 'float32', name='image')]
              labels = [Input([None, 1], 'int64', name='label')]
           
              model = LeNet()
              optim = fluid.optimizer.Adam(
                  learning_rate=0.001, parameter_list=model.parameters())
              model.prepare(
                  optim,
                  CrossEntropy(),
                  Accuracy(topk=(1, 2)),
                  inputs=inputs,
                  labels=labels,
                  device=device)
              model.fit(train_loader,
                        val_loader,
                        epochs=2,
                        save_dir='mnist_checkpoint')
        """

        assert train_data is not None, \
                "train_data must be given!"

        if isinstance(train_data, Dataset):
            train_sampler = DistributedBatchSampler(
                train_data,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last)
            train_loader = DataLoader(
                train_data,
                batch_sampler=train_sampler,
                places=self._place,
                num_workers=num_workers,
                return_list=True)
        else:
            train_loader = train_data

        if eval_data is not None and isinstance(eval_data, Dataset):
            eval_sampler = DistributedBatchSampler(
                eval_data, batch_size=batch_size)
            eval_loader = DataLoader(
                eval_data,
                batch_sampler=eval_sampler,
                places=self._place,
                num_workers=num_workers,
                return_list=True)
        elif eval_data is not None:
            eval_loader = eval_data
        else:
            eval_loader = None

        do_eval = eval_loader is not None
        self._test_dataloader = eval_loader

        steps = self._len_data_loader(train_loader)
        cbks = config_callbacks(
            callbacks,
            model=self,
            epochs=epochs,
            steps=steps,
            log_freq=log_freq,
            save_freq=save_freq,
            save_dir=save_dir,
            verbose=verbose,
            metrics=self._metrics_name(), )

        cbks.on_begin('train')
        for epoch in range(epochs):

            cbks.on_epoch_begin(epoch)
            logs = self._run_one_epoch(train_loader, cbks, 'train')
            cbks.on_epoch_end(epoch, logs)

            if do_eval and epoch % eval_freq == 0:

                eval_steps = self._len_data_loader(eval_loader)
                cbks.on_begin('eval', {
                    'steps': eval_steps,
                    'metrics': self._metrics_name()
                })

                eval_logs = self._run_one_epoch(eval_loader, cbks, 'eval')

                cbks.on_end('eval', eval_logs)

        cbks.on_end('train', logs)
        self._test_dataloader = None

    def evaluate(
            self,
            eval_data,
            batch_size=1,
            log_freq=10,
            verbose=2,
            num_workers=0,
            callbacks=None, ):
        """
        Evaluate the loss and metrics of the model on input dataset.

        Args:
            eval_data (Dataset|DataLoader): An iterable data loader is used for
                evaluation. An instance of paddle.io.Dataset or 
                paddle.io.Dataloader is recomended.
            batch_size (int): Integer number. The batch size of train_data
                and eval_data.  When eval_data is the instance of Dataloader,
                this argument will be ignored. Default: 1.
            log_freq (int): The frequency, in number of steps, the eval logs
                are printed. Default: 10.
            verbose (int): The verbosity mode, should be 0, 1, or 2. 0 = silent,
                1 = progress bar, 2 = one line per epoch. Default: 2.
            num_workers (int): The number of subprocess to load data,
                0 for no subprocess used and loading data in main process. When
                train_data and eval_data are both the instance of Dataloader,
                this parameter will be ignored. Default: 0.
            callbacks (Callback|None): A list of `Callback` instances to apply
                during training. If None, `ProgBarLogger` and `ModelCheckpoint`
                are automatically inserted. Default: None.
        Returns:
            dict: Result of metric. The key is the names of Metric,
                value is a scalar or numpy.array.

        Examples:
        .. code-block:: python

            # declarative mode
            import numpy as np
            from paddle.incubate.hapi.metrics import Accuracy
            from paddle.incubate.hapi.datasets import MNIST
            from paddle.incubate.hapi.vision.transforms import Compose,Resize
            from paddle.incubate.hapi.vision.models import LeNet
            from paddle.incubate.hapi.model import Input, set_device


            inputs = [Input([-1, 1, 28, 28], 'float32', name='image')]
            labels = [Input([None, 1], 'int64', name='label')]

            val_dataset = MNIST(mode='test')

            model = LeNet()
            model.prepare(metrics=Accuracy(), inputs=inputs, labels=labels)

            result = model.evaluate(val_dataset, batch_size=64)
            print(result)

            # imperative mode
            import paddle.fluid.dygraph as dg
            place = set_device('cpu')
            with dg.guard(place) as g:
                model = LeNet()
                model.prepare(metrics=Accuracy(), inputs=inputs, labels=labels)

                result = model.evaluate(val_dataset, batch_size=64)
                print(result)
                
        """

        if eval_data is not None and isinstance(eval_data, Dataset):
            eval_sampler = DistributedBatchSampler(
                eval_data, batch_size=batch_size)
            eval_loader = DataLoader(
                eval_data,
                batch_sampler=eval_sampler,
                places=self._place,
                num_workers=num_workers,
                return_list=True)
        else:
            eval_loader = eval_data

        self._test_dataloader = eval_loader

        cbks = config_callbacks(
            callbacks,
            model=self,
            log_freq=log_freq,
            verbose=verbose,
            metrics=self._metrics_name(), )

        eval_steps = self._len_data_loader(eval_loader)
        cbks.on_begin('eval',
                      {'steps': eval_steps,
                       'metrics': self._metrics_name()})

        logs = self._run_one_epoch(eval_loader, cbks, 'eval')

        cbks.on_end('eval', logs)

        self._test_dataloader = None

        eval_result = {}
        for k in self._metrics_name():
            eval_result[k] = logs[k]

        return eval_result

    def predict(self,
                test_data,
                batch_size=1,
                num_workers=0,
                stack_outputs=False,
                callbacks=None):
        """
        Compute the output predictions on testing data.

        Args:
            test_data (Dataset|DataLoader): An iterable data loader is used for
                predict. An instance of paddle.io.Dataset or paddle.io.Dataloader
                is recomended.
            batch_size (int): Integer number. The batch size of train_data and eval_data.
                When train_data and eval_data are both the instance of Dataloader, this
                argument will be ignored. Default: 1.
            num_workers (int): The number of subprocess to load data, 0 for no subprocess 
                used and loading data in main process. When train_data and eval_data are
                both the instance of Dataloader, this argument will be ignored. Default: 0.
            stack_output (bool): Whether stack output field like a batch, as for an output
                filed of a sample is in shape [X, Y], test_data contains N samples, predict
                output field will be in shape [N, X, Y] if stack_output is True, and will
                be a length N list in shape [[X, Y], [X, Y], ....[X, Y]] if stack_outputs
                is False. stack_outputs as False is used for LoDTensor output situation,
                it is recommended set as True if outputs contains no LoDTensor. Default: False.
        Returns:
            list: output of models.

        Examples:
        .. code-block:: python

            # declarative mode
            import numpy as np
            from paddle.incubate.hapi.metrics import Accuracy
            from paddle.incubate.hapi.datasets import MNIST
            from paddle.incubate.hapi.vision.transforms import Compose,Resize
            from paddle.incubate.hapi.vision.models import LeNet
            from paddle.incubate.hapi.model import Input, set_device

            class MnistDataset(MNIST):
                def __init__(self, mode, return_label=True):
                    super(MnistDataset, self).__init__(mode=mode)
                    self.return_label = return_label

                def __getitem__(self, idx):
                    img = np.reshape(self.images[idx], [1, 28, 28])
                    if self.return_label:
                        return img, np.array(self.labels[idx]).astype('int64')
                    return img,

                def __len__(self):
                    return len(self.images)

            inputs = [Input([-1, 1, 28, 28], 'float32', name='image')]

            test_dataset = MnistDataset(mode='test', return_label=False)

            model = LeNet()
            model.prepare(inputs=inputs)

            result = model.predict(test_dataset, batch_size=64)
            print(result)

            # imperative mode
            import paddle.fluid.dygraph as dg
            place = set_device('cpu')
            with dg.guard(place) as g:
                model = LeNet()
                model.prepare(inputs=inputs)

                result = model.predict(test_dataset, batch_size=64)
                print(result)
        """

        if test_data is not None and isinstance(test_data, Dataset):
            test_sampler = DistributedBatchSampler(
                test_data, batch_size=batch_size)
            test_loader = DataLoader(
                test_data,
                batch_sampler=test_sampler,
                places=self._place,
                num_workers=num_workers,
                return_list=True)
        else:
            test_loader = test_data

        self._test_dataloader = test_loader

        cbks = config_callbacks(callbacks, model=self, verbose=1)

        test_steps = self._len_data_loader(test_loader)
        logs = {'steps': test_steps}

        cbks.on_begin('test', logs)

        outputs = []

        logs, outputs = self._run_one_epoch(test_loader, cbks, 'test')

        outputs = list(zip(*outputs))

        # NOTE: for lod tensor output, we should not stack outputs
        # for stacking may lose its detail info
        if stack_outputs:
            outputs = [np.vstack(outs) for outs in outputs]

        self._test_dataloader = None

        cbks.on_end('test', logs)
        return outputs

    def save_inference_model(self,
                             save_dir,
                             model_filename=None,
                             params_filename=None,
                             model_only=False):
        """
        Save inference model must in static mode.

        Args:
            save_dir (str): The directory path to save the inference model.
            model_filename (str|None): The name of file to save the inference
                model itself. If is set None, a default filename
                :code:`__model__` will be used.
            params_filename (str|None): The name of file to save all related
                parameters. If it is set None, parameters will be saved
                in separate files .
            model_only (bool): If True, It will save inference model only,
                and do not save parameters. Default: False.

        Returns:
            list: The fetch variables' name list
        """
        assert not fluid.in_dygraph_mode(
        ), 'Save inference model must in static mode!'

        prog = self._adapter._progs.get('test', None)
        assert prog, \
            "Model is not ready, please call `model.prepare()` first"

        infer_prog = prog.clone(for_test=True)

        input_names = [v.name for v in self._adapter._input_vars['test']]
        endpoints = self._adapter._endpoints['test']['output']

        return fluid.io.save_inference_model(
            save_dir,
            input_names,
            endpoints,
            self._adapter._executor,
            main_program=infer_prog,
            model_filename=model_filename,
            params_filename=params_filename,
            program_only=model_only)

    def _run_one_epoch(self, data_loader, callbacks, mode, logs={}):
        outputs = []
        for step, data in enumerate(data_loader):
            # data might come from different types of data_loader and have
            # different format, as following:
            # 1. DataLoader in static graph:
            #    [[input1, input2, ..., label1, lable2, ...]]
            # 2. DataLoader in dygraph
            #    [input1, input2, ..., label1, lable2, ...]
            # 3. custumed iterator yield concated inputs and labels:
            #   [input1, input2, ..., label1, lable2, ...]
            # 4. custumed iterator yield seperated inputs and labels:
            #   ([input1, input2, ...], [label1, lable2, ...])
            # To handle all of these, flatten (nested) list to list.
            data = flatten(data)
            # LoDTensor.shape is callable, where LoDTensor comes from
            # DataLoader in static graph
            batch_size = data[0].shape()[0] if callable(data[
                0].shape) else data[0].shape[0]

            callbacks.on_batch_begin(mode, step, logs)

            if mode != 'test':
                outs = getattr(self, mode + '_batch')(data[:len(self._inputs)],
                                                      data[len(self._inputs):])
                # losses
                loss = outs[0] if self._metrics else outs
                metrics = [[l[0] for l in loss]]

                # metrics
                for metric in self._metrics:
                    res = metric.accumulate()
                    metrics.extend(to_list(res))

                assert len(self._metrics_name()) == len(metrics)
                for k, v in zip(self._metrics_name(), metrics):
                    logs[k] = v
            else:
                if self._inputs is not None:
                    outs = getattr(self,
                                   mode + '_batch')(data[:len(self._inputs)])
                else:
                    outs = getattr(self, mode + '_batch')(data)

                outputs.append(outs)

            logs['step'] = step
            if mode == 'train' or self._adapter._merge_count.get(
                    mode + '_batch', 0) <= 0:
                logs['batch_size'] = batch_size * ParallelEnv().nranks
            else:
                logs['batch_size'] = self._adapter._merge_count[mode + '_batch']

            callbacks.on_batch_end(mode, step, logs)
        self._reset_metrics()

        if mode == 'test':
            return logs, outputs
        return logs

    def _reset_metrics(self):
        for metric in self._metrics:
            metric.reset()

    def _metrics_name(self):
        metrics_name = ['loss']
        for m in self._metrics:
            metrics_name.extend(to_list(m.name()))
        return metrics_name

    def _len_data_loader(self, data_loader):
        try:
            steps = len(data_loader)
        except Exception:
            steps = None
        return steps
