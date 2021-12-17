#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import copy
import time
import contextlib
import logging
import functools
import numpy as np
from itertools import chain
from functools import reduce
from types import MethodType
from collections import deque, OrderedDict

import paddle
from paddle import nn
from paddle.autograd import PyLayer
import paddle.fluid.core as core
import paddle.distributed as dist
from paddle.fluid.framework import ParamBase

from .sharding_utils import Type
from ..pp_utils.utils import _all_gather

# CUDA alignment 256 bytes
alignment = {"gpu": 256, }
align = {
    Type.fp16.value: 2,
    Type.fp32.value: 4,
}


class ShardingStage3(nn.Layer):
    """ 
    A wrapper for Sharding Stage3 Layer in Dygraph. 

    .. warning: ShardingStage3 encapsulates the layer strategy and integrates it into the nn.Layer.

    .. ZeRO: https://arxiv.org/pdf/1910.02054.pdf.
    """

    # TODO (Baibaifan) 
    # Feature Notes::
    # 1. The model supports the segmentation of parameters by global ranks in layers.
    # 2. Support communication flow and computing flow.
    # 3. Support offload function.
    # 4. Support the establishment of independent communication groups.

    def __init__(self,
                 layer,
                 optimizer,
                 group,
                 sync_buffers=False,
                 device="gpu",
                 accumulate_grads=False,
                 not_update_list=[]):
        super().__init__()

        # Default configs
        assert core.is_compiled_with_cuda(), "Only support CUDA."
        self._layer = layer
        self._default_device = device
        self.__sync_buffers = sync_buffers
        self._accumulate_grads = accumulate_grads

        assert not isinstance(
            optimizer, list), "Multiple optimizers are not supported now."
        self._optim = optimizer
        self._ori_parameter_list = self._optim._parameter_list
        self._ori_param_groups = self._optim._param_groups
        self._not_update_list = not_update_list

        # Communication group establishment
        assert group is not None, "Distributed communication group is must be gived."
        self._group = group
        self._world_size_scaling = 1.0 / self._group.nranks
        assert self._group.nranks > 1, "Training must be distributed, ranks must be greater than 1."
        self._rank = self._group.rank
        self._global_root_rank = 0  # picking rank 0 as the reference
        self._global_ranks = self._group.ranks

        # Parameter segmentation for global ranks
        # After flatten -> self._param2buffer_size, self._param2buffer, self._trainable_params
        self._param2buffer_size = dict()  # {param.name: size}
        self._param2buffer = dict(
        )  # {param.name: [(start0, end0),(start1, end1), ...]}
        self._trainable_params = dict()  # {layer.name: [trainable_params]}

        self._segment_rank_params(self._layer)

        # In the first step, record the execution order of the layer
        self._order_tracer = OrderedDict()
        self._order_tracer["order"] = 0
        self._order_tracer["layer"] = []

        # Register task flow
        self._task_flow = TaskFlow()

        # Register forward hooks
        self._register_forward_hooks(self._layer)

        # Register backward parameter hooks
        self._register_backward_hooks()

        # Redefine optimizer step and clear function
        self._redefine_opt_step()
        self._redefine_opt_clear()

    @property
    def not_update_list(self):
        return self._not_update_list

    @not_update_list.setter
    def not_update_list(self, not_update_list):
        self._not_update_list = not_update_list

    def _clear_gradients(self):
        assert len(self._trainable_params.keys()) > 0
        current_layer_params = self._layer.parameters(include_sublayers=True)
        trainable_params = list(
            filter(lambda x: x.trainable, current_layer_params))
        for param in trainable_params:
            assert hasattr(
                param, "fw_storage"
            ), "Find {} don't have fw_storage attribute.".format(param.name)

            # param.bw_storage.zero_()
            param.fw_storage.clear_gradient(False)
            param.fw_storage._gradient_set_empty(False)
            param.bw_storage._clear()

    # Update param memery slice
    def _update_params_slice(self):
        update_list, not_update_list = self._update_params()
        # Not_update_list post processing
        if not isinstance(self._optim._param_groups[0], dict):
            slice_params = [param.fw_storage for param in update_list]
            self._optim._parameter_list = slice_params
            self._optim._param_groups = slice_params
        else:
            params_name_list = list(map(lambda p: p.name, update_list))
            for param_group in self._optim._param_groups:
                slice_p = []
                for p in param_group['params']:
                    assert hasattr(
                        p, "fw_storage"
                    ), "Find {} don't have fw_storage attribute.".format(p.name)
                    if p.name in params_name_list:
                        slice_p.append(p.fw_storage)
                    param_group['params'] = slice_p

    def forward(self, *inputs, **kwargs):
        """
        A wrapper for Sharding Stage3 layer.
        """
        # 1.Sync layer's buffers state
        if self.__sync_buffers:
            self._sync_buffers()

        # 2.Normal FW on the base model
        fw = self._layer(*inputs, **kwargs)

        return fw

    def _segment_rank_params(self, layer):
        """
        Flatten parameters according to layer.
        """
        current_layer_params = _current_layer_params(layer)
        if current_layer_params:
            self._flatten_layer_params(layer, current_layer_params)

        for _, sub_layer in layer.named_children():
            self._segment_rank_params(sub_layer)

    def _flatten_layer_params(self, layer, current_layer_params):
        """
        Parameter segmentation and memory integration.
        """

        def _add_manage_info(trainable_param):
            return _PartitionParam(trainable_param)

        trainable_params = list(
            filter(lambda x: x.trainable, current_layer_params))
        assert id(layer) not in self._trainable_params.keys()
        self._trainable_params[id(layer)] = list(
            map(_add_manage_info, trainable_params))

        for param in self._trainable_params[id(layer)]:
            if param.name in self._param2buffer.keys():
                continue
            self._param2buffer[param.name] = []
            # 1.Params alignment
            offset = 0
            # CUDA alignment 256 bytes
            size = param._numel() * align[param.dtype]
            remaining = size % alignment[self._default_device]
            ali = 0 if remaining == 0 else alignment[
                self._default_device] - remaining
            align_ = ali // align[param.dtype]

            offset = align_ + param._numel()
            buffer_size = offset if offset % self._group.nranks == 0 else offset + self._group.nranks - (
                offset % self._group.nranks)
            self._param2buffer_size[param.name] = buffer_size

            # 2.Combination param buffer
            assert buffer_size % self._group.nranks == 0
            pre_buffer = buffer_size // self._group.nranks

            for rank_ in range(self._group.nranks):
                self._param2buffer[param.name].append(
                    (rank_ * pre_buffer, (rank_ + 1) * pre_buffer))

            # 3.Flatten layer params and release other rank buffer
            self._param_storage(param, buffer_size)

    def _param_storage(self, param, buffer_size):
        """
        This is a function to simplify the handling of parameter InternalStorages.
        """
        assert isinstance(buffer_size, int)
        value = np.zeros(
            buffer_size,
            dtype=np.float16) if Type.fp16.value == param.dtype else np.zeros(
                buffer_size, dtype=np.float32)
        buffer = core.VarBase(value=value, place=core.CPUPlace())

        param_shape = param.shape
        origin_state = param.stop_gradient
        param.stop_gradient = True
        param.flatten_()
        param.stop_gradient = origin_state
        start, end = self._param2buffer[param.name][self._rank]

        # Copy the current param value
        tmp_var = core.VarBase(
            tensor=buffer._slice(0, param._numel()), place=core.CPUPlace())
        param_cpu = param.cpu()
        tmp_var.value().get_tensor().set(param_cpu.value().get_tensor(),
                                         core.CPUPlace())
        param.value().get_tensor()._set_dims(param_shape)
        param._clear()

        # current rank param_storage
        param.fw_storage = core.VarBase(
            buffer._slice(start, end), param.name + "@slice")
        param.status = "part"

    def _register_forward_hooks(self, layer):
        """
        Register pylayer to manage memory slices.
        There are four stages:
        1.
        2.
        3.
        4.
        """
        current_layer_params = _current_layer_params(layer)
        if current_layer_params:
            self._register_forward_all_hooks(layer, self._task_flow)

        for _, sub_layer in layer.named_children():
            self._register_forward_hooks(sub_layer)

    def _register_forward_all_hooks(self, sub_layer, task_flow):
        def _forward_pre_hook(layer, inputs):
            return ForwardPreHooks.apply(
                inputs, layer, self._order_tracer, self._trainable_params,
                self._param2buffer, self._rank, self._group, task_flow)

        def _forward_post_hook(layer, inputs, outputs):
            return ForwardPostHooks.apply(
                outputs, layer, self._order_tracer, self._trainable_params,
                self._param2buffer, self._param2buffer_size, self._rank,
                self._group, task_flow)

        # register previous forward hooks
        sub_layer.register_forward_pre_hook(_forward_pre_hook)

        # register post forward hooks
        sub_layer.register_forward_post_hook(_forward_post_hook)

    @paddle.no_grad()
    def _sync_buffers(self):
        """
        Sync all the param buffers from all ranks (exp: batch norm statistics).
        """

        for buffer in self._layer.buffers(include_sublayers=True):
            dist.broadcast(
                buffer,
                self._global_root_rank,
                self._group,
                use_calc_stream=True)
        # Multi stream operation will be supported later
        dist.wait(tensor=buffer, group=self._group, use_calc_stream=True)

    def __getattr__(self, name):
        """Forward missing attributes to wrapped layer."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._layer, name)

    def _update_params(self):
        """
        Update parameters to optimizer memory slice.
        """
        update_list, not_update_list = [], []
        assert len(self._trainable_params.keys()) > 0
        current_layer_params = self._layer.parameters(include_sublayers=True)
        trainable_params = list(
            filter(lambda x: x.trainable, current_layer_params))
        for param in trainable_params:
            assert hasattr(
                param,
                "fw_storage"), "Find {} don't have fw_storage attribute".format(
                    param.name)

            if self._accumulate_grads:
                param.bw_storage.scale_(scale=self._world_size_scaling)
            param.fw_storage = _VarBaseWrapper(param)
            param.fw_storage._copy_gradient_from(param.bw_storage)
            if param.name in self._not_update_list:
                not_update_list.append(param)
            else:
                update_list.append(param)
        return update_list, not_update_list

    def restore_parameters(self):
        """
        Get the full parameters and return the corresponding task flows.
        """
        assert len(self._trainable_params.keys()) > 0
        current_layer_params = self._layer.parameters(include_sublayers=True)
        trainable_params = list(
            filter(lambda x: x.trainable, current_layer_params))
        for param in trainable_params:
            if param.use_count > 0:
                continue
            assert hasattr(
                param,
                "fw_storage"), "Find {} don't have fw_storage attribute".format(
                    param.name)

            full_param = _all_gather(
                param.fw_storage, self._group, use_calc_stream=True)
            dist.wait(
                tensor=full_param, group=self._group, use_calc_stream=True)
            core.VarBase(full_param._slice(0, param._numel()))._share_buffer_to(
                param)
            param.value().get_tensor()._set_dims(param.shape)
            param.fw_storage._clear()
            param.fw_storage = None
            param.status = "all"
            param.use_count += 1

        self._optim._parameter_list = self._ori_parameter_list
        self._optim._param_groups = self._ori_param_groups

    def _register_backward_hooks(self):
        current_layer_params = self._layer.parameters(include_sublayers=True)
        trainable_params = list(
            filter(lambda x: x.trainable, current_layer_params))

        for param in trainable_params:
            allreduce_function = self._get_allreduce_fn(param)
            param._register_backward_hook(allreduce_function)

    def _get_allreduce_fn(self, param):
        @paddle.no_grad()
        def reduce(*_):
            if param.name in self._task_flow.full_grad.keys():
                full_grad = self._task_flow.full_grad[param.name]
                if not self._accumulate_grads:
                    full_grad.scale_(scale=self._world_size_scaling)
                # Only support sync allreduce current rank's layer now
                dist.all_reduce(
                    tensor=full_grad, group=self._group, use_calc_stream=True)
                dist.wait(
                    tensor=full_grad, group=self._group, use_calc_stream=True)

                start, end = self._param2buffer[param.name][self._rank]
                if not self._accumulate_grads or param.bw_storage is None:
                    param.bw_storage = core.VarBase(
                        full_grad._slice(start, end)).detach().clone()
                else:
                    param.bw_storage.add_(
                        core.VarBase(full_grad._slice(start, end)).detach()
                        .clone())
                param.clear_gradient(False)
                param._gradient_set_empty(False)
                tmp_var = self._task_flow.full_grad.pop(param.name)
                tmp_var._clear()

            if param.name in self._task_flow.full_param.keys():
                if param.status == "all":
                    param.use_count = 0
                    param._clear()
                    start, end = self._param2buffer[param.name][self._rank]
                    param.fw_storage = core.VarBase(
                        self._task_flow.full_param[param.name]._slice(
                            start, end), param.name + "@slice").detach().clone()
                    param.status = "part"
                    tmp_var = self._task_flow.full_param.pop(param.name)
                    tmp_var._clear()

        return reduce

    def _redefine_opt_step(self):
        params_slice_func = self._update_params_slice
        opt_step = self._optim.step

        def _opt_step(self):
            params_slice_func()
            opt_step()

        self._optim.step = MethodType(_opt_step, self._optim)

    def _redefine_opt_clear(self):
        clear_func = self._clear_gradients

        def _opt_clear(self):
            clear_func()

        self._optim.clear_grad = MethodType(_opt_clear, self._optim)


class ForwardPreHooks(PyLayer):
    @staticmethod
    def forward(ctx, inputs, layer, order_tracer, trainable_params,
                param2buffer, rank, group, task_flow):
        # Record layer's id
        layer_id = id(layer)
        use_calc, sync_wait = False, False

        ctx.rank = rank
        ctx.layer = layer
        ctx.task_flow = task_flow
        ctx.trainable_params = trainable_params
        ctx.param2buffer = param2buffer

        if layer_id not in order_tracer.keys():
            use_calc, sync_wait = True, True

            # Whether to use calc stream
            task_flow.use_calc[layer_id] = use_calc
        else:
            # Whether to use calc stream
            task_flow.use_calc[layer_id] = use_calc
            order_ = order_tracer[layer_id]
            if not order_:
                # Allgather current layer in the 1st step of the 1st layer
                _allgather_buffer(
                    layer_id,
                    trainable_params,
                    group,
                    use_calc_stream=use_calc,
                    task_flow=task_flow,
                    sync_wait=True)

            if layer_id == order_tracer["layer"][-1]:
                return inputs
            layer_id = order_tracer["layer"][order_ + 1]
        _allgather_buffer(
            layer_id,
            trainable_params,
            group,
            use_calc_stream=use_calc,
            task_flow=task_flow,
            sync_wait=sync_wait)

        return inputs

    @staticmethod
    def backward(ctx, *args):
        rank = ctx.rank
        layer = ctx.layer
        task_flow = ctx.task_flow
        trainable_params = ctx.trainable_params
        param2buffer = ctx.param2buffer

        # release current layer full params
        _release_param(layer, trainable_params, param2buffer, rank, task_flow)

        return args


class ForwardPostHooks(PyLayer):
    @staticmethod
    def forward(ctx, inputs, layer, order_tracer, trainable_params,
                param2buffer, param2buffer_size, rank, group, task_flow):

        # release current layer full params
        _release_param(layer, trainable_params, param2buffer, rank, task_flow)

        layer_id = id(layer)
        use_calc_stream = task_flow.use_calc[layer_id]
        if layer_id not in order_tracer.keys():
            order_ = order_tracer["order"]
            order_tracer[layer_id] = order_
            order_tracer["order"] += 1
            order_tracer["layer"].append(layer_id)

        # wait next layer params
        if not use_calc_stream:
            next_index = order_tracer[layer_id] + 1
            if next_index < order_tracer["order"]:
                next_layer_id = order_tracer["layer"][next_index]
                _wait_layer(trainable_params, next_layer_id, task_flow, group,
                            use_calc_stream)

        #Record bw info 
        ctx.order_tracer = order_tracer
        ctx.task_flow = task_flow
        ctx.group = group
        ctx.layer = layer
        ctx.trainable_params = trainable_params
        ctx.param2buffer_size = param2buffer_size

        return inputs

    @staticmethod
    def backward(ctx, *args):
        # Load context value
        order_tracer = ctx.order_tracer
        task_flow = ctx.task_flow
        group = ctx.group
        layer = ctx.layer
        trainable_params = ctx.trainable_params
        param2buffer_size = ctx.param2buffer_size
        layer_id = id(layer)
        last_layer = layer_id == ctx.order_tracer["layer"][-1]
        use_calc, sync_wait = False, False

        # Create params's grad
        _create_params_grad(layer, trainable_params, param2buffer_size,
                            task_flow)

        # Allgather params in last layer or wait next layer params
        if last_layer:
            _allgather_buffer(
                layer_id,
                trainable_params,
                group,
                use_calc_stream=use_calc,
                task_flow=task_flow,
                sync_wait=True)
        else:
            _wait_layer(trainable_params, layer_id, task_flow, group, use_calc)

        # Whether to use calc stream
        task_flow.use_calc[layer_id] = use_calc
        if layer_id != order_tracer["layer"][0]:
            layer_next_id = order_tracer["layer"][order_tracer[layer_id] - 1]
            _allgather_buffer(
                layer_next_id,
                trainable_params,
                group,
                use_calc_stream=use_calc,
                task_flow=task_flow,
                sync_wait=sync_wait)

        return args


class TaskFlow:
    """
    Task flows, one way linked list for task acquisition.
    """

    def __init__(self,
                 full_param=dict(),
                 full_grad=dict(),
                 use_calc=dict(),
                 callback=None):
        self.full_param = full_param
        self.full_grad = full_grad
        self.use_calc = use_calc
        self.callback = callback


def _release_param(layer, trainable_params, param2buffer, rank, task_flow):
    for param in trainable_params[id(layer)]:
        param.use_count -= 1
        if param.use_count == 0:
            param._clear()
            if param.name in task_flow.full_param.keys():
                start, end = param2buffer[param.name][rank]
                param.fw_storage = core.VarBase(
                    task_flow.full_param[param.name]._slice(start, end),
                    param.name + "@slice").detach().clone()
                param.status = "part"
                tmp_var = task_flow.full_param.pop(param.name)
                tmp_var._clear()
    return


def _wait_layer(trainable_params, next_layer_id, task_flow, group,
                use_calc_stream):
    for param in trainable_params[next_layer_id]:
        if param.status == "all":
            continue
        if param.name in task_flow.full_param.keys():
            full_param = task_flow.full_param[param.name]
            dist.wait(
                tensor=full_param, group=group, use_calc_stream=use_calc_stream)
            core.VarBase(full_param._slice(0, param._numel()))._share_buffer_to(
                param)
            param.value().get_tensor()._set_dims(param.shape)
            param.fw_storage._clear()
            param.fw_storage = None
            param.status = "all"
            param.use_count += 1
    return task_flow


def _allgather_buffer(layer_id,
                      trainable_params,
                      group,
                      use_calc_stream,
                      task_flow,
                      sync_wait=False):
    for param in trainable_params[layer_id]:
        if param.status == "all":
            param.use_count += 1
            continue
        full_param = _all_gather(
            param.fw_storage, group, use_calc_stream=use_calc_stream)
        # Allgather current layer in the 1st step 
        if sync_wait:
            dist.wait(
                tensor=full_param, group=group, use_calc_stream=use_calc_stream)
            core.VarBase(full_param._slice(0, param._numel()))._share_buffer_to(
                param)
            param.value().get_tensor()._set_dims(param.shape)
            param.fw_storage._clear()
            param.fw_storage = None
            param.status = "all"
            param.use_count += 1
        task_flow.full_param[param.name] = full_param
    return task_flow


@paddle.no_grad()
def _create_params_grad(layer, trainable_params, param2buffer_size, task_flow):
    for param in trainable_params[id(layer)]:
        if param.name in task_flow.full_grad.keys():
            continue
        assert isinstance(param2buffer_size[param.name], int)
        temp_grad = paddle.zeros(
            [param2buffer_size[param.name]], dtype=param.dtype)
        param._copy_gradient_from(
            core.VarBase(temp_grad._slice(0, param._numel())))
        task_flow.full_grad[param.name] = temp_grad
    return task_flow


def _PartitionParam(param):
    if not hasattr(param, "fw_storage"):
        setattr(param, "fw_storage", None)
        setattr(param, "bw_storage", None)
        setattr(param, "status", "all")
        setattr(param, "use_count", 0)
    return param


def _VarBaseWrapper(param):
    varbase = param.fw_storage
    tmp_param = ParamBase(
        shape=varbase.shape, dtype=varbase.dtype, name="slice@" + param.name)
    varbase._share_buffer_to(tmp_param)
    tmp_param.regularizer = param.regularizer
    tmp_param.optimize_attr['learning_rate'] = param.optimize_attr[
        'learning_rate']
    varbase._clear()
    return tmp_param


def _current_layer_params(layer):
    return layer.parameters(
        include_sublayers=False) + list(layer.extra_parameters) if hasattr(
            layer, "extra_parameters") else layer.parameters(
                include_sublayers=False)
