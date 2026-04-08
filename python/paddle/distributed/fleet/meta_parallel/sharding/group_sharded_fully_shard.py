# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
from collections import OrderedDict
from types import MethodType

import numpy as np

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.distributed import collective, fleet
from paddle.framework import core
from paddle.nn import ClipGradByGlobalNorm

from .group_sharded_stage3 import (
    ForwardPostHooks,
    ForwardPreHooks,
    OrderedSet,
    TaskFlow,
    _current_layer_params,
    _PartitionParam,
    _TensorWrapper,
    _UnsliceParam,
    align,
    alignment,
)
from .group_sharded_storage import GradStorage
from .group_sharded_utils import GroupShardedClipGrad, Type, device_guard


def _OptimizerWrapper(optimizer, offload, group, update_params_slice):
    if not hasattr(optimizer, "_optim"):
        optimizer._optim = optimizer
        optimizer.offload = offload
        optimizer._group = group
        optimizer.update_scaler = None
        optimizer.update_slice = update_params_slice
    return optimizer


class FullyShardOptimizer:
    def __init__(
        self,
        optimizer,
        group=None,
        sync_buffers=False,
        device="xpu" if core.is_compiled_with_xpu() else "gpu",
        segment_size=2**20,
        pretrain_sync_models=True,
        offload=False,
        sync_comm=False,
        dp_group=None,
        exclude_layer=None,
    ):
        self._default_device = device
        self.__sync_buffers = sync_buffers
        self._offload = offload
        self._sync_comm = sync_comm

        # stage3 support some layer set by users to be unslice
        # _exclude_layer=[layer_name or id(layer)]
        self._exclude_layer = [] if exclude_layer is None else exclude_layer
        assert isinstance(self._exclude_layer, (list, tuple)), (
            "the exclude_layers must be a list with layers' name or layers' id"
        )

        # segmentation size
        assert segment_size >= 0, "segment_size must be GE than 0."
        self._segment_size = segment_size

        global param2dtype
        param2dtype = {}

        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_sharding_parallel_group()
        # Communication group establishment
        self._group = (
            collective.new_group(collective._get_global_group().ranks)
            if group is None
            else group
        )
        self._dp_group = dp_group
        self._world_size_scaling = 1.0 / self._group.nranks
        assert self._group.nranks > 1, (
            "Training must be distributed, ranks must be greater than 1."
        )
        self._rank = self._group.rank
        self._global_root_rank = self._group.ranks[
            0
        ]  # picking ranks index 0 as the reference

        # Parameter segmentation for global ranks
        self._unslice_params = OrderedSet()  # param's numel <= segment_size
        self._unslice_params2align = {}  # {param.name: param's align}
        self._grad_storages = {}  # {param.dtype: GradStorage}

        assert not isinstance(optimizer, list), (
            "Multiple optimizers are not supported now."
        )
        self._optim = _OptimizerWrapper(
            optimizer,
            self._offload,
            self._group,
            self._update_params_slice,
        )
        self._ori_parameter_list = self._optim._parameter_list
        self._ori_param_groups = self._optim._param_groups

        for p in self._ori_parameter_list:
            del p._need_shard
            if p._numel() > self._segment_size:
                pass
            elif p.trainable:
                self._unslice_params.add(_UnsliceParam(p))

        # check main_grad
        self._check_main_grad()

        # Replace optimizer's _grad_clip
        if isinstance(self._optim._grad_clip, ClipGradByGlobalNorm):
            logging.warning(
                "While using ClipGradByGlobalNorm in GroupShardedStage3, the grad clip of original optimizer will be changed."
            )
            if self.use_main_grad:
                self._optim._inner_opt._grad_clip = GroupShardedClipGrad(
                    self._optim._inner_opt._grad_clip,
                    paddle.get_device(),
                    self._group,
                )
            else:
                self._optim._grad_clip = GroupShardedClipGrad(
                    self._optim._grad_clip, paddle.get_device(), self._group
                )
            if self._optim._parameter_list and isinstance(
                self._optim._parameter_list[0], dict
            ):
                for item in self._optim._param_groups:
                    if "grad_clip" in item.keys():
                        item["grad_clip"] = self._optim._grad_clip

        # Add unslice params to master_weight in fp16
        self._setup_master_weights_for_unslice()

        # Redefine optimizer step and clear function
        self._redefine_opt_step()
        self._redefine_opt_clear()

    def _check_main_grad(self):
        self.use_main_grad = None
        for param in self._ori_parameter_list:
            if self.use_main_grad is None and hasattr(param, "main_grad"):
                self.use_main_grad = True
            if self.use_main_grad:
                assert hasattr(param, "main_grad"), (
                    "Params have different main grad attributes."
                )

    def _setup_master_weights_for_unslice(self):
        for param in self._unslice_params:
            # Update optimizer master weights
            if (
                param.dtype == Type.fp16.value or param.dtype == Type.bf16.value
            ) and not self._offload:
                master_tensor = paddle.cast(param, Type.fp32.value)
                master_tensor.name = param.name
                self._optim._master_weights[param.name] = master_tensor

    def _clear_gradients(self):
        current_layer_params = self._ori_parameter_list
        # 1.Handle param's slice
        trainable_params = list(
            filter(
                lambda p: p.trainable and p not in self._unslice_params,
                current_layer_params,
            )
        )
        for param in trainable_params:
            if not hasattr(param, "fw_storage"):
                continue
            assert hasattr(param, "fw_storage"), (
                f"Find {param.name} don't have fw_storage attribute."
            )
            if self.use_main_grad:
                param.fw_storage.main_grad._clear()
                param.fw_storage.main_grad = None
            else:
                param.fw_storage.clear_gradient(False)
            param.bw_storage._clear()
            param.bw_storage = None

    # Update param memory slice
    def _update_params_slice(self):
        update_list = self._update_params()

        if not isinstance(self._optim._param_groups[0], dict):
            slice_params = [param.fw_storage for param in update_list]
            self._optim._parameter_list = slice_params + list(
                self._unslice_params
            )
            self._optim._param_groups = slice_params + list(
                self._unslice_params
            )
        else:
            for param_group in self._optim._param_groups:
                p_group = []
                for p in param_group['params']:
                    if hasattr(p, "fw_storage"):
                        p_group.append(p.fw_storage)
                    else:
                        p_group.append(p)

                param_group['params'] = p_group

    def _update_params(self):
        """
        Update parameters to optimizer memory slice.
        """
        update_list = []
        current_layer_params = self._ori_parameter_list
        trainable_params = list(
            filter(
                lambda p: p.trainable and p not in self._unslice_params,
                current_layer_params,
            )
        )
        # 1.Handle param's slice
        for param in trainable_params:
            assert hasattr(param, "fw_storage"), (
                f"Find {param.name} don't have fw_storage attribute"
            )

            param.fw_storage = _TensorWrapper(param)
            if self.use_main_grad:
                param.fw_storage.main_grad = param.bw_storage
            else:
                assert param.fw_storage.grad is None
                param.fw_storage._copy_gradient_from(param.bw_storage)
            update_list.append(param)

        return update_list

    def _redefine_opt_step(self):
        params_slice_func = self._update_params_slice
        opt_step = self._optim.step

        def _opt_step(self):
            if not self.update_scaler:
                params_slice_func()
            opt_step()

        self._optim.step = MethodType(_opt_step, self._optim)

    def _redefine_opt_clear(self):
        clear_func = self._clear_gradients

        def _opt_clear(self):
            clear_func()

        self._optim.clear_grad = MethodType(_opt_clear, self._optim)


class FullyShard(nn.Layer):
    """
    A wrapper for Sharding Stage3 Layer in Dygraph.

    .. warning: GroupShardedStage3 encapsulates the layer strategy and integrates it into the nn.Layer.

    .. ZeRO: https://arxiv.org/pdf/1910.02054.pdf.
    """

    def __init__(
        self,
        layer,
        group=None,
        sync_buffers=False,
        device="xpu" if core.is_compiled_with_xpu() else "gpu",
        segment_size=2**20,
        pretrain_sync_models=True,
        offload=False,
        sync_comm=False,
        dp_group=None,
        exclude_layer=None,
    ):
        super().__init__()

        # Default configs
        assert (
            core.is_compiled_with_cuda()
            or core.is_compiled_with_xpu()
            or (device in core.get_all_custom_device_type())
        ), "Only support CUDA / XPU / CustomDevice."

        self._layer = layer
        self._default_device = device
        self.__sync_buffers = sync_buffers
        self._offload = offload
        self._sync_comm = sync_comm

        # stage3 support some layer set by users to be unslice
        self._exclude_layer = [] if exclude_layer is None else exclude_layer
        assert isinstance(self._exclude_layer, (list, tuple)), (
            "the exclude_layers must be a list with layers' name or layers' id"
        )

        # segmentation size
        assert segment_size >= 0, "segment_size must be GE than 0."
        self._segment_size = segment_size

        global DEV
        DEV = (
            "cpu"
            if paddle.get_device() == "cpu"
            else paddle.get_device().split(":")[0]
        )
        global DEV_ID
        DEV_ID = (
            0
            if paddle.get_device() == "cpu"
            else int(paddle.get_device().split(":")[1])
        )
        global param2dtype
        param2dtype = {}

        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_sharding_parallel_group()

        self._group = (
            collective.new_group(collective._get_global_group().ranks)
            if group is None
            else group
        )
        self._dp_group = dp_group
        self._world_size_scaling = 1.0 / self._group.nranks
        assert self._group.nranks > 1, (
            "Training must be distributed, ranks must be greater than 1."
        )
        self._rank = self._group.rank
        self._global_root_rank = self._group.ranks[
            0
        ]  # picking ranks index 0 as the reference

        # Parameter segmentation for global ranks
        # After flatten -> self._param2buffer_size, self._param2buffer, self._trainable_params
        self._param2buffer_size = {}  # {param.name: size}
        self._param2buffer = {}  # {param.name: [(start0, end0),(start1, end1), ...]}
        self._trainable_params = {}  # {id(layer): [trainable_params]}
        self._unslice_params = OrderedSet()  # param's numel <= segment_size
        self._unslice_params2align = {}  # {param.name: param's align}
        self._grad_storages = {}  # {param.dtype: GradStorage}

        self._ori_parameter_list = self._layer.parameters()
        for param in self._ori_parameter_list:
            param._need_shard = True
        # check main_grad
        self._check_main_grad()

        # Synchronous all ranks models
        if pretrain_sync_models:
            self._sync_params_and_buffers()

        self._segment_rank_params(self._layer)

        # In the first step, record the execution order of the layer
        self._order_tracer = OrderedDict()
        self._order_tracer["order"] = 0
        self._order_tracer["layer"] = []

        # Add unslice params GradStorage
        self._handle_unslice_params()

        # Register task flow
        self._task_flow = TaskFlow()

        # Register forward hooks
        self._register_forward_hooks(self._layer)

        # Register backward parameter hooks
        self._register_backward_hooks()

    def _handle_unslice_params(self):
        buffer_size = {}
        buffer_size[Type.bf16.value] = 0
        buffer_size[Type.fp32.value] = 0
        buffer_size[Type.fp16.value] = 0
        for param in self._unslice_params:
            param2dtype[param.name] = param.dtype
            p_align = self._param2align(param)
            self._unslice_params2align[param.name] = p_align
            buffer_size[param.dtype] += param._numel() + p_align

        # Create unslice_params'grad
        for param in sorted(self._unslice_params, key=lambda p: p.name):
            if param.dtype not in self._grad_storages.keys():
                self._grad_storages[param.dtype] = GradStorage(
                    buffer_size[param.dtype],
                    dtype=(
                        param.dtype
                        if not self.use_main_grad
                        else paddle.float32
                    ),
                    device=self._default_device,
                    destination=self._rank,
                    param2align=self._unslice_params2align,
                )
            self._grad_storages[param.dtype].add_grad(
                param, self._unslice_params2align[param.name]
            )

    def _check_main_grad(self):
        self.use_main_grad = None
        for param in self._layer.parameters():
            if self.use_main_grad is None and hasattr(param, "main_grad"):
                self.use_main_grad = True
            if self.use_main_grad:
                assert hasattr(param, "main_grad"), (
                    "Params have different main grad attributes."
                )

    @paddle.autograd.no_grad()
    def _sync_params_and_buffers(self):
        """
        Sync all model states for all ranks
        """

        for p in self._layer.parameters():
            dist.broadcast(
                p, src=self._global_root_rank, group=self._group, sync_op=True
            )
            if self._dp_group is not None and self._dp_group.nranks > 1:
                dist.broadcast(
                    p,
                    src=self._dp_group.ranks[0],
                    group=self._dp_group,
                    sync_op=True,
                )

    def _sync_grad_storages_hook(self):
        for grad_storage in self._grad_storages.values():
            grad_storage.buffer.scale_(scale=self._world_size_scaling)
            dist.all_reduce(tensor=grad_storage.buffer, group=self._group)
            if self._dp_group is not None and self._dp_group.nranks > 1:
                grad_storage.buffer.scale_(scale=(1.0 / self._dp_group.nranks))
                dist.all_reduce(
                    tensor=grad_storage.buffer, group=self._dp_group
                )

    def forward(self, *inputs, **kwargs):
        """
        A wrapper for Sharding Stage3 layer.
        """
        # add hook to sync grad storage
        for grad_storage in self._grad_storages.values():
            grad_storage.buffer.zero_()
            grad_storage.manual_release()
            grad_storage.rebuild()
        core.eager._add_backward_final_hook(self._sync_grad_storages_hook)

        # 1.Sync layer's buffers state
        if self.__sync_buffers:
            self._sync_buffers()

        # 2.Normal FW on the base model
        fw = self._layer(*inputs, **kwargs)

        return fw

    def _segment_rank_params(self, layer, name="last_layer"):
        """
        Flatten parameters according to layer.
        """
        current_layer_params = _current_layer_params(layer)
        if current_layer_params:
            self._flatten_layer_params(layer, current_layer_params)

        for name, sub_layer in layer.named_children():
            self._segment_rank_params(sub_layer, name)

    def _flatten_layer_params(self, layer, current_layer_params):
        """
        Parameter segmentation and memory integration.
        """

        if id(layer) in self._trainable_params.keys():
            return

        def _add_manage_info(trainable_param):
            return _PartitionParam(trainable_param)

        current_params = []
        for p in current_layer_params:
            if p._numel() > self._segment_size:
                current_params.append(_add_manage_info(p))
            elif p.trainable:
                self._unslice_params.add(_UnsliceParam(p))

        self._trainable_params[id(layer)] = current_params

        for param in self._trainable_params[id(layer)]:
            if param.name in self._param2buffer.keys():
                continue
            self._param2buffer[param.name] = []
            # 1.Params alignment
            align_ = self._param2align(param)

            offset = align_ + param._numel()
            buffer_size = (
                offset
                if offset % self._group.nranks == 0
                else offset + self._group.nranks - (offset % self._group.nranks)
            )
            self._param2buffer_size[param.name] = buffer_size

            # 2.Combination param buffer
            assert buffer_size % self._group.nranks == 0
            pre_buffer = buffer_size // self._group.nranks

            for rank_ in range(self._group.nranks):
                self._param2buffer[param.name].append(
                    (rank_ * pre_buffer, (rank_ + 1) * pre_buffer)
                )

            # Record param's dtype
            param2dtype[param.name] = param.dtype
            # 3.Flatten layer params and release other rank buffer
            self._param_storage(param, buffer_size)

    def _param_storage(self, param, buffer_size):
        """
        This is a function to simplify the handling of parameter InternalStorages.
        """
        assert isinstance(buffer_size, int)
        value = (
            np.zeros(buffer_size, dtype=np.float16)
            if (
                Type.fp16.value == param.dtype or Type.bf16.value == param.dtype
            )
            else np.zeros(buffer_size, dtype=np.float32)
        )
        buffer = core.eager.Tensor(value=value, place=core.CPUPlace())
        if Type.bf16.value == param.dtype:
            buffer = buffer.cast(Type.bf16.value)

        param_shape = param.shape
        origin_state = param.stop_gradient
        param.stop_gradient = True
        param.flatten_()
        param.stop_gradient = origin_state
        start, end = self._param2buffer[param.name][self._rank]

        # Copy the current param value
        with device_guard():
            tmp_var = buffer._slice(0, param._numel())
        param_cpu = param.cpu()
        tmp_var.get_tensor().set(param_cpu.get_tensor(), core.CPUPlace())
        del tmp_var
        param.get_tensor()._set_dims(param_shape)

        # Current rank param_storage
        param.fw_storage = core.eager.Tensor(
            value=buffer._slice(start, end), name="slice@" + param.name
        )
        param.status = "part"
        param._clear_data()

    def _register_forward_hooks(self, layer):
        """
        Register PyLayer to manage memory slices.
        There are four stages:
        FW
        1. Before the forward layers, synchronize the full parameters.
        2. After the forward layers, release the full parameter and keep the parameter slice.
        BW
        3. Before the backward layers, synchronize the full parameters and create param's grad.
        4. After the gradient accumulation, release the full parameter and keep the parameter slice.
        """
        current_layer_params = _current_layer_params(layer)
        if current_layer_params:
            # the layer in self._exclude_layer will be added hooks.
            if not (
                id(layer) in self._exclude_layer
                or layer.__class__.__name__ in self._exclude_layer
            ):
                self._register_forward_all_hooks(layer, self._task_flow)

        for _, sub_layer in layer.named_children():
            self._register_forward_hooks(sub_layer)

    def _register_forward_all_hooks(self, sub_layer, task_flow):
        def _forward_pre_hook(layer, inputs):
            return ForwardPreHooks(
                layer,
                self._order_tracer,
                self._trainable_params,
                self._param2buffer_size,
                self._group,
                self._sync_comm,
                self._offload,
                task_flow,
            )

        def _forward_post_hook(layer, inputs, outputs):
            if isinstance(outputs, paddle.Tensor):
                outputs = (outputs,)
            return ForwardPostHooks.apply(
                *outputs,
                layer=layer,
                order_tracer=self._order_tracer,
                trainable_params=self._trainable_params,
                param2buffer=self._param2buffer,
                param2buffer_size=self._param2buffer_size,
                rank=self._rank,
                group=self._group,
                sync_comm=self._sync_comm,
                offload=self._offload,
                task_flow=task_flow,
            )

        # register previous forward hooks
        sub_layer.register_forward_pre_hook(_forward_pre_hook)

        # register post forward hooks
        sub_layer.register_forward_post_hook(_forward_post_hook)

    @paddle.autograd.no_grad()
    def _sync_buffers(self):
        """
        Sync all the param buffers from all ranks (exp: batch norm statistics).
        """

        for buffer in self._layer.buffers(include_sublayers=True):
            dist.broadcast(
                buffer, self._global_root_rank, self._group, sync_op=True
            )
            if self._dp_group is not None and self._dp_group.nranks > 1:
                dist.broadcast(
                    buffer,
                    self._dp_group.ranks[0],
                    self._dp_group,
                    sync_op=True,
                )

    def __getattr__(self, name):
        """Forward missing attributes to wrapped layer."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._layer, name)

    def _register_backward_hooks(self):
        current_layer_params = self._layer.parameters(include_sublayers=True)
        trainable_params = list(
            filter(
                lambda p: p.trainable and p not in self._unslice_params,
                current_layer_params,
            )
        )

        for param in trainable_params:
            allreduce_function = self._get_allreduce_fn(param)
            param._register_backward_hook(allreduce_function)

    def _get_allreduce_fn(self, param):
        @paddle.autograd.no_grad()
        def allreduce_(*_):
            assert param.trainable, (
                "the param must be trainable for grad allreduced"
            )
            if param.name in self._task_flow.full_grad.keys():
                full_grad = self._task_flow.full_grad[param.name]
                # Only support sync allreduce current rank's layer now
                full_grad.scale_(scale=self._world_size_scaling)
                dist.all_reduce(tensor=full_grad, group=self._group)
                if self._dp_group is not None and self._dp_group.nranks > 1:
                    full_grad.scale_(scale=1.0 / self._dp_group.nranks)
                    dist.all_reduce(tensor=full_grad, group=self._dp_group)

                start, end = self._param2buffer[param.name][self._rank]
                if param.bw_storage is None:
                    param.bw_storage = (
                        full_grad._slice(start, end).detach().clone()
                    )
                else:
                    param.bw_storage = paddle.add(
                        param.bw_storage,
                        full_grad._slice(start, end).detach().clone(),
                    )

                if self.use_main_grad:
                    param.main_grad = None
                else:
                    param.clear_gradient(False)
                del self._task_flow.full_grad[param.name]

            if param.name in self._task_flow.full_param.keys():
                if param.status == "all":
                    param.use_count = 0
                    param._clear_data()
                    start, end = self._param2buffer[param.name][self._rank]
                    param.fw_storage = (
                        self._task_flow.full_param[param.name][0]
                        ._slice(start, end)
                        .detach()
                        .clone()
                    )
                    param.status = "part"
                    del self._task_flow.full_param[param.name]

        return allreduce_

    def _param2align(self, param):
        # CUDA alignment 256 bytes
        size = param._numel() * align[param.dtype]
        device_alignment = alignment[self._default_device]
        remaining = size % device_alignment
        ali = 0 if remaining == 0 else device_alignment - remaining
        align_ = ali // align[param.dtype]
        return align_
