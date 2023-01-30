#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

<<<<<<< HEAD
import logging
from collections import OrderedDict
from types import MethodType

import numpy as np

import paddle
import paddle.distributed as dist
from paddle import framework, nn
from paddle.autograd import PyLayer
from paddle.distributed import collective
from paddle.fluid.framework import EagerParamBase
from paddle.framework import core
from paddle.nn import ClipGradByGlobalNorm

from .group_sharded_storage import GradStorage
from .group_sharded_utils import GroupShardedClipGrad, Type, device_guard
=======
import time
import logging
import numpy as np
from types import MethodType
from collections import OrderedDict

import paddle
from paddle import nn
from paddle.autograd import PyLayer
import paddle.fluid.core as core
import paddle.fluid.framework as framework
from paddle.fluid.framework import EagerParamBase
from paddle.fluid.clip import ClipGradByGlobalNorm
from paddle.distributed import collective

from .group_sharded_storage import GradStorage
from .group_sharded_utils import Type, GroupShardedClipGrad, device_guard
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


def _all_gather(tensor, buffer_size, group):
    """
<<<<<<< HEAD
    The main difference with paddle.distributed.all_gather:
=======
    The main difference with paddle.distributed.all_gather: 
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    no need to pass in tensor_list, the returned tensor is spliced
    """

    assert group is not None
    if framework.in_dygraph_mode():
        out = paddle.zeros([buffer_size], dtype=tensor.dtype)
        task = group.process_group.all_gather(tensor, out)
        return out, task


# CUDA alignment 256 bytes
alignment = {
    "gpu": 256,
}
align = {
    Type.fp16.value: 2,
    Type.fp32.value: 4,
}

global CHECK_LAYER
CHECK_LAYER = dict()  # Help to check layer's id -> layer's name


class GroupShardedStage3(nn.Layer):
<<<<<<< HEAD
    """
    A wrapper for Sharding Stage3 Layer in Dygraph.
=======
    """ 
    A wrapper for Sharding Stage3 Layer in Dygraph. 
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    .. warning: GroupShardedStage3 encapsulates the layer strategy and integrates it into the nn.Layer.

    .. ZeRO: https://arxiv.org/pdf/1910.02054.pdf.
    """

    # TODO (Baibaifan)
    # Feature Notes::
    # 1. The model supports the segmentation of parameters by global ranks in layers.
    # 2. Support communication flow and computing flow.
    # 3. Support offload function.
    # 4. Support the establishment of independent communication groups.

<<<<<<< HEAD
    def __init__(
        self,
        layer,
        optimizer,
        group=None,
        sync_buffers=False,
        device="gpu",
        segment_size=2**20,
        pertrain_sync_models=True,
        offload=False,
        sync_comm=False,
        dp_group=None,
        exclude_layer=None,
    ):
=======
    def __init__(self,
                 layer,
                 optimizer,
                 group=None,
                 sync_buffers=False,
                 device="gpu",
                 segment_size=2**20,
                 pertrain_sync_models=True,
                 offload=False,
                 sync_comm=False):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        super().__init__()

        # Default configs
        assert core.is_compiled_with_cuda(), "Only support CUDA."
        self._layer = layer
        self._default_device = device
        self.__sync_buffers = sync_buffers
        self._offload = offload
        self._sync_comm = sync_comm
<<<<<<< HEAD

        # stage3 support some layer set by users to be unslice
        # _exclude_layer=[layer_name or id(layer)]
        self._exclude_layer = [] if exclude_layer is None else exclude_layer
        assert isinstance(
            self._exclude_layer, (list, tuple)
        ), "the exclude_layers must be a list with layers' name or layers' id"

=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        # segmentation size
        assert segment_size >= 0, "segment_size must be GE than 0."
        self._segment_size = segment_size

        global DEV
<<<<<<< HEAD
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
=======
        DEV = "cpu" if paddle.get_device() == "cpu" else paddle.get_device(
        ).split(":")[0]
        global DEV_ID
        DEV_ID = 0 if paddle.get_device() == "cpu" else int(
            paddle.get_device().split(":")[1])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        global param2dtype
        param2dtype = dict()

        # Communication group establishment
<<<<<<< HEAD
        self._group = (
            collective.new_group(collective._get_global_group().ranks)
            if group is None
            else group
        )
        self._dp_group = dp_group
        self._world_size_scaling = 1.0 / self._group.nranks
        assert (
            self._group.nranks > 1
        ), "Training must be distributed, ranks must be greater than 1."
        self._rank = self._group.rank
        self._global_root_rank = self._group.ranks[
            0
        ]  # picking ranks index 0 as the reference
=======
        self._group = collective.new_group(
            collective._get_global_group().ranks) if group is None else group
        self._world_size_scaling = 1.0 / self._group.nranks
        assert self._group.nranks > 1, "Training must be distributed, ranks must be greater than 1."
        self._rank = self._group.rank
        self._global_root_rank = self._group.ranks[
            0]  # picking ranks index 0 as the reference
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        # Parameter segmentation for global ranks
        # After flatten -> self._param2buffer_size, self._param2buffer, self._trainable_params
        self._param2buffer_size = dict()  # {param.name: size}
<<<<<<< HEAD
        self._param2buffer = (
            dict()
=======
        self._param2buffer = dict(
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        )  # {param.name: [(start0, end0),(start1, end1), ...]}
        self._trainable_params = dict()  # {id(layer): [trainable_params]}
        self._unslice_params = set()  # param's numel <= segment_size
        self._unslice_params2align = dict()  # {param.name: param's align}
        self._grad_storages = dict()  # {param.dtype: GradStorage}

        assert not isinstance(
<<<<<<< HEAD
            optimizer, list
        ), "Multiple optimizers are not supported now."
        self._optim = _OptimizerWrapper(
            optimizer, self._offload, self._group, self._update_params_slice
        )
=======
            optimizer, list), "Multiple optimizers are not supported now."
        self._optim = _OptimizerWrapper(optimizer, self._offload, self._group,
                                        self._update_params_slice)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self._ori_parameter_list = self._optim._parameter_list
        self._ori_param_groups = self._optim._param_groups

        # Replace optimizer's _grad_clip
        if isinstance(self._optim._grad_clip, ClipGradByGlobalNorm):
            logging.warning(
                "While using ClipGradByGlobalNorm in GroupShardedStage3, the grad clip of original optimizer will be changed."
            )
            self._optim._grad_clip = GroupShardedClipGrad(
<<<<<<< HEAD
                self._optim._grad_clip, paddle.get_device(), self._group
            )
            if self._optim._parameter_list and isinstance(
                self._optim._parameter_list[0], dict
            ):
=======
                self._optim._grad_clip, paddle.get_device(), self._group)
            if self._optim._parameter_list and isinstance(
                    self._optim._parameter_list[0], dict):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                for item in self._optim._param_groups:
                    if "grad_clip" in item.keys():
                        item["grad_clip"] = self._optim._grad_clip

        # Synchronous all ranks models
        if pertrain_sync_models:
            self._sync_params_and_buffers()

        self._segment_rank_params(self._layer)

        # Add unslice params to master_weight in fp16
        self._handle_unslice_params()

        # In the first step, record the execution order of the layer
        self._order_tracer = OrderedDict()
        self._order_tracer["order"] = 0
        self._order_tracer["layer"] = list()

        # Register task flow
        self._task_flow = TaskFlow()

        # Register forward hooks
        self._register_forward_hooks(self._layer)

        # Register backward parameter hooks
        self._register_backward_hooks()

        # Redefine optimizer step and clear function
        self._redefine_opt_step()
        self._redefine_opt_clear()

    @paddle.autograd.no_grad()
    def _sync_params_and_buffers(self):
        """
        Sync all model states for all ranks
        """

        for p in self._layer.parameters():
<<<<<<< HEAD
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
=======
            collective.broadcast(p,
                                 src=self._global_root_rank,
                                 group=self._group,
                                 sync_op=True)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def _clear_gradients(self):
        assert len(self._trainable_params.keys()) > 0
        current_layer_params = self._layer.parameters(include_sublayers=True)
        # 1.Handle param's slice
        trainable_params = list(
<<<<<<< HEAD
            filter(
                lambda p: p.trainable and p not in self._unslice_params,
                current_layer_params,
            )
        )
        for param in trainable_params:
            assert hasattr(
                param, "fw_storage"
            ), "Find {} don't have fw_storage attribute.".format(param.name)
=======
            filter(lambda p: p.trainable and p not in self._unslice_params,
                   current_layer_params))
        for param in trainable_params:
            assert hasattr(param, "fw_storage"
                           ), "Find {} don't have fw_storage attribute.".format(
                               param.name)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            param.fw_storage.clear_gradient(False)
            param.bw_storage._clear()
            param.bw_storage = None
        # 2.Handle unslice param
        if not self._offload:
            for grad_storage in self._grad_storages.values():
                grad_storage.buffer.zero_()
        else:
            for param in list(self._unslice_params):
                param.clear_gradient(False)
                tmp_var = param.cuda(DEV_ID)

<<<<<<< HEAD
                if (
                    tmp_var.dtype == Type.fp32.value
                    and param2dtype[param.name] == Type.fp16.value
                ):
=======
                if tmp_var.dtype == Type.fp32.value and param2dtype[
                        param.name] == Type.fp16.value:
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                    tmp_var = paddle.cast(tmp_var, Type.fp16.value)
                tmp_var._share_buffer_to(param)
                del tmp_var
            for grad_storage in self._grad_storages.values():
                grad_storage.manumal_relase()
                grad_storage.rebuild()

    # Update param memery slice
    def _update_params_slice(self):
        update_list = self._update_params()

        if not isinstance(self._optim._param_groups[0], dict):
            slice_params = [param.fw_storage for param in update_list]
            self._optim._parameter_list = slice_params + list(
<<<<<<< HEAD
                self._unslice_params
            )
            self._optim._param_groups = slice_params + list(
                self._unslice_params
            )
=======
                self._unslice_params)
            self._optim._param_groups = slice_params + list(
                self._unslice_params)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        else:
            for param_group in self._optim._param_groups:
                p_group = []
                for p in param_group['params']:
                    if hasattr(p, "fw_storage"):
                        p_group.append(p.fw_storage)
                    else:
                        p_group.append(p)

                param_group['params'] = p_group

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

    def set_state_dict(self, state_dict, use_structured_name=True):
<<<<<<< HEAD
        self._layer.set_state_dict(
            state_dict, use_structured_name=use_structured_name
        )

    def state_dict(
        self,
        destination=None,
        include_sublayers=True,
        structured_name_prefix="",
    ):
        return self._layer.state_dict(
            destination=destination,
            include_sublayers=include_sublayers,
            structured_name_prefix=structured_name_prefix,
        )
=======
        self._layer.set_state_dict(state_dict,
                                   use_structured_name=use_structured_name)

    def state_dict(self,
                   destination=None,
                   include_sublayers=True,
                   structured_name_prefix=""):
        return self._layer.state_dict(
            destination=destination,
            include_sublayers=include_sublayers,
            structured_name_prefix=structured_name_prefix)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def _handle_unslice_params(self):
        buffer_size = dict()
        buffer_size[Type.fp32.value] = 0
        buffer_size[Type.fp16.value] = 0
        for param in self._unslice_params:
            # Updata optimizer master weights
            if param.dtype == Type.fp16.value and not self._offload:
                master_tensor = paddle.cast(param, Type.fp32.value)
                master_tensor.name = param.name
                self._optim._master_weights[param.name] = master_tensor
            if self._offload:
                param.master_weight = paddle.cast(param, Type.fp32.value).cpu()
            param2dtype[param.name] = param.dtype
            p_align = self._param2align(param)
            self._unslice_params2align[param.name] = p_align
            buffer_size[param.dtype] += param._numel() + p_align

        # Create unslice_params'grad
        for param in sorted(list(self._unslice_params), key=lambda p: p.name):
            if param.dtype not in self._grad_storages.keys():
                self._grad_storages[param.dtype] = GradStorage(
                    buffer_size[param.dtype],
                    dtype=param.dtype,
                    device=self._default_device,
                    destination=self._rank,
<<<<<<< HEAD
                    parm2align=self._unslice_params2align,
                )
            self._grad_storages[param.dtype].add_grad(
                param, self._unslice_params2align[param.name]
            )
=======
                    parm2align=self._unslice_params2align)
            self._grad_storages[param.dtype].add_grad(
                param, self._unslice_params2align[param.name])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def _segment_rank_params(self, layer, name="last_layer"):
        """
        Flatten parameters according to layer.
        """
        current_layer_params = _current_layer_params(layer)
        if current_layer_params:
            CHECK_LAYER[id(layer)] = name
            self._flatten_layer_params(layer, current_layer_params)

        for name, sub_layer in layer.named_children():
            self._segment_rank_params(sub_layer, name)

    def _flatten_layer_params(self, layer, current_layer_params):
        """
        Parameter segmentation and memory integration.
        """

<<<<<<< HEAD
        if id(layer) in self._trainable_params.keys():
            return

        # the layer in self._exclude_layer will be unsliced.
        if (
            id(layer) in self._exclude_layer
            or layer.__class__.__name__ in self._exclude_layer
        ):
            for p in current_layer_params:
                if p.trainable:
                    self._unslice_params.add(_UnsliceParam(p))
            return

=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def _add_manage_info(trainable_param):
            return _PartitionParam(trainable_param)

        current_params = list()
        for p in current_layer_params:
<<<<<<< HEAD
            if p._numel() > self._segment_size:
=======
            if p.trainable and p._numel() > self._segment_size:
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                current_params.append(_add_manage_info(p))
            elif p.trainable:
                self._unslice_params.add(_UnsliceParam(p))

<<<<<<< HEAD
=======
        assert id(layer) not in self._trainable_params.keys()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self._trainable_params[id(layer)] = current_params

        for param in self._trainable_params[id(layer)]:
            if param.name in self._param2buffer.keys():
                continue
            self._param2buffer[param.name] = []
            # 1.Params alignment
            align_ = self._param2align(param)

            offset = align_ + param._numel()
<<<<<<< HEAD
            buffer_size = (
                offset
                if offset % self._group.nranks == 0
                else offset + self._group.nranks - (offset % self._group.nranks)
            )
=======
            buffer_size = offset if offset % self._group.nranks == 0 else offset + self._group.nranks - (
                offset % self._group.nranks)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            self._param2buffer_size[param.name] = buffer_size

            # 2.Combination param buffer
            assert buffer_size % self._group.nranks == 0
            pre_buffer = buffer_size // self._group.nranks

            for rank_ in range(self._group.nranks):
                self._param2buffer[param.name].append(
<<<<<<< HEAD
                    (rank_ * pre_buffer, (rank_ + 1) * pre_buffer)
                )
=======
                    (rank_ * pre_buffer, (rank_ + 1) * pre_buffer))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            # Record param's dtype
            param2dtype[param.name] = param.dtype
            # 3.Flatten layer params and release other rank buffer
            self._param_storage(param, buffer_size)

    def _param_storage(self, param, buffer_size):
        """
        This is a function to simplify the handling of parameter InternalStorages.
        """
        assert isinstance(buffer_size, int)
<<<<<<< HEAD
        value = (
            np.zeros(buffer_size, dtype=np.float16)
            if Type.fp16.value == param.dtype
            else np.zeros(buffer_size, dtype=np.float32)
        )
=======
        value = np.zeros(
            buffer_size,
            dtype=np.float16) if Type.fp16.value == param.dtype else np.zeros(
                buffer_size, dtype=np.float32)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        buffer = core.eager.Tensor(value=value, place=core.CPUPlace())

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
        if self._offload:
            with device_guard():
                tmp_tensor = buffer._slice(start, end)
<<<<<<< HEAD
            param.fw_storage = core.eager.Tensor(
                value=tmp_tensor,
                place=core.CPUPlace(),
                name="slice@" + param.name,
            )
            if param.trainable:
                with device_guard():
                    param.master_weight = paddle.cast(
                        param.fw_storage, Type.fp32.value
                    )
        else:
            param.fw_storage = core.eager.Tensor(
                value=buffer._slice(start, end), name="slice@" + param.name
            )
        param.status = "part"

        # Updata optimizer master weights
        if (
            param.trainable
            and param.dtype == Type.fp16.value
            and not self._offload
        ):
=======
            param.fw_storage = core.eager.Tensor(value=tmp_tensor,
                                                 place=core.CPUPlace(),
                                                 name="slice@" + param.name)
            with device_guard():
                param.master_weight = paddle.cast(param.fw_storage,
                                                  Type.fp32.value)
        else:
            param.fw_storage = core.eager.Tensor(value=buffer._slice(
                start, end),
                                                 name="slice@" + param.name)
        param.status = "part"

        # Updata optimizer master weights
        if param.dtype == Type.fp16.value and not self._offload:
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            master_tensor = paddle.cast(param.fw_storage, Type.fp32.value)
            master_tensor.name = param.name
            self._optim._master_weights[param.fw_storage.name] = master_tensor
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
<<<<<<< HEAD
            # the layer in self._exclude_layer will be added hooks.
            if not (
                id(layer) in self._exclude_layer
                or layer.__class__.__name__ in self._exclude_layer
            ):
                self._register_forward_all_hooks(layer, self._task_flow)
=======
            self._register_forward_all_hooks(layer, self._task_flow)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        for _, sub_layer in layer.named_children():
            self._register_forward_hooks(sub_layer)

    def _register_forward_all_hooks(self, sub_layer, task_flow):
<<<<<<< HEAD
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
            return ForwardPostHooks.apply(
                outputs,
                layer,
                self._order_tracer,
                self._trainable_params,
                self._param2buffer,
                self._param2buffer_size,
                self._rank,
                self._group,
                self._sync_comm,
                self._offload,
                task_flow,
            )
=======

        def _forward_pre_hook(layer, inputs):
            return ForwardPreHooks(layer, self._order_tracer,
                                   self._trainable_params,
                                   self._param2buffer_size, self._group,
                                   self._sync_comm, self._offload, task_flow)

        def _forward_post_hook(layer, inputs, outputs):
            return ForwardPostHooks.apply(outputs, layer, self._order_tracer,
                                          self._trainable_params,
                                          self._param2buffer,
                                          self._param2buffer_size, self._rank,
                                          self._group, self._sync_comm,
                                          self._offload, task_flow)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

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
<<<<<<< HEAD
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
=======
            collective.broadcast(buffer,
                                 self._global_root_rank,
                                 self._group,
                                 sync_op=True)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

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
        update_list = []
        assert len(self._trainable_params.keys()) > 0
        current_layer_params = self._layer.parameters(include_sublayers=True)
        trainable_params = list(
<<<<<<< HEAD
            filter(
                lambda p: p.trainable and p not in self._unslice_params,
                current_layer_params,
            )
        )
        # 1.Handle param's slice
        for param in trainable_params:
            assert hasattr(
                param, "fw_storage"
            ), "Find {} don't have fw_storage attribute".format(param.name)

=======
            filter(lambda p: p.trainable and p not in self._unslice_params,
                   current_layer_params))
        # 1.Handle param's slice
        for param in trainable_params:
            assert hasattr(
                param,
                "fw_storage"), "Find {} don't have fw_storage attribute".format(
                    param.name)
            # Gradient average
            if self._offload:
                with device_guard():
                    param.bw_storage.scale_(scale=self._world_size_scaling)
            else:
                param.bw_storage.scale_(scale=self._world_size_scaling)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            param.fw_storage = _VarBaseWrapper(param)
            assert param.fw_storage.grad is None
            param.fw_storage._copy_gradient_from(param.bw_storage)
            update_list.append(param)

        # 2.Handle unslice param
        for grad_storage in self._grad_storages.values():
            grad_storage.buffer.scale_(scale=self._world_size_scaling)
<<<<<<< HEAD
            dist.all_reduce(tensor=grad_storage.buffer, group=self._group)
            if self._dp_group is not None and self._dp_group.nranks > 1:
                grad_storage.buffer.scale_(scale=(1.0 / self._dp_group.nranks))
                dist.all_reduce(
                    tensor=grad_storage.buffer, group=self._dp_group
                )

=======
            collective.all_reduce(tensor=grad_storage.buffer, group=self._group)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        if self._offload:
            for param in list(self._unslice_params):
                param._clear_data()
                param.master_weight._share_buffer_to(param)

            for grad_storage in self._grad_storages.values():
                for p in grad_storage._params:
                    tmp_g = _device2cpu(p.grad, convert_dtype=True)
                    p.clear_gradient(False)
                    p._copy_gradient_from(tmp_g)
                    del tmp_g
                grad_storage.buffer._clear()

        return update_list

    def get_all_parameters(self, convert2cpu=False):
        """
        Get the full parameters and return the corresponding task flows.
        """
        assert len(self._trainable_params.keys()) > 0
        current_layer_params = self._layer.parameters(include_sublayers=True)
        trainable_params = list(
<<<<<<< HEAD
            filter(
                lambda p: p.trainable and p not in self._unslice_params,
                current_layer_params,
            )
        )
        t_flow = _allgather_buffer(
            trainable_params,
            self._group,
            param2buffer_size=self._param2buffer_size,
            use_calc_stream=True,
            task_flow=TaskFlow(),
            sync_wait=True,
            offload=self._offload,
            convert2cpu=convert2cpu,
        )
=======
            filter(lambda p: p.trainable and p not in self._unslice_params,
                   current_layer_params))
        t_flow = _allgather_buffer(trainable_params,
                                   self._group,
                                   param2buffer_size=self._param2buffer_size,
                                   use_calc_stream=True,
                                   task_flow=TaskFlow(),
                                   sync_wait=True,
                                   offload=self._offload,
                                   convert2cpu=convert2cpu)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        if convert2cpu:
            for param in trainable_params:
                t_flow.full_param[param.name][0]._share_buffer_to(param)

        self._optim._parameter_list = self._ori_parameter_list
        self._optim._param_groups = self._ori_param_groups

    def _register_backward_hooks(self):
        current_layer_params = self._layer.parameters(include_sublayers=True)
        trainable_params = list(
<<<<<<< HEAD
            filter(
                lambda p: p.trainable and p not in self._unslice_params,
                current_layer_params,
            )
        )
=======
            filter(lambda p: p.trainable and p not in self._unslice_params,
                   current_layer_params))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        for param in trainable_params:
            allreduce_function = self._get_allreduce_fn(param)
            param._register_backward_hook(allreduce_function)

    def _get_allreduce_fn(self, param):
<<<<<<< HEAD
        @paddle.autograd.no_grad()
        def allreduce_(*_):
            assert (
                param.trainable
            ), "the param must be trainable for grad allreduced"
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
=======

        @paddle.autograd.no_grad()
        def allreduce_(*_):
            if param.name in self._task_flow.full_grad.keys():
                full_grad = self._task_flow.full_grad[param.name]
                # Only support sync allreduce current rank's layer now
                collective.all_reduce(tensor=full_grad, group=self._group)

                start, end = self._param2buffer[param.name][self._rank]
                if param.bw_storage is None:
                    param.bw_storage = full_grad._slice(start,
                                                        end).detach().clone()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                    if self._offload:
                        param.bw_storage = _device2cpu(param.bw_storage, True)
                else:
                    if self._offload:
                        cpu_grad = _device2cpu(
<<<<<<< HEAD
                            full_grad._slice(start, end).detach().clone(), True
                        )
                        with device_guard():
                            param.bw_storage = paddle.add(
                                param.bw_storage, cpu_grad
                            )
                    else:
                        param.bw_storage = paddle.add(
                            param.bw_storage,
                            full_grad._slice(start, end).detach().clone(),
                        )
=======
                            full_grad._slice(start, end).detach().clone(), True)
                        with device_guard():
                            param.bw_storage = paddle.add(
                                param.bw_storage, cpu_grad)
                    else:
                        param.bw_storage = paddle.add(
                            param.bw_storage,
                            full_grad._slice(start, end).detach().clone())
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                param.clear_gradient(False)
                del self._task_flow.full_grad[param.name]

            if param.name in self._task_flow.full_param.keys():
                if param.status == "all":
                    param.use_count = 0
                    param._clear_data()
                    start, end = self._param2buffer[param.name][self._rank]
<<<<<<< HEAD
                    param.fw_storage = (
                        self._task_flow.full_param[param.name][0]
                        ._slice(start, end)
                        .detach()
                        .clone()
                    )
=======
                    param.fw_storage = self._task_flow.full_param[
                        param.name][0]._slice(start, end).detach().clone()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                    param.status = "part"
                    del self._task_flow.full_param[param.name]

                    if self._offload:
                        param.fw_storage._clear_data()
                        param.master_weight._share_buffer_to(param.fw_storage)

        return allreduce_

    def _param2align(self, param):
        # CUDA alignment 256 bytes
        size = param._numel() * align[param.dtype]
        remaining = size % alignment[self._default_device]
<<<<<<< HEAD
        ali = (
            0 if remaining == 0 else alignment[self._default_device] - remaining
        )
=======
        ali = 0 if remaining == 0 else alignment[
            self._default_device] - remaining
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        align_ = ali // align[param.dtype]
        return align_

    def _redefine_opt_step(self):
        params_slice_func = self._update_params_slice
        opt_step = self._optim.step

        def _opt_step(self):
            if not self.update_scaler:
                params_slice_func()
            if self.offload:
                with device_guard():
                    opt_step()
            else:
                opt_step()

        def _opt_minimize(self):
            raise RuntimeError(
                "optimizer.minimize() not support now, please use optimizer.step()"
            )

        self._optim.step = MethodType(_opt_step, self._optim)
        self._optim.minimize = MethodType(_opt_minimize, self._optim)

    def _redefine_opt_clear(self):
        clear_func = self._clear_gradients

        def _opt_clear(self):
            clear_func()

        self._optim.clear_grad = MethodType(_opt_clear, self._optim)


<<<<<<< HEAD
def ForwardPreHooks(
    layer,
    order_tracer,
    trainable_params,
    param2buffer_size,
    group,
    sync_comm,
    offload,
    task_flow,
):
=======
def ForwardPreHooks(layer, order_tracer, trainable_params, param2buffer_size,
                    group, sync_comm, offload, task_flow):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # Record layer's id
    layer_id = id(layer)
    use_calc, sync_wait = False, False

    if layer_id not in order_tracer.keys() or sync_comm:
        use_calc, sync_wait = True, True

        # Whether to use calc stream
        task_flow.use_calc[layer_id] = use_calc
    else:
        # Whether to use calc stream
        task_flow.use_calc[layer_id] = use_calc
        # wait current layer params
<<<<<<< HEAD
        _wait_layer(
            trainable_params[layer_id],
            task_flow,
            group,
            param2buffer_size,
            use_calc,
            offload,
        )

        if layer_id == order_tracer["layer"][-1]:
            return
        order_ = order_tracer[layer_id]
        layer_id = order_tracer["layer"][order_ + 1]

    _allgather_buffer(
        trainable_params[layer_id],
        group,
        param2buffer_size=param2buffer_size,
        use_calc_stream=use_calc,
        task_flow=task_flow,
        sync_wait=sync_wait,
        offload=offload,
    )
=======
        _wait_layer(trainable_params[layer_id], task_flow, group,
                    param2buffer_size, use_calc, offload)

        if layer_id == order_tracer["layer"][-1]: return
        order_ = order_tracer[layer_id]
        layer_id = order_tracer["layer"][order_ + 1]

    _allgather_buffer(trainable_params[layer_id],
                      group,
                      param2buffer_size=param2buffer_size,
                      use_calc_stream=use_calc,
                      task_flow=task_flow,
                      sync_wait=sync_wait,
                      offload=offload)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    return


class ForwardPostHooks(PyLayer):
<<<<<<< HEAD
    @staticmethod
    def forward(
        ctx,
        inputs,
        layer,
        order_tracer,
        trainable_params,
        param2buffer,
        param2buffer_size,
        rank,
        group,
        sync_comm,
        offload,
        task_flow,
    ):

        layer_id = id(layer)
        # release current layer full params
        _release_param(
            trainable_params[layer_id], param2buffer, rank, task_flow, offload
        )
=======

    @staticmethod
    def forward(ctx, inputs, layer, order_tracer, trainable_params,
                param2buffer, param2buffer_size, rank, group, sync_comm,
                offload, task_flow):

        layer_id = id(layer)
        # release current layer full params
        _release_param(trainable_params[layer_id], param2buffer, rank,
                       task_flow, offload)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        if layer_id not in order_tracer.keys():
            order_ = order_tracer["order"]
            order_tracer[layer_id] = order_
            order_tracer["order"] += 1
            order_tracer["layer"].append(layer_id)

<<<<<<< HEAD
        # Record fw info
=======
        #Record fw info
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        ctx.order_tracer = order_tracer
        ctx.task_flow = task_flow
        ctx.group = group
        ctx.layer_id = layer_id
        ctx.sync_comm = sync_comm
        ctx.trainable_params = trainable_params
        ctx.param2buffer_size = param2buffer_size
        ctx.offload = offload

        return inputs

    @staticmethod
    def backward(ctx, *args):
        # Load context value
        order_tracer = ctx.order_tracer
        task_flow = ctx.task_flow
        group = ctx.group
        layer_id = ctx.layer_id
        trainable_params = ctx.trainable_params
        param2buffer_size = ctx.param2buffer_size
        sync_comm = ctx.sync_comm
        offload = ctx.offload
        use_calc, sync_wait = False, False

        # Allgather params synchronization
        if sync_comm:
            use_calc, sync_wait = True, True
<<<<<<< HEAD
            _allgather_buffer(
                trainable_params[layer_id],
                group,
                param2buffer_size=param2buffer_size,
                use_calc_stream=use_calc,
                task_flow=task_flow,
                sync_wait=sync_wait,
                offload=offload,
            )
        else:
            _wait_layer(
                trainable_params[layer_id],
                task_flow,
                group,
                param2buffer_size,
                use_calc,
                offload,
            )

        # Create params's grad
        _create_params_grad(
            trainable_params[layer_id], param2buffer_size, task_flow
        )
=======
            _allgather_buffer(trainable_params[layer_id],
                              group,
                              param2buffer_size=param2buffer_size,
                              use_calc_stream=use_calc,
                              task_flow=task_flow,
                              sync_wait=sync_wait,
                              offload=offload)
        else:
            _wait_layer(trainable_params[layer_id], task_flow, group,
                        param2buffer_size, use_calc, offload)

        # Create params's grad
        _create_params_grad(trainable_params[layer_id], param2buffer_size,
                            task_flow)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        # Whether to use calc stream
        task_flow.use_calc[layer_id] = use_calc
        if layer_id != order_tracer["layer"][0] and not sync_comm:
            layer_next_id = order_tracer["layer"][order_tracer[layer_id] - 1]
<<<<<<< HEAD
            _allgather_buffer(
                trainable_params[layer_next_id],
                group,
                param2buffer_size=param2buffer_size,
                use_calc_stream=use_calc,
                task_flow=task_flow,
                sync_wait=sync_wait,
                offload=offload,
            )
=======
            _allgather_buffer(trainable_params[layer_next_id],
                              group,
                              param2buffer_size=param2buffer_size,
                              use_calc_stream=use_calc,
                              task_flow=task_flow,
                              sync_wait=sync_wait,
                              offload=offload)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        return args


class TaskFlow:
    """
    Task flows, one way linked list for task acquisition.
    """

<<<<<<< HEAD
    def __init__(
        self,
        full_param=dict(),
        full_grad=dict(),
        use_calc=dict(),
        callback=None,
    ):
=======
    def __init__(self,
                 full_param=dict(),
                 full_grad=dict(),
                 use_calc=dict(),
                 callback=None):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.full_param = full_param
        self.full_grad = full_grad
        self.use_calc = use_calc
        self.callback = callback


<<<<<<< HEAD
def _release_param(
    trainable_params, param2buffer, rank, task_flow, offload=False
):
=======
def _release_param(trainable_params,
                   param2buffer,
                   rank,
                   task_flow,
                   offload=False):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    for param in trainable_params:
        # async communicate share weight not clear
        param.use_count -= 1
        if param.use_count == 0:
            param._clear_data()
            if param.name in task_flow.full_param.keys():
                start, end = param2buffer[param.name][rank]
                with paddle.amp.auto_cast(enable=False):
<<<<<<< HEAD
                    param.fw_storage = (
                        task_flow.full_param[param.name][0]
                        ._slice(start, end)
                        .detach()
                        .clone()
                    )
=======
                    param.fw_storage = task_flow.full_param[
                        param.name][0]._slice(start, end).detach().clone()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                param.status = "part"
                del task_flow.full_param[param.name]

                if offload:
                    param.fw_storage = _device2cpu(param.fw_storage)
    return


<<<<<<< HEAD
def _wait_layer(
    trainable_params,
    task_flow,
    group,
    param2buffer_size,
    use_calc_stream,
    offload=False,
):
=======
def _wait_layer(trainable_params,
                task_flow,
                group,
                param2buffer_size,
                use_calc_stream,
                offload=False):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    for param in trainable_params:
        if param.status == "all":
            param.use_count += 1
            continue
        if param.name in task_flow.full_param.keys():
            full_param, task = task_flow.full_param[param.name]
            task.wait()
            full_param._slice(0, param._numel())._share_buffer_to(param)
            param.fw_storage._clear()
            param.fw_storage = None
            param.status = "all"
            param.use_count += 1
        else:
<<<<<<< HEAD
            _allgather_buffer(
                trainable_params,
                group,
                param2buffer_size=param2buffer_size,
                use_calc_stream=True,
                task_flow=task_flow,
                sync_wait=True,
                offload=offload,
            )
=======
            _allgather_buffer(trainable_params,
                              group,
                              param2buffer_size=param2buffer_size,
                              use_calc_stream=True,
                              task_flow=task_flow,
                              sync_wait=True,
                              offload=offload)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            break
    return task_flow


<<<<<<< HEAD
def _allgather_buffer(
    trainable_params,
    group,
    param2buffer_size,
    use_calc_stream,
    task_flow,
    sync_wait=False,
    offload=False,
    convert2cpu=False,
):
=======
def _allgather_buffer(trainable_params,
                      group,
                      param2buffer_size,
                      use_calc_stream,
                      task_flow,
                      sync_wait=False,
                      offload=False,
                      convert2cpu=False):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    for param in trainable_params:
        if param.status == "all":
            param.use_count += 1
            continue

        if offload:
            param.fw_storage = _cpu2device(param)

        buffer_size = param2buffer_size[param.name]
        with paddle.amp.auto_cast(enable=False):
            full_param, task = _all_gather(param.fw_storage, buffer_size, group)

        # Allgather current layer in the 1st step synchronously
        if sync_wait:
            with paddle.amp.auto_cast(enable=False):
                task.wait()
            full_param._slice(0, param._numel())._share_buffer_to(param)
            param.fw_storage._clear()
            param.fw_storage = None
            param.status = "all"
            param.use_count += 1
        task_flow.full_param[param.name] = (full_param, task)

        # parameter converts to cpu
        if convert2cpu:
            p_name = param.name
            param = _device2cpu(param)
            del task_flow.full_param[p_name]
            task_flow.full_param[p_name] = (param, None)

    return task_flow


@paddle.autograd.no_grad()
def _create_params_grad(trainable_params, param2buffer_size, task_flow):
    for param in trainable_params:
<<<<<<< HEAD
        if not param.trainable:
            continue
        if param.name in task_flow.full_grad.keys():
            continue
        assert isinstance(param2buffer_size[param.name], int)
        temp_grad = paddle.zeros(
            [param2buffer_size[param.name]], dtype=param.dtype
        )
=======
        if param.name in task_flow.full_grad.keys():
            continue
        assert isinstance(param2buffer_size[param.name], int)
        temp_grad = paddle.zeros([param2buffer_size[param.name]],
                                 dtype=param.dtype)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        temp_tensor = temp_grad._slice(0, param._numel())
        temp_tensor.get_tensor()._set_dims(param.shape)
        param._copy_gradient_from(temp_tensor)
        del temp_tensor
        task_flow.full_grad[param.name] = temp_grad
    return task_flow


def _PartitionParam(param):
    if not hasattr(param, "fw_storage"):
        setattr(param, "fw_storage", None)
        setattr(param, "bw_storage", None)
        setattr(param, "master_weight", None)
        setattr(param, "status", "all")
        setattr(param, "use_count", 0)
    return param


def _UnsliceParam(param):
    if not hasattr(param, "unslice"):
        setattr(param, "unslice", True)
        setattr(param, "master_weight", None)
    return param


def _VarBaseWrapper(param):
    varbase = param.fw_storage
<<<<<<< HEAD
    tmp_param = EagerParamBase(
        shape=varbase.shape, dtype=varbase.dtype, name="slice@" + param.name
    )
    varbase._share_buffer_to(tmp_param)
    tmp_param.regularizer = param.regularizer
    tmp_param.optimize_attr['learning_rate'] = param.optimize_attr[
        'learning_rate'
    ]
=======
    tmp_param = EagerParamBase(shape=varbase.shape,
                               dtype=varbase.dtype,
                               name="slice@" + param.name)
    varbase._share_buffer_to(tmp_param)
    tmp_param.regularizer = param.regularizer
    tmp_param.optimize_attr['learning_rate'] = param.optimize_attr[
        'learning_rate']
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    varbase._clear()
    return tmp_param


def _OptimizerWrapper(optimizer, offload, group, update_params_slice):
    if not hasattr(optimizer, "_optim"):
        setattr(optimizer, "_optim", optimizer)
        setattr(optimizer, "offload", offload)
        setattr(optimizer, "_group", group)
        setattr(optimizer, "update_scaler", None)
        setattr(optimizer, "update_slice", update_params_slice)
    return optimizer


def _device2cpu(trans_param, convert_dtype=False):
    if convert_dtype:
        trans_param = paddle.cast(trans_param, Type.fp32.value)
    tmp_p = trans_param.cpu()
    trans_param._clear_data()
    return tmp_p


def _cpu2device(param):
    tmp_p = param.fw_storage.cuda(DEV_ID)
<<<<<<< HEAD
    if (
        tmp_p.dtype == Type.fp32.value
        and param2dtype[param.name] == Type.fp16.value
    ):
=======
    if tmp_p.dtype == Type.fp32.value and param2dtype[
            param.name] == Type.fp16.value:
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        tmp_p = paddle.cast(tmp_p, Type.fp16.value)
    return tmp_p


def _current_layer_params(layer):
<<<<<<< HEAD
    return (
        layer.parameters(include_sublayers=False) + list(layer.extra_parameters)
        if hasattr(layer, "extra_parameters")
        else layer.parameters(include_sublayers=False)
    )
=======
    return layer.parameters(
        include_sublayers=False) + list(layer.extra_parameters) if hasattr(
            layer, "extra_parameters") else layer.parameters(
                include_sublayers=False)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
