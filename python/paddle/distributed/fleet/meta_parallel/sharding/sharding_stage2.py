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
import contextlib
import logging
import time
import functools
from itertools import chain
from functools import reduce
from collections import deque

import paddle
from paddle import nn
import paddle.distributed as dist

from .sharding_optimizer import ShardingOptimizer
from .internal_storage import GradStorage
from .sharding_utils import Taskflow, GpuInfo, Type


def _trainable(param):
    return param.trainable


class ShardingStage2(nn.Layer):
    """ 
    A wrapper for Sharding Stage2 Layer in Dygraph. 
    .. warning: ShardingStage2 encapsulates the layer strategy and integrates it into the nn.Layer.
    .. this file is inspired by: https://github.com/facebookresearch/fairscale/blob/main/fairscale/nn/data_parallel/sharded_ddp.py.
    .. ZeRO: https://arxiv.org/pdf/1910.02054.pdf.
    """

    # TODO (Baibaifan) 
    # TO support following featrues in future:
    # 1. Unified memory for param and param.grad to InternalStorage.
    # 2. Divide param.grad according to rank to centrally apply for and release GPU memory.
    # 3. Dynamically adjust training parameters and models。
    # 4. Support offload function.
    # 5. Support the establishment of independent communication groups.

    def __init__(
            self,
            layer,
            sharding_optimizer,
            group,
            sync_buffers=False,
            pertrain_sync_models=True,
            buffer_max_size=2**24,  #16MB param/ranks
            auto_refresh_trainable=True,
            device="gpu",
            post_hook=False):
        super().__init__()

        # training options
        self._layer = layer
        self._sharding_optimizers = [sharding_optimizer] if not isinstance(
            sharding_optimizer, list) else sharding_optimizer
        self._sync_buffers = sync_buffers
        self._auto_refresh_trainable = auto_refresh_trainable
        self._post_hook = post_hook

        # Gradient accumulation, Gradient flip
        self._accumulate_grads = True if self._sharding_optimizers[
            0].accumulation_steps is not None else False
        self._grads_flipped = True if self._sharding_optimizers[
            0].accumulation_steps is not None else False

        # Communication related attributes
        assert group is not None, "Distributed communication group is must be gived"
        self._group = group
        self._world_size_scaling = 1.0 / self._group.nranks
        assert self._group.nranks > 1, "Training must be distributed, ranks must be greater than 1"
        self._rank = self._group.rank
        self._global_root_rank = 0  # picking rank 0 as the reference
        self._global_ranks = self._group.ranks
        self._default_device = device

        # Global statistical parameters
        self._all_params = list(
            chain(
                * [optim.local_params for optim in self._sharding_optimizers]))
        self._trainable_params = []
        self._grad_reduced = []
        self._trainable_param2rank = {}
        self._trainable_mask = list(map(_trainable, self._all_params))

        # Set grad storage size
        self._buffer_max_size = buffer_max_size
        self._use_grad_storage = self._buffer_max_size > 0
        self._grad_storages = {}  # {dtype: {rank: GradStorage}}
        self._has_grad_storage = []
        self._grad_storage_list = []
        model_size = sum([p.numel() for p in self._layer.parameters()]).item()
        if Type.fp16.value == self._all_params[0].dtype:
            print(
                "GradStorage size: {:.2f}M parameters, Model size {:.2f}M parameters".
                format(self._buffer_max_size / 2**20, model_size / 2**19))
        elif Type.fp32.value == self._all_params[0].dtype:
            print(
                "GradStorage size: {:.2f}M parameters, Model size {:.2f}M parameters".
                format(self._buffer_max_size / 2**20, model_size / 2**18))

        # Set backward pass hooks
        self._bw_hooks = []

        # Synchronous all ranks models
        if pertrain_sync_models:
            self._sync_params_and_buffers()

        # Set tasks flow
        self._tasks_flow = deque()

    def forward(self, *inputs, **kwargs):
        """
        A wrapper for Sharding Stage2 layer.
        - Fresh trainable params or rebuild grad storage
        - Sync layer's buffer params
        - Clear all flags states
        - Forward for origin layers
        """

        # Whether to need to reset trainable parameters
        needs_fresh = len(self._bw_hooks) == 0 and self.training

        if self._auto_refresh_trainable:
            needs_fresh |= self._detect_train_change()

        # Front hook
        if not self._post_hook:
            self._init_internal_storage(needs_fresh)

        # Sync layer's buffers state
        if self._sync_buffers:
            self.sync_buffers()

        # Normal FW on the base model
        fw = self._layer(*inputs, **kwargs)

        # Post hook
        if self._post_hook:
            self._init_internal_storage(needs_fresh)

        return fw

    def _init_internal_storage(self, needs_fresh):
        """
        Judge Fresh trainable params or rebuild grad storage.
        """
        if needs_fresh:
            self.fresh_trainable()
        else:
            self.build_grad_storages()

        # Clear all flags state 
        self._clear_counters()

    def to(self, device=None, dtype=None, blocking=True):
        """
        Synchronously or asynchronously convert the data type of the layer, the device is not supported now.
        """

        assert isinstance(device, str), "Device must be type str"
        assert device == self._default_device, "New devices are not supported, because of the optimizer state is not sync"

        self._layer.to(device=device, dtype=dtype, blocking=blocking)

        # Re-build the buckets, hooks, etc..
        self.fresh_trainable()

    def fresh_trainable(self):
        """ Whether to update training parameters """

        # Make sure that this is not done while gradients are waiting to be reduced (if no_sync context for instance)
        if reduce(lambda x, y: x or y, self._grad_reduced, False):
            logging.warning("Grads waiting to be reduced.")

        self._trainable_params = list(
            filter(lambda x: x.trainable, self._all_params))
        self._trainable_params.sort(key=lambda x: x.numel())

        self._trainable_param2rank = {}
        for optim in self._sharding_optimizers:
            # Need to be wrappered for Sharding Stage2 Optimizer
            if len(optim.param_storages.keys()) == 0:
                optim.update_opt_status()

            # Get the parameters split by the optimizer according to rank
            for per_rank_params in optim.dtype_rank_params.values(
            ):  # all the params from all ranks
                for params in per_rank_params:
                    for param in filter(lambda x: x.trainable, params):
                        self._trainable_param2rank[param] = optim.param2rank[
                            param]

        self._setup_use_grad_storage()
        self._setup_backward_hooks()

    @paddle.no_grad()
    def sync_buffers(self):
        """
        Sync all the param buffers from all ranks (exp: batch norm statistics).
        """

        for buffer in self._layer.buffers(include_sublayers=True):
            dist.broadcast(
                buffer,
                self._global_root_rank,
                self._group,
                use_calc_stream=True)
        dist.wait(tensor=buffer, group=self._group, use_calc_stream=True)

    def __getattr__(self, name):
        """Forward missing attributes to wrapped layer."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._layer, name)

    @paddle.no_grad()
    def _clear_counters(self):
        """Reset all the grad reduce and call counters"""
        if self.training:
            self._grad_reduced = [True for _ in self._trainable_params]

        if self._use_grad_storage:
            for grad_storage in self._grad_storage_list:
                grad_storage.reset_checked_in()

        if not self._accumulate_grads:
            self._grads_flipped = False

    def _get_reduce_fn(self, index, param, dst_rank):
        """
        There are two ways to reduce gradient.
        - 1. Do not use use_grad_storage or exceeded buffer_max_size will be reduced separately.
        - 2. Use grad_storage Reduce the storage to get the full gradient from different ranks.
        """

        if not self._use_grad_storage or not self._grad_reduced[index]:
            # Direct reduction
            @paddle.no_grad()
            def reduce(*_):
                # Skip gradient reduction, do not change status information
                if not self._accumulate_grads and self._grad_reduced[index]:
                    assert param.grad is not None, "Parameter gradient cannot be None"

                    # Change reduce information
                    self._grad_reduced[index] = False
                    paddle.multiply(param.grad,
                                    paddle.to_tensor(
                                        [self._world_size_scaling]))

                    # Clear the gradient that does not belong to the current rank through the callback function
                    def cleanup():
                        if dst_rank != self._rank:
                            param._clear_gradient()

                    # Synchronize the reduce parameter gradient
                    self._tasks_flow.append(
                        Taskflow(
                            task=dist.reduce(
                                tensor=param.grad,
                                dst=dst_rank,
                                group=self._group,
                                use_calc_stream=True),
                            callback=cleanup))
                    dist.wait(
                        tensor=param.grad,
                        group=self._group,
                        use_calc_stream=True)

                    # Clear the task flow and trigger callback to clear the redundant gradient
                    self._clear_task_flow()
        else:
            # Buffer reduction
            @paddle.no_grad()
            def reduce(*_):
                # Skip gradient reduction, do not change status information
                if not self._accumulate_grads and self._grad_reduced[index]:
                    assert param.grad is not None, "Parameter gradient cannot be None"

                    # Change reduce information
                    self._grad_reduced[index] = False
                    grad_storage = self._grad_storages[param.dtype][dst_rank]
                    grad_storage.params_checked_in += 1

                    if grad_storage.all_checked_in:
                        assert grad_storage.buffer is not None

                        # Normalize all ranks grad_storage
                        paddle.multiply(grad_storage.buffer,
                                        paddle.to_tensor(
                                            [self._world_size_scaling]))

                        # Clearing up the grad_storage buffer
                        def cleanup():
                            if dst_rank != self._rank:
                                for p in grad_storage._params:
                                    p._clear_gradient()
                                grad_storage.buffer.value().get_tensor()._clear(
                                )

                        # Reduce the bucket
                        grad_storage.sent = True
                        self._tasks_flow.append(
                            Taskflow(
                                task=dist.reduce(
                                    tensor=grad_storage.buffer,
                                    dst=grad_storage.destination,
                                    group=self._group,
                                    use_calc_stream=True),
                                callback=cleanup))
                        dist.wait(
                            tensor=grad_storage.buffer,
                            group=self._group,
                            use_calc_stream=True)

                    # Clear the task flow and trigger callback to clear the redundant gradient
                    self._clear_task_flow()

        return reduce

    def _setup_backward_hooks(self):
        """
        Set the backward hook to synchronize the gradients of all rank by reduce group ranks.
        """

        # Remove previous backward hooks
        while len(self._bw_hooks) > 0:
            self._bw_hooks.pop().remove()

        # Go through the parameters, attach the hook
        self._grad_accs = []
        if not self.training:
            return

        for index, param in enumerate(self._trainable_params):
            dst_rank = self._trainable_param2rank[param]

            reduce_function = self._get_reduce_fn(index, param, dst_rank)

            self._bw_hooks.append(
                param._register_backward_hook(reduce_function))

    @paddle.no_grad()
    def _sync_params_and_buffers(self):
        """
        Sync all model states for all ranks
        """

        for t in self._layer.state_dict().values():
            dist.broadcast(
                t,
                src=self._global_root_rank,
                group=self._group,
                use_calc_stream=True)
            dist.wait(tensor=t, group=self._group, use_calc_stream=True)

    def _setup_use_grad_storage(self):
        """
        Integrate the parameters gradient into a continuous memory according to rank, and support the update of training parameters.
        """

        if not self._use_grad_storage:
            return

        # According to parameters's numel sort, allocate memory of parameter gradient to continuous memory according to rank
        self._grad_storages = {}
        self._grad_reduced = [False for _ in self._trainable_params]

        for index, param in enumerate(self._trainable_params):
            dst_rank = self._trainable_param2rank[param]

            if param.dtype not in self._grad_storages.keys():
                self._grad_storages[param.dtype] = {}

            if dst_rank not in self._grad_storages[param.dtype].keys():
                self._grad_storages[param.dtype][dst_rank] = GradStorage(
                    self._buffer_max_size,
                    dtype=param.dtype,
                    device=self._default_device,
                    destination=dst_rank)

            # Criteria to decide whether this parameter is to be put in GradStorage
            if self._grad_storages[param.dtype][dst_rank].can_add_grad_view(
                    param):
                self._grad_storages[param.dtype][dst_rank].add_grad(param)
                self._grad_reduced[index] = True

        self._grad_storage_list = list(
            chain(* [
                self._grad_storages[dtype].values()
                for dtype in self._grad_storages.keys()
            ]))

        # Shrink the grad_storages to remove lost space in the end
        for grad_storage in self._grad_storage_list:
            grad_storage.rearrange()

    def _clear_task_flow(self):
        """Try to consume the previous tasks"""
        while len(self._tasks_flow) > 0:
            task = self._tasks_flow.popleft()
            if task.callback is not None:
                task.callback()

    def _detect_train_change(self):
        # Current trainable parameters
        trainable_mask = list(map(_trainable, self._all_params))

        # Whether parameters trainability changed
        trainability_changed = trainable_mask != self._trainable_mask

        # The whole model is not trainable but we still have grad hooks
        trainability_changed |= not self.training and len(self._bw_hooks) > 0

        if trainability_changed:
            logging.warning(
                "Trainable params changed, because of eval/train mode or parameter freezing/unfreeze."
            )
            self._trainable_mask = trainable_mask

        return trainability_changed

    def build_grad_storages(self):
        """
        Rebuild grad storages
        """

        only_fp32 = False

        if len(self._grad_storages.keys(
        )) == 1 and Type.fp32.value in self._grad_storages.keys():
            only_fp32 = True

        if not only_fp32:
            # Rebuild fp16 grad storages
            for dst_rank, grad_storage in self._grad_storages[
                    Type.fp16.value].items():
                grad_storage.manumal_relase()
                grad_storage.rebuild()

        # Rebuild fp32 grad storages
        for dst_rank, grad_storage in self._grad_storages[
                Type.fp32.value].items():
            grad_storage.manumal_relase()
            grad_storage.rebuild()
