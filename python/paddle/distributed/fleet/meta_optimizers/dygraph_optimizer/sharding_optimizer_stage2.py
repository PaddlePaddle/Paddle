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

# The file has been adapted from fairscale file:
# https://github.com/facebookresearch/fairscale/blob/main/fairscale/optim/oss.py
# Git commit hash: 8acbec718f3c70a6b9785470bb9e05cd84fc3f8e
# We retain the following license from the original files:

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
import numpy as np
from itertools import chain
from functools import reduce
from collections import OrderedDict

import paddle
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.optimizer import Optimizer
from paddle.fluid.clip import ClipGradByGlobalNorm
from paddle.distributed.collective import _get_global_group, new_group, broadcast, wait

from ...utils.internal_storage import ParamStorage, GradStorage
from ...meta_parallel.sharding.sharding_utils import Type, device_guard, ShardingClipGrad

# CUDA alignment 256 bytes, cpu alignment 4096 bytes
alignment = {"gpu": 256, "cpu": 4096}
align = {
    Type.fp16.value: 2,
    Type.fp32.value: 4,
}


class ShardingOptimizerStage2(Optimizer):
    """
    A wrapper for Sharding Stage2 Optimizer in Dygraph.

    .. warning: ShardingOptimizer encapsulates the optimization strategy and integrates it into the optimizer.

    .. ZeRO: 1.https://arxiv.org/pdf/1910.02054.pdf 2.https://arxiv.org/pdf/1910.02054.pdf.

    """

    # TODO (Baibaifan)
    # Feature Notes:
    # 1. Unified memory for parameters and parameters.grad to InternalStorage.
    # 2. Support the segmentation of optimizer parameters and partial updating of parameters.
    # 3. Dynamically adjust training parameters and models.
    # 4. Support offload function.
    # 5. Support the establishment of independent communication groups.
    # 6. Broadcast_fp16 is not supported now.
    def __init__(self,
                 params,
                 optim,
                 group=None,
                 offload=False,
                 device="gpu",
                 pertrain_sync_models=True,
                 **kw):

        super().__init__(optim._learning_rate, params, kw)

        # Segmentation information
        self._dtype_rank_params = OrderedDict(
        )  # {dtype:[param1,param2]} device, rank, params
        self._param2rank = {}
        self.__segment_params = []
        self._rank_buffer_size = {}  # {dtype: {rank: numel+alignment}}
        self._param2align = {}  # {param.name: align}

        # Default information
        self._optim_defaults = kw
        self._optim = optim

        assert hasattr(self._optim, "_master_weights"
                       ), "Must use optimizer with _master_weights attribute"
        self._local_params = params
        self._default_device = device
        self._pfp16 = len(
            list(
                filter(lambda x: x.trainable and x.dtype == Type.fp16.value,
                       self._local_params))) > 0

        self.group = new_group(
            _get_global_group().ranks) if group is None else group

        self.world_size = self.group.nranks
        self.rank = self.group.rank
        self._global_root_rank = self.group.ranks[0]

        # Synchronous all ranks models
        if pertrain_sync_models:
            self._sync_params_and_buffers()

        self.param_storages = {}  # {dtype: {rank: InternalStorage}}

        if isinstance(self._optim._grad_clip, ClipGradByGlobalNorm):
            logging.warning(
                "While using ClipGradByGlobalNorm in ShardingOptimizer, the grad clip of original optimizer will be changed."
            )
            self._optim._grad_clip = ShardingClipGrad(self._optim._grad_clip,
                                                      paddle.get_device(),
                                                      self.group)
            if self._optim._parameter_list and isinstance(
                    self._optim._parameter_list[0], dict):
                for item in self._optim._param_groups:
                    if "grad_clip" in item.keys():
                        item["grad_clip"] = ShardingClipGrad(
                            self._optim._grad_clip, paddle.get_device(),
                            self.group)

        if offload:
            assert self._pfp16, "Only support offload strategy while using \'Adam\', \'AdamW\' and \'Momentum\' optimizer with AMP/Pure FP16"

        self.offload = offload  # Using for offload
        self.offload_device = "cpu"
        self.offload_buffer_size = 0
        self.offload_param2align = {}
        self.offload_params = None
        self.offload_grads = None

        self._master_params = {}

        # Update optimizer parameters and adjust parameter storage and use according to rank.
        self._update_opt_status()

    @paddle.autograd.no_grad()
    def _sync_params_and_buffers(self):
        """
        Sync all model states for all ranks
        """

        for p in self._local_params:
            broadcast(p,
                      src=self._global_root_rank,
                      group=self.group,
                      use_calc_stream=True)

        # Multi stream operation will be supported later
        wait(tensor=p, group=self.group, use_calc_stream=True)

    def _generate_master_params(self, trainable_params):
        if self.offload:
            for param in trainable_params:
                if param.name not in self._master_params.keys():
                    self._master_params[param.name] = core.VarBase(
                        name=param.name,
                        value=param.cast(dtype=Type.fp32.value).numpy(),
                        place=core.CPUPlace(),
                        stop_gradient=param.stop_gradient)
        else:
            for param in trainable_params:
                if param.dtype == Type.fp16.value:
                    self._optim._master_weights[param.name] = paddle.cast(
                        param, Type.fp32.value)

    def _update_opt_status(self):
        """Update optimizer status and parameter storage information, and special functions to be developed.
        """
        # func 1
        self._integration_params()

        # fun 2 TODO

    # Segement helpers

    def _segment_params(self):
        """
        Divide all optimizer parameters equally into rank.
        """
        if len(self.__segment_params) == 0:
            self.__segment_params, param_lists = [
                [] for _ in range(self.world_size)
            ], [[] for _ in range(self.world_size)]
            sizes = [0] * self.world_size
            for param in self._local_params:
                # Add this param to rank with smallest size.
                rank = sizes.index(min(sizes))
                param_lists[rank].append(param)

                # Statistical real numels
                sizes[rank] += np.prod(param.shape) if param.trainable else 0

            for rank, params in enumerate(param_lists):
                self.__segment_params[rank].extend(params)
        return self.__segment_params

    @property
    def local_params(self):
        return self._local_params

    @property
    def param2rank(self):
        """Map the params to the rank which owns them"""
        if len(self._param2rank) == 0:
            for rank, params in enumerate(self._segment_params()):
                for param in params:
                    self._param2rank[param.name] = rank
        return self._param2rank

    @property
    def dtype_rank_params(self):
        """
        Divide the parameters into groups according to rank and dtype.
        """
        if len(self._dtype_rank_params) == 0:
            # Assign the parameters of each rank according to the type
            for param in self._local_params:
                if param.dtype not in self._dtype_rank_params.keys():
                    self._dtype_rank_params[param.dtype] = [
                        [] for _ in range(self.world_size)
                    ]
                self._dtype_rank_params[param.dtype][self.param2rank[
                    param.name]].append(param)

            # Sort per rank params by size
            for dtype in self._dtype_rank_params.keys():
                for rank_params in self._dtype_rank_params[dtype]:
                    rank_params.sort(key=lambda x: np.prod(x.shape))

        return self._dtype_rank_params

    @property
    def rank_buffer_size(self):
        """
        Count the memory size of the parameters corresponding to rank under the corresponding dtype.
        """
        # CUDA alignment 256 bytes
        if len(self._rank_buffer_size) == 0:
            for dtype in self.dtype_rank_params.keys():
                if dtype not in self._rank_buffer_size.keys():
                    self._rank_buffer_size[dtype] = {}
                for dst_rank, per_rank_params in enumerate(
                        self.dtype_rank_params[dtype]):
                    if dst_rank not in self._rank_buffer_size[dtype].keys():
                        self._rank_buffer_size[dtype][dst_rank] = 0
                    for param in per_rank_params:
                        if not param.trainable:
                            continue
                        size = np.prod(param.shape) * align[dtype]
                        remaining = size % alignment[self._default_device]
                        ali = 0 if remaining == 0 else alignment[
                            self._default_device] - remaining
                        align_ = ali // align[dtype]
                        self._rank_buffer_size[dtype][dst_rank] += np.prod(
                            param.shape) + align_
                        self._param2align[param.name] = align_

        return self._rank_buffer_size

    def _integration_params(self):
        """
        Integrate the parameters into a continuous memory according to rank, and support the update of training parameters.
        """

        for dtype, per_rank_params in self.dtype_rank_params.items():
            if dtype not in self.param_storages.keys():
                self.param_storages[dtype] = {}

            for dst_rank, params in enumerate(per_rank_params):
                if len(params) > 0:

                    # Merge all the trainable params in a single InternalStorage
                    trainable_params = list(
                        filter(lambda x: x.trainable, params))
                    if self._pfp16 and dst_rank == self.rank:
                        self._generate_master_params(trainable_params)
                    if trainable_params:
                        param_storage = ParamStorage(
                            size=self.rank_buffer_size[dtype][dst_rank],
                            dtype=dtype,
                            device=self._default_device)

                        param_storage.add_rank_params(trainable_params,
                                                      self._param2align)
                        self.param_storages[dtype][dst_rank] = param_storage

        # Clear the InternalStorage keys which are not in use anymore
        dtype_in_use = list(self.dtype_rank_params.keys())
        dtype_to_pop = list(
            filter(lambda x: x not in dtype_in_use, self.param_storages.keys()))
        for d in dtype_to_pop:
            self.param_storages.pop(d)

        if self.offload:
            self._optim._master_weights = self._master_params
            cpu_master_params = [p for p in self._master_params.values()]
            for param in cpu_master_params:
                size = np.prod(param.shape) * align[Type.fp32.value]
                remaining = size % alignment[self.offload_device]
                ali = 0 if remaining == 0 else alignment[
                    self.offload_device] - remaining
                align_ = ali // align[Type.fp32.value]
                self.offload_buffer_size += np.prod(param.shape) + align_
                self.offload_param2align[param.name] = align_

            if cpu_master_params:
                with device_guard(self.rank, self.offload_device):
                    self.offload_params = ParamStorage(
                        size=self.offload_buffer_size,
                        dtype=Type.fp32.value,
                        device=self.offload_device)
                    self.offload_params.add_rank_params(
                        cpu_master_params, self.offload_param2align, False)
                    self.offload_params.buffer.stop_gradient = False

                    self.offload_grads = GradStorage(
                        size=self.offload_buffer_size,
                        dtype=Type.fp32.value,
                        device=self.offload_device,
                        destination=self.rank,
                        parm2align=self.offload_param2align,
                        convert_cpu=True)
                    for p in cpu_master_params:
                        self.offload_grads.add_grad(
                            p, self.offload_param2align[p.name])

                    self._optim._master_weights[
                        self.offload_params.buffer.
                        name] = self.offload_params.buffer

    def _offload_acc_grad(self, param_name, grad_fp32_cpu):
        """accumulate grads with offload strategy"""
        with device_guard(self.rank, self.offload_device):
            if param_name in self._master_params.keys():
                if self._master_params[param_name].grad is None:
                    self._master_params[param_name]._copy_gradient_from(
                        grad_fp32_cpu)
                else:
                    self._master_params[param_name].grad.add_(grad_fp32_cpu)

        self.offload_params.buffer._copy_gradient_from(
            self.offload_grads.buffer)

    def _offload_scale_grad(self, scale_size):
        """scale grads with offload strategy"""
        with device_guard(self.rank, self.offload_device):
            self.offload_grads.buffer.scale_(scale=scale_size)

    def _offload_clear_grad(self):
        """clear grads with offload strategy"""
        with device_guard(self.rank, self.offload_device):
            self.offload_grads.buffer.zero_()

    def step(self):
        """
        A wrapper for Optimizer's step function to finish the update operation of the optimizer.
        """

        if self.offload:
            params_list = [self.offload_params.buffer]

            #TODO(Baibaifan): Offload will support param_groups later
            if not isinstance(self._optim._param_groups[0], dict):
                self._optim._parameter_list = params_list
                self._optim._param_groups = params_list

        # Run the optimizer of the current rank step
        if self.offload:
            with device_guard(device=self.offload_device):
                self._optim.step()

            dev_id = int(paddle.get_device().split(":")[1])
            for param in self._local_params:
                if param.name in self._master_params.keys():
                    param.set_value(
                        self._master_params[param.name].cuda(dev_id).cast(
                            dtype=param.dtype))
        else:
            self._optim.step()

        # Synchronize all the updated shards in between the ranks
        self._broadcast_params()

    def minimize(self):
        raise RuntimeError(
            "optimizer.minimize() not support now, please use optimizer.step()")

    def set_state_dict(self, state_dict):
        self._optim.set_state_dict(state_dict)

    def state_dict(self):
        return self._optim.state_dict()

    def _clear_cache(self):
        self.__segment_params.clear()
        self._dtype_rank_params.clear()
        self._param2rank.clear()

    @paddle.autograd.no_grad()
    def _broadcast_params(self):
        """Broadcast the parameters of the current rank to each rank"""

        assert self._default_device == "gpu", "Only supported gpu"

        # Exchange all the shards with the other ranks
        for dtype_per_rank in self.param_storages.values():
            for dst_rank, internal_storage in dtype_per_rank.items():
                broadcast(tensor=internal_storage.buffer,
                          src=self.group.ranks[dst_rank],
                          group=self.group,
                          use_calc_stream=True)

            # Multi stream operation will be supported later
            wait(tensor=internal_storage.buffer,
                 group=self.group,
                 use_calc_stream=True)
