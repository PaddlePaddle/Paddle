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
#Taken and modified for fairscale from:
#    https://github.com/facebookresearch/fairscale/blob/main/fairscale/optim/oss.py
#Commit: 8acbec718f3c70a6b9785470bb9e05cd84fc3f8e

import io
import copy
import time
import logging
import numpy as np
from math import inf
from itertools import chain
from functools import reduce
from collections import OrderedDict

import paddle
import paddle.distributed as dist
from paddle.optimizer import SGD, Optimizer

from .internal_storage import ParamStorage
from .sharding_utils import Type

__all__ = ["ShardingOptimizer"]

# CUDA alignment 256 bytes
alignment = {"gpu": 256, }
align = {
    Type.fp16.value: 2,
    Type.fp32.value: 4,
}


class ShardingOptimizer(Optimizer):
    """
    A wrapper for Sharding Stage2 Optimizer in Dygraph. 

    .. warning: ShardingOptimizer encapsulates the optimization strategy and integrates it into the optimizer.

    .. ZeRO: 1.https://arxiv.org/pdf/1910.02054.pdf 2.https://arxiv.org/pdf/1910.02054.pdf.

    """

    # TODO (Baibaifan) 
    # Feature Notes:
    # 1. Unified memory for parameters and parameters.grad to InternalStorage.
    # 2. Support the segmentation of optimizer parameters and partial updating of parameters.
    # 3. Dynamically adjust training parameters and models。
    # 4. Support offload function.
    # 5. Support the establishment of independent communication groups.
    # 6. Broadcast_fp16 is not supported now.
    def __init__(self,
                 params,
                 optim,
                 group,
                 broadcast_fp16=False,
                 offload=False,
                 device="gpu",
                 accumulation_steps=None,
                 **kw):

        super().__init__(optim._learning_rate, params, kw)

        # Segmentation information
        self._dtype_rank_params = OrderedDict(
        )  # {dtype:[param1,param2]} device, rank, params
        self._param2rank = {}
        self._segment_params = []
        self._rank_buffer_size = {}  # {dtype: {rank: numel+alignment}}
        self._param2align = {}  # {param.name: align}

        # Default information
        self._optim_defaults = kw
        self._optim = optim
        self._local_params = params
        self._default_device = device
        self._accumulation_steps = accumulation_steps

        assert group is not None, "Distributed communication group is must be gived"
        self.group = group
        self.world_size = group.nranks
        self.rank = group.rank

        self.broadcast_fp16 = broadcast_fp16
        self.param_storages = {}  # {dtype: {rank: InternalStorage}}
        self.offload = offload  # Using for offload

        # Update optimizer parameters and adjust parameter storage and use according to rank.
        self.update_opt_status()

    def update_opt_status(self):
        """Update optimizer status and parameter storage information, and special functions to be developed.
        """
        # func 1
        self._integration_params()

        # fun 2 TODO

    # Segement helpers

    def segment_params(self):
        """
        Divide all optimizer parameters equally into rank.
        """
        if len(self._segment_params) == 0:
            self._segment_params, param_lists = [
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
                # param_group_rank = copy.copy(params)
                self._segment_params[rank].extend(params)
        return self._segment_params

    @property
    def local_params(self):
        return self._local_params

    @property
    def accumulation_steps(self):
        return self._accumulation_steps

    @property
    def param2rank(self):
        """Map the params to the rank which owns them"""
        if len(self._param2rank) == 0:
            for rank, params in enumerate(self.segment_params()):
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
                    self._dtype_rank_params[
                        param.dtype] = [[] for _ in range(self.world_size)]
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

    def step(self):
        """
        A wrapper for Optimizer's step function to finish the update operation of the optimizer.
        """

        # Synchronize optimizer parameters for the current rank
        if len(self.dtype_rank_params.keys(
        )) == 1 and Type.fp32.value in self.dtype_rank_params.keys():
            self._optim._parameter_list = self.dtype_rank_params[
                Type.fp32.value][self.rank]
        elif len(self.dtype_rank_params.keys(
        )) == 1 and Type.fp16.value in self.dtype_rank_params.keys():
            self._optim._parameter_list = self.dtype_rank_params[
                Type.fp16.value][self.rank]
        else:
            self._optim._parameter_list = self.dtype_rank_params[
                Type.fp16.value][self.rank] + self.dtype_rank_params[
                    Type.fp32.value][self.rank]

        # Run the optimizer of the current rank step
        loss = self._optim.step()

        # Synchronize all the updated shards in between the ranks
        self._broadcast_params()

        # Return full parameters to optimizer parameters
        self._optim._parameter_list = self._local_params

        return loss

    def _clear_cache(self):
        self._segment_params.clear()
        self._dtype_rank_params.clear()
        self._param2rank.clear()

    def _clear_gradients(self):
        """
        Clean up the gradient of the optimizer's full parameters.
        """
        for dtype in self._optim.param_storages.keys():
            self._optim.param_storages[dtype][self.rank].buffer.zero_()

    @paddle.no_grad()
    def _broadcast_params(self):
        """Broadcast the parameters of the current rank to each rank"""

        assert self._default_device == "gpu", "Only supported gpu"

        # # Multi stream operation will be supported later
        # paddle.device.cuda.synchronize()

        # Convert fp32 internal storage to fp16
        if self.broadcast_fp16:
            for dst_rank, internal_storage in self.param_storages[
                    Type.fp32.value].items():
                internal_storage.to(dtype=Type.fp16.value,
                                    device=self._default_device,
                                    keep_alignment=False)
            # Multi stream operation will be supported later
            # paddle.device.cuda.synchronize()

            # Exchange all the shards with the other ranks
        for dtype_per_rank in self.param_storages.values():
            for dst_rank, internal_storage in dtype_per_rank.items():
                dist.broadcast(
                    tensor=internal_storage.buffer,
                    src=dst_rank,
                    group=self.group,
                    use_calc_stream=True)

            # Multi stream operation will be supported later
            dist.wait(
                tensor=internal_storage.buffer,
                group=self.group,
                use_calc_stream=True)

            # Convert fp32 internal storage to fp32
        if self.broadcast_fp16:
            for dst_rank, internal_storage in self.param_storages[
                    Type.fp32.value].items():
                internal_storage.to(dtype=Type.fp32.value,
                                    device=self._default_device,
                                    keep_alignment=True)
            # Multi stream operation will be supported later
            # paddle.device.cuda.synchronize()
