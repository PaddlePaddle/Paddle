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
from __future__ import annotations

import os

import paddle

from ..utils.log_util import get_sync_logger

_use_four_directions = os.environ.get(
    'PADDLE_USE_FOUR_DIRECTIONS_P2P', paddle.base.core.is_compiled_with_xpu()
)
_use_four_directions = False  # xpu use the same p2p method as gpu
if _use_four_directions:
    pass
else:
    pass

from paddle.distributed import fleet
from paddle.distributed.fleet.utils.tensor_fusion_helper import (
    HOOK_ACTION,
)

g_profile_pipeline_details_steps = int(
    os.getenv("FLAGS_profile_pipeline_details_steps", "0")
)

__all__ = []


def profile_pipeline_details(msg):
    GB = 1024.0 * 1024.0 * 1024.0
    if paddle.base.core.is_compiled_with_cuda():
        memory_allocated_size = paddle.device.cuda.memory_allocated() / GB
        memory_reserved_size = paddle.device.cuda.memory_reserved() / GB
    else:
        memory_allocated_size, memory_reserved_size = 0, 0
    get_sync_logger().info(
        f"{msg}: memory_allocated_size={memory_allocated_size:.2f}, memory_reserved_size={memory_reserved_size:.2f}"
    )


def get_action(is_dp, shard_split_param=False):
    if is_dp:
        return HOOK_ACTION.ALL_REDUCE
    if shard_split_param:
        return HOOK_ACTION.REDUCE_SCATTER
    return HOOK_ACTION.REDUCE


def _get_align_mode_scale():
    hcg = fleet.get_hybrid_communicate_group()
    data_parallel_world_size = hcg.get_data_parallel_world_size()
    sharding_parallel_world_size = hcg.get_sharding_parallel_world_size()
    return max(data_parallel_world_size, 1) * max(
        sharding_parallel_world_size, 1
    )


# assume only the first stage and last stage need data, and data consumption is ordered
# to be replaced by real micro dataset from reader
class FakeMicroDataset:
    def __init__(
        self,
        data,
        is_first_stage,
        is_last_stage,
        acc_steps,
        micro_batch_size,
    ):
        self._data = data
        self._index = 0
        self._acc_steps = acc_steps
        self._is_first_stage = is_first_stage
        self._is_last_stage = is_last_stage
        self._micro_batch_size = micro_batch_size

    def __iter__(self):
        return self

    def __next__(self):
        if self._index >= self._acc_steps:
            raise StopIteration
        assert self._is_first_stage or self._is_last_stage
        micro_batch_data = self._load_micro_batch(self._index)
        self._index += 1

        if self._index >= self._acc_steps:
            self._data = None  # clearup

        return micro_batch_data

    def _load_micro_batch(self, micro_step):
        inputs = self._data

        data = None
        label = None
        if self._is_first_stage:
            assert len(inputs) == 2, "length of input should be 2"
            data = self._load_micro_batch_impl(inputs[0], micro_step)

        if self._is_last_stage:
            assert len(inputs) == 2, "length of input should be 2"
            label = self._load_micro_batch_impl(inputs[1], micro_step)

        return (data, label)

    def _load_micro_batch_impl(self, inputs, micro_step):
        begin = micro_step * self._micro_batch_size
        end = begin + self._micro_batch_size

        if isinstance(inputs, tuple):
            output = []
            for data in inputs:
                if isinstance(data, list):
                    assert len(data) == self._acc_steps, (
                        f"length of data should be {self._acc_steps}, but it is {len(data)}"
                    )
                    output.append(
                        data[micro_step].detach()
                        if data[micro_step] is not None
                        else None
                    )
                elif data is not None:
                    self._check_data_valid(data)
                    output.append(data[begin:end, :].detach())
                else:
                    output.append(None)
            return tuple(output)
        elif isinstance(inputs, dict):
            output_dict = {}
            for key, data in inputs.items():
                if isinstance(data, list):
                    assert len(data) == self._acc_steps, (
                        f"length of data should be {self._acc_steps}, but it is {len(data)}"
                    )
                    output_dict[key] = (
                        data[micro_step].detach()
                        if data[micro_step] is not None
                        else None
                    )
                elif data is not None:
                    self._check_data_valid(data)
                    output_dict[key] = data[begin:end, :].detach()
                else:
                    output_dict[key] = None
            return output_dict
        elif isinstance(inputs, list):
            assert len(inputs) == self._acc_steps, (
                f"length of data should be {self._acc_steps}, but it is {len(inputs)}"
            )
            if isinstance(inputs[micro_step], list):
                return [
                    tensor.detach() if tensor is not None else None
                    for tensor in inputs[micro_step]
                ]
            return inputs[micro_step].detach()
        elif inputs is not None:
            self._check_data_valid(inputs)
            return inputs[begin:end, :].detach()
        else:
            return None

    def _check_data_valid(self, data):
        batch_size = data.shape[0]
        assert self._micro_batch_size * self._acc_steps == batch_size, (
            "batch_size needs to be divisible by micro_batch_size. Currently, "
            f"batch_size = {batch_size}, micro_batch_size = {self._micro_batch_size}, accumulate_steps = {self._acc_steps}."
        )


# A wrapper for pipeline dataser, to avoid GPU memory leaks.
class PipelineDatasetPreprocessor:
    def __init__(self, function):
        self.function = function

    def __call__(self):
        return self.function()
