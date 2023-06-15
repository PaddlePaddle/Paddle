# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from collections import OrderedDict

import numpy as np

import paddle
from paddle import _legacy_C_ops
from paddle.distributed.parallel import _split_tensors
from paddle.fluid import core
from paddle.framework import base as imperative_base

__all__ = []


class HOOK_ACTION:
    ALL_REDUCE = 0
    REDUCE = 1


FLOAT_TYPE_DICT = {
    paddle.float16: "float16",
    paddle.float32: "float32",
    paddle.float64: "float64",
    paddle.bfloat16: "bfloat16",
}

PADDLE_TO_NUMBER = {
    paddle.float16: 0,
    paddle.float32: 1,
    paddle.float64: 2,
    paddle.int32: 3,
    paddle.int64: 4,
    paddle.bfloat16: 5,
}

NUMBER_TO_DTYPE = {
    0: "float16",
    1: "float32",
    2: "float64",
    3: "int32",
    4: "int64",
    5: "bfloat16",
}


def is_float_tensor(tensor):
    """Is a float tensor"""
    return tensor.dtype in FLOAT_TYPE_DICT.keys()


def get_tensor_dtype(dtype):
    assert dtype in FLOAT_TYPE_DICT.keys()
    return FLOAT_TYPE_DICT[dtype]


def paddle_2_number(dtype):
    assert dtype in PADDLE_TO_NUMBER.keys()
    return PADDLE_TO_NUMBER[dtype]


def number_2_dtype(number):
    assert number in NUMBER_TO_DTYPE.keys()
    return NUMBER_TO_DTYPE[number]


def get_tensor_bytes(tensor):
    """Get the bytes a tensor occupied."""
    elem_size = None
    if tensor.dtype == paddle.float32:
        elem_size = 4
    elif tensor.dtype == paddle.float64:
        elem_size = 8
    elif tensor.dtype == paddle.int64:
        elem_size = 8
    elif tensor.dtype == paddle.int32:
        elem_size = 4
    elif tensor.dtype == paddle.float16:
        elem_size = 2
    elif tensor.dtype == paddle.int8:
        elem_size = 1
    else:
        raise ValueError(f"unknown data type: {tensor.dtype}")
    return tensor.numel() * elem_size


def _all_gather(tensor, group=None, use_calc_stream=True):
    """
    The main difference with paddle.distributed.all_gather:
    no need to pass in tensor_list, the returned tensor is spliced
    """
    if group is not None and not group.is_member():
        return
    ring_id = 0 if group is None else group.id
    nranks = (
        paddle.distributed.collective._get_global_group().nranks
        if group is None
        else group.nranks
    )
    return _legacy_C_ops.c_allgather(
        tensor,
        'use_calc_stream',
        use_calc_stream,
        'ring_id',
        ring_id,
        'nranks',
        nranks,
    )


class FusedCommBuffer:
    def __init__(
        self,
        id,
        params,
        comm_group,
        acc_steps=1,
        act=None,
        dst=-1,
    ):
        self._id = id
        self._params = params
        self._acc_steps = acc_steps
        self._comm_group = comm_group

        self._tasks = []
        self._grads = []
        self._params_step_dict = {}
        self._params_checked_in = 0
        self._coalesced_grads_and_grad_vars = []

        self._act = act
        if self._act == HOOK_ACTION.ALL_REDUCE:
            assert dst == -1
        elif self._act == HOOK_ACTION.REDUCE:
            assert dst != -1
        else:
            raise ValueError(
                "The act should be allreudce for dp or reduce for sharding."
            )
        self._dst = dst

        self._init_step_dict()

    def _init_step_dict(self):
        for p in self._params:
            self._params_step_dict[p.name] = 0

    def _reset_params_checked_in(self):
        self._tasks.clear()
        self._grads.clear()
        self._init_step_dict()
        self._params_checked_in = 0
        self._coalesced_grads_and_grad_vars.clear()

    @property
    def _all_params_checked_in(self):
        return (
            len(self._params) == self._params_checked_in
            and len(self._params_step_dict) == 0
        )

    def add_grad(self, param):
        assert param.name in self._params_step_dict

        if self._params_step_dict[param.name] == 0:
            if getattr(param, "main_grad", None) is not None:
                assert param.grad is None
                self._grads.append(param.main_grad)
            else:
                self._grads.append(param.grad)

        self._params_step_dict[param.name] += 1

        if self._params_step_dict[param.name] == self._acc_steps:
            self._params_checked_in += 1
            self._params_step_dict.pop(param.name)

        if self._all_params_checked_in:
            self._fused_comm_grads()

    @imperative_base.no_grad
    def _fused_comm_grads(self):
        assert self._all_params_checked_in
        flattened_vars = []
        g_var_shapes = []

        for g_var in self._grads:
            g_var_shapes.append(g_var.shape)
            flattened_vars.append(
                paddle.reshape(x=g_var, shape=[np.prod(g_var.shape)])
            )

        coalesced_grad = paddle.concat(flattened_vars)
        self._coalesced_grads_and_grad_vars.append(
            [coalesced_grad, self._grads, g_var_shapes]
        )

        for coalesced_grad, _, _ in self._coalesced_grads_and_grad_vars:
            if self._act == HOOK_ACTION.ALL_REDUCE:
                task = paddle.distributed.all_reduce(
                    coalesced_grad, group=self._comm_group, sync_op=False
                )
            elif self._act == HOOK_ACTION.REDUCE:
                task = paddle.distributed.reduce(
                    coalesced_grad,
                    dst=self._dst,
                    group=self._comm_group,
                    sync_op=False,
                )
            self._tasks.append(task)

    @imperative_base.no_grad
    def scale_and_split_grads(self):
        for task in self._tasks:
            task.wait()

        scale_factor = 1.0 / self._comm_group.nranks
        for coalesced_grad, _, _ in self._coalesced_grads_and_grad_vars:
            coalesced_grad.scale_(scale_factor)

        _split_tensors(self._coalesced_grads_and_grad_vars)
        self._reset_params_checked_in()


def assign_group_by_size(parameters, group_size=128 * 1024 * 1024):

    group_idx = 0
    memory_counter = 0
    var_groups = OrderedDict()
    dtype = parameters[0].dtype

    for var in parameters:
        bytes = np.prod(var.shape) * core.size_of_dtype(var.dtype)
        if memory_counter < group_size and dtype == var.dtype:
            memory_counter += bytes
        else:
            memory_counter = bytes
            dtype = var.dtype
            group_idx += 1
        var_groups.setdefault(group_idx, []).append(var)

    return var_groups
