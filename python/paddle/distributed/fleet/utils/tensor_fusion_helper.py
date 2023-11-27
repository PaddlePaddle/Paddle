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

import weakref
from collections import OrderedDict

import numpy as np

import paddle
from paddle.fluid import core
from paddle.framework import base as imperative_base

alignment = {
    "gpu": 256,
}
align = {
    paddle.float16.value: 2,
    paddle.bfloat16.value: 2,
    paddle.float32.value: 4,
}

__all__ = []


class HOOK_ACTION:
    ALL_REDUCE = 0
    REDUCE = 1
    REDUCE_SCATTER = 2


def flatten_dense_tensors(parameters, use_main_grad=False, release_grad=False):
    from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_storage import (
        GradStorage,
    )

    _buffer_size = 0
    _param2align = {}
    _param2offset = {}
    dtype = paddle.float32 if use_main_grad else parameters[0].dtype

    for param in parameters:
        assert param.trainable, "param must be trainable..."
        size = np.prod(param.shape) * align[dtype]
        remaining = size % alignment["gpu"]
        ali = 0 if remaining == 0 else alignment["gpu"] - remaining
        align_ = ali // align[dtype]
        _param2offset[param.name] = _buffer_size
        _buffer_size += np.prod(param.shape) + align_
        _param2align[param.name] = align_

    if release_grad:
        return None, _buffer_size, _param2offset

    # process gradient
    grad_storage = GradStorage(
        size=_buffer_size,
        dtype=dtype,
        device="gpu",
        destination="0",
        parm2align=_param2align,
    )

    for param in parameters:
        grad_storage.add_grad(param, _param2align[param.name])

    return grad_storage.buffer, _buffer_size, _param2offset


class ShardingGradView:
    def __init__(
        self,
        param,
        param_buffer,
        grad_buffer,
        index,
        padded_size,
        sharding_degree,
        rank,
        use_main_grad=False,
        release_grad=False,
    ):
        self._param = param
        self._param_buffer = param_buffer
        self._grad_buffer = grad_buffer
        self._index = index
        self._padded_size = padded_size
        self._sharding_degree = sharding_degree
        self._rank = rank
        self._use_main_grad = use_main_grad
        self._release_grad = release_grad
        shard_size = param_buffer._numel() // sharding_degree
        rank_begin = rank * shard_size
        rank_end = rank_begin + shard_size

        param_begin = max(self._index, rank_begin)
        param_end = min(self._index + self._padded_size, rank_end)
        self._param_begin = param_begin
        self._param_end = param_end

        self._slice_grad = None

        if not self._release_grad:
            self._link_grad_to_buffer()

        # share param buffer
        self._share_param_buffer()

    def _slice_grad_from_buffer(self):
        assert self._grad_buffer is not None
        if self._param_begin < self._param_end:
            self._slice_grad = self._grad_buffer._slice(
                self._param_begin, self._param_end
            )
        tmp_grad = self._grad_buffer._slice(
            self._index, self._index + self._param._numel()
        )
        return tmp_grad

    def _link_grad_to_buffer(self):
        tmp_grad = self._slice_grad_from_buffer()
        tmp_grad.get_tensor()._set_dims(self._param.shape)
        if not self._use_main_grad:
            self._param._copy_gradient_from(tmp_grad)
        else:
            self._param.main_grad = tmp_grad

    def _share_param_buffer(self):
        param_shape = self._param.shape
        stop_gradient = self._param.stop_gradient
        self._param.stop_gradient = True
        self._param.flatten_()
        self._param_buffer[
            self._index : self._index + self._param._numel()
        ] = self._param
        self._param.get_tensor()._set_dims(param_shape)
        self._param.stop_gradient = stop_gradient
        self._param_buffer._slice(
            self._index, self._index + self._param._numel()
        )._share_buffer_to(self._param)

    def fill_slice_param(self, slice_param):
        slice_begin = self._param_begin
        slice_end = self._param_end
        if slice_param._is_initialized():
            assert self._param_buffer._is_shared_buffer_with(slice_param)
            assert len(slice_param.shape) == 1
            assert slice_param.shape[0] == (slice_end - slice_begin)
        slice_begin = self._param_begin
        slice_end = self._param_end
        slice_buffer = self._param_buffer._slice(slice_begin, slice_end)
        slice_param.get_tensor()._set_dims([slice_end - slice_begin])
        slice_buffer._share_buffer_to(slice_param)

    def assign_slice_grad(self, slice_param):
        assert self._param_buffer._is_shared_buffer_with(self._param)
        slice_grad = self._slice_grad
        if slice_grad is None:
            return
        self.fill_slice_param(slice_param)
        if hasattr(self._param, "main_grad"):
            if not hasattr(slice_param, "main_grad"):
                slice_param.main_grad = slice_grad
            else:
                assert slice_param.main_grad is slice_grad
        elif slice_grad is not None:
            if slice_param.grad is None:
                slice_param._copy_gradient_from(slice_grad)
            else:
                assert slice_param.grad._is_shared_buffer_with(slice_grad)

    def _reset_grad_buffer(self):
        if self._slice_grad is not None:
            self._slice_grad._clear_dataptr()
            self._slice_grad = None

        if self._grad_buffer is not None:
            self._grad_buffer._clear_dataptr()
            self._grad_buffer = None


def build_reduce_scatter_buffer(
    parameters, sharding_degree, rank, use_main_grad=False, release_grad=False
):
    total_buffer_size = 0
    param2index = {}
    dtype = parameters[0].dtype

    def get_padded_size(param):
        size = np.prod(param.shape)
        align_size = alignment["gpu"] // align[dtype]
        align_size = align_size * sharding_degree
        padded_size = ((size + align_size - 1) // align_size) * align_size
        return padded_size

    for param in parameters:
        assert param.trainable, "param must be trainable..."
        param2index[param.name] = total_buffer_size
        total_buffer_size += get_padded_size(param)

    grad_dtype = paddle.float32 if use_main_grad else dtype

    param_buffer = paddle.zeros(shape=[total_buffer_size], dtype=dtype)
    grad_buffer = (
        paddle.zeros(shape=[total_buffer_size], dtype=grad_dtype)
        if not release_grad
        else None
    )

    sharding_grad_view = {}
    for param in parameters:
        padded_size = get_padded_size(param)
        grad_view = ShardingGradView(
            param,
            param_buffer,
            grad_buffer,
            param2index[param.name],
            padded_size,
            sharding_degree,
            rank,
            use_main_grad,
            release_grad,
        )
        # hack main_grad
        sharding_grad_view[param.name] = grad_view
    return sharding_grad_view, total_buffer_size, param_buffer, grad_buffer


def get_grad_address(param, use_main_grad):
    addr = None
    if use_main_grad:
        if param.main_grad is not None:
            addr = param.main_grad.data_ptr()
    else:
        if (param.grad is not None) and param.grad._is_initialized():
            addr = param.grad.data_ptr()
    return addr


class FusedCommBuffer:
    def __init__(
        self,
        id,
        params,
        comm_group,
        acc_steps=1,
        act=None,
        dst=-1,
        release_grads=False,
    ):
        self._id = id
        self._params = params
        self._acc_steps = acc_steps
        self._comm_group = comm_group
        self._release_grads = release_grads

        self.use_main_grad = all(
            hasattr(param, "main_grad") for param in self._params
        )
        assert (
            self.use_main_grad
        ), "All parameters in FusedCommBuffer must have main_grad."

        self._task = None
        self._dtype = (
            paddle.float32 if self.use_main_grad else self._params[0].dtype
        )
        self._params_step_dict = {}
        self._params_checked_in = 0
        self._params_to_addr = {}

        self._act = act
        if self._act == HOOK_ACTION.ALL_REDUCE:
            assert dst == -1
        elif self._act == HOOK_ACTION.REDUCE_SCATTER:
            assert dst == -1
        elif self._act == HOOK_ACTION.REDUCE:
            assert dst != -1
        else:
            raise ValueError(
                "The act should be allreudce for dp or reduce for sharding."
            )
        self._dst = dst

        self._init_step_dict()
        if self._act != HOOK_ACTION.REDUCE_SCATTER:
            (
                self.grad_storage,
                self.buffer_size,
                self.param2offset,
            ) = flatten_dense_tensors(
                self._params, self.use_main_grad, self._release_grads
            )
        else:
            (
                self._sharding_param_grad_view,
                self.buffer_size,
                self.param_storage,
                self.grad_storage,
            ) = build_reduce_scatter_buffer(
                self._params,
                self._comm_group.nranks,
                self._comm_group.rank,
                use_main_grad=self.use_main_grad,
                release_grad=self._release_grads,
            )
            # hack, pass comm buffer to dygraph sharding optimizer
            self._params[0].comm_buffer_ref = weakref.ref(self)

        if not self._release_grads:
            self._record_addr()

    def _record_addr(self):
        for param in self._params:
            self._params_to_addr[param.name] = get_grad_address(
                param, self.use_main_grad
            )

    def _clear_grad_storage(self):
        self.grad_storage._clear_dataptr()
        self.grad_storage = None
        if self._act == HOOK_ACTION.REDUCE_SCATTER:
            for param in self._params:
                self._sharding_param_grad_view[param.name]._reset_grad_buffer()

    def _init_step_dict(self):
        for p in self._params:
            self._params_step_dict[p.name] = 0

    def _reset_params_checked_in(self):
        self._task = None
        self._init_step_dict()
        self._params_checked_in = 0

    @property
    def _all_params_checked_in(self):
        return (
            len(self._params) == self._params_checked_in
            and len(self._params_step_dict) == 0
        )

    def _copy_grad_to_buffer(self, param):

        if self._params_step_dict[param.name] > 0:
            return

        if self.grad_storage is None:
            assert self._params_step_dict[param.name] == 0

            self.grad_storage = paddle.zeros(
                [self.buffer_size], dtype=self._dtype
            )

        if self._act == HOOK_ACTION.REDUCE_SCATTER:
            self._sharding_param_grad_view[
                param.name
            ]._grad_buffer = self.grad_storage
            tmp_var = self._sharding_param_grad_view[
                param.name
            ]._slice_grad_from_buffer()
        else:
            grad_end = self.param2offset[param.name] + np.prod(param.shape)
            assert grad_end <= self.buffer_size
            tmp_var = self.grad_storage._slice(
                self.param2offset[param.name], grad_end
            )

        grad_var = param.main_grad if self.use_main_grad else param.grad
        grad_var.stop_gradient = True
        grad_var.flatten_()

        tmp_var.add_(grad_var)
        tmp_var.get_tensor()._set_dims(param.shape)

        if self.use_main_grad:
            param.main_grad._clear()
            param.main_grad = tmp_var
            param.main_grad.name = "main_grad@" + param.name
        else:
            param._copy_gradient_from(tmp_var)

    def add_grad(self, param, use_comm=True):
        assert (
            param.name in self._params_step_dict
        ), "{} not in params_dict, please check accumulation_steps".format(
            param.name
        )

        if not self._release_grads:
            current_ptr = get_grad_address(param, self.use_main_grad)
            if self._params_to_addr[param.name] != current_ptr:
                raise ValueError(
                    "The address of the grad/main_grad of the param has been changed during training, "
                    "which is not allowed for dp/sharding overlap with pp. "
                    "This may be caused by some non-inplace operations on the grad/main_grad. "
                    "Please use the inplace version of the operations or disable the overlapping."
                )
        else:
            self._copy_grad_to_buffer(param)

        self._params_step_dict[param.name] += 1

        if self._params_step_dict[param.name] == self._acc_steps:
            self._params_checked_in += 1
            self._params_step_dict.pop(param.name)

        if self._all_params_checked_in and use_comm:
            self.comm_grads()

    @imperative_base.no_grad
    def assign_slice_grad(self, param, slice_param):
        assert self._act == HOOK_ACTION.REDUCE_SCATTER
        assert param.name in self._sharding_param_grad_view
        grad_view = self._sharding_param_grad_view[param.name]
        grad_view.assign_slice_grad(slice_param)

    @imperative_base.no_grad
    def sync_params(self):
        assert self._act == HOOK_ACTION.REDUCE_SCATTER
        full_buffer = self.param_storage
        group = self._comm_group
        shard_size = full_buffer._numel() // group.nranks
        begin = shard_size * group.rank
        end = begin + shard_size
        slice_buffer = full_buffer._slice(begin, end)
        group.process_group.all_gather(slice_buffer, full_buffer).wait()

    @property
    def params(self):
        return self._params

    @imperative_base.no_grad
    def comm_grads(self):
        assert self._all_params_checked_in, (
            "Not all params checked in."
            "Parameter number: {}, Check-in number: {}".format(
                len(self._params), self._params_checked_in
            )
        )
        self._comm_grads()

    @imperative_base.no_grad
    def _comm_grads(self):
        if self._act == HOOK_ACTION.ALL_REDUCE:
            task = paddle.distributed.all_reduce(
                self.grad_storage, group=self._comm_group, sync_op=False
            )

        elif self._act == HOOK_ACTION.REDUCE:
            task = paddle.distributed.reduce(
                self.grad_storage,
                dst=self._dst,
                group=self._comm_group,
                sync_op=False,
            )
        elif self._act == HOOK_ACTION.REDUCE_SCATTER:
            shard_size = self.grad_storage._numel() // self._comm_group.nranks
            begin = shard_size * self._comm_group.rank
            end = begin + shard_size
            reduce_scattered = self.grad_storage._slice(begin, end)
            task = paddle.distributed.reduce_scatter(
                reduce_scattered,
                self.grad_storage,
                group=self._comm_group,
                sync_op=False,
            )
        self._task = task

    @imperative_base.no_grad
    def scale_and_split_grads(self):
        assert self._task is not None, "Task is not initialized. "
        self._task.wait()
        scale_factor = 1.0 / self._comm_group.nranks
        self.grad_storage.scale_(scale_factor)

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
