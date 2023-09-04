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
import itertools
import os
from collections import OrderedDict

import numpy as np

import paddle
from paddle.framework import base as imperative_base
from paddle.framework import core


class HOOK_ACTION:
    ALL_REDUCE = 0
    REDUCE = 1


alignment = {
    "gpu": 256,
}

align = {
    paddle.float16.value: 2,
    paddle.bfloat16.value: 2,
    paddle.float32.value: 4,
}


def assign_group_by_size(parameters, group_size=128 * 1024 * 1024):
    is_sparse_gradient = [False] * len(parameters)

    group_indices = core.eager_assign_group_by_size(
        parameters, is_sparse_gradient, [group_size, group_size]
    )

    var_groups = OrderedDict()
    for group_idx, indices in enumerate(group_indices):
        for index in indices:
            var_groups.setdefault(group_idx, []).append(parameters[index])
    return var_groups


def flatten_dense_tensors(
    parameters, use_main_grad=False, fuse_param=True, warp_buffer=False
):
    from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_storage import (
        GradStorage,
        ParamStorage,
    )

    _buffer_size = 0
    _param2align = {}
    dtype = parameters[0].dtype

    for param in parameters:
        assert param.trainable, "param must be trainable..."
        size = np.prod(param.shape) * align[dtype]
        remaining = size % alignment["gpu"]
        ali = 0 if remaining == 0 else alignment["gpu"] - remaining
        align_ = ali // align[dtype]
        _buffer_size += np.prod(param.shape) + align_
        _param2align[param.name] = align_

    if fuse_param:
        param_storage = ParamStorage(
            size=_buffer_size, dtype=dtype, device="gpu"
        )
        param_storage.add_rank_params(parameters, _param2align)

    # process gradient
    grad_dtype = paddle.float32 if use_main_grad else dtype
    grad_storage = GradStorage(
        size=_buffer_size,
        dtype=grad_dtype,
        device="gpu",
        destination="0",
        parm2align=_param2align,
    )

    for param in parameters:
        grad_storage.add_grad(param, _param2align[param.name])

    if warp_buffer:
        if fuse_param:
            param_storage.warp_buffer()
        grad_storage.warp_buffer()

    if fuse_param:
        if not use_main_grad:
            # param_storage --> grad_storage
            param_storage.buffer._copy_gradient_from(grad_storage.buffer)
        else:
            param_storage.buffer.main_grad = grad_storage.buffer
        param_storage.buffer.stop_gradient = False
        return param_storage, grad_storage
    else:
        return grad_storage


def bw_hook_func(buffer, param):
    @paddle.autograd.no_grad()
    def fused_comm(*_):
        buffer.add_grad(param)

    return fused_comm


class FusedCommBuffer:
    def __init__(
        self,
        id,
        params,
        comm_group,
        acc_steps=1,
        act=None,
        dst=-1,
        use_main_grad=None,
        fuse_param=False,
        scale_after_comm=True,
    ):
        self._id = id
        self._params = params
        self._acc_steps = acc_steps
        self._comm_group = comm_group
        self._scale_after_comm = scale_after_comm
        self._fuse_param = fuse_param

        self.use_main_grad = (
            use_main_grad
            if use_main_grad is not None
            else hasattr(self._params[0], "main_grad")
        )

        self._task = None
        self._params_step_dict = {}
        self._params_checked_in = 0
        self._grads_to_addr = {}

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

        if self._fuse_param:
            self.param_storage, self.grad_storage = flatten_dense_tensors(
                self._params,
                use_main_grad=use_main_grad,
                fuse_param=True,
                warp_buffer=True,
            )
            self.param_storage = self.param_storage.buffer
            self.grad_storage = self.grad_storage.buffer
        else:
            self.param_storage = None
            self.grad_storage = flatten_dense_tensors(
                self._params,
                use_main_grad=self.use_main_grad,
                fuse_param=False,
                warp_buffer=False,
            ).buffer

        self._record_addr()

    def _record_addr(self):
        for param in self._params:
            addr = (
                param.main_grad.data_ptr()
                if self.use_main_grad
                else param.grad.data_ptr()
            )
            self._grads_to_addr[param.name] = addr

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

    def add_grad(self, param, use_comm=True):
        assert param.name in self._params_step_dict
        current_ptr = (
            param.main_grad.data_ptr()
            if self.use_main_grad
            else param.grad.data_ptr()
        )
        if self._grads_to_addr[param.name] != current_ptr:
            raise ValueError(
                "The address of the grad/main_grad of the param has been changed during training, "
                "which is not allowed for dp/sharding overlap with pp. "
                "This may be caused by some non-inplace operations on the grad/main_grad. "
                "Please use the inplace version of the operations or disable the overlapping."
            )

        self._params_step_dict[param.name] += 1

        if self._params_step_dict[param.name] == self._acc_steps:
            self._params_checked_in += 1
            self._params_step_dict.pop(param.name)

        if self._all_params_checked_in and use_comm:
            self.comm_grads()

    @imperative_base.no_grad
    def comm_grads(self):
        assert self._all_params_checked_in, (
            "Not all params checked in."
            "Parameter number: {}, Check-in number: {}".format(
                len(self._params), self._params_checked_in
            )
        )

        if not self._scale_after_comm:
            scale_factor = 1.0 / self._comm_group.nranks
            self.grad_storage.scale_(scale_factor)

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

        self._task = task

    @imperative_base.no_grad
    def scale_grads(self):
        assert self._task is not None, "Task is not initialized."
        self._task.wait()

        if self._scale_after_comm:
            scale_factor = 1.0 / self._comm_group.nranks
            self.grad_storage.scale_(scale_factor)

        self._reset_params_checked_in()


def obtain_storage(
    parameters,
    use_main_grad=False,
    clip=True,
    dist=False,
    fuse_param=True,
    comm_overlap=False,
    act=None,
    comm_group=None,
    dst=-1,
    acc_steps=1,
    scale_after_comm=False,
):
    if len(parameters) < 1:
        return [], []

    var_groups = assign_group_by_size(parameters, group_size=256 * 1024 * 1024)
    storage = []
    buffers = []
    for group_idx, parameters in var_groups.items():
        comm_buffer = FusedCommBuffer(
            group_idx,
            parameters,
            comm_group=comm_group,
            acc_steps=acc_steps,
            act=act,
            dst=dst,
            use_main_grad=use_main_grad,
            fuse_param=fuse_param,
            scale_after_comm=scale_after_comm,
        )
        if fuse_param:
            param_buffer = comm_buffer.param_storage
            param_buffer.need_clip = clip
            param_buffer.is_distributed = dist
            storage.append(param_buffer)
        if comm_overlap:
            for param in parameters:
                param._register_backward_hook(bw_hook_func(comm_buffer, param))
            buffers.append(comm_buffer)

    return storage, buffers


def filter_params(params, is_fp32, is_distributed, need_clip):
    params = list(
        filter(
            lambda x: x.is_distributed
            if is_distributed
            else (not x.is_distributed),
            params,
        )
    )
    params = list(
        filter(
            lambda x: getattr(x, 'need_clip', True)
            if need_clip
            else (not getattr(x, 'need_clip', True)),
            params,
        )
    )
    params = list(
        filter(
            lambda x: x.dtype == paddle.float32
            if is_fp32
            else x.dtype != paddle.float32,
            params,
        )
    )
    dtype = None
    for p in params:
        if dtype is None:
            dtype = p.dtype
        else:
            assert dtype == p.dtype

    return params, dtype


def fused_parameters(
    parameters,
    use_main_grad=False,
    fuse_param=True,
    comm_overlap=False,
    comm_group=None,
    dst=-1,
    acc_step=1,
    scale_after_comm=False,
):
    """
    Fuse gradients. Fuse parameters if be enabled. Prepare for comm overlap if be enabled.
    :param parameters: all parameters to be fused.
    :param use_main_grad: does the gradient use main grad or not
    :param comm_overlap: enable comm overlap or not
    :param comm_group: the comm group for comm overlap
    :param dst: the dst for comm overlap
    :param acc_step: acc steps, using for comm overlap
    :param fuse_param: fuse param or not
    :param scale_after_comm: if enable comm overlap, specify the location of grad scale
    :return: param storage if fused, comm buffers is comm overlap
    """
    g_shard_use_reduce = int(os.environ.get("FLAGS_shard_use_reduce", 1))
    act = (
        HOOK_ACTION.ALL_REDUCE if not g_shard_use_reduce else HOOK_ACTION.REDUCE
    )
    if comm_overlap:
        assert comm_group is not None
    if act == HOOK_ACTION.REDUCE:
        assert dst != -1
    elif act == HOOK_ACTION.ALL_REDUCE:
        dst = -1
    param_groups = []
    attrs = []

    is_fp32 = [True, False]
    is_distributed = [True, False]
    need_clip = [True, False]

    no_fp32_dtype = None
    for fp32, dist, clip in itertools.product(
        is_fp32, is_distributed, need_clip
    ):
        params, dtype = filter_params(parameters, fp32, dist, clip)
        if not fp32:
            if no_fp32_dtype is None:
                no_fp32_dtype = dtype
            elif dtype is not None:
                assert no_fp32_dtype == dtype
        attrs.append([dtype, dist, clip])
        param_groups.append(params)

    decay_fused = []
    all_fused = []
    all_buffers = []
    for params, attr in zip(param_groups, attrs):
        decay_params = []
        other_params = []

        for param in params:
            if not any(nd in param.name for nd in ["bias", "norm", "b_0"]):
                decay_params.append(param)
            else:
                other_params.append(param)

        is_distributed = attr[1]
        need_clip = attr[2]
        decay, decay_buffers = obtain_storage(
            decay_params,
            use_main_grad=use_main_grad,
            clip=need_clip,
            dist=is_distributed,
            fuse_param=fuse_param,
            comm_overlap=comm_overlap,
            act=act,
            comm_group=comm_group,
            dst=dst,
            acc_steps=acc_step,
            scale_after_comm=scale_after_comm,
        )
        other, other_buffers = obtain_storage(
            other_params,
            fuse_param=fuse_param,
            comm_overlap=comm_overlap,
            use_main_grad=use_main_grad,
            clip=need_clip,
            dist=is_distributed,
            act=act,
            comm_group=comm_group,
            dst=dst,
            acc_steps=acc_step,
            scale_after_comm=scale_after_comm,
        )
        decay_fused += decay
        all_fused += decay
        all_fused += other
        all_buffers += decay_buffers
        all_buffers += other_buffers

    return decay_fused, all_fused, all_buffers
