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
import weakref
from collections import OrderedDict

import numpy as np

import paddle
from paddle.framework import (
    base as imperative_base,
    core,
)


class HOOK_ACTION:
    ALL_REDUCE = 0
    REDUCE = 1
    REDUCE_SCATTER = 2


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
    ):
        self._param = param
        self._param_buffer = param_buffer
        self._grad_buffer = grad_buffer
        self._index = index
        self._padded_size = padded_size
        self._sharding_degree = sharding_degree
        self._rank = rank
        shard_size = param_buffer._numel() // sharding_degree
        rank_begin = rank * shard_size
        rank_end = rank_begin + shard_size

        param_begin = max(self._index, rank_begin)
        param_end = min(self._index + self._padded_size, rank_end)
        self._param_begin = param_begin
        self._param_end = param_end

        self._slice_grad = None
        if param_begin < param_end:
            self._slice_grad = grad_buffer._slice(param_begin, param_end)

        # share grad buffer
        tmp_grad = grad_buffer._slice(self._index, self._index + param._numel())
        tmp_grad.get_tensor()._set_dims(param.shape)
        if not use_main_grad:
            self._param._copy_gradient_from(tmp_grad)
        else:
            self._param.main_grad = tmp_grad

        # share param buffer
        self._share_param_buffer()

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


def build_reduce_scatter_buffer(
    parameters, sharding_degree, rank, use_main_grad=False
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
    grad_buffer = paddle.zeros(shape=[total_buffer_size], dtype=grad_dtype)

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
        )
        # hack main_grad
        sharding_grad_view[param.name] = grad_view
    return sharding_grad_view, param_buffer, grad_buffer


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
        else:
            assert not self._fuse_param, "not supported"
            (
                self._sharding_param_grad_view,
                self.param_storage,
                self.grad_storage,
            ) = build_reduce_scatter_buffer(
                self._params,
                self._comm_group.nranks,
                self._comm_group.rank,
                use_main_grad=self.use_main_grad,
            )
            # hack, for parameter sync in dygraph sharding optimizer after step
            self._params[0].comm_buffer_ref = weakref.ref(self)
        self._record_addr()

    def _record_addr(self):
        for param in self._params:
            self._grads_to_addr[param.name] = get_grad_address(
                param, self.use_main_grad
            )

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

        current_ptr = get_grad_address(param, self.use_main_grad)

        if self._grads_to_addr[param.name] != current_ptr:
            raise ValueError(
                "The address of the grad/main_grad of the param has been changed during training, "
                "which is not allowed for dp/sharding overlap with pp. "
                "This may be caused by some non-inplace operations on the grad/main_grad. Here are some examples: "
                "1. The grad/main_grad of the param is changed by other operations, such as: clear_grad, "
                "2. Using non-inplace operations on the grad/main_grad, such as: add, sub, mul, div, etc. "
            )

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
    def scale_grads(self):
        assert self._task is not None, "Task is not initialized."
        self._task.wait()

        if self._scale_after_comm:
            scale_factor = 1.0 / self._comm_group.nranks
            self.grad_storage.scale_(scale_factor)

        self._reset_params_checked_in()

    @imperative_base.no_grad
    def scale_and_split_grads(self):
        assert self._task is not None, "Task is not initialized. "
        self._task.wait()
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


def _fused_parameters_impl(
    parameters,
    use_main_grad=False,
    fuse_param=True,
    comm_overlap=False,
    comm_group=None,
    act=None,
    dst=-1,
    acc_step=1,
    scale_after_comm=False,
    apply_decay_param_fun=None,
):
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
            if apply_decay_param_fun is not None and apply_decay_param_fun(
                param.name
            ):
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


def fused_parameters(
    parameters,
    use_main_grad=False,
    fuse_param=True,
    comm_overlap=False,
    comm_group=None,
    act=None,
    dst=-1,
    acc_step=1,
    scale_after_comm=False,
    group_params=False,
    apply_decay_param_fun=None,
):
    """
    Fuse gradients. Fuse parameters if be enabled. Prepare for comm overlap if be enabled.
    :param parameters: all parameters to be fused.
    :param use_main_grad: does the gradient use main grad or not
    :param comm_overlap: enable comm overlap or not
    :param comm_group: the comm group for comm overlap
    :param act: the comm operation, could be chosen from reduce and allreduce
    :param dst: the dst for comm overlap
    :param acc_step: acc steps, using for comm overlap
    :param fuse_param: fuse param or not
    :param scale_after_comm: if enable comm overlap, specify the location of grad scale
    :param group_params: the format of the input parameters is param group
    :param apply_decay_param_fun: the funtion to filter decay param
    :return: param storage if fused, comm buffers if comm overlap, param groups if use group params
    """
    if act is None:
        g_shard_use_reduce = int(os.environ.get("FLAGS_shard_use_reduce", 1))
        act = (
            HOOK_ACTION.ALL_REDUCE
            if not g_shard_use_reduce
            else HOOK_ACTION.REDUCE
        )
    if comm_overlap:
        if comm_group is None:
            assert (
                act == HOOK_ACTION.ALL_REDUCE
            ), "Only allreduce action can use default comm group"
            comm_group = paddle.distributed.collective._get_default_group()
    if act == HOOK_ACTION.REDUCE:
        assert dst != -1
    elif act == HOOK_ACTION.ALL_REDUCE:
        dst = -1

    if group_params:
        updated_parameters = []
        comm_buffers = []
        for idx, group_param in enumerate(parameters):
            assert isinstance(
                group_param, dict
            ), "For group params, each group should be a dictionary."
            assert (
                'params' in group_param.keys()
            ), "For group params, each group should have parameters."
            real_param = group_param['params']
            (
                group_decay_fused,
                group_all_fused,
                group_all_buffers,
            ) = _fused_parameters_impl(
                real_param,
                use_main_grad=use_main_grad,
                fuse_param=fuse_param,
                comm_overlap=comm_overlap,
                comm_group=comm_group,
                act=act,
                dst=dst,
                acc_step=acc_step,
                scale_after_comm=scale_after_comm,
                apply_decay_param_fun=apply_decay_param_fun,
            )
            if comm_overlap:
                comm_buffers.extend(group_all_buffers)
            for fused_tensor in group_all_fused:
                fused_tensor.optimize_attr = real_param[0].optimize_attr
            group_param['params'] = group_all_fused
            updated_parameters.append(group_param)
        return updated_parameters, comm_buffers
    else:
        decay_fused, all_fused, all_buffers = _fused_parameters_impl(
            parameters,
            use_main_grad=use_main_grad,
            fuse_param=fuse_param,
            comm_overlap=comm_overlap,
            comm_group=comm_group,
            act=act,
            dst=dst,
            acc_step=acc_step,
            scale_after_comm=scale_after_comm,
            apply_decay_param_fun=apply_decay_param_fun,
        )

        return decay_fused, all_fused, all_buffers
