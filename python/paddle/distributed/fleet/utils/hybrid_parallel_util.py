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

from collections import OrderedDict

import numpy as np

import paddle
from paddle import framework
from paddle.distributed.parallel import (
    _split_tensors,
    build_groups,
    in_dygraph_mode,
    sync_params_buffers,
)

# (TODO: GhostScreaming) It will be removed later.
from paddle.fluid import core

from .log_util import logger

__all__ = []


def _apply_collective_grads(parameters, comm_group, bucket_size, scale=None):
    grad_var_set = set()
    grad_vars = []
    sparse_grad_vars = []

    for param in parameters:
        if param.trainable and (param._grad_ivar() is not None):
            g_var = param._grad_ivar()
            assert (
                not g_var._is_sparse()
            ), "Now, it doesn't support sparse parameters"
            grad_vars.append(g_var)
            assert g_var not in grad_var_set
            grad_var_set.add(g_var)

    coalesced_grads_and_vars = build_groups(grad_vars, bucket_size)

    nranks = (
        paddle.distributed.get_world_size()
        if comm_group is None
        else comm_group.nranks
    )

    scale = nranks if scale is None else 1.0 / scale
    scale = None if scale == 1.0 else scale

    for coalesced_grad, _, _ in coalesced_grads_and_vars:
        # need to div nranks
        if scale is not None:
            div_factor = paddle.to_tensor(scale, dtype=coalesced_grad.dtype)
            paddle.fluid.framework._dygraph_tracer().trace_op(
                type="elementwise_div",
                inputs={'X': coalesced_grad, 'Y': div_factor},
                outputs={'Out': coalesced_grad},
                attrs={'axis': -1},
            )
        paddle.distributed.all_reduce(coalesced_grad, group=comm_group)

    _split_tensors(coalesced_grads_and_vars)


def _apply_collective_grads_eager(
    parameters, comm_group, bucket_size, scale=None
):
    grad_var_set = set()
    grad_vars = []

    for param in parameters:
        g_var = None
        if param.trainable and (param._grad_ivar() is not None):
            g_var = param._grad_ivar()
        if param.trainable and hasattr(param, "main_grad"):
            assert param._grad_ivar() is None, "param.grad is not None"
            g_var = param.main_grad
        if g_var is not None:
            assert (
                not g_var.is_sparse()
            ), "Now, it doesn't support sparse parameters"
            grad_vars.append(g_var)
            assert g_var not in grad_var_set
            grad_var_set.add(g_var)

    coalesced_grads_and_vars = build_groups(grad_vars, bucket_size)

    nranks = (
        paddle.distributed.get_world_size()
        if comm_group is None
        else comm_group.nranks
    )

    scale = 1.0 / nranks if scale is None else scale
    scale = None if scale == 1.0 else scale

    for coalesced_grad, _, _ in coalesced_grads_and_vars:
        # need to div nranks
        if scale is not None:
            coalesced_grad.scale_(scale)
        paddle.distributed.all_reduce(coalesced_grad, group=comm_group)

    _split_tensors(coalesced_grads_and_vars)


def _broadcast_data_help(data, shape, dtype, hcg):
    model_parallel_group = hcg.get_model_parallel_group()
    src_rank = hcg.get_model_parallel_group_src_rank()
    mp_rank = hcg.get_model_parallel_rank()

    shape_gpu = paddle.to_tensor(shape, dtype="int32")
    paddle.distributed.broadcast(
        shape_gpu, src=src_rank, group=model_parallel_group, sync_op=True
    )

    if mp_rank != 0:
        input_data = paddle.zeros(shape_gpu, dtype=dtype)
    else:
        input_data = data

    paddle.distributed.broadcast(
        input_data, src=src_rank, group=model_parallel_group, sync_op=True
    )

    if mp_rank != 0:
        if in_dygraph_mode():
            data._clear_data()
            input_data._share_buffer_to(data)
        else:
            data.value().get_tensor()._clear()
            data.value().get_tensor()._share_data_with(
                input_data.value().get_tensor()
            )


def _broadcast_object_list_help(object_list, hcg):
    model_parallel_group = hcg.get_model_parallel_group()
    src_rank = hcg.get_model_parallel_group_src_rank()
    mp_rank = hcg.get_model_parallel_rank()

    paddle.distributed.broadcast_object_list(
        object_list, src=src_rank, group=model_parallel_group
    )


def broadcast_input_data(hcg, *inputs, **kwargs):
    cur_device = paddle.get_device()
    dev = cur_device.split(":")[0]
    assert dev in [
        "xpu",
        "gpu",
        "npu",
    ], f"Only support xpu, gpu and npu now, but this is {dev}"
    dev_idx = int(cur_device.split(':')[1])
    if dev == "gpu":
        place = paddle.CUDAPlace(dev_idx)
    else:
        place = eval(f"paddle.{dev.upper()}Place")(dev_idx)

    for v in inputs:
        if isinstance(v, core.eager.Tensor):
            with framework.no_grad():
                if in_dygraph_mode() and not eval(f"v.place.is_{dev}_place")():
                    v_gpu = v._copy_to(place, True)
                    v._clear_data()
                    v_gpu._share_buffer_to(v)
                _broadcast_data_help(v, v.shape, v.dtype, hcg)
        else:
            _broadcast_object_list_help(v, hcg)

    for k, v in kwargs.items():
        if isinstance(v, core.eager.Tensor):
            with framework.no_grad():
                if in_dygraph_mode() and not eval(f"v.place.is_{dev}_place")():
                    v_gpu = v._copy_to(place, True)
                    v._clear_data()
                    v_gpu._share_buffer_to(v)
                _broadcast_data_help(v, v.shape, v.dtype, hcg)
            kwargs[k] = v
        else:
            kwargs[k] = _broadcast_object_list_help(v, hcg)
    return inputs, kwargs


def broadcast_mp_parameters(model, hcg):
    model_parallel_group = hcg.get_model_parallel_group()
    src_rank = hcg.get_model_parallel_group_src_rank()
    sync_params_buffers(
        model, model_parallel_group, src_rank, is_model_parallel=True
    )


def broadcast_dp_parameters(model, hcg):
    data_parallel_group = hcg.get_data_parallel_group()
    src_rank = hcg.get_data_parallel_group_src_rank()
    sync_params_buffers(
        model, data_parallel_group, src_rank, is_model_parallel=False
    )


def fused_allreduce_gradients_with_group(
    parameter_list, group, bucket_size=128 * 1024 * 1024, scale=None
):
    apply_func = (
        _apply_collective_grads_eager
        if in_dygraph_mode()
        else _apply_collective_grads
    )
    with framework.no_grad():
        apply_func(parameter_list, group, bucket_size, scale)


def fused_allreduce_gradients(parameter_list, hcg):
    data_parallel_group = None if hcg is None else hcg.get_data_parallel_group()
    logger.debug("dp start fuse allreduce gradients")
    fused_allreduce_gradients_with_group(parameter_list, data_parallel_group)


def sharding_reduce_gradients(parameter_list, hcg):
    # TODO allreduce --> reduce
    # TODO merge grad / nrank with dp
    logger.debug("sharding start gradients sync")
    with framework.no_grad():

        sharding_nrank = hcg.get_sharding_parallel_group().nranks
        for param in parameter_list:
            if param.trainable and (param._grad_ivar() is not None):
                param.grad.scale_(1.0 / sharding_nrank)
                paddle.distributed.all_reduce(
                    param.grad,
                    group=hcg.get_sharding_parallel_group(),
                    sync_op=True,
                )


def broadcast_sharding_parameters(model, hcg):
    # TODO TO save memory, use un-fused broadcast to avoid potentional OOM
    logger.debug("sharding start init parameters sync")
    sharding_parallel_group = hcg.get_sharding_parallel_group()
    src_rank = hcg.get_sharding_parallel_group_src_rank()
    sync_params_buffers(
        model, sharding_parallel_group, src_rank, is_model_parallel=False
    )


class FusedAllReduceBuffer:
    def __init__(self, params, comm_group, acc_steps=1):
        self._params = params
        self._acc_steps = acc_steps
        self._comm_group = comm_group

        self._tasks = []
        self._grads = []
        self._params_step_dict = {}
        self._params_checked_in = 0
        self._coalesced_grads_and_grad_vars = []

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
            self._fused_allreduce_grads()

    def _fused_allreduce_grads(self):
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
            self._tasks.append(
                paddle.distributed.all_reduce(
                    coalesced_grad, group=self._comm_group, sync_op=False
                )
            )

    def scale_and_split_grads(self):
        for task in self._tasks:
            task.wait()

        scale_factor = 1.0 / self._comm_group.nranks
        for coalesced_grad, _, _ in self._coalesced_grads_and_grad_vars:
            coalesced_grad.scale_(scale_factor)

        _split_tensors(self._coalesced_grads_and_grad_vars)
        self._reset_params_checked_in()


def assign_group_by_size(parameters, group_size=256 * 1024 * 1024):
    is_sparse_gradient = [False] * len(parameters)

    group_indices = core.eager_assign_group_by_size(
        parameters, is_sparse_gradient, [group_size, group_size]
    )

    var_groups = OrderedDict()
    for group_idx, indices in enumerate(group_indices):
        for index in indices:
            var_groups.setdefault(group_idx, []).append(parameters[index])
    return var_groups


def bw_hook_func(buffer, param):
    @paddle.autograd.no_grad()
    def fused_allreduce(*_):
        buffer.add_grad(param)

    return fused_allreduce


def register_allreduce_overlap_hook(
    parameter_list, optimizer, comm_group, acc_steps
):
    parameter_list = [p for p in parameter_list if not p.stop_gradient]
    if len(parameter_list) < 1:
        return

    var_groups = assign_group_by_size(parameter_list)
    for group_idx, parameters in var_groups.items():
        buffer = FusedAllReduceBuffer(parameters, comm_group, acc_steps)
        optimizer._comm_buffers.append(buffer)
        for param in parameters:
            param._register_backward_hook(bw_hook_func(buffer, param))
