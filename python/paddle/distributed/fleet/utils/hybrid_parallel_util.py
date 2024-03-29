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

import paddle
from paddle import framework

# (TODO: GhostScreaming) It will be removed later.
from paddle.base import core
from paddle.distributed.parallel import (
    _split_tensors,
    build_groups,
    in_dynamic_mode,
    sync_params_buffers,
)

from .log_util import logger

__all__ = []


def obtain_optimizer_parameters_list(optimizer):
    if getattr(optimizer, '_param_groups', None) and isinstance(
        optimizer._param_groups[0], dict
    ):
        parameters_list = []
        for group in optimizer._param_groups:
            for param in group['params']:
                parameters_list.append(param)
    else:
        parameters_list = list(optimizer._parameter_list)

    return parameters_list


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
            paddle.base.framework._dygraph_tracer().trace_op(
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
        if in_dynamic_mode():
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
    assert (
        dev
        in [
            "xpu",
            "gpu",
        ]
        or dev in paddle.device.get_all_custom_device_type()
    ), f"Only support xpu, gpu and custom_device now, but this is {dev}"
    dev_idx = int(cur_device.split(':')[1])
    if dev == "gpu":
        place = paddle.CUDAPlace(dev_idx)
    elif dev in paddle.device.get_all_custom_device_type():
        place = paddle.CustomPlace(dev, dev_idx)
        dev = 'custom'
    else:
        place = eval(f"paddle.{dev.upper()}Place")(dev_idx)

    for v in inputs:
        if isinstance(v, core.eager.Tensor):
            with framework.no_grad():
                if in_dynamic_mode() and not eval(f"v.place.is_{dev}_place")():
                    v_gpu = v._copy_to(place, True)
                    v._clear_data()
                    v_gpu._share_buffer_to(v)
                _broadcast_data_help(v, v.shape, v.dtype, hcg)
        else:
            _broadcast_object_list_help(v, hcg)

    for k, v in kwargs.items():
        if isinstance(v, core.eager.Tensor):
            with framework.no_grad():
                if in_dynamic_mode() and not eval(f"v.place.is_{dev}_place")():
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
        if in_dynamic_mode()
        else _apply_collective_grads
    )
    with framework.no_grad():
        apply_func(parameter_list, group, bucket_size, scale)


def fused_allreduce_gradients(parameter_list, hcg):
    group = None
    scale = None
    if hcg is not None:
        dp_enabled = hcg.get_data_parallel_world_size() > 1
        sep_enabled = hcg.get_sep_parallel_world_size() > 1
        assert (
            dp_enabled or sep_enabled
        ), f"dp_enabled {dp_enabled}; sep_enabled {sep_enabled}"
        group = None
        # sep all reduce is not scaled
        scale = 1.0
        if dp_enabled:
            group = hcg.get_data_parallel_group()
            scale = scale / group.nranks
        if sep_enabled:
            sep_group = hcg.get_sep_parallel_group()
            dp_sep_group = hcg.get_dp_sep_parallel_group()
            group = sep_group if group is None else dp_sep_group

    logger.debug("dp or sep start fuse allreduce gradients")
    fused_allreduce_gradients_with_group(parameter_list, group, scale=scale)


def broadcast_sharding_parameters(model, hcg):
    # TODO TO save memory, use un-fused broadcast to avoid potential OOM
    logger.debug("sharding start init parameters sync")
    sharding_parallel_group = hcg.get_sharding_parallel_group()
    src_rank = hcg.get_sharding_parallel_group_src_rank()
    sync_params_buffers(
        model, sharding_parallel_group, src_rank, is_model_parallel=False
    )


def broadcast_sep_parameters(model, hcg):
    # TODO TO save memory, use un-fused broadcast to avoid potential OOM
    logger.debug("sep start init parameters sync")
    sep_group = hcg.get_sep_parallel_group()
    src_rank = hcg.get_sep_parallel_group_src_rank()
    sync_params_buffers(model, sep_group, src_rank, is_model_parallel=False)


def unwrap_optimizer(optimizer, optimizer_instances=()):
    _inner_opt = optimizer
    while isinstance(_inner_opt, optimizer_instances):
        _inner_opt = _inner_opt._inner_opt
    return _inner_opt
