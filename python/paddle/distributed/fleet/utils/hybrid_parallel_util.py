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
import six
import numpy as np
import warnings

from paddle import framework
import paddle
from paddle.fluid import core
from paddle.fluid.dygraph.parallel import _split_tensors, _coalesce_tensors
from collections import OrderedDict


def _apply_collective_grads(parameters, comm_group, bucket_size):
    grad_var_set = set()
    grad_vars = []
    sparse_grad_vars = []

    for param in parameters:
        if param.trainable and (param._grad_ivar() is not None):
            g_var = param._grad_ivar()
            assert not g_var._is_sparse(
            ), "Now, it doesn't support sparse parameters"
            grad_vars.append(g_var)
            assert g_var not in grad_var_set
            grad_var_set.add(g_var)

    mega_bytes = bucket_size
    group_idx = 0
    memory_counter = 0
    grad_var_groups = OrderedDict()
    dtype = grad_vars[0].dtype
    for g_var in grad_vars:
        # NOTE: the dtype of the same group should be the same.
        bytes = np.prod(g_var.shape) * core.size_of_dtype(g_var.dtype)
        if memory_counter < mega_bytes and dtype == g_var.dtype:
            memory_counter += bytes
        else:
            memory_counter = 0
            group_idx += 1
        grad_var_groups.setdefault(group_idx, []).append(g_var)

    coalesced_grads_and_vars = _coalesce_tensors(grad_var_groups)

    for coalesced_grad, _, _ in coalesced_grads_and_vars:
        paddle.distributed.all_reduce(coalesced_grad, group=comm_group)

    _split_tensors(coalesced_grads_and_vars)


def _sync_params_buffers(model, comm_group, src_rank, is_model_parallel):
    model_vars = []
    for _, param in model.state_dict().items():
        if not isinstance(param, core.VarBase):
            raise TypeError("The data type of '%s' must be Varbase" %
                            param.name)
        # is_distributed param not need to sync 
        if is_model_parallel and param.is_distributed:
            continue
        model_vars.append(param.detach())

    if len(model_vars) == 0:
        return

    mega_bytes = 128 * 1024 * 1024
    group_idx = 0
    memory_counter = 0
    var_groups = OrderedDict()
    dtype = model_vars[0].dtype

    for var in model_vars:
        bytes = np.prod(var.shape) * core.size_of_dtype(var.dtype)
        if memory_counter < mega_bytes and dtype == var.dtype:
            memory_counter += bytes
        else:
            memory_counter = 0
            dtype = var.dtype
            group_idx += 1
        var_groups.setdefault(group_idx, []).append(var)

    coalesced_vars = _coalesce_tensors(var_groups)

    for coalesced_var, _, _ in coalesced_vars:
        with framework.no_grad():
            paddle.distributed.broadcast(
                coalesced_var,
                src=src_rank,
                group=comm_group,
                use_calc_stream=True)

    for coalesced_var, origin_vars, var_shapes in coalesced_vars:
        var_len = [np.prod(v_shape) for v_shape in var_shapes]
        paddle.fluid.framework._dygraph_tracer().trace_op(
            type='split',
            inputs={'X': coalesced_var},
            outputs={'Out': origin_vars},
            attrs={'sections': var_len,
                   'axis': 0})


def broadcast_input_data(hcg, *inputs, **kwargs):
    model_parallel_group = hcg.get_model_parallel_group()
    src_rank = hcg.get_model_parallel_group_src_rank()

    for input_ in inputs:
        if isinstance(input_, core.VarBase):
            with framework.no_grad():
                paddle.distributed.broadcast(
                    input_,
                    src=src_rank,
                    group=model_parallel_group,
                    use_calc_stream=True)
        else:
            print("it doesn't support data type {}".format(type(input_)))

    for k, v in kwargs.items():
        if isinstance(v, core.VarBase):
            with framework.no_grad():
                paddle.distributed.broadcast(
                    v,
                    src=src_rank,
                    group=model_parallel_group,
                    use_calc_stream=True)
            kwargs[k] = v
        else:
            print("it doesn't support data type {}".format(type(v)))
    return inputs, kwargs


def broadcast_mp_parameters(model, hcg):
    model_parallel_group = hcg.get_model_parallel_group()
    src_rank = hcg.get_model_parallel_group_src_rank()
    _sync_params_buffers(
        model, model_parallel_group, src_rank, is_model_parallel=True)


def broadcast_dp_parameters(model, hcg):
    data_parallel_group = hcg.get_data_parallel_group()
    src_rank = hcg.get_data_parallel_group_src_rank()
    _sync_params_buffers(
        model, data_parallel_group, src_rank, is_model_parallel=False)


def fused_allreduce_gradients(parameter_list, hcg, bucket_size=50000000):
    data_parallel_group = hcg.get_data_parallel_group()
    with framework.no_grad():
        _apply_collective_grads(parameter_list, data_parallel_group,
                                bucket_size)
