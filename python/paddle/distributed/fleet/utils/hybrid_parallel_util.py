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
from paddle.fluid.dygraph.parallel import _split_tensors, sync_params_buffers, build_groups
from collections import OrderedDict
from .log_util import logger


def _apply_collective_grads(parameters, comm_group):
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

    coalesced_grads_and_vars = build_groups(grad_vars, 128 * 1024 * 1024)

    for coalesced_grad, _, _ in coalesced_grads_and_vars:
        # need to div nranks
        coalesced_grad = coalesced_grad / comm_group.nranks
        paddle.distributed.all_reduce(coalesced_grad, group=comm_group)

    _split_tensors(coalesced_grads_and_vars)


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
            logger.error("it doesn't support data type {}".format(type(input_)))

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
            logger.error("it doesn't support data type {}".format(type(v)))
    return inputs, kwargs


def broadcast_mp_parameters(model, hcg):
    model_parallel_group = hcg.get_model_parallel_group()
    src_rank = hcg.get_model_parallel_group_src_rank()
    sync_params_buffers(
        model, model_parallel_group, src_rank, is_model_parallel=True)


def broadcast_dp_parameters(model, hcg):
    data_parallel_group = hcg.get_data_parallel_group()
    src_rank = hcg.get_data_parallel_group_src_rank()
    sync_params_buffers(
        model, data_parallel_group, src_rank, is_model_parallel=False)


def fused_allreduce_gradients(parameter_list, hcg):
    data_parallel_group = hcg.get_data_parallel_group()
    logger.debug("dp start fuse allreduce gradients")
    with framework.no_grad():
        _apply_collective_grads(parameter_list, data_parallel_group)
