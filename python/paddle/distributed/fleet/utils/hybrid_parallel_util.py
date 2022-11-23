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

from paddle import framework
import paddle
from paddle.fluid import core
from paddle.fluid.dygraph.parallel import (
    _split_tensors,
    sync_params_buffers,
    build_groups,
)
from paddle.fluid.framework import in_dygraph_mode, _in_legacy_dygraph
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

    nranks = paddle.distributed.get_world_size(
    ) if comm_group is None else comm_group.nranks
    for coalesced_grad, _, _ in coalesced_grads_and_vars:
        # need to div nranks
<<<<<<< HEAD
        div_factor = paddle.to_tensor(nranks, dtype=coalesced_grad.dtype)
        paddle.fluid.framework._dygraph_tracer().trace_op(
            type="elementwise_div",
            inputs={
                'X': coalesced_grad,
                'Y': div_factor
            },
            outputs={'Out': coalesced_grad},
            attrs={'axis': -1})
=======
        if scale is not None:
            div_factor = paddle.to_tensor(scale, dtype=coalesced_grad.dtype)
            paddle.fluid.framework._dygraph_tracer().trace_op(
                type="elementwise_div",
                inputs={'X': coalesced_grad, 'Y': div_factor},
                outputs={'Out': coalesced_grad},
                attrs={'axis': -1},
            )
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
        paddle.distributed.all_reduce(coalesced_grad, group=comm_group)

    _split_tensors(coalesced_grads_and_vars)


def _apply_collective_grads_eager(
    parameters, comm_group, bucket_size, scale=None
):
    grad_var_set = set()
    grad_vars = []

    for param in parameters:
        if param.trainable and (param._grad_ivar() is not None):
            g_var = param._grad_ivar()
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

<<<<<<< HEAD
    nranks = paddle.distributed.get_world_size(
    ) if comm_group is None else comm_group.nranks
    for coalesced_grad, _, _ in coalesced_grads_and_vars:
        # need to div nranks
        coalesced_grad.scale_(1.0 / nranks)
=======
    for coalesced_grad, _, _ in coalesced_grads_and_vars:
        # need to div nranks
        if scale is not None:
            coalesced_grad.scale_(scale)
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
        paddle.distributed.all_reduce(coalesced_grad, group=comm_group)

    _split_tensors(coalesced_grads_and_vars)


def _broadcast_data_help(data, shape, dtype, hcg):
    model_parallel_group = hcg.get_model_parallel_group()
    src_rank = hcg.get_model_parallel_group_src_rank()
    mp_rank = hcg.get_model_parallel_rank()

    shape_gpu = paddle.to_tensor(shape, dtype="int32")
<<<<<<< HEAD
    paddle.distributed.broadcast(shape_gpu,
                                 src=src_rank,
                                 group=model_parallel_group,
                                 use_calc_stream=True)
=======
    paddle.distributed.broadcast(
        shape_gpu, src=src_rank, group=model_parallel_group, sync_op=True
    )
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91

    if mp_rank != 0:
        input_data = paddle.zeros(shape_gpu, dtype=dtype)
    else:
        input_data = data

<<<<<<< HEAD
    paddle.distributed.broadcast(input_data,
                                 src=src_rank,
                                 group=model_parallel_group,
                                 use_calc_stream=True)
=======
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
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91


def broadcast_input_data(hcg, *inputs, **kwargs):
    cur_device = paddle.get_device()
    for v in inputs:
        if isinstance(v, (core.VarBase, core.eager.Tensor)):
            with framework.no_grad():
                if (
                    "gpu" in cur_device
                    and in_dygraph_mode()
                    and not v.place.is_gpu_place()
                ):
                    v_gpu = v.cuda(int(cur_device.split(":")[1]))
                    v._clear_data()
                    v_gpu._share_buffer_to(v)
                _broadcast_data_help(v, v.shape, v.dtype, hcg)
        else:
            logger.error("it doesn't support data type {}".format(type(v)))

    for k, v in kwargs.items():
        if isinstance(v, (core.VarBase, core.eager.Tensor)):
            with framework.no_grad():
                if (
                    "gpu" in cur_device
                    and in_dygraph_mode()
                    and not v.place.is_gpu_place()
                ):
                    v_gpu = v.cuda(int(cur_device.split(":")[1]))
                    v._clear_data()
                    v_gpu._share_buffer_to(v)
                _broadcast_data_help(v, v.shape, v.dtype, hcg)
            kwargs[k] = v
        else:
            logger.error("it doesn't support data type {}".format(type(v)))
    return inputs, kwargs


def broadcast_mp_parameters(model, hcg):
    model_parallel_group = hcg.get_model_parallel_group()
    src_rank = hcg.get_model_parallel_group_src_rank()
<<<<<<< HEAD
    sync_params_buffers(model,
                        model_parallel_group,
                        src_rank,
                        is_model_parallel=True)
=======
    sync_params_buffers(
        model, model_parallel_group, src_rank, is_model_parallel=True
    )
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91


def broadcast_dp_parameters(model, hcg):
    data_parallel_group = hcg.get_data_parallel_group()
    src_rank = hcg.get_data_parallel_group_src_rank()
<<<<<<< HEAD
    sync_params_buffers(model,
                        data_parallel_group,
                        src_rank,
                        is_model_parallel=False)
=======
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
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91


def fused_allreduce_gradients(parameter_list, hcg):
    data_parallel_group = None if hcg is None else hcg.get_data_parallel_group()
    logger.debug("dp start fuse allreduce gradients")
<<<<<<< HEAD
    apply_func = _apply_collective_grads_eager if in_dygraph_mode(
    ) else _apply_collective_grads
    with framework.no_grad():
        apply_func(parameter_list, data_parallel_group)
=======
    fused_allreduce_gradients_with_group(parameter_list, data_parallel_group)
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91


def sharding_reduce_gradients(parameter_list, hcg):
    # TODO allreduce --> reduce
    # TODO merge grad / nrank with dp
    logger.debug("sharding start gradients sync")
    with framework.no_grad():

        sharding_nrank = hcg.get_sharding_parallel_group().nranks
        for param in parameter_list:
            if param.trainable and (param._grad_ivar() is not None):
                if in_dygraph_mode():
                    param.grad.scale_(1.0 / sharding_nrank)
                    paddle.distributed.all_reduce(
                        param.grad,
                        group=hcg.get_sharding_parallel_group(),
<<<<<<< HEAD
                        use_calc_stream=True)
=======
                        sync_op=True,
                    )
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91

                elif _in_legacy_dygraph():
                    g_var = param._grad_ivar()
                    # need use trace_op to allreduce
                    # paddle.distributed.all_reduce(
                    #     g_var, group=hcg.get_sharding_parallel_group(), use_calc_stream=True)
                    paddle.fluid.framework._dygraph_tracer().trace_op(
                        type="c_allreduce_sum",
                        inputs={'X': g_var},
                        outputs={'Out': g_var},
                        attrs={
                            'ring_id': hcg.get_sharding_parallel_group().id,
<<<<<<< HEAD
                            'use_calc_stream': True
                        })

                    # grad / sharding_rank
                    div_factor = paddle.to_tensor(sharding_nrank,
                                                  dtype=g_var.dtype)
                    paddle.fluid.framework._dygraph_tracer().trace_op(
                        type="elementwise_div",
                        inputs={
                            'X': g_var,
                            'Y': div_factor
                        },
                        outputs={'Out': g_var},
                        attrs={'axis': -1})
=======
                            'use_calc_stream': True,
                        },
                    )

                    # grad / sharding_rank
                    div_factor = paddle.to_tensor(
                        sharding_nrank, dtype=g_var.dtype
                    )
                    paddle.fluid.framework._dygraph_tracer().trace_op(
                        type="elementwise_div",
                        inputs={'X': g_var, 'Y': div_factor},
                        outputs={'Out': g_var},
                        attrs={'axis': -1},
                    )
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91


def broadcast_sharding_parameters(model, hcg):
    # TODO TO save memory, use un-fused broadcast to avoid potentional OOM
    logger.debug("sharding start init parameters sync")
    sharding_parallel_group = hcg.get_sharding_parallel_group()
    src_rank = hcg.get_sharding_parallel_group_src_rank()
<<<<<<< HEAD
    sync_params_buffers(model,
                        sharding_parallel_group,
                        src_rank,
                        is_model_parallel=False)
=======
    sync_params_buffers(
        model, sharding_parallel_group, src_rank, is_model_parallel=False
    )
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
