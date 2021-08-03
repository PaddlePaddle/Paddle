# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.fluid import core
from paddle.autograd import PyLayer
from paddle.fluid import framework
import contextlib
from paddle.distributed.fleet.utils.recompute import check_recompute_necessary, detach_variable, swith_rng_state
import paddle.distributed as dist
import numpy as np
import logging
logger = logging.getLogger(__name__)
formatter = logging.Formatter(
    fmt='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

__all__ = []

FLOAT_TYPE_DICT = {
    paddle.float16: "float16",
    paddle.float32: "float32",
    paddle.float64: "float64",
}

PADDLE_TO_NUMBER = {
    paddle.float16: 0,
    paddle.float32: 1,
    paddle.float64: 2,
    paddle.int32: 3,
    paddle.int64: 4
}

NUMBER_TO_DTYPE = {
    0: "float16",
    1: "float32",
    2: "float64",
    3: "int32",
    4: "int64"
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
        raise ValueError("unknown data type: {}".format(tensor.dtype))
    return tensor.numel() * elem_size


_hcg = None
_recompute_offload = False
_recompute_partition = False


def _initialize_recompute_setting(is_offload, is_partition):
    global _recompute_offload, _recompute_partition

    _recompute_offload = is_offload
    _recompute_partition = is_partition


def _initialize_recompute_hcg(hcg):
    global _hcg, _mp_degree
    _hcg = hcg


def _get_partition_start_end(tensor):
    global _hcg
    mp_degree = _hcg.get_model_parallel_world_size()
    mp_rank = _hcg.get_model_parallel_rank()

    tensor_numel = np.prod(tensor.shape)
    assert tensor_numel != 0, "can't recompute zero element"
    assert tensor_numel % mp_degree == 0
    part_size = tensor_numel // mp_degree
    start = mp_rank * part_size
    end = start + part_size
    return start, end


def split_tensor_into_1d_equal_chunks(tensor):
    global _hcg

    mp_degree = _hcg.get_model_parallel_world_size()
    mp_rank = _hcg.get_model_parallel_rank()
    tensor_numel = paddle.numel(tensor)
    assert tensor_numel != 0, "can't recompute zero element"
    assert tensor_numel % mp_degree == 0

    data = tensor.flatten_()
    partition_size = tensor_numel // mp_degree
    start_index = partition_size * mp_rank
    end_index = start_index + partition_size
    return data[start_index:end_index]


def gather_split_1d_tensor(tensor):
    global _hcg
    mp_degree = _hcg.get_model_parallel_world_size()
    mp_rank = _hcg.get_model_parallel_rank()
    mp_group = _hcg.get_model_parallel_group()

    if mp_degree < 2:
        return tensor

    tensor_list = []
    paddle.distributed.all_gather(tensor_list, tensor, group=mp_group)
    gathered = paddle.concat(x=tensor_list, axis=-1)
    return gathered


class _HPRecomputeFunction(PyLayer):
    @staticmethod
    def forward(ctx, run_function, all_outputs, *args):
        check_recompute_necessary(args)

        # store for recomputing 
        ctx.run_function = run_function

        # save input for backward
        ctx.inputs = []
        ctx.tensor_indices = []
        ctx.tensor_shapes = []
        tensor_inputs = []

        cur_device = paddle.get_device()
        if 'gpu:' not in cur_device:
            raise RuntimeError(
                "Recompute with RNG perserve is not support current device: {}.".
                format(cur_device))
        ctx.fw_cuda_rng_state = paddle.get_cuda_rng_state()

        # TODO support AMP
        tracer = framework._dygraph_tracer()
        ctx.is_fw_autocast = tracer._enable_autocast
        ctx.amp_white_list, ctx.amp_black_list = tracer._get_amp_op_list()

        with paddle.no_grad():
            outputs = run_function(*args)

        for i, arg in enumerate(args):
            if paddle.is_tensor(arg):
                if _recompute_partition:
                    ctx.tensor_shapes.append(arg.shape)
                    state = arg.stop_gradient
                    partition = split_tensor_into_1d_equal_chunks(arg.detach(
                    )).clone()
                    arg = partition.cpu() if _recompute_offload else partition
                    arg.stop_gradient = state
                else:
                    arg = arg.cpu() if _recompute_offload else arg
                tensor_inputs.append(arg)
                ctx.tensor_indices.append(i)
                ctx.inputs.append(None)
            else:
                ctx.inputs.append(arg)

        ctx.save_for_backward(*tensor_inputs)

        if paddle.is_tensor(outputs):
            all_outputs += [outputs]
            return outputs
        else:
            all_outputs += outputs
            return tuple(outputs)

    @staticmethod
    def backward(ctx, *args):
        with paddle.fluid.dygraph.guard():
            # Restore inputs
            inputs = list(ctx.inputs)
            tensor_indices = ctx.tensor_indices
            tensor_shapes = ctx.tensor_shapes
            tensors = list(ctx.saved_tensor())

            device_id = dist.ParallelEnv().device_id
            for i, idx in enumerate(tensor_indices):
                if _recompute_partition:
                    state = tensors[i].stop_gradient
                    tensors[i] = gather_split_1d_tensor(tensors[i]).reshape_(
                        tensor_shapes[i])
                    tensors[i].stop_gradient = state
                inputs[idx] = tensors[i].cuda(
                    device_id) if _recompute_offload else tensors[i]

            tracer = framework._dygraph_tracer()
            tracer._has_grad = True

            # NOTE support AMP
            # need restore auto_cast state as well as w/b list
            with swith_rng_state(ctx.fw_cuda_rng_state):
                with paddle.amp.auto_cast(
                        enable=ctx.is_fw_autocast,
                        custom_white_list=ctx.amp_white_list,
                        custom_black_list=ctx.amp_black_list):
                    detached_inputs = detach_variable(tuple(inputs))
                    outputs = ctx.run_function(*detached_inputs)

            if isinstance(outputs, core.VarBase):
                outputs = (outputs, )
            assert len(outputs) == len(args)

            forward_outputs_with_grad = []
            backward_inputs = list(args)
            for i in range(len(outputs)):
                if isinstance(outputs[i],
                              core.VarBase) and not outputs[i].stop_gradient:
                    forward_outputs_with_grad.append(outputs[i])
            if len(forward_outputs_with_grad) == 0:
                raise RuntimeError(
                    "none of output has stop_gradient=False, this recompute() is not necessary"
                )

            assert len(backward_inputs) == len(
                forward_outputs_with_grad
            ), "number of forward outputs is [{}], but the backward got [{}] inputs".format(
                len(forward_outputs_with_grad), len(backward_inputs))

            # actually backward            
            paddle.autograd.backward(forward_outputs_with_grad, backward_inputs)

            grads = list(inp._grad_ivar() for inp in detached_inputs
                         if isinstance(inp, core.VarBase))
            return grads


def _hp_recompute(function, *args):
    all_outputs = []
    _HPRecomputeFunction.apply(function, all_outputs, *args)
    if len(all_outputs) == 1:
        return all_outputs[0]
    else:
        return tuple(all_outputs)
