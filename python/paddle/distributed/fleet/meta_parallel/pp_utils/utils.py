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


class _HPRecomputeFunction(PyLayer):
    @staticmethod
    def forward(ctx, run_function, all_outputs, *args):
        check_recompute_necessary(args)

        # store for recomputing 
        ctx.run_function = run_function

        # save input for backward
        ctx.inputs = []
        ctx.tensor_indices = []
        tensor_inputs = []
        for i, arg in enumerate(args):
            if paddle.is_tensor(arg):
                tensor_inputs.append(arg)
                ctx.tensor_indices.append(i)
                ctx.inputs.append(None)
            else:
                ctx.inputs.append(arg)
        ctx.save_for_backward(*tensor_inputs)

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
            tensors = ctx.saved_tensor()
            for i, idx in enumerate(tensor_indices):
                inputs[idx] = tensors[i]

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

            # run backward() with only tensor that requires grad
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
