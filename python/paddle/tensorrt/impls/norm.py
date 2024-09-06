# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved
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
import logging
import os
import sys

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import tensorrt as trt

from paddle.base.log_helper import get_logger
from paddle.tensorrt.register import converter_registry

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s'
)
from paddle.tensorrt.converter_utils import (
    append_ones,
    get_axes_for_reduce_op,
    get_dynamic_dims,
    has_dynamic_shape,
)


@converter_registry.register(
    "pd_op.layer_norm", trt_version="trt_version_ge=8.6"
)
def layernorm_converter(network, paddle_op, inputs):
    input_a, scale, bias = inputs

    begin_norm_axis = paddle_op.attrs().get("begin_norm_axis", 0)
    epsilon = paddle_op.attrs().get("epsilon", 1e-5)
    assert len(paddle_op.operands()) == 3
    scale_shape = paddle_op.operands()[1].source().shape

    scale_tensor = network.add_constant(scale_shape, scale).get_output(0)
    bias_shape = paddle_op.operands()[2].source().shape
    bias_tensor = network.add_constant(bias_shape, bias).get_output(0)

    # dims = list(range( len(input_a.shape) - len(normalized_shape), len(input_a.shape)))
    dims = list(range(len(input_a.shape)))[begin_norm_axis:]
    axes = get_axes_for_reduce_op(dims)

    scale_tensor = append_ones(
        network,
        scale_tensor,
        f"{scale_tensor.name}_broadcast",
        len(input_a.shape) - len(scale_tensor.shape),
    )

    bias_tensor = append_ones(
        network,
        bias_tensor,
        f"{bias_tensor.name}_broadcast",
        len(input_a.shape) - len(bias_tensor.shape),
    )

    layer_norm = network.add_normalization(
        input_a, scale_tensor, bias_tensor, axes
    )
    layer_norm.epsilon = epsilon
    layer_norm.compute_precision = trt.float32

    return layer_norm.get_output(0)


@converter_registry.register("pd_op.batch_norm", trt_version="8.x")
@converter_registry.register("pd_op.batch_norm_", trt_version="8.x")
def batch_norm_converter(network, paddle_op, inputs):
    input_tensor, mean, variance, scale, bias = inputs

    scale_shape = paddle_op.operands()[3].source().shape

    power = np.ones(scale_shape, dtype='float32')
    power = trt.Weights(power)
    input_tensor_shape = paddle_op.operands()[0].source().shape
    if has_dynamic_shape(input_tensor_shape):
        assert (
            input_tensor.shape[1] != -1
        ), "Channel dim can't be dynamic for batch norm."
    # For BatchNorm1d ,reshape 1d to 2d
    output_shape = input_tensor_shape

    if not network.has_implicit_batch_dimension and len(input_tensor_shape) < 4:
        assert (
            len(get_dynamic_dims(input_tensor.shape)) <= 1
        ), "BatchNorm1D with more than one dynamic dims is not currently supported."
        reshape_layer = network.add_shuffle(input_tensor)
        if len(input_tensor_shape) == 2:
            reshape_layer.reshape_dims = (
                input_tensor_shape[0],
                input_tensor_shape[1],
                1,
                1,
            )
        else:  # len(input_tensor_shape) ==3
            reshape_layer.reshape_dims = (
                input_tensor_shape[0],
                input_tensor_shape[1],
                input_tensor_shape[2],
                1,
            )
        input_tensor = reshape_layer.get_output(0)
    # (self: tensorrt.tensorrt.INetworkDefinition, input: tensorrt.tensorrt.ITensor, mode: tensorrt.tensorrt.ScaleMode, shift: tensorrt.tensorrt.Weights = None, scale: tensorrt.tensorrt.Weights = None, power: tensorrt.tensorrt.Weights = None) -> tensorrt.tensorrt.IScaleLayer
    batch_norm_layer = network.add_scale(
        input_tensor, trt.ScaleMode.CHANNEL, bias, scale, power
    )
    # For BatchNorm1d,reshape output back to 1d
    if not network.has_implicit_batch_dimension and len(output_shape) < 4:
        reshape_output_layer = network.add_shuffle(
            batch_norm_layer.get_output(0)
        )
        reshape_output_layer.reshape_dims = tuple(output_shape)
        batch_norm_layer = reshape_output_layer

    return batch_norm_layer.get_output(0)
