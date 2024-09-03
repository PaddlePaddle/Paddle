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
    broadcast,
    get_axes_for_reduce_op,
)


@converter_registry.register("pd_op.add", trt_version="8.x")
@converter_registry.register("pd_op.add_", trt_version="8.x")
def add_converter(network, paddle_op, inputs):
    weight_shape = paddle_op.operands()[1].source().shape
    input_shape = paddle_op.operands()[0].source().shape

    weight_tensor = inputs[1]
    input_tensor = inputs[0]
    if type(inputs[1]) == trt.Weights:
        weight_tensor = network.add_constant(
            weight_shape, inputs[1]
        ).get_output(0)
    if type(inputs[0]) == trt.Weights:
        input_tensor = network.add_constant(input_shape, inputs[0]).get_output(
            0
        )
    lhs_val, rhs_val = broadcast(
        network,
        input_tensor,
        weight_tensor,
        input_tensor.name,
        weight_tensor.name,
    )

    out = network.add_elementwise(
        lhs_val, rhs_val, trt.ElementWiseOperation.SUM
    )
    return out.get_output(0)


@converter_registry.register("pd_op.scale", trt_version="8.x")
def scale_converter(network, paddle_op, inputs):
    scale = paddle_op.operands()[1].source().get_defining_op().attrs()["value"]
    bias = paddle_op.attrs().get("bias", 0.0)
    power = paddle_op.attrs().get("power", 1.0)

    # Convert scale, bias, and power to TensorRT weights
    scale_weight = trt.Weights(np.array([scale], dtype=np.float32))
    bias_weight = trt.Weights(np.array([bias], dtype=np.float32))
    power_weight = trt.Weights(np.array([power], dtype=np.float32))

    scale_layer = network.add_scale(
        inputs[0],
        mode=trt.ScaleMode.UNIFORM,
        shift=bias_weight,
        scale=scale_weight,
        power=power_weight,
    )
    return scale_layer.get_output(0)


@converter_registry.register("pd_op.max", trt_version="8.x")
def max_converter(network, paddle_op, inputs):
    input_tensor = inputs[0]
    axis = paddle_op.operands()[1].source().get_defining_op().attrs()["value"]
    input_shape = paddle_op.operands()[0].source().shape
    keepdim = paddle_op.attrs()["keepdim"]
    if network.has_implicit_batch_dimension:
        assert (
            axis != 0
        ), "can't reduce on axis == 0 when network has implicit batch dimension"
    output_shape = []
    if len(axis) == 0:
        axis = list(range(len(input_shape)))
    for i in range(len(axis)):
        if axis[i] < 0:
            axis[i] = len(input_shape) + axis[i]
    layer = network.add_reduce(
        input_tensor,
        trt.ReduceOperation.MAX,
        axes=get_axes_for_reduce_op(axis),
        keep_dims=keepdim,
    )
    return layer.get_output(0)
