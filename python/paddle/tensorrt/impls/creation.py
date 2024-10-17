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

import numpy as np
import tensorrt as trt

from paddle.tensorrt.converter_utils import (
    add_1D_constant_layer,
    cast_tensor,
    trt_floor_div,
    trt_max,
    trt_reduce_to_scalar,
    trt_reshape,
    trt_sub,
)
from paddle.tensorrt.register import converter_registry


@converter_registry.register("pd_op.full_int_array", trt_version="8.x")
def full_int_array_converter(network, paddle_op, inputs):
    value = paddle_op.attrs()["value"]
    if len(value) == 0:
        return ()
    value_weight = trt.Weights(np.array(value, dtype=np.int32))
    full_int_array_layer = network.add_constant([len(value)], value_weight)
    return full_int_array_layer.get_output(0)


@converter_registry.register("pd_op.full", trt_version="8.x")
def full_converter(network, paddle_op, inputs):
    shape = paddle_op.attrs()["shape"]
    value = paddle_op.attrs().get("value", 1.0)
    full_layer = network.add_constant(
        shape, np.full(shape, value, dtype=np.float32)
    )
    return full_layer.get_output(0)


@converter_registry.register("pd_op.arange", trt_version="8.x")
def arange_converter(network, paddle_op, inputs):
    start, end, step = inputs
    zero_tensor = add_1D_constant_layer(network, 0, np.int32)

    delta = trt_sub(network, end, start)

    f_quotient_tensor = trt_floor_div(network, delta, step)

    if start.dtype == trt.DataType.FLOAT:
        quotient_tensor = cast_tensor(network, f_quotient_tensor, trt.int32)
    else:
        quotient_tensor = f_quotient_tensor

    number_tensor = trt_max(network, quotient_tensor, zero_tensor)

    reshape_start_layer = trt_reshape(network, start, (1,))

    start_tensor = trt_reduce_to_scalar(network, reshape_start_layer)

    fill_layer = network.add_fill(shape=(), op=trt.FillOperation.LINSPACE)
    fill_layer.set_input(0, number_tensor)
    fill_layer.set_input(1, start_tensor)
    fill_layer.set_input(2, step)

    return fill_layer.get_output(0)
