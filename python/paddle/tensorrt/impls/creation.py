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

import paddle
from paddle.tensorrt.converter_utils import (
    add_1D_constant_layer,
    cast_tensor,
    trt_cast,
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


@converter_registry.register("pd_op.assign", trt_version="8.x")
@converter_registry.register("pd_op.assign_out_", trt_version="8.x")
def assign_converter(network, paddle_op, inputs):
    input_tensor = inputs[0]
    identity_layer = network.add_identity(input_tensor)
    return identity_layer.get_output(0)


@converter_registry.register("pd_op.assign_value_", trt_version="8.x")
def assign_value_converter(network, paddle_op, inputs):
    attrs = paddle_op.attrs()
    shape = attrs['shape']
    dtype = attrs['dtype']
    values = attrs['values']

    if dtype == paddle.float32:
        dtype = np.float32

    constant_layer = network.add_constant(shape, np.array(values, dtype=dtype))
    return constant_layer.get_output(0)


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


@converter_registry.register("pd_op.full_like", trt_version="8.x")
def full_like_converter(network, paddle_op, inputs):
    shape = tuple(paddle_op.operands()[0].source().shape)
    ndims = len(shape)
    value = paddle_op.attrs().get("value", 1.0)
    out_dtype = int(paddle_op.attrs().get("dtype", None))
    # Reference paddle/phi/common/data_type.h enum DataType
    if out_dtype == 1:
        out_dtype = trt.bool
    elif out_dtype == 7:
        out_dtype = trt.int32
    elif out_dtype == 9:
        out_dtype = trt.int32
    elif out_dtype == 10:
        out_dtype = trt.float32
    elif out_dtype == 11:
        out_dtype = trt.float32
    elif out_dtype == 15:
        out_dtype = trt.float16
    else:
        raise RuntimeError(
            f"cast converter currently doesn't support dtype: {out_dtype}"
        )

    value_tensor = network.add_constant(
        (1,), np.array([value], dtype=np.float32)
    ).get_output(0)
    value_tensor = trt_cast(network, value_tensor, out_dtype)

    shuffle_layer = network.add_shuffle(value_tensor)
    shuffle_layer.reshape_dims = (1,) * ndims

    start_vec = np.zeros((ndims,), dtype=np.int32)
    start_tensor = network.add_constant((ndims,), start_vec).get_output(0)
    shape_tensor = network.add_shape(inputs[0]).get_output(0)
    stride_tensor = network.add_constant(
        (ndims,), np.ones((ndims,), dtype=np.int32)
    ).get_output(0)

    slice_layer = network.add_slice(
        shuffle_layer.get_output(0),
        start_vec,
        [1] * ndims,
        np.ones((ndims,), dtype=np.int32),
    )
    slice_layer.mode = trt.SliceMode.FILL
    slice_layer.set_input(1, start_tensor)
    slice_layer.set_input(2, shape_tensor)
    slice_layer.set_input(3, stride_tensor)
    fill_constant = network.add_input("value", dtype=out_dtype, shape=())
    slice_layer.set_input(4, fill_constant)
    return slice_layer.get_output(0)
