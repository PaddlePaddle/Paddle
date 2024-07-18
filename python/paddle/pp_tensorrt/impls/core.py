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
from register import converter_registry

from paddle.base.log_helper import get_logger

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s'
)
from converter_utils import (
    append_ones,
    broadcast,
    get_axes_for_reduce_op,
    get_dynamic_dims,
    has_dynamic_shape,
)

# def get_trt_tensor(
#     network, input_val, name, dtype=None
# ) -> trt.tensorrt.ITensor:
#     if isinstance(input_val, (torch.Tensor, int, float)):
#         return create_constant(network, input_val, name, dtype)
#     elif not isinstance(input_val, trt.tensorrt.ITensor):
#         raise RuntimeError(
#             f"Received input {input_val} of name {name} that "
#             "is not part of the TensorRT region!"
#         )
#     else:
#         return input_val


@converter_registry.register("pd_op.add", trt_version="8.x")
@converter_registry.register("pd_op.elementwise_add", trt_version="8.x")
def add_converter(network, paddle_op, inputs):
    input_a, input_b = inputs
    input_b_shape = paddle_op.operands()[1].source().shape
    if type(input_b) == trt.Weights:
        input_b = network.add_constant(input_b_shape, input_b).get_output(0)

    # check if input_b should reshape
    if len(input_b_shape) < len(input_a.shape):
        reshape_dims = [1] * (len(input_a.shape) - len(input_b_shape)) + list(
            input_b_shape
        )
        reshape_layer = network.add_shuffle(input_b)
        reshape_layer.reshape_dims = reshape_dims
        input_b = reshape_layer.get_output(0)

    output = network.add_elementwise(
        input_a, input_b, trt.ElementWiseOperation.SUM
    )
    return output


@converter_registry.register("pd_op.relu", trt_version="8.x")
def relu_converter(network, paddle_op, inputs):
    out = network.add_activation(inputs[0], trt.ActivationType.RELU)
    return out


@converter_registry.register("pd_op.matmul", trt_version="8.x")
def matmul_converter(network, paddle_op, inputs):
    weight_shape = paddle_op.operands()[1].source().shape
    transpose_x = paddle_op.attrs()["transpose_x"]
    transpose_y = paddle_op.attrs()["transpose_y"]
    self_matrix_op = (
        trt.MatrixOperation.TRANSPOSE
        if transpose_x
        else trt.MatrixOperation.NONE
    )
    other_matrix_op = (
        trt.MatrixOperation.TRANSPOSE
        if transpose_y
        else trt.MatrixOperation.NONE
    )

    weight_tensor = inputs[1]
    if type(inputs[1]) == trt.Weights:
        weight_tensor = network.add_constant(
            weight_shape, inputs[1]
        ).get_output(0)
    lhs_val, rhs_val = broadcast(
        network, inputs[0], weight_tensor, inputs[0].name, weight_tensor.name
    )
    out = network.add_matrix_multiply(
        lhs_val, self_matrix_op, rhs_val, other_matrix_op
    )
    return out


@converter_registry.register("pd_op.full_int_array", trt_version="8.x")
def full_int_array_converter(network, paddle_op, inputs):
    shape = paddle_op.attrs()["value"]
    shape_weight = trt.Weights(np.array(shape, dtype=np.int32))
    full_int_array_tensor = network.add_constant([len(shape)], shape_weight)
    return full_int_array_tensor


@converter_registry.register("pd_op.reshape", trt_version="8.x")
def reshape_converter(network, paddle_op, inputs):
    input_tensor, shape_tensor = inputs
    shuffle_layer = network.add_shuffle(input_tensor)
    try:
        reshape_dims = (
            paddle_op.operands()[1].source().get_defining_op().attrs()["value"]
        )
        shuffle_layer.reshape_dims = tuple(reshape_dims)
    except Exception:
        shuffle_layer.set_input(1, shape_tensor)

    return shuffle_layer


@converter_registry.register("pd_op.transpose", trt_version="8.x")
def transpose_converter(network, paddle_op, inputs):
    perm = paddle_op.attrs()["perm"]
    transposed_tensor = network.add_shuffle(inputs[0])
    transposed_tensor.second_transpose = perm
    return transposed_tensor


@converter_registry.register("pd_op.full")
def full_converter(network, paddle_op, inputs):
    shape = paddle_op.attrs()["shape"]
    value = paddle_op.attrs().get("value", 1.0)  # 默认值为1.0
    full_tensor = network.add_constant(
        shape, np.full(shape, value, dtype=np.float32)
    )
    return full_tensor


@converter_registry.register("pd_op.scale", trt_version="8.x")
def scale_converter(network, paddle_op, inputs):
    scale = paddle_op.attrs()["scale"]
    bias = paddle_op.attrs().get("bias", 0.0)
    power = paddle_op.attrs().get("power", 1.0)

    scale_layer = network.add_scale(
        inputs[0],
        mode=trt.ScaleMode.UNIFORM,
        shift=bias,
        scale=scale,
        power=power,
    )
    return scale_layer


@converter_registry.register("pd_op.softmax", trt_version="8.x")
def softmax_converter(network, paddle_op, inputs):
    axis = paddle_op.attrs().get("axis", 0)
    if axis < 0:
        axis = len(inputs[0].shape) + axis

    softmax_layer = network.add_softmax(inputs[0])
    softmax_layer.axes = 1 << axis
    return softmax_layer


@converter_registry.register("pd_op.layer_norm", trt_version="8.x")
def layernorm_converter(network, paddle_op, inputs):
    input_a, scale, bias = inputs
    begin_norm_axis = paddle_op.attrs().get("begin_norm_axis", 0)
    epsilon = paddle_op.attrs().get("epsilon", 0.0)
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
    _logger.info(
        f"!!! layernorm, {input_a.shape}, {scale_tensor.shape}, {bias_tensor.shape}"
    )

    layer_norm = network.add_normalization(
        input_a, scale_tensor, bias_tensor, axes
    )
    layer_norm.epsilon = epsilon
    layer_norm.compute_precision = trt.float32

    return layer_norm


@converter_registry.register("pd_op.conv2d", trt_version="8.x")
def conv2d_converter(network, paddle_op, inputs):
    input_tensor, weight = inputs
    weight_shape = paddle_op.operands()[1].source().shape

    padding = paddle_op.attrs().get("paddings", [0, 0])
    stride = paddle_op.attrs().get("strides", [1, 1])
    dilation = paddle_op.attrs().get("dilations", [1, 1])
    groups = paddle_op.attrs().get("groups", 1)

    # weight_tensor = network.add_constant(weight_shape, weight).get_output(0)
    kernel_shape = trt.Dims((weight_shape[2], weight_shape[3]))

    conv_layer = network.add_convolution_nd(
        input_tensor, weight_shape[0], kernel_shape, weight
    )
    conv_layer.stride_nd = stride
    conv_layer.padding_nd = padding
    conv_layer.dilation_nd = dilation
    conv_layer.num_groups = groups

    return conv_layer


@converter_registry.register("pd_op.pool2d", trt_version="8.x")
def pool2d_converter(network, paddle_op, inputs):
    input_tensor = inputs[0]
    pooling_type = paddle_op.attrs().get("pooling_type", "max")
    padding = paddle_op.attrs().get("paddings", [0, 0])
    stride = paddle_op.attrs().get("strides", [1, 1])
    ceil_mode = paddle_op.attrs().get("ceil_mode", False)
    exclusive = paddle_op.attrs().get("exclusive")
    adaptive = paddle_op.attrs().get("adaptive")
    padding_algorithm = paddle_op.attrs().get("padding_algorithm")

    input_shape = input_tensor.shape

    # TODO attention for these codes
    if not paddle_op.attrs().get("kernel_size") and len(inputs) == 2:
        # the size of pool2d inputs is 2, means kernel size is the second input.
        # kernel_size_tensor = inputs[1]
        full_int_op = paddle_op.operands()[1].source().get_defining_op()
        if full_int_op.name() == "pd_op.full_int_array":
            kernel_size = full_int_op.attrs().get("value")
        else:
            raise Exception(
                "the defining op of kernel size must be pd_op.full_int_array"
            )
    else:
        kernel_size = paddle_op.attrs().get("kernel_size")

    if len(stride) == 0 or stride[0] is None:
        stride = kernel_size

    if pooling_type == "max":
        pooling_type = trt.PoolingType.MAX
    elif pooling_type == "avg":
        pooling_type = trt.PoolingType.AVERAGE
    else:
        raise ValueError(f"Unsupported pooling type: {pooling_type}")

    if padding_algorithm == "VALID":
        padding = [0, 0]

    if adaptive:
        output_size = kernel_size
        stride = tuple(input_shape[-2 + i] // output_size[i] for i in range(2))
        kernel_size = tuple(
            input_shape[-2 + i] - (output_size[i] - 1) * stride[i]
            for i in range(2)
        )

        pool_layer = network.add_pooling_nd(
            input_tensor, pooling_type, window_size=kernel_size
        )
        pool_layer.stride_nd = stride
        if pooling_type == "max":
            pool_layer.padding_nd = padding
    else:
        pool_layer = network.add_pooling(
            input_tensor, pooling_type, window_size=kernel_size
        )
        pool_layer.stride = stride
        pool_layer.padding = padding
        if exclusive:
            pool_layer.average_count_excludes_padding = True
        else:
            pool_layer.average_count_excludes_padding = False
        if ceil_mode:
            pool_layer.padding_mode = trt.PaddingMode.EXPLICIT_ROUND_UP

    return pool_layer


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

    return batch_norm_layer


# @converter_registry.register("pd_op.flatten", trt_version="8.x")
# def flatten_converter(network, paddle_op, inputs):
#     input_val = inputs[0]
#     input_val_shape = paddle_op.operands()[0].source().shape
#     num_dims = len(input_val_shape) + (
#         1 if network.has_implicit_batch_dimension else 0
#     )

#     start_axis = paddle_op.attrs().get("start_axis")
#     end_axis = paddle_op.attrs().get("stop_axis")
#     if network.has_implicit_batch_dimension:
#         assert (
#             start_axis != 0
#         ), "Can't flatten batch dimension when it's implicit."
#         start_axis -= 1
#         end_axis -= 1

#     flatten_layer = network.add_shuffle(input_val)
#     # If there're dynamic shapes then we need to use shape layers
#     # to figure out the final shape after flatten. We first slice
#     # the input shape to three parts:
#     #   1. dimensions before start_axis
#     #   2. dimensions between start_axis and end_axis
#     #   3. dimensions after end_axis
#     # Part 1 and 3 might not exist if start_axis is 0 or end_axis is
#     # last dim. Then we do a reduced multiplication over part 2 to
#     # get flattened dim. Finally, we concatenate the three parts to
#     # get the final shape.
#     if has_dynamic_shape(input_val_shape):
#         input_shape_layer = network.add_shape(input_val)
#         final_shapes = []

#         # Shapes before start_axis
#         if start_axis > 0:
#             predix_shape_layer = network.add_slice(
#                 input_shape_layer.get_output(0),
#                 start=(0,),
#                 shape=(start_axis,),
#                 stride=(1,),
#             )
#             final_shapes.append(prefix_shape_layer.get_output(0))

#         flatten_shape_layer = network.add_slice(
#             input_shape_layer.get_output(0),
#             start=(start_axis,),
#             shape=(end_axis - start_axis + 1,),
#             stride=(1,),
#         )
#         flatten_shape_layer = network.add_reduce(
#             flatten_shape_layer.get_output(0),
#             trt_ReduceOperation.PROD,
#             axes=get_axes_for_reduce_op(0, False),
#             keep_dims=True,
#         )
#         final_shapes.append(flatten_shape_layer.get_output(0))

#         # Shapes after start_axis
#         if end_axis < len(input_val_shape) - 1:
#             suffix_shape_layer = network.add_slice(
#                 input_shape_layer.get_output(0),
#                 start=(end_axis + 1,),
#                 shape=(len(input_val_shape) - end_axis - 1,),
#                 stride=(1,),
#             )
#             final_shapes.append(suffix_shape_layer.get_output(0))

#         final_shape_layer = network.add_concatenation(final_shapes)
#         final_shape_layer.axis = 0
#         flatten_layer.set_input(1, final_shape_layer.get_output(0))
#     else:
#         final_shape = []
#         flatten_dim = 1
#         for i, s in enumerate(input_val_shape):
#             if i >= start_axis and i <= end_axis:
#                 flatten_dim *= s
#             elif i == end_axis + 1:
#                 final_shape.append(flatten_dim)
#                 final_shape.append(s)
#             else:
#                 final_shape.append(s)

#         if end_axis == len(input_val_shape) - 1:
#             final_shape.append(flatten_dim)

#         flatten_layer.reshape_dims = tuple(final_shape)

#     return flatten_layer
