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

import os
import numpy as np
import logging
import sys

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


def get_trt_tensor(
    network, input_val, name, dtype=None
) -> trt.tensorrt.ITensor:
    if isinstance(input_val, (torch.Tensor, int, float)):
        return create_constant(network, input_val, name, dtype)
    elif not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(
            f"Received input {input_val} of name {name} that "
            "is not part of the TensorRT region!"
        )
    else:
        return input_val


def has_dynamic_shape(shape):
    return any(s == -1 for s in shape)


def append_ones(network, input, name, num_prepend_ones):
    layer = network.add_shuffle(input)

    if has_dynamic_shape(input.shape):
        input_shape_layer = network.add_shape(input)
        input_shape_layer.name = f"{name}_broadcast_orig_shape"
        prepend_shape_layer = network.add_constant(
            (num_prepend_ones,), np.ones((num_prepend_ones,), dtype=np.int32)
        )
        prepend_shape_layer.name = f"{name}_broadcast_prepend_ones"
        reshape_dim_layer = network.add_concatenation(
            [prepend_shape_layer.get_output(0), input_shape_layer.get_output(0)]
        )
        reshape_dim_layer.axis = 0
        reshape_dim_layer.name = f"{name}_broadcast_final_shape"
        layer.set_input(1, reshape_dim_layer.get_output(0))
    else:
        layer.reshape_dims = (1,) * num_prepend_ones + tuple(input.shape)

    layer.name = name
    return layer.get_output(0)


def broadcast(network, a, b, a_name, b_name, preset_diff=0):
    a_shape = tuple(a.shape)
    b_shape = tuple(b.shape)

    diff = len(a_shape) - len(b_shape) - preset_diff
    if diff > 0:
        b = append_ones(network, b, f"{b_name}_broadcast", diff)
    elif diff < 0:
        a = append_ones(network, a, f"{a_name}_broadcast", -diff)

    return a, b


def get_axes_for_reduce_op(
    dim,
    has_implicit_batch_dimension=False,
):
    if isinstance(dim, int):
        dim = (dim,)

    if has_implicit_batch_dimension:
        assert (
            0 not in dim
        ), "Can't reduce over batch dimension when it's implicit."

    axes = 0
    for d in dim:
        axes |= 1 << (d - (1 if has_implicit_batch_dimension else 0))

    return axes


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
        layer.reshape_dims = tuple(reshape_dims)
    except Exception:
        shuffle_layer.set_input(1, shape_tensor)

    return shuffle_layer


@converter_registry.register("pd_op.transpose", trt_version="8.x")
def transpose_converter(network, paddle_op, inputs):
    perm = paddle_op.attrs()["perm"]
    transposed_tensor = network.add_shuffle(inputs[0])
    transposed_tensor.second_transpose = perm
    return transposed_tensor


@converter_registry.register("pd_op.full", trt_version="8.x")
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

    if pooling_type == "max":
        pooling_type = trt.PoolingType.MAX
    elif pooling_type == "avg":
        pooling_type = trt.PoolingType.AVERAGE
    else:
        raise ValueError(f"Unsupported pooling type: {pooling_type}")

    pool_layer = network.add_pooling_nd(
        input_tensor, pooling_type, window_size=kernel_size
    )
    pool_layer.stride_nd = stride
    pool_layer.padding_nd = padding

    return pool_layer


@converter_registry.register("pd_op.batch_norm", trt_version="8.x")
@converter_registry.register("pd_op.batch_norm_", trt_version="8.x")
def batch_norm_converter(network, paddle_op, inputs):
    input_tensor, mean, variance, scale, bias = inputs

    scale_shape = paddle_op.operands()[3].source().shape
    power = np.ones(scale_shape, dtype='float32')
    power = trt.Weights(power)

    # (self: tensorrt.tensorrt.INetworkDefinition, input: tensorrt.tensorrt.ITensor, mode: tensorrt.tensorrt.ScaleMode, shift: tensorrt.tensorrt.Weights = None, scale: tensorrt.tensorrt.Weights = None, power: tensorrt.tensorrt.Weights = None) -> tensorrt.tensorrt.IScaleLayer
    batch_norm_layer = network.add_scale(
        input_tensor, trt.ScaleMode.CHANNEL, bias, scale, power
    )

    return batch_norm_layer
