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


import tensorrt as trt

from paddle.tensorrt.converter_utils import has_dynamic_shape
from paddle.tensorrt.register import converter_registry


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

    return conv_layer.get_output(0)


@converter_registry.register("pd_op.conv2d_transpose", trt_version="8.x")
def conv2d_transpose_converter(network, paddle_op, inputs):
    """
    问题：现在convWD_TRANSPOSE 返回的答案是0 需要排查
    """
    input_tensor, weight, output_size = inputs
    input_tensor_shape = paddle_op.operands()[0].source().shape
    weight_shape = paddle_op.operands()[1].source().shape
    if has_dynamic_shape(input_tensor.shape):
        assert (
            input_tensor.shape[1] != -1
        ), "Channel dim can't be dynamic for transpose convolution."

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
    kernel_shape = trt.Dims((weight_shape[2], weight_shape[3]))
    if network.has_explicit_precision:
        dummy_weight = trt.Weights()
        layer = network.add_deconvolution_nd(
            input=input_tensor,
            num_output_maps=weight_shape[1] * groups,
            kernel_shape=weight_shape[2:],
            kernel=dummy_weight,
            # bias=bias,
        )
        layer.set_input(1, weight)
    else:
        layer = network.add_deconvolution_nd(
            input=input_tensor,
            num_output_maps=weight_shape[1] * groups,
            kernel_shape=kernel_shape,
            kernel=weight,
            # bias=bias,
        )
    layer.stride_nd = stride
    layer.padding_nd = padding
    layer.dilation_nd = dilation
    if groups is not None:
        layer.num_groups = groups
    return conv_layer.get_output(0)
