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

from paddle.tensorrt.register import converter_registry


@converter_registry.register("pd_op.depthwise_conv2d", trt_version="8.x")
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
