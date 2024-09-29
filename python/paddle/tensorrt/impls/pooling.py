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

    return pool_layer.get_output(0)
