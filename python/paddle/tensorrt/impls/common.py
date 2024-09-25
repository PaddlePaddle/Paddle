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

from paddle.tensorrt.converter_utils import get_shape_tensor_element
from paddle.tensorrt.register import converter_registry


@converter_registry.register("pd_op.bilinear_interp", trt_version="8.x")
def bilinear_interp_converter(network, paddle_op, inputs):

    input_tensor = inputs[0]
    input_shape = paddle_op.operands()[0].source().shape
    data_format = paddle_op.attrs().get("data_format")
    interp_method = paddle_op.attrs().get("interp_method")
    align_corners = paddle_op.attrs().get("align_corners")
    align_mode = paddle_op.attrs().get("align_mode")
    out_h = paddle_op.attrs().get("out_h")
    out_w = paddle_op.attrs().get("out_w")
    out_d = paddle_op.attrs().get("out_d")
    scale_attr = paddle_op.attrs().get("scale")

    trt_major, trt_minor, trt_patch = trt.__version__.split(".")
    trt_version_float = float(f"{trt_major}.{trt_minor}")

    # 创建 Resize 层
    resize_layer = network.add_resize(input_tensor)
    if align_mode == 0:
        if trt_version_float >= 8.6:
            resize_layer.resize_mode = trt.InterpolationMode.LINEAR
        else:
            resize_layer.resize_mode = trt.ResizeMode.LINEAR

    if (align_corners is not None) and align_corners:
        resize_layer.coordinate_transformation = (
            trt.ResizeCoordinateTransformation.ALIGN_CORNERS
        )
    else:
        resize_layer.coordinate_transformation = (
            trt.ResizeCoordinateTransformation.HALF_PIXEL
        )

    scale_h = -1.0
    scale_w = -1.0

    if scale_attr and len(scale_attr) > 1:
        scale_h = scale_attr[0]
        scale_w = scale_attr[1]
    elif len(scale_attr) == 1:
        scale_h = scale_w = scale_attr[0]

    if data_format == "NCHW":
        h_axis = 2
        w_axis = 3
    elif data_format == "NHWC":
        h_axis = 1
        w_axis = 2

    # 获取输入张量的形状（动态形状）
    input_shape_tensor = network.add_shape(input_tensor).get_output(0)

    if scale_w > 0 and scale_h > 0:
        in_dim = input_tensor.shape
        if in_dim[h_axis] > 0 and in_dim[w_axis] > 0:
            out_h = int(in_dim[h_axis] * scale_h)
            out_w = int(in_dim[w_axis] * scale_w)

    else:
        if out_h > 0 and out_w > 0 and not (scale_w > 0 and scale_h > 0):
            in_dim = input_tensor.shape
            if in_dim[h_axis] > 0 and in_dim[w_axis] > 0:
                scale_h = float(out_h) / float(in_dim[h_axis])
                scale_w = float(out_w) / float(in_dim[w_axis])

    scales = [1.0] * len(input_tensor.shape)
    if data_format == "NCHW":
        scales[1] = 1.0
        scales[2] = scale_h
        scales[3] = scale_w
    elif data_format == "NHWC":
        scales[1] = scale_h
        scales[2] = scale_w
        scales[3] = 1.0

    outsize_tensor = None
    output_itensors = []
    if len(inputs) > 1 and inputs[1] is not None:
        outsize_tensor = inputs[1]
        if outsize_tensor is not None:
            input_shape_tensor = network.add_shape(input_tensor).get_output(0)
            batch_dim = get_shape_tensor_element(network, input_shape_tensor, 0)
            output_itensors.append(batch_dim)
            if data_format == "NCHW":
                channel_dim = get_shape_tensor_element(
                    network, input_shape_tensor, 1
                )
                output_itensors.append(channel_dim)
                output_itensors.append(outsize_tensor)
            elif data_format == "NHWC":
                channel_dim = get_shape_tensor_element(
                    network, input_shape_tensor, 3
                )
                output_itensors.append(outsize_tensor)
                output_itensors.append(channel_dim)
            output_size_tensor = network.add_concatenation(
                output_itensors
            ).get_output(0)
            resize_layer.set_input(1, output_size_tensor)
    else:
        resize_layer.scales = scales

    return resize_layer.get_output(0)
