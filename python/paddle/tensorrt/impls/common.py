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

from math import floor
from typing import Sequence

import tensorrt as trt

from paddle.tensorrt.register import converter_registry


@converter_registry.register("pd_op.nearest_interp", trt_version="8.x")
def nearest_interp_converter(network, paddle_op, inputs):
    input_tensor = inputs[0]
    input_shape = paddle_op.operands()[0].source().shape
    intput_shape_size = len(input_shape)
    scale_factor = paddle_op.attrs().get("scale")
    align_corners = paddle_op.attrs().get("align_corners")
    interp_method = paddle_op.attrs().get("interp_method")
    interp_method = paddle_op.attrs().get("align_mode")
    data_format = paddle_op.attrs().get("data_format")
    out_h = paddle_op.attrs().get("out_h")
    out_w = paddle_op.attrs().get("out_w")
    out_d = paddle_op.attrs().get("out_d")
    if len(scale_factor) == 0:
        if out_d != -1:
            size = [out_d, out_h, out_w]
            scale_factor = [
                out_d / input_shape[2],
                out_h / input_shape[3],
                out_w / input_shape[4],
            ]
        else:
            size = [out_h, out_w]
            scale_factor = [out_h / input_shape[2], out_w / input_shape[3]]
    else:
        if intput_shape_size == 5:
            if align_corners:
                size = [
                    floor(input_shape[2] * scale_factor[0]),
                    floor(input_shape[3] * scale_factor[1]),
                    floor(input_shape[4] * scale_factor[2]),
                ]
            else:
                size = [
                    round(input_shape[2] * scale_factor[0]),
                    round(input_shape[3] * scale_factor[1]),
                    round(input_shape[4] * scale_factor[2]),
                ]

        elif intput_shape_size == 4:
            if align_corners:
                size = [
                    floor(input_shape[2] * scale_factor[0]),
                    floor(input_shape[3] * scale_factor[1]),
                ]
            else:
                size = [
                    round(input_shape[2] * scale_factor[0]),
                    round(input_shape[3] * scale_factor[1]),
                ]
    layer = network.add_resize(input_tensor)
    layer.resize_mode = trt.InterpolationMode.NEAREST
    if network.has_implicit_batch_dimension:
        if size is not None:
            if not isinstance(size, Sequence):
                layer.shape = [input_shape[0]] + [size] * intput_shape_size
            else:
                layer.shape = [input_shape[0], *list(size)]
        if scale_factor is not None:
            if not isinstance(scale_factor, Sequence):
                layer.scales = [1] + [scale_factor] * intput_shape_size
            else:
                layer.scales = [1, *list(scale_factor)]
    else:
        if size is not None:
            if not isinstance(size, Sequence):
                layer.shape = [input_shape[0], input_shape[1]] + [
                    size
                ] * intput_shape_size
            else:
                layer.shape = [input_shape[0], input_shape[1], *list(size)]
        if scale_factor is not None:
            if not isinstance(scale_factor, Sequence):
                layer.scales = [1, 1] + [scale_factor] * intput_shape_size
            else:
                layer.scales = [1, 1, *list(scale_factor)]
    if (align_corners is not None) and align_corners:
        layer.coordinate_transformation = (
            trt.ResizeCoordinateTransformation.ALIGN_CORNERS
        )
    return layer.get_output(0)
