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
    WithFp16,
    get_trt_plugin,
    unary_op_converter,
)
from paddle.tensorrt.register import converter_registry


@converter_registry.register("pd_op.sqrt", trt_version="trt_version_ge=8.0")
@converter_registry.register("pd_op.sqrt_", trt_version="trt_version_ge=8.0")
@converter_registry.register("pd_op.floor", trt_version="trt_version_ge=8.0")
@converter_registry.register("pd_op.exp", trt_version="trt_version_ge=8.0")
@converter_registry.register("pd_op.abs", trt_version="trt_version_ge=8.0")
@converter_registry.register("pd_op.abs_", trt_version="trt_version_ge=8.0")
@converter_registry.register("pd_op.sin", trt_version="trt_version_ge=8.0")
@converter_registry.register("pd_op.cos", trt_version="trt_version_ge=8.0")
@converter_registry.register("pd_op.sinh", trt_version="trt_version_ge=8.0")
@converter_registry.register("pd_op.cosh", trt_version="trt_version_ge=8.0")
@converter_registry.register("pd_op.asinh", trt_version="trt_version_ge=8.0")
@converter_registry.register("pd_op.acosh", trt_version="trt_version_ge=8.0")
@converter_registry.register("pd_op.atanh", trt_version="trt_version_ge=8.0")
@converter_registry.register("pd_op.ceil", trt_version="trt_version_ge=8.0")
@converter_registry.register(
    "pd_op.reciprocal", trt_version="trt_version_ge=8.0"
)
@converter_registry.register("pd_op.erf", trt_version="trt_version_ge=8.0")
@converter_registry.register("pd_op.rsqrt", trt_version="trt_version_ge=8.0")
@converter_registry.register("pd_op.sign", trt_version="trt_version_ge=8.2")
@converter_registry.register("pd_op.round", trt_version="trt_version_ge=8.2")
def UnaryOpConverter(network, paddle_op, inputs):
    layer_output = unary_op_converter(network, paddle_op, inputs)
    return layer_output


@converter_registry.register("pd_op.yolo_box", trt_version="trt_version_ge=8.0")
def YoloBoxOpConverter(network, paddle_op, inputs):
    x, imgSize = inputs
    class_num = paddle_op.attrs().get("class_num")
    anchors = paddle_op.attrs().get("anchors")
    downsample_ratio = paddle_op.attrs().get("downsample_ratio")
    conf_thresh = paddle_op.attrs().get("conf_thresh")
    clip_bbox = paddle_op.attrs().get("clip_bbox")
    scale_x_y = paddle_op.attrs().get("scale_x_y")
    iou_aware = paddle_op.attrs().get("iou_aware")
    iou_aware_factor = paddle_op.attrs().get("iou_aware_factor")
    type_id = int(WithFp16())
    input_dim = x.shape
    input_h = input_dim[2]
    input_w = input_dim[3]
    plugin_fields = [
        trt.PluginField(
            "type_id",
            np.array([type_id], dtype=np.int32),
            trt.PluginFieldType.INT32,
        ),
        trt.PluginField(
            "anchors",
            np.array(anchors, dtype=np.int32),
            trt.PluginFieldType.INT32,
        ),
        trt.PluginField(
            "class_num",
            np.array(class_num, dtype=np.int32),
            trt.PluginFieldType.INT32,
        ),
        trt.PluginField(
            "conf_thresh",
            np.array(conf_thresh, dtype=np.float32),
        ),
        trt.PluginField(
            "downsample_ratio",
            np.array(downsample_ratio, dtype=np.int32),
        ),
        trt.PluginField(
            "clip_bbox",
            np.array(clip_bbox, dtype=np.bool),
        ),
        trt.PluginField(
            "scale_x_y",
            np.array(scale_x_y, dtype=np.float32),
        ),
        trt.PluginField(
            "iou_aware",
            np.array(iou_aware, dtype=np.bool),
        ),
        trt.PluginField(
            "iou_aware_factor",
            np.array(iou_aware_factor, dtype=np.float32),
        ),
        trt.PluginField(
            "h",
            np.array(input_h, dtype=np.int32),
        ),
        trt.PluginField(
            "w",
            np.array(input_w, dtype=np.int32),
        ),
    ]
    plugin_field_collection = trt.PluginFieldCollection(plugin_fields)
    plugin_name = "pir_yolo_box_plugin"
    plugin_version = "1"
    plugin = get_trt_plugin(
        plugin_name, plugin_field_collection, plugin_version
    )

    yolo_box_inputs = [x, imgSize]
    yolo_box_layer = network.add_plugin_v2(yolo_box_inputs, plugin)
    out0 = yolo_box_layer.get_output(0)
    out1 = yolo_box_layer.get_output(1)
    return (out0, out1)
