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
    get_trt_plugin,
)
from paddle.tensorrt.register import converter_registry


@converter_registry.register("pd_op.multiclass_nms3", trt_version="8.x")
def multiclass_nms3_converter(network, paddle_op, inputs):
    bboxes = inputs[0]
    scores = inputs[1]
    background_label = paddle_op.attrs().get("background_label")
    score_threshold = paddle_op.attrs().get("score_threshold")
    nms_top_k = paddle_op.attrs().get("nms_top_k")
    nms_threshold = paddle_op.attrs().get("nms_threshold")
    keep_top_k = paddle_op.attrs().get("keep_top_k")
    normalized = paddle_op.attrs().get("normalized")
    num_classes = scores.shape[1]

    bboxes_dims = bboxes.shape
    bboxes_expand_dims = [bboxes_dims[0], bboxes_dims[1], 1, bboxes_dims[2]]
    bboxes_expand_layer = network.add_shuffle(bboxes)
    bboxes_expand_layer.reshape_dims = trt.Dims(bboxes_expand_dims)

    scores_transpose_layer = network.add_shuffle(scores)
    scores_transpose_layer.first_transpose = (0, 2, 1)

    # create multiclass num3 plugin
    batch_nms_inputs = [
        bboxes_expand_layer.get_output(0),
        scores_transpose_layer.get_output(0),
    ]
    plugin_fields = [
        trt.PluginField(
            "shareLocation",
            np.array([1], dtype=np.int32),
            trt.PluginFieldType.INT32,
        ),
        trt.PluginField(
            "backgroundLabelId",
            np.array(background_label, dtype=np.int32),
            trt.PluginFieldType.INT32,
        ),
        trt.PluginField(
            "numClasses",
            np.array(num_classes, dtype=np.int32),
            trt.PluginFieldType.INT32,
        ),
        trt.PluginField(
            "topK",
            np.array(nms_top_k, dtype=np.int32),
            trt.PluginFieldType.INT32,
        ),
        trt.PluginField(
            "keepTopK",
            np.array(keep_top_k, dtype=np.int32),
            trt.PluginFieldType.INT32,
        ),
        trt.PluginField(
            "scoreThreshold",
            np.array(score_threshold, dtype=np.float32),
            trt.PluginFieldType.FLOAT32,
        ),
        trt.PluginField(
            "iouThreshold",
            np.array(nms_threshold, dtype=np.float32),
            trt.PluginFieldType.FLOAT32,
        ),
        trt.PluginField(
            "isNormalized",
            np.array(normalized, dtype=np.int32),
            trt.PluginFieldType.INT32,
        ),
        trt.PluginField(
            "clipBoxes",
            np.array([0], dtype=np.int32),
            trt.PluginFieldType.INT32,
        ),
    ]
    plugin_field_collection = trt.PluginFieldCollection(plugin_fields)
    plugin_name = "BatchedNMSDynamic_TRT"
    plugin_version = "1"
    plugin = get_trt_plugin(
        plugin_name, plugin_field_collection, plugin_version
    )
    batch_nms_layer = network.add_plugin_v2(batch_nms_inputs, plugin)

    # dynamic shape: [bs, keep_topk, 4], [bs, keep_topk], [bs, keep_topk]
    nmsed_boxes = batch_nms_layer.get_output(1)
    nmsed_scores = batch_nms_layer.get_output(2)
    nmsed_classes = batch_nms_layer.get_output(3)
    nmsed_scores_transpose_layer = network.add_shuffle(nmsed_scores)
    nmsed_classes_reshape_layer = network.add_shuffle(nmsed_classes)
    nmsed_scores_transpose_layer.reshape_dims = trt.Dims(
        [bboxes_dims[0], keep_top_k, 1]
    )
    nmsed_classes_reshape_layer.reshape_dims = trt.Dims(
        [bboxes_dims[0], keep_top_k, 1]
    )

    concat_inputs = [
        nmsed_classes_reshape_layer.get_output(0),
        nmsed_scores_transpose_layer.get_output(0),
        nmsed_boxes,
    ]
    nms_concat_layer = network.add_concatenation(inputs=concat_inputs)
    nms_concat_layer.axis = 2
    nms_concat_output = nms_concat_layer.get_output(0)
    nms_shuffle_layer = network.add_shuffle(nms_concat_output)
    nms_shuffle_layer.reshape_dims = trt.Dims(
        [bboxes_dims[0], nms_concat_output.shape[-1]]
    )

    # add fake index as output to be consistent with the outputs of multiclass_nms3
    shape_weight = trt.Weights(np.array([0], dtype=np.int32))
    constant_layer = network.add_constant([1, 1], shape_weight)

    return (
        nms_shuffle_layer.get_output(0),
        constant_layer.get_output(0),
        batch_nms_layer.get_output(0),
    )
