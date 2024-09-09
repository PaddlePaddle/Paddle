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

from paddle.base.log_helper import get_logger
from paddle.tensorrt.register import converter_registry

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s'
)
from paddle.tensorrt.converter_utils import (
    get_trt_plugin,
)


@converter_registry.register("pd_op.relu", trt_version="8.x")
def relu_converter(network, paddle_op, inputs):
    relu_layer = network.add_activation(inputs[0], trt.ActivationType.RELU)
    return relu_layer.get_output(0)


@converter_registry.register("pd_op.softmax", trt_version="8.x")
def softmax_converter(network, paddle_op, inputs):
    axis = paddle_op.attrs().get("axis", 0)
    if axis < 0:
        axis = len(inputs[0].shape) + axis

    softmax_layer = network.add_softmax(inputs[0])
    softmax_layer.axes = 1 << axis
    return softmax_layer.get_output(0)


@converter_registry.register("pd_op.gelu", trt_version="8.x")
def gelu_converter(network, paddle_op, inputs):
    input_val = inputs[0]
    approximate = paddle_op.attrs()["approximate"]
    if approximate:
        raise RuntimeError(
            "GeLU converter currently doesn't support fast gelu compute"
        )

    plugin_name = "CustomGeluPluginDynamic"
    type_id = trt.PluginField(
        "type_id", np.array(0, dtype=np.int32), trt.PluginFieldType.INT32
    )

    filed_collection = trt.PluginFieldCollection([type_id])
    plugin_version = "1"

    plugin = get_trt_plugin(plugin_name, filed_collection, plugin_version)

    layer = network.add_plugin_v2([input_val], plugin)
    return layer.get_output(0)
