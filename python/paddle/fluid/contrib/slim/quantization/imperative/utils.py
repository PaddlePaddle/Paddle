#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle

op_real_in_out_name = {
    "conv2d": [["Input", "Filter"], ["Output"]],
    "depthwise_conv2d": [["Input", "Filter"], ["Output"]],
    "pool2d": [["X"], ["Out"]],
    "elementwise_add": [["X", "Y"], ["Out"]],
    "softmax": [["X"], ["Out"]],
    "relu": [["X"], ["Out"]],
    "relu6": [["X"], ["Out"]],
    "leaky_relu": [["X"], ["Out"]],
    "prelu": [["X"], ["Out"]],
    "tanh": [["X"], ["Out"]],
    "batch_norm": [["X"], ["Y"]],
    "sigmoid": [["X"], ["Out"]],
    "swish": [["X"], ["Out"]],
}

supported_quant_layers_map = {
    'Conv2D': paddle.nn.Conv2D,
    'Linear': paddle.nn.Linear,
    'AdaptiveAvgPool2D': paddle.nn.AdaptiveAvgPool2D,
    'AdaptiveMaxPool2D': paddle.nn.AdaptiveMaxPool2D,
    'AvgPool2D': paddle.nn.AvgPool2D,
    'MaxPool2D': paddle.nn.MaxPool2D,
    'Hardswish': paddle.nn.Hardswish,
    'LeakyReLU': paddle.nn.LeakyReLU,
    'PReLU': paddle.nn.PReLU,
    'ReLU': paddle.nn.ReLU,
    'ReLU6': paddle.nn.ReLU6,
    'Sigmoid': paddle.nn.Sigmoid,
    'Softmax': paddle.nn.Softmax,
    'Swish': paddle.nn.Swish,
    'Tanh': paddle.nn.Tanh,
    'Hardswish': paddle.nn.Hardswish,
    'BatchNorm': paddle.nn.BatchNorm,
    'GroupNorm': paddle.nn.GroupNorm,
    'LayerNorm': paddle.nn.LayerNorm,
}

fake_quantize_dequantize_types = [
    "fake_quantize_dequantize_abs_max",
    "fake_quantize_dequantize_channel_wise_abs_max",
    "fake_quantize_dequantize_moving_average_abs_max"
]

out_scale_layers_list = (
    paddle.nn.Conv2D, paddle.nn.Linear, paddle.nn.MaxPool2D,
    paddle.nn.BatchNorm, paddle.nn.BatchNorm2D, paddle.nn.SyncBatchNorm,
    paddle.nn.LeakyReLU, paddle.nn.PReLU, paddle.nn.ReLU, paddle.nn.ReLU6,
    paddle.nn.Sigmoid, paddle.nn.Softmax, paddle.nn.Tanh, paddle.nn.Swish)
