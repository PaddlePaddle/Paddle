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
import numpy as np

quant_input_layers_map = {
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

quant_output_layers = (
    paddle.nn.Conv2D, paddle.nn.Conv2DTranspose, paddle.nn.Linear,
    paddle.nn.AdaptiveAvgPool2D, paddle.nn.AdaptiveMaxPool2D,
    paddle.nn.AvgPool2D, paddle.nn.MaxPool2D, paddle.nn.BatchNorm,
    paddle.nn.BatchNorm2D, paddle.nn.LayerNorm, paddle.nn.SyncBatchNorm,
    paddle.nn.ELU, paddle.nn.GELU, paddle.nn.Hardshrink, paddle.nn.Hardsigmoid,
    paddle.nn.Hardswish, paddle.nn.Hardtanh, paddle.nn.LeakyReLU,
    paddle.nn.LogSigmoid, paddle.nn.LogSoftmax, paddle.nn.Maxout,
    paddle.nn.PReLU, paddle.nn.ReLU, paddle.nn.ReLU6, paddle.nn.SELU,
    paddle.nn.Sigmoid, paddle.nn.Softmax, paddle.nn.Softplus,
    paddle.nn.Softshrink, paddle.nn.Softsign, paddle.nn.Swish, paddle.nn.Tanh,
    paddle.nn.Tanhshrink, paddle.nn.ThresholdedReLU, paddle.nn.Upsample)

weight_op_types = [
    "conv2d", "depthwise_conv2d", "matmul", "conv2d_transpose",
    "depthwise_conv2d_transpose"
]


def load_variable_data(scope, var_name):
    '''
    Load variable value from scope
    '''
    var_node = scope.find_var(var_name)
    assert var_node is not None, \
        "Can not find " + var_name + " in the scope."
    return np.array(var_node.get_tensor())


def find_previous_op(block, var_name):
    """
    Find the previous op for the input variable.
    """
    for op in block.ops:
        if var_name in op.output_arg_names:
            return op


def find_next_ops(block, var_name):
    """
    Find all followed ops for the input variable.
    """
    res_ops = []
    for op in block.ops:
        if var_name in op.input_arg_names:
            res_ops.append(op)
    return res_ops
