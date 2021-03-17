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

from paddle.nn import Linear, Conv2D
from paddle.fluid.dygraph.nn import Pool2D
from paddle.nn.layer.activation import ReLU, LeakyReLU, Sigmoid, ReLU6
from paddle.nn.layer.activation import Tanh, Softmax, PReLU, Swish

_op_real_in_out_name = {
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

_quant_layers_map = {
    'Conv2D': Conv2D,
    'Linear': Linear,
    'Pool2D': Pool2D,
    'ReLU': ReLU,
    'LeakyReLU': LeakyReLU,
    'ReLU6': ReLU6,
    'Softmax': Softmax,
    'Tanh': Tanh,
    'Swish': Swish
}
