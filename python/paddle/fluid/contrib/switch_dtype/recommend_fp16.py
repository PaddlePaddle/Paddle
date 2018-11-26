#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

# NOTES: for some operator, it's output's range is larger then it's input's
# i.g. |f(x)| >> x, it is recommended that those operators should not use Fp16,
# because it's result may be overflow or underflow.

# white_list is the collection of operators which is safe when using fp16 computation.
white_list = [
    #
    'relu',
    'sigmoid',
    'tanh',
    #
    'pool2d',
    'pool3d',
    'conv2d',
    'conv3d',
    'conv_transpose2d',
    'conv_transpose3d',
    #
    'matmul',
]

# black_list is the collection of operators which may be overflows or
# when using fp16 computation.
black_list = [
    #
    'exp',
    'square',
    'log',
    'pow',
    #
    'mean',
    'sum',
    'reduce_sum',
    #
    'cos_sim',
    'layer_norm',
    'softmax',
    'softmax_with_cross_entropy',
    'cross_entropy',
]

# gray_list is the collection of operators whose gradient may be overflow or
# underflow when using fp16 computation.
gray_list = [
    # because elementwise_x_grad maybe use reduce_sum operation
    # which may cause overflows.
    "elementwise_add",
    "elementwise_sub",
    "elementwise_mul",
    "elementwise_div",
]
