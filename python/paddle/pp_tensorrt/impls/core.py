
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

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import tensorrt as trt
from register import converter_registry

@converter_registry.register("pd_op.add")
@converter_registry.register("pd_op.elementwise_add")
def add_converter(network, paddle_op, inputs):
    weight_shape = paddle_op.operands()[1].source().shape
    weight_tensor = network.add_constant(weight_shape, inputs[1]).get_output(0)

    input_shape = paddle_op.operands()[0].source().shape
    # reshape_layer = network.add_reshape(weight_tensor, input_shape).get_output(0)
    shuffle_layer = network.add_shuffle(weight_tensor)
    shuffle_layer.reshape_dims = input_shape
    reshape_layer = shuffle_layer.get_output(0)

    out = network.add_elementwise(inputs[0], reshape_layer, trt.ElementWiseOperation.SUM)
    return out

@converter_registry.register("pd_op.relu")
def relu_converter(network, paddle_op, inputs):
    out = network.add_activation(inputs[0], trt.ActivationType.RELU)
    return out

@converter_registry.register("pd_op.matmul")
def matmul_converter(network, paddle_op, inputs):
    weight_shape = paddle_op.operands()[1].source().shape
    weight_tensor = network.add_constant(weight_shape, inputs[1]).get_output(0)
    out = network.add_matrix_multiply(inputs[0], trt.MatrixOperation.NONE, weight_tensor, trt.MatrixOperation.NONE)
    return out