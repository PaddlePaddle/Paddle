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

from paddle.tensorrt.converter_utils import (
    broadcast,
)
from paddle.tensorrt.register import converter_registry


@converter_registry.register("pd_op.matmul", trt_version="8.x")
def matmul_converter(network, paddle_op, inputs):
    weight_shape = paddle_op.operands()[1].source().shape
    transpose_x = paddle_op.attrs()["transpose_x"]
    transpose_y = paddle_op.attrs()["transpose_y"]
    self_matrix_op = (
        trt.MatrixOperation.TRANSPOSE
        if transpose_x
        else trt.MatrixOperation.NONE
    )
    other_matrix_op = (
        trt.MatrixOperation.TRANSPOSE
        if transpose_y
        else trt.MatrixOperation.NONE
    )

    weight_tensor = inputs[1]
    if type(inputs[1]) == trt.Weights:
        weight_tensor = network.add_constant(
            weight_shape, inputs[1]
        ).get_output(0)

    if len(weight_shape) == 1:
        layer = network.add_shuffle(weight_tensor)
        layer.reshape_dims = (*tuple(weight_shape), 1)
        weight_tensor = layer.get_output(0)

    lhs_val, rhs_val = broadcast(
        network, inputs[0], weight_tensor, inputs[0].name, weight_tensor.name
    )
    out = network.add_matrix_multiply(
        lhs_val, self_matrix_op, rhs_val, other_matrix_op
    )
    return out.get_output(0)


@converter_registry.register("pd_op.transpose", trt_version="8.x")
def transpose_converter(network, paddle_op, inputs):
    perm = paddle_op.attrs()["perm"]
    transposed_tensor = network.add_shuffle(inputs[0])
    transposed_tensor.second_transpose = perm
    return transposed_tensor.get_output(0)
