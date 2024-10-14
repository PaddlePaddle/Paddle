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
    add_elementwise_layer,
    trt_cast,
)
from paddle.tensorrt.register import converter_registry


@converter_registry.register("pd_op.greater_than", trt_version="8.x")
def greater_than_converter(network, paddle_op, inputs):
    layer_output = add_elementwise_layer(
        network, paddle_op, inputs, trt.ElementWiseOperation.GREATER
    )
    return trt_cast(network, layer_output, trt.float32)


@converter_registry.register("pd_op.less_than", trt_version="8.x")
def less_than_converter(network, paddle_op, inputs):
    layer_output = add_elementwise_layer(
        network, paddle_op, inputs, trt.ElementWiseOperation.LESS
    )
    return trt_cast(network, layer_output, trt.float32)
