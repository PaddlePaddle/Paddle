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

from paddle.tensorrt.register import converter_registry


@converter_registry.register("pd_op.full_int_array", trt_version="8.x")
def full_int_array_converter(network, paddle_op, inputs):
    shape = paddle_op.attrs()["value"]
    shape_weight = trt.Weights(np.array(shape, dtype=np.int32))
    full_int_array_layer = network.add_constant([len(shape)], shape_weight)
    return full_int_array_layer.get_output(0)


@converter_registry.register("pd_op.full", trt_version="8.x")
def full_converter(network, paddle_op, inputs):
    shape = paddle_op.attrs()["shape"]
    value = paddle_op.attrs().get("value", 1.0)  # 默认值为1.0
    full_layer = network.add_constant(
        shape, np.full(shape, value, dtype=np.float32)
    )
    return full_layer.get_output(0)
