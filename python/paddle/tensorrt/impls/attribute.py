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

from paddle.tensorrt.register import converter_registry


@converter_registry.register("pd_op.shape", trt_version="8.x")
def shape_converter(network, paddle_op, inputs):
    input_tensor = inputs[0]
    shape_layer = network.add_shape(input_tensor)
    return shape_layer.get_output(0)
