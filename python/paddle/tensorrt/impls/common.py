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


@converter_registry.register("pd_op.dropout", trt_version="8.x")
def dropout_converter(network, paddle_op, inputs):
    input_x = inputs[0]
    p_defining_op = paddle_op.operands()[2].source().get_defining_op()
    dropout_prob = p_defining_op.attrs()["value"]
    downgrade_in_infer = paddle_op.attrs().get("mode")

    if downgrade_in_infer == "upscale_in_train":
        shuffle_layer = network.add_shuffle(input_x)
        return shuffle_layer.get_output(0)

    weight_data = np.array([1 - dropout_prob]).astype("float32")
    scale_weights = trt.Weights(weight_data)
    shift_weights = trt.Weights(np.array([0]).astype("float32"))
    power_weights = trt.Weights(np.array([1]).astype("float32"))

    scale_layer = network.add_scale(
        input_x,
        mode=trt.ScaleMode.UNIFORM,
        shift=shift_weights,
        scale=scale_weights,
        power=power_weights,
    )
    return scale_layer.get_output(0)
