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
import logging
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import tensorrt as trt

from paddle.base.log_helper import get_logger
from paddle.tensorrt.register import converter_registry

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s'
)


@converter_registry.register("pd_op.nonzero", trt_version="8.x")
def non_zero_converter(network, paddle_op, inputs):
    input_tensor = inputs[0]
    cast_layer = network.add_cast(input_tensor, trt.float32)
    non_zero_layer = network.add_non_zero(cast_layer.get_output(0))

    return non_zero_layer.get_output(0)
