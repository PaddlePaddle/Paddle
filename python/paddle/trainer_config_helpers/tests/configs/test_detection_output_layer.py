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

from paddle.trainer_config_helpers import *

settings(batch_size=1000, learning_rate=1e-5)

input_loc = data_layer(name='input_loc', size=16, height=16, width=1)

input_conf = data_layer(name='input_conf', size=8, height=1, width=8)

priorbox = data_layer(name='priorbox', size=32, height=4, width=8)

detout = detection_output_layer(
    input_loc=input_loc,
    input_conf=input_conf,
    priorbox=priorbox,
    num_classes=21,
    nms_threshold=0.45,
    nms_top_k=400,
    keep_top_k=200,
    confidence_threshold=0.01,
    background_id=0,
    name='test_detection_output')

outputs(detout)
