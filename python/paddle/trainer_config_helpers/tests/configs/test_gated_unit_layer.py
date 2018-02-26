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

data = data_layer(name='input', size=256)
glu = gated_unit_layer(
    size=512,
    input=data,
    act=TanhActivation(),
    gate_attr=ExtraLayerAttribute(error_clipping_threshold=100.0),
    gate_param_attr=ParamAttr(initial_std=1e-4),
    gate_bias_attr=ParamAttr(initial_std=1),
    inproj_attr=ExtraLayerAttribute(error_clipping_threshold=100.0),
    inproj_param_attr=ParamAttr(initial_std=1e-4),
    inproj_bias_attr=ParamAttr(initial_std=1),
    layer_attr=ExtraLayerAttribute(error_clipping_threshold=100.0))

outputs(glu)
