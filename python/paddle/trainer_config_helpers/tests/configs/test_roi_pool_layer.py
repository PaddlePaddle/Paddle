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

data = data_layer(name='data', size=3 * 14 * 14, height=14, width=14)

rois = data_layer(name='rois', size=10)

conv = img_conv_layer(
    input=data,
    filter_size=3,
    num_channels=3,
    num_filters=16,
    padding=1,
    act=LinearActivation(),
    bias_attr=True)

roi_pool = roi_pool_layer(
    input=conv,
    rois=rois,
    pooled_width=7,
    pooled_height=7,
    spatial_scale=1. / 16)

outputs(roi_pool)
