# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

NEED_FIX_FP64_CHECK_GRAD_THRESHOLD_OP_LIST = [
    'affine_channel', 'bilinear_interp', 'cross_entropy', 'elementwise_mul',
    'grid_sampler', 'group_norm', 'gru', 'gru_unit', 'lstm', 'lstmp', 'norm',
    'pool3d', 'reduce_prod', 'selu', 'sigmoid_cross_entropy_with_logits',
    'sigmoid_focal_loss', 'soft_relu', 'softmax_with_cross_entropy', 'unpool',
    'yolov3_loss'
]

NEED_FIX_FP64_CHECK_OUTPUT_THRESHOLD_OP_LIST = ['bilinear_interp']
