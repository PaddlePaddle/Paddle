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

NOT_CHECK_OP_LIST = [
    # The increment's input must be 1-d and only has one data
    'increment',
    # elementwise ops have cases(y_shape: (1) or (1,1)) to test broadcast
    'elementwise_add',
    'elementwise_sub',
    'elementwise_mul',
    'elementwise_div',
    'elementwise_max',
    'elementwise_min',
    'elementwise_pow',
    'fused_elemwise_activation',
    # prelu op's input alpha must be 1-d and only has one data in 'all' mode
    'prelu'
]

NEED_TO_FIX_OP_LIST = [
    'bilinear_tensor_product',
    'conv2d_transpose',
    'deformable_conv',
    'deformable_conv',
    'grid_sampler',
    'hierarchical_sigmoid',
    'lstmp',
    'margin_rank_loss',
    'matmul',
    'mul',
    'row_conv',
    'scatter',
    'smooth_l1_loss',
    'soft_relu',
    'spp',
    'squared_l2_distance',
    'tree_conv',
    'var_conv_2d',
]
