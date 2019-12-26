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

# check no_grad_set is None
NOT_CHECK_OP_LIST = []
# no_grad_set has value in NEED_TO_FIX_OP_LIST
NEED_TO_FIX_OP_LIST = [
    'row_conv', 'mul', 'smooth_l1_loss', 'multiplex', 'sequence_conv',
    'conv_shift', 'margin_rank_loss', 'lstm', 'lstmp', 'lod_reset',
    'filter_by_instag', 'elementwise_div', 'elementwise_max', 'elementwise_min',
    'elementwise_add', 'elementwise_sub', 'elementwise_mul', 'affine_channel',
    'fused_emb_seq_pool', 'huber_loss', 'rank_loss',
    'fused_elemwise_activation', 'prelu', 'cos_sim', 'deformable_conv',
    'matmul', 'hsigmoid', 'kldiv_loss', 'affine_grad', 'conv3d_transpose',
    'conv2d_transpose', 'spectral_norm', 'cross_entropy2', 'linear_chain_crf',
    'lookup_table_v2', 'conv2d', 'gru_unit', 'lookup_table', 'conv3d',
    'deformable_conv_v1', 'depthwise_conv2d', 'depthwise_conv2d_transpose',
    'hierarchical_sigmoid', 'affine_grid', 'data_norm'
]
