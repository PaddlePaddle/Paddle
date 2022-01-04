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
NOT_CHECK_OP_LIST = ['deformable_conv', 'row_conv', 'kron']

# TODO(Shixiaowei02): Check if the items do not need fix.
# no_grad_set has value in NEED_TO_FIX_OP_LIST
# yapf: disable
NEED_TO_FIX_OP_LIST = [
    'affine_channel',
    'affine_grid',
    'backward',
    'batch_norm',
    'conv2d',
    'conv2d_transpose',
    'conv3d',
    'conv3d_transpose',
    'cos_sim',
    'cross_entropy',
    'cross_entropy2',
    'data_norm',
    'deformable_conv_v1',
    'depthwise_conv2d',
    'depthwise_conv2d_transpose',
    'dot',
    'elementwise_add',
    'elementwise_div',
    'elementwise_max',
    'elementwise_min',
    'elementwise_mul',
    'elementwise_sub',
    'elementwise_pow',
    'elementwise_fmin',
    'elementwise_fmax',
    'filter_by_instag',
    'fused_elemwise_activation',
    'fused_emb_seq_pool',
    'fused_embedding_seq_pool',
    'gru_unit',
    'hierarchical_sigmoid',
    'hsigmoid',
    'huber_loss',
    'instance_norm',
    'kldiv_loss',
    'linear_chain_crf',
    'lod_reset',
    'lookup_table',
    'lookup_table_v2',
    'lstm',
    'lstmp',
    'margin_rank_loss',
    'matmul',
    'matmul_v2',
    'mul',
    'multiplex',
    'rank_loss',
    'sequence_conv',
    'smooth_l1_loss',
    'spectral_norm',
    'complex',
]
# yapf: enable
