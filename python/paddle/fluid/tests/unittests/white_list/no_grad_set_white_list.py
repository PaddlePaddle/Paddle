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
NOT_CHECK_OP_LIST = [
    'affine_channel',
    'affine_grid',
    'batch_norm',
    'conv_shift',
    'cos_sim',
    'cross_entropy',
    'data_norm',
    'deformable_conv',
    'deformable_conv_v1',
    'elementwise_add',
    'elementwise_div',
    'elementwise_max',
    'elementwise_min',
    'elementwise_mul',
    'elementwise_sub',
    'fused_elemwise_activation',
    'fused_emb_seq_pool',
    'fused_embedding_seq_pool',
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
    'mul',
    'multiplex',
    'rank_loss',
    'smooth_l1_loss',
    'spectral_norm'
]
# TODO(Shixiaowei02): Check if the items do not need fix.
# no_grad_set has value in NEED_TO_FIX_OP_LIST
# yapf: disable
NEED_TO_FIX_OP_LIST = [
    'backward',
    'conv2d',
    'conv2d_transpose',
    'conv3d',
    'conv3d_transpose',
    'cross_entropy2',
    'depthwise_conv2d',
    'depthwise_conv2d_transpose',
    'filter_by_instag',
    'gru_unit',
    'hierarchical_sigmoid',
    'hsigmoid',
    'prelu',
    'row_conv',
    'sequence_conv'
]
# yapf: enable
