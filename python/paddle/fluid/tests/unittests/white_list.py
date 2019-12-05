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

Grad_White_List = []
Need_To_Fix_Op_List = [
    'row_conv', 'mul', 'smooth_l1_loss', 'multiplex', 'sequence_conv',
    'conv_shift', 'margin_rank_loss', 'lstm', 'lstmp', 'lod_reset',
    'filter_by_instag', 'elementwise_div', 'elementwise_max', 'elementwise_min',
    'elementwise_add', 'elementwise_sub', 'elementwise_mul', 'affine_channel',
    'fused_emb_seq_pool', 'huber_loss', 'rank_loss',
    'fused_elemwise_activation', 'prelu', 'cos_sim', 'deformable_conv',
    'matmul', 'hissigmod', 'kldiv_loss', 'affine_grad', 'conv3d_transpose',
    'conv2d_transpose', 'spectral_norm', 'cross_entropy2', 'linear_chain_crf',
    'lookup_table_v2', 'conv2d', 'gru_unit', 'lookup_table', 'conv3d',
    'deformable_conv_v1'
]
