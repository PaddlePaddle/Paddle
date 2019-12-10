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

need_to_fix_check_shape_op_list = [
    'elementwise_sub', 'gather', 'mean', 'pad2d', 'scatter',
    'sequence_expand_as', 'sequence_expand', 'sequence_pad', 'sequence_unpad',
    'sequence_scatter', 'squared_l2_distance', 'gather_nd', 'log_loss',
    'sequence_topk_avg_pooling', 'matmul', 'transpose2', 'crop_tensor',
    'expand', 'center_loss', 'softmax', 'scale', 'elementwise_max',
    'hierarchical_sigmoid', 'rank_loss', 'reshape2', 'conv_shift', 'spp', 'pad',
    'add_position_encoding', 'modified_huber_loss', 'affine_grid', 'sign',
    'squeeze', 'smooth_l1_loss', 'margin_rank_loss', 'multiplex',
    'sequence_softmax', 'reduce_sum', 'nce', 'sigmoid_focal_loss', 'sum',
    'elementwise_pow', 'pad_constant_like', 'squeeze2', 'norm', 'unsqueeze',
    'fused_elemwise_activation', 'tree_conv', 'huber_loss', 'group_norm',
    'bilinear_tensor_product', 'flatten2', 'unsqueeze2', 'mul', 'im2sequence',
    'kldiv_loss', 'unpool', 'cos_sim', 'strided_slice', 'cast', 'reverse',
    'hinge_loss', 'flatten', 'scatter_nd_add', 'expand_as',
    'teacher_student_sigmoid_loss', 'label_smooth', 'spectral_norm',
    'elementwise_min', 'abs', 'acos', 'warpctc', 'nearest_interp', 'data_norm',
    'match_matrix_tensor', 'var_conv_2d', 'fused_embedding_seq_pool',
    'increment', 'elementwise_mul'
]
