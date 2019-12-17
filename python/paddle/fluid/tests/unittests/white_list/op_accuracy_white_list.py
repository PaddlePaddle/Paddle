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

# For op in FP16_CHECK_OP_LIST, the op test of fp16 precision should inherit OpTestFp16
FP16_CHECK_OP_LIST = [
    'abs', 'acos', 'asin', 'atan', 'brelu', 'concat', 'cos', 'elementwise_div',
    'elementwise_mul', 'elu', 'exp', 'gelu', 'hard_shrink', 'hard_swish', 'log',
    'logsigmoid', 'mean', 'mul', 'pad', 'pool2d', 'pow', 'reciprocal', 'relu',
    'relu6', 'scale', 'sigmoid', 'sin', 'slice', 'soft_relu', 'softmax',
    'softmax_with_cross_entropy', 'softshrink', 'softsign', 'sqrt', 'square',
    'stanh', 'sum', 'swish', 'tanh', 'tanh_shrink', 'thresholded_relu'
]

# For op in NO_FP64_CHECK_GRAD_OP_LIST, the op test requires check_grad with fp64 precision
NO_FP64_CHECK_GRAD_OP_LIST = [
    'abs', 'acos', 'add_position_encoding', 'affine_grid', 'asin', 'atan',
    'bilinear_interp', 'bilinear_tensor_product', 'brelu', 'center_loss',
    'clip', 'concat', 'conv2d', 'conv2d_transpose', 'conv3d',
    'conv3d_transpose', 'conv_shift', 'cos', 'cos_sim', 'crop', 'crop_tensor',
    'cross_entropy', 'cross_entropy2', 'cudnn_lstm', 'cvm', 'data_norm',
    'deformable_conv', 'deformable_conv_v1', 'deformable_psroi_pooling',
    'depthwise_conv2d', 'depthwise_conv2d_transpose', 'dropout',
    'elementwise_add', 'elementwise_div', 'elementwise_max', 'elementwise_min',
    'elementwise_mul', 'elementwise_pow', 'elementwise_sub', 'elu', 'exp',
    'expand', 'flatten', 'flatten2', 'fused_elemwise_activation',
    'fused_embedding_seq_pool', 'gather', 'gather_nd', 'gelu', 'grid_sampler',
    'group_norm', 'hard_shrink', 'hard_sigmoid', 'hard_swish',
    'hierarchical_sigmoid', 'hinge_loss', 'huber_loss', 'im2sequence',
    'increment', 'kldiv_loss', 'l1_norm', 'leaky_relu', 'lod_reset', 'log',
    'log_loss', 'logsigmoid', 'lookup_table', 'lookup_table_v2', 'lrn',
    'margin_rank_loss', 'match_matrix_tensor', 'matmul',
    'max_pool2d_with_index', 'max_pool3d_with_index', 'maxout', 'mean', 'minus',
    'modified_huber_loss', 'mul', 'multiplex', 'nce', 'nearest_interp', 'pad',
    'pad2d', 'pad_constant_like', 'pixel_shuffle', 'pool2d', 'pool3d', 'pow',
    'prelu', 'prroi_pool', 'psroi_pool', 'rank_loss', 'reciprocal',
    'reduce_max', 'reduce_min', 'relu', 'relu6', 'reshape2', 'reverse',
    'roi_align', 'roi_perspective_transform', 'roi_pool', 'row_conv', 'rsqrt',
    'scale', 'scatter', 'scatter_nd_add', 'seed', 'selu', 'sequence_concat',
    'sequence_conv', 'sequence_expand', 'sequence_expand_as', 'sequence_pad',
    'sequence_pool', 'sequence_reshape', 'sequence_reverse', 'sequence_scatter',
    'sequence_slice', 'sequence_softmax', 'sequence_topk_avg_pooling',
    'sequence_unpad', 'shuffle_channel', 'sigmoid',
    'sigmoid_cross_entropy_with_logits', 'sigmoid_focal_loss', 'sign', 'sin',
    'slice', 'smooth_l1_loss', 'soft_relu', 'softmax', 'softshrink', 'softsign',
    'space_to_depth', 'spectral_norm', 'split', 'spp', 'sqrt', 'square',
    'squared_l2_distance', 'squared_l2_norm', 'squeeze', 'squeeze2', 'stack',
    'stanh', 'strided_slice', 'swish', 'tanh', 'tanh_shrink',
    'teacher_student_sigmoid_loss', 'temporal_shift', 'thresholded_relu',
    'transpose2', 'tree_conv', 'trilinear_interp', 'unfold', 'unpool',
    'unsqueeze', 'unsqueeze2', 'unstack', 'var_conv_2d', 'warpctc',
    'yolov3_loss'
]

# For cases in NO_FP64_CHECK_GRAD_CASES, the op test requires check_grad with fp64 precision
NO_FP64_CHECK_GRAD_CASES = ['TestFSPOp']
