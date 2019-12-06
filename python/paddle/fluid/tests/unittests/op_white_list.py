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

CHECK_FP16_OP_LIST = [
    'concat', 'pad', 'elementwise_mul', 'elementwise_div',
    'softmax_with_cross_entropy'
]

CHECK_GRAD_FP64_OP_WHITE_LIST = [
    'abs', 'accuracy', 'acos', 'adadelta', 'adagrad', 'adam', 'adamax',
    'add_position_encoding', 'affine_grid', 'anchor_generator', 'arg_max',
    'arg_min', 'argsort', 'asin', 'assign_value', 'atan', 'attention_lstm',
    'auc', 'bilinear_interp', 'bilinear_tensor_product', 'bipartite_match',
    'box_clip', 'box_coder', 'box_decoder_and_assign', 'brelu', 'cast', 'ceil',
    'center_loss', 'chunk_eval', 'clip', 'clip_by_norm',
    'collect_fpn_proposals', 'concat', 'conv2d_transpose', 'conv3d_transpose',
    'conv_shift', 'cos', 'cos_sim', 'crf_decoding', 'crop', 'crop_tensor',
    'ctc_align', 'cvm', 'data_norm', 'decayed_adagrad', 'deformable_conv',
    'deformable_conv_v1', 'deformable_psroi_pooling', 'density_prior_box',
    'depthwise_conv2d_transpose', 'dequantize', 'dequantize_abs_max',
    'detection_map', 'diag', 'distribute_fpn_proposals', 'dpsgd', 'dropout',
    'edit_distance', 'elementwise_add', 'elementwise_div',
    'elementwise_floordiv', 'elementwise_max', 'elementwise_min',
    'elementwise_mod', 'elementwise_mul', 'elementwise_pow', 'elementwise_sub',
    'elu', 'equal', 'exp', 'expand', 'eye',
    'fake_channel_wise_dequantize_max_abs',
    'fake_channel_wise_quantize_abs_max', 'fake_dequantize_max_abs',
    'fake_quantize_abs_max', 'fake_quantize_dequantize_moving_average_abs_max',
    'fake_quantize_moving_average_abs_max', 'fake_quantize_range_abs_max', 'fc',
    'fill', 'fill_any_like', 'fill_constant', 'fill_constant_batch_size_like',
    'fill_zeros_like', 'fill_zeros_like2', 'flatten', 'flatten2', 'floor',
    'ftrl', 'fused_elemwise_activation', 'fused_embedding_fc_lstm',
    'fused_embedding_seq_pool', 'fusion_gru', 'fusion_lstm',
    'fusion_repeated_fc_relu', 'fusion_seqconv_eltadd_relu',
    'fusion_seqexpand_concat_fc', 'fusion_seqpool_concat',
    'fusion_seqpool_cvm_concat', 'fusion_squared_mat_sub', 'gather',
    'gather_nd', 'gather_tree', 'gelu', 'generate_mask_labels',
    'generate_proposal_labels', 'generate_proposals', 'greater_equal',
    'greater_than', 'grid_sampler', 'hard_shrink', 'hard_sigmoid', 'hard_swish',
    'hash', 'hierarchical_sigmoid', 'hinge_loss', 'huber_loss', 'im2sequence',
    'iou_similarity', 'is_empty', 'isfinite', 'isinf', 'isnan', 'kldiv_loss',
    'l1_norm', 'lamb', 'lars_momentum', 'leaky_relu', 'less_equal', 'less_than',
    'linspace', 'locality_aware_nms', 'lod_reset', 'log', 'log_loss',
    'logical_and', 'logical_not', 'logical_or', 'logical_xor', 'logsigmoid',
    'lookup_table', 'lookup_table_v2', 'lrn', 'margin_rank_loss',
    'match_matrix_tensor', 'matmul', 'max_pool2d_with_index',
    'max_pool3d_with_index', 'maxout', 'mean', 'mean_iou', 'merge_ids',
    'mine_hard_examples', 'minus', 'modified_huber_loss', 'momentum',
    'moving_average_abs_max_scale', 'mul', 'multiclass_nms', 'multiclass_nms2',
    'multiplex', 'nce', 'nearest_interp', 'not_equal', 'one_hot', 'one_hot_v2',
    'pad', 'pad2d', 'pad_constant_like', 'pixel_shuffle',
    'polygon_box_transform', 'pool2d', 'pool3d', 'positive_negative_pair',
    'pow', 'precision_recall', 'prelu', 'prior_box', 'proximal_adagrad',
    'proximal_gd', 'prroi_pool', 'psroi_pool', 'quantize', 'range', 'rank_loss',
    'reciprocal', 'ref_by_trainer_id', 'relu', 'relu6', 'requantize',
    'reshape2', 'retinanet_detection_output', 'retinanet_target_assign',
    'reverse', 'roi_align', 'roi_perspective_transform', 'roi_pool', 'round',
    'row_conv', 'rpn_target_assign', 'rsqrt', 'scale', 'scatter',
    'scatter_nd_add', 'selu', 'sequence_concat', 'sequence_conv',
    'sequence_enumerate', 'sequence_erase', 'sequence_expand',
    'sequence_expand_as', 'sequence_mask', 'sequence_pad', 'sequence_pool',
    'sequence_reshape', 'sequence_reverse', 'sequence_scatter',
    'sequence_slice', 'sequence_softmax', 'sequence_topk_avg_pooling',
    'sequence_unpad', 'sgd', 'shape', 'shard_index', 'shuffle_channel',
    'sigmoid', 'sigmoid_cross_entropy_with_logits', 'sigmoid_focal_loss',
    'sign', 'similarity_focus', 'sin', 'size', 'slice', 'smooth_l1_loss',
    'soft_relu', 'softmax', 'softshrink', 'spectral_norm', 'split', 'split_ids',
    'spp', 'sqrt', 'squared_l2_distance', 'squared_l2_norm', 'squeeze',
    'squeeze2', 'stack', 'stanh', 'sum', 'tanh', 'target_assign',
    'teacher_student_sigmoid_loss', 'temporal_shift', 'top_k', 'transpose2',
    'tree_conv', 'trilinear_interp', 'unfold', 'unique', 'unique_with_counts',
    'unpool', 'unsqueeze', 'unsqueeze2', 'unstack', 'var_conv_2d', 'warpctc',
    'where', 'yolo_box'
]

CHECK_GRAD_FP16_OP_WHITE_LIST = []
