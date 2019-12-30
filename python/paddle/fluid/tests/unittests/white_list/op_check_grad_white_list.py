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

# Grad op is not registered for Ops in EMPTY_GRAD_OP_LIST, so check grad 
# will not be required.
EMPTY_GRAD_OP_LIST = [
    'fill_zeros_like2', 'gaussian_random_batch_size_like',
    'fill_constant_batch_size_like', 'iou_similarity', 'where',
    'uniform_random_batch_size_like', 'box_coder', 'equal', 'greater_equal',
    'greater_than', 'less_equal', 'sequence_enumerate', 'logical_and',
    'logical_not', 'logical_or', 'logical_xor', 'unique',
    'fusion_seqconv_eltadd_relu', 'prior_box', 'decayed_adagrad',
    'crf_decoding', 'mine_hard_examples', 'fusion_seqpool_concat',
    'fused_embedding_fc_lstm', 'top_k', 'uniform_random', 'multihead_matmul',
    'edit_distance', 'shard_index', 'generate_proposals', 'density_prior_box',
    'round', 'floor', 'ceil', 'precision_recall', 'proximal_adagrad', 'cast',
    'isinf', 'isfinite', 'isnan', 'fill_constant', 'fusion_seqpool_cvm_concat',
    'accuracy', 'fc', 'sgd', 'anchor_generator',
    'fake_channel_wise_quantize_abs_max',
    'fake_quantize_dequantize_moving_average_abs_max', 'fake_quantize_abs_max',
    'fake_quantize_range_abs_max', 'moving_average_abs_max_scale',
    'fake_quantize_moving_average_abs_max', 'fill_any_like', 'one_hot',
    'gather_tree', 'lookup_sparse_table', 'lamb', 'fusion_squared_mat_sub',
    'range', 'box_decoder_and_assign', 'one_hot_v2', 'shape',
    'fusion_transpose_flatten_concat', 'lars_momentum', 'momentum',
    'fusion_lstm', 'assign_value', 'polygon_box_transform',
    'retinanet_detection_output', 'generate_proposal_labels', 'ctc_align',
    'sequence_erase', 'fake_channel_wise_dequantize_max_abs',
    'fake_dequantize_max_abs', 'generate_mask_labels', 'elementwise_floordiv',
    'sum', 'ftrl', 'fusion_repeated_fc_relu', 'size', 'bipartite_match',
    'elementwise_mod', 'multiclass_nms2', 'multiclass_nms', 'fill_zeros_like',
    'adadelta', 'conv2d_fusion', 'adamax', 'sampling_id', 'dpsgd',
    'target_assign', 'random_crop', 'mean_iou', 'reduce_all', 'reduce_any',
    'attention_lstm', 'fusion_seqexpand_concat_fc', 'dequantize_abs_max',
    'clip_by_norm', 'diag', 'yolo_box', 'adam', 'fusion_gru',
    'locality_aware_nms', 'ref_by_trainer_id', 'linspace', 'box_clip',
    'similarity_focus', 'detection_map', 'sequence_mask', 'coalesce_tensor',
    'arg_min', 'arg_max', 'split_ids', 'adagrad', 'fill', 'argsort',
    'dequantize', 'merge_ids', 'fused_fc_elementwise_layernorm',
    'retinanet_target_assign', 'rpn_target_assign', 'requantize',
    'distribute_fpn_proposals', 'auc', 'quantize', 'positive_negative_pair',
    'hash', 'less_than', 'not_equal', 'eye', 'chunk_eval', 'is_empty',
    'proximal_gd', 'collect_fpn_proposals', 'unique_with_counts', 'seed'
]
