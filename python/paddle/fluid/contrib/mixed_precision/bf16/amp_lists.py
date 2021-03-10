#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import copy

__all__ = ["AutoMixedPrecisionListsBF16"]


class AutoMixedPrecisionListsBF16(object):
    """
    AutoMixedPrecisionListsBF16 is a class for fp32/bf16 list. It can update
    pre-defined fp32 list and bf16 list according to users' custom fp32
    bf16 lists. The lists are used for an algorithm which determines op's
    execution mode (fp32 or bf16).

    Args:
        custom_bf16_list (set): Users' custom bf16 list.
        custom_fp32_list (set): Users' custom fp32 list.
        custom_fp32_varnames (set): Users' custom fp32 variables' names.
    """

    def __init__(self,
                 custom_bf16_list=None,
                 custom_fp32_list=None,
                 custom_fp32_varnames=None):
        self._custom_bf16_list = custom_bf16_list
        self._custom_fp32_list = custom_fp32_list
        self.bf16_list = copy.copy(bf16_list)
        self.fp32_list = copy.copy(fp32_list)
        self.gray_list = copy.copy(gray_list)
        self.unsupported_list = copy.copy(unsupported_list)
        self.fp32_varnames = copy.copy(custom_fp32_varnames)
        self._update_list()

    def _update_list(self):
        """
        Update fp32 and bf16 list according to users' custom list.
        """
        if self._custom_bf16_list and self._custom_fp32_list:
            for op_name in self._custom_bf16_list:
                if op_name in self._custom_fp32_list:
                    raise ValueError("Custom bf16 list overlap "
                                     "custom fp32 list")
        if self._custom_bf16_list:
            for op_name in self._custom_bf16_list:
                if op_name in self.fp32_list:
                    self.fp32_list.remove(op_name)
                elif op_name in self.gray_list:
                    self.gray_list.remove(op_name)
                self.bf16_list.add(op_name)
        if self._custom_fp32_list:
            for op_name in self._custom_fp32_list:
                if op_name in self.bf16_list:
                    self.bf16_list.remove(op_name)
                elif op_name in self.gray_list:
                    self.gray_list.remove(op_name)
                self.fp32_list.add(op_name)
                self.unsupported_list.add(op_name)


# always bf16
bf16_list = {'elementwise_add', }

# depends on the prev_op type
gray_list = {'reshape2', }

# always fp32
fp32_list = {
    'conv2d',
    'matmul',
    'matmul_v2',
    'mul',
    'exp',
    'square',
    'log',
    'mean',
    'sum',
    'cos_sim',
    'softmax',
    'softmax_with_cross_entropy',
    'sigmoid_cross_entropy_with_logits',
    'cross_entropy',
    'cross_entropy2',
    'lookup_table',
    'lookup_table_v2',
    # 'elementwise_add',
    'elementwise_sub',
    'elementwise_mul',
    'elementwise_div',
    'elementwise_max',
    'elementwise_min',
    'elementwise_pow',
    'elementwise_mod',
    'elementwise_floordiv',
    'batch_norm',
    'layer_norm',
    'tanh',
    'sigmoid',
    'top_k',
    'pool2d',
    'pool3d',
    'dropout',
    'relu',
    'relu6',
    'leaky_relu',
    'soft_relu',
    'flatten2',
    'stack',
    'unstack',
    'uniform_random',
    'uniform_random_batch_size_like',
    'gaussian_random',
    'gaussian_random_batch_size_like',
    'slice',
    'rank',
    'scale',
    'transpose2',
    # 'reshape2',
    'gather',
    'fill_constant',
    'get_tensor_from_selected_rows',
    'sign',
    'cast',
    'fused_bn_add_activation',
}

# The set of ops that don't support bf16 calculation
unsupported_list = {
    # from python/paddle/fluid/layers/io.py
    'send',
    'send_barrier',
    'recv',
    'fetch_barrier',
    'create_py_reader',
    'create_double_buffer_reader',
    'read',
    'load',

    # from python/paddle/fluid/control_flow.py
    'increment',
    'less_than',
    'less_equal',
    'greater_than',
    'greater_equal',
    'equal',
    'not_equal',
    'read_from_array',
    'shrink_rnn_memory',
    'lod_array_length',
    'logical_and',
    'logical_or',
    'logical_xor',
    'logical_not',
    'print',
    'conditional_block',
    'while',
    'ifelse',
    'is_empty',
    'lstm',
    'cudnn_lstm',
    'lstmp',
    'gru',
    'gru_unit',
    'linear_chain_crf',
    'crf_decoding',
    'bpr_loss',
    'chunk_eval',
    'sequence_conv',
    'sequence_softmax',
    # Depthwise conv2d isn't fast and safe currently.
    # ref: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/grappler/optimizers/auto_mixed_precision_lists.h#L79
    'depthwise_conv2d',
    # Tensor Core kernels are not available for 3D convolutions currently.
    'conv3d',
    'sequence_pool',
    'sequence_concat',
    'sequence_slice',
    'data_norm',
    'group_norm',
    'spectral_norm',
    'depthwise_conv2d_transpose',
    'sequence_expand',
    'conv_transposed2d',
    'conv_transposed3d',
    'sequence_expand_as',
    'sequence_pad',
    'sequence_unpad',
    'sequence_erase',
    'beam_search',
    'beam_search_decode',
    'lstm_unit',
    'reduce_sum',
    'reduce_mean',
    'reduce_max',
    'reduce_min',
    'reduce_prod',
    'reduce_all',
    'reduce_any',
    'split',
    'edit_distance',
    'ctc_align',
    'warpctc',
    'sequence_reshape',
    'nce',
    'hierarchical_sigmoid',
    'im2sequence',
    'row_conv',
    'multiplex',
    'sample_logits',
    'one_hot',
    'smooth_l1_loss',
    'squeeze2',
    'unsqueeze2',
    'lod_reset',
    'lrn',
    'pad',
    'pad_constant_like',
    'label_smooth',
    'scatter',
    'sequence_scatter',
    'random_crop',
    'mean_iou',
    'selu',
    'crop',
    'affine_grid',
    'rank_loss',
    'margin_rank_loss',
    'pad2d',
    'elu',
    'pow',
    'stanh',
    'hard_sigmoid',
    'swish',
    'prelu',
    'brelu',
    'sequence_enumerate',
    'sequence_mask',
    'expand',
    'sampling_id',
    'maxout',
    'space_to_depth',
    'sequence_reverse',
    'similarity_focus',
    'hash',
    'grid_sampler',
    'log_loss',
    'teacher_student_sigmoid_loss',
    'add_position_encoding',
    'bilinear_tensor_product',
    'shuffle_channel',
    'temporal_shift',
    'psroi_pool',
    'huber_loss',
    'kldiv_loss',
    'tree_conv',
    'pixel_shuffle',
    'fsp',
    'cvm',
    'affine_channel',
    'roi_pool',
    'roi_align',
    'anchor_generator',
    'generate_proposals',
    'generate_proposal_labels',
    'generate_mask_labels',
    'lookup_table',
    'lookup_table_v2',
}
