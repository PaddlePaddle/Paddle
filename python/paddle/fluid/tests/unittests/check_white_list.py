#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve
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

compile_vs_runtime_white_list = [
    'lod_reset', 'sequence_pool', 'sequence_slice', 'generate_mask_labels', 'sequence_reshape',
    'generate_proposals', 'mine_hard_examples', 'retinanet_detection_output', 'ctc_align', 'fusion_seqpool_cvm_concat',
    'gru', 'sequence_erase', 'rpn_target_assign', 'filter_by_instag',
    'fusion_seqpool_concat', 'multiclass_nms', 'im2sequence', 'generate_proposal_labels',
    'distribute_fpn_proposals', 'detection_map', 'locality_aware_nms', 'multiclass_nms', 'rpn_target_assign',
    'var_conv_2d'
]
