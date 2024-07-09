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

# If the output after infershape() is a lod_tensor, commenly its lod_level
# should be equal during compile time and run time.
# For ops in this whitelist, the equality check of lod_level between
# compiletime&runtime will be skipped. Ops in this whitelist need to declare
# reasons for skipping compile_vs_runtime test or be fixed later.

COMPILE_RUN_OP_WHITE_LIST = [
    'sequence_pool',
    'sequence_slice',
    'generate_proposals',
    'retinanet_detection_output',
    'ctc_align',
    'fusion_seqpool_concat',
    'fusion_seqpool_cvm_concat',
    'gru',
    'rpn_target_assign',
    'retinanet_target_assign',
    'im2sequence',
    'generate_proposal_labels',
    'detection_map',
    'var_conv_2d',
]
