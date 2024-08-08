# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

export FLAGS_multi_block_attention_min_partition_size=1024

export FLAGS_mmha_use_flash_decoding=true
GLOG_v=1 python test_fused_multi_transformer_op.py &> test_fused_mt.log
# export FLAGS_mmha_use_flash_decoding=false
# GLOG_v=1 python test_fused_multi_transformer_op.py &> test_fused_mt_mmha.log
