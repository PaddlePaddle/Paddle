#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve
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
# compiletime&runtime will be skipped. Ops in this whitelist need to declear
# reasons for skipping compile_vs_runtime test or be fixed later.

#!/usr/bin/env python
from __future__ import print_function
import sys

# For ops in this whitelist, the check of instance size is 0 input will be skipped.
# Ops in this whitelist need to be fixed later.
NEED_TO_FIX_OP_LIST = [
    'sequence_concat',
    'sequence_conv',
    'sequence_enumerate',
    'sequence_erase',
    'sequence_expand_as',
    'sequence_expand',
    'sequence_mask',
    'sequence_pad',
    'sequence_reshape',
    'sequence_reverse',
    'sequence_scatter',
    'sequence_slice',
    'sequence_softmax',
    'sequence_topk_avg_pooling',
    'sequence_unpad',
]

op_name = sys.argv[1]
print(op_name in NEED_TO_FIX_OP_LIST)
