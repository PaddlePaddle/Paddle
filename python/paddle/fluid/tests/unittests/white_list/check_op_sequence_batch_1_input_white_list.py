#!/usr/bin/env python
from __future__ import print_function
import sys

NEED_TO_FIX_OP_LIST = [
    'sequence_concat',
    'sequence_conv',
    'sequence_enumerate',
    'sequence_erase',
    'sequence_expand_as',
    'sequence_expand',
    'sequence_mask',
    'sequence_pad',
    # 'sequence_pool',
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
