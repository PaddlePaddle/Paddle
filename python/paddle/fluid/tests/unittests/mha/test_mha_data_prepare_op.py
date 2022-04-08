# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
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

from __future__ import print_function

import sys
import unittest
import numpy as np
import paddle

sys.path.append("..")
from op_test import OpTest, skip_check_grad_ci


class TestMHADataPrepareOp(OpTest):
    def setUp(self):
        self.op_type = "mha_data_prepare"
        self.init_dtype_type()

        batch_size = 32
        nheads = 64
        max_seqlen = 128
        min_seqlen = 100
        seqlen = np.random.randint(
            low=min_seqlen,
            high=max_seqlen + 1,
            size=(batch_size, ),
            dtype=np.int32)
        attn_mask = np.ones(
            (batch_size, nheads, max_seqlen, max_seqlen), dtype=np.int32)
        for i in range(batch_size):
            attn_mask[0, :, :, seqlen[i]:] = 0

        qo_seqlen = np.sum(attn_mask[:, 0, 0, :] == 1, axis=1, dtype='int32')
        kv_seqlen = qo_seqlen
        qo_kv_seqlen = np.concatenate((qo_seqlen, kv_seqlen))

        low_windows = np.full((max_seqlen, ), 0, dtype=np.int32)
        high_windows = np.full((max_seqlen, ), max_seqlen, dtype=np.int32)
        low_high_windows = np.concatenate((low_windows, high_windows))
        self.inputs = {'attn_mask': attn_mask}

        self.outputs = {
            'qo_kv_seqlen': qo_kv_seqlen,
            'qo_kv_seqlen_host': qo_kv_seqlen,
            'low_high_windows_host': low_high_windows
        }

    def init_dtype_type(self):
        self.dtype = np.int32
        self.atol = 1e-4

    def test_check_output(self):
        self.check_output_with_place(place=paddle.CUDAPlace(0), atol=self.atol)


if __name__ == '__main__':
    unittest.main()
