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

        batch_size = 128
        qo_seqlen = np.full((batch_size, ), 128, dtype=np.int32)
        kv_seqlen = np.full((batch_size, ), 128, dtype=np.int32)

        max_seqlen = 128
        lo_windows = np.full((max_seqlen, ), 0, dtype=np.int32)
        high_windows = np.full((max_seqlen, ), max_seqlen, dtype=np.int32)

        qo_kv_seqlen = np.concatenate((qo_seqlen, kv_seqlen))
        low_high_windows = np.concatenate((lo_windows, high_windows))

        self.inputs = {
            'qo_kv_seqlen': qo_kv_seqlen,
            'low_high_windows': low_high_windows
        }

        self.outputs = {
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
