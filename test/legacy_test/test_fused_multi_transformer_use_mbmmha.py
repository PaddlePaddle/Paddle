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

import os

import numpy as np

os.environ["FLAGS_mmha_use_flash_decoding"] = "true"
import random
import unittest

from test_fused_multi_transformer_op import TestFusedMultiTransformerOp
from test_sparse_attention_op import get_cuda_version

import paddle
from paddle.base.framework import default_main_program

seed = 42
random.seed(seed)
default_main_program().random_seed = seed
np.random.seed(seed)
paddle.seed(seed)


@unittest.skipIf(
    not paddle.is_compiled_with_cuda()
    or get_cuda_version() < 11030
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "FusedMultiTransformer requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class TestFusedMultiTransformerOpUseMBMMHA(TestFusedMultiTransformerOp):
    def config(self):
        super().config()
        self.has_cache_kv = True
        self.gen_cache_kv = False
        self.remove_padding = True
        self.query_length = 1
        self.key_length, self.value_length = 1, 1
        self.cache_length = 2049
        self.layers = 2
        self.rotary_emb_dims = 2
        self.x_type = np.float16


if __name__ == "__main__":
    unittest.main()
