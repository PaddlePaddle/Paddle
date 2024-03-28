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

import unittest

from c_embedding_op_base import (
    TestCEmbeddingCPU,
    TestCEmbeddingOpBase,
    TestCEmbeddingOpComplex64,
    TestCEmbeddingOpComplex128,
    TestCEmbeddingOpFP32,
)

TestCEmbeddingCPU()

TestCEmbeddingOpBase()

TestCEmbeddingOpFP32()

TestCEmbeddingOpComplex64()

TestCEmbeddingOpComplex128()

if __name__ == "__main__":
    unittest.main()
