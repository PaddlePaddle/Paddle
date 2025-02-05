#  Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

from utils import compare_legacy_with_pt

import paddle


class TestSparseEmbeddingAPIError(unittest.TestCase):
    @compare_legacy_with_pt
    def test_errors(self):
        with paddle.base.dygraph.guard():
            # The size of input in sparse_embedding should not be 0.
            def test_0_size():
                input = paddle.to_tensor([], dtype='int64')
                paddle.static.nn.sparse_embedding(
                    input,
                    [2097152, 2097152, 2097152, 2097152],
                    padding_idx=2097152,
                )

            self.assertRaises(ValueError, test_0_size)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
