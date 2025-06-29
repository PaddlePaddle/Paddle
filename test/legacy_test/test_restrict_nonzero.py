# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np

import paddle
from paddle.base import core


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestRestrictNonzero(unittest.TestCase):
    def test_restrict_nonzero(self):
        # test dynamic
        paddle.disable_static()
        x = paddle.to_tensor(
            [[-1, 2, 3, -1], [0, 1, 2, -1], [0, -1, 1, -1]]
        ).flatten()
        num_tokens_per_expert_list = [2, 2, 2, 1]
        ref_out = (x + (x == -1).cast('int64') * 256).argsort()[:7]
        out = paddle.concat(
            [
                paddle.tensor.search._restrict_nonzero(x == i, total_true_num)
                for i, total_true_num in enumerate(num_tokens_per_expert_list)
            ]
        ).flatten()
        np.testing.assert_equal(actual=out.numpy(), desired=ref_out.numpy())


if __name__ == '__main__':
    unittest.main()
