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

import unittest

import numpy as np
from op_test import get_device_place, is_custom_device

import paddle
from paddle.base import core
from paddle.incubate.nn.functional import (
    variable_length_memory_efficient_attention,
)


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device())
    or core.is_compiled_with_rocm(),
    "variable_length_memory_efficient_attention zero-size regressions require CUDA",
)
class TestVariableLengthMemoryEfficientAttentionZeroSize(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.place = get_device_place()

    def tearDown(self):
        paddle.enable_static()

    def test_zero_effective_sequence_length_returns_zero_output(self):
        query = paddle.to_tensor(
            np.arange(1 * 1 * 31 * 64, dtype=np.float16).reshape(
                [1, 1, 31, 64]
            ),
            place=self.place,
        )
        key = paddle.to_tensor(
            np.ones([1, 1, 31, 64], dtype=np.float16), place=self.place
        )
        value = paddle.to_tensor(
            np.full([1, 1, 31, 64], 2.0, dtype=np.float16), place=self.place
        )
        mask = paddle.to_tensor(
            np.zeros([1, 1, 50, 50], dtype=np.float16), place=self.place
        )
        seq_lens = paddle.to_tensor([0, 1], dtype="int32", place=self.place)
        kv_seq_lens = paddle.to_tensor([1, 1], dtype="int32", place=self.place)

        out = variable_length_memory_efficient_attention(
            query,
            key,
            value,
            seq_lens,
            kv_seq_lens,
            mask=mask,
            scale=0.125,
        )

        self.assertEqual(list(out.shape), [1, 1, 31, 64])
        np.testing.assert_array_equal(
            out.numpy(), np.zeros([1, 1, 31, 64], dtype=np.float16)
        )


if __name__ == "__main__":
    unittest.main()
