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
from op_test import get_cuda_version, get_device_place, is_custom_device

import paddle
from paddle.base import core
from paddle.nn.functional.flash_attention import flashmask_attention

is_sm8x = (
    (core.is_compiled_with_cuda() or is_custom_device())
    and paddle.device.cuda.get_device_capability()[0] == 8
    and paddle.device.cuda.get_device_capability()[1] >= 0
)

is_sm90 = (
    (core.is_compiled_with_cuda() or is_custom_device())
    and paddle.device.cuda.get_device_capability()[0] == 9
    and paddle.device.cuda.get_device_capability()[1] == 0
)


def is_flashattn_supported():
    return (
        (core.is_compiled_with_cuda() or is_custom_device())
        and not core.is_compiled_with_rocm()
        and get_cuda_version() >= 11040
        and (is_sm8x or is_sm90)
    )


@unittest.skipIf(
    not is_flashattn_supported(),
    "flashmask zero-size regressions require supported CUDA flash attention",
)
class TestFlashMaskAttentionZeroSize(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.place = get_device_place()
        self.seq_len = 16
        self.num_heads = 8
        self.head_dim = 96
        self.startend_row_indices = paddle.to_tensor(
            np.zeros([1, 1, self.seq_len, 1], dtype=np.int32),
            place=self.place,
        )

    def tearDown(self):
        paddle.enable_static()

    def test_q_zero_heads_returns_empty_output(self):
        out = flashmask_attention(
            paddle.to_tensor(
                np.zeros([1, self.seq_len, 0, self.head_dim], dtype=np.float16),
                place=self.place,
            ),
            paddle.to_tensor(
                np.zeros(
                    [1, self.seq_len, self.num_heads, self.head_dim],
                    dtype=np.float16,
                ),
                place=self.place,
            ),
            paddle.to_tensor(
                np.zeros(
                    [1, self.seq_len, self.num_heads, self.head_dim],
                    dtype=np.float16,
                ),
                place=self.place,
            ),
            startend_row_indices=self.startend_row_indices,
            causal=True,
        )
        self.assertEqual(list(out.shape), [1, self.seq_len, 0, self.head_dim])
        self.assertEqual(out.numel(), 0)

    def test_k_zero_heads_returns_zero_filled_output(self):
        out = flashmask_attention(
            paddle.to_tensor(
                np.arange(
                    self.seq_len * self.num_heads * self.head_dim,
                    dtype=np.float16,
                ).reshape([1, self.seq_len, self.num_heads, self.head_dim]),
                place=self.place,
            ),
            paddle.to_tensor(
                np.zeros([1, self.seq_len, 0, self.head_dim], dtype=np.float16),
                place=self.place,
            ),
            paddle.to_tensor(
                np.ones(
                    [1, self.seq_len, self.num_heads, self.head_dim],
                    dtype=np.float16,
                ),
                place=self.place,
            ),
            startend_row_indices=self.startend_row_indices,
            causal=True,
        )
        self.assertEqual(
            list(out.shape), [1, self.seq_len, self.num_heads, self.head_dim]
        )
        np.testing.assert_array_equal(
            out.numpy(),
            np.zeros(
                [1, self.seq_len, self.num_heads, self.head_dim],
                dtype=np.float16,
            ),
        )


if __name__ == "__main__":
    unittest.main()
