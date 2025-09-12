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

import itertools
import unittest

import numpy as np

import paddle
from paddle.incubate.nn.functional.moe_gate_dispatch_and_quant import (
    math_moe_gate_dispatch_and_quant,
    moe_gate_dispatch_and_quant,
)


class TestMoeOpsFP8(unittest.TestCase):
    def setUp(self):
        paddle.seed(42)

    def single_test(self, seq_len, expert_num, moe_k, cap):
        capacity = int(cap * seq_len // expert_num)

        hidden_sizes = [256, 512, 640, 2048]
        use_pad_options = [True, False]
        use_pow2_scale_options = [True, False]

        for hidden_size, use_pad, use_pow2_scale in itertools.product(
            hidden_sizes, use_pad_options, use_pow2_scale_options
        ):
            x = paddle.randn([seq_len, hidden_size], dtype="bfloat16")
            gate_logtis = paddle.randn([seq_len, expert_num], dtype="float32")

            (
                out_fp8,
                scale,
                combine_weights,
                scatter_index,
                expert_offset,
                expert_id,
            ) = moe_gate_dispatch_and_quant(
                x,
                gate_logtis,
                corr_bias=None,
                k=moe_k,
                capacity=capacity,
                use_pad=use_pad,
                use_pow2_scale=use_pow2_scale,
            )

            (
                out_fp8_ref,
                scale_ref,
                combine_weights_ref,
                scatter_index_ref,
                expert_offset_ref,
                expert_id_ref,
            ) = math_moe_gate_dispatch_and_quant(
                x,
                gate_logtis,
                corr_bias=None,
                k=moe_k,
                capacity=capacity,
                use_pad=use_pad,
                use_pow2_scale=use_pow2_scale,
            )

            np.testing.assert_equal(
                combine_weights._md5sum(), combine_weights_ref._md5sum()
            )
            np.testing.assert_equal(
                scatter_index._md5sum(), scatter_index_ref._md5sum()
            )
            np.testing.assert_equal(
                expert_offset._md5sum(), expert_offset_ref._md5sum()
            )
            np.testing.assert_equal(
                expert_id._md5sum(), expert_id_ref._md5sum()
            )

            np.testing.assert_equal(scale.shape, scale_ref.shape)
            np.testing.assert_equal(out_fp8.shape, out_fp8_ref.shape)

            np.testing.assert_equal(scale._md5sum(), scale_ref._md5sum())
            np.testing.assert_equal(
                out_fp8.astype("float32")._md5sum(),
                out_fp8_ref.astype("float32")._md5sum(),
            )

    def test_moe_gate_dispatch_and_quant(self):
        self.single_test(seq_len=4096, expert_num=1, moe_k=1, cap=1)
        self.single_test(seq_len=4096, expert_num=64, moe_k=8, cap=8)
        self.single_test(seq_len=128, expert_num=16, moe_k=8, cap=8)


if __name__ == "__main__":
    unittest.main()
