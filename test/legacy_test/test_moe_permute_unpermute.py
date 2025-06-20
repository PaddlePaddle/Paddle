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
from paddle.nn.functional import moe_permute, moe_unpermute


def fabricate_dispatch_result(
    seqlen,
    token_length,
    topk,
    num_experts,
    data_type="bfloat16",
    broadcast_ratio=0.5,
):
    """Helper function to generate test data."""
    hidden_states = paddle.randn([seqlen, token_length], dtype=data_type)

    scale = paddle.empty([0])
    if data_type == "float8_e4m3fn":
        scale_cols = (token_length + 127) // 128
        scale = paddle.randn([seqlen, scale_cols], dtype="float32")

    # Calculate expert counts with normal distribution
    expected_experts = max(1, min(broadcast_ratio * num_experts, topk))
    std_dev = max(1, expected_experts / 6)
    experts_count = paddle.normal(expected_experts, std_dev, [seqlen])
    experts_count = paddle.clip(
        paddle.round(experts_count), 1, min(topk, num_experts)
    )
    experts_count = paddle.cast(experts_count, "int32")

    # Preallocate results
    expert_routemap_topk = paddle.full([seqlen, topk], -1, dtype="int32")
    expert_prob_topk = paddle.zeros([seqlen, topk], dtype="float32")

    # Batch generate expert indices and probabilities
    for i in range(seqlen):
        count = experts_count[i].item()
        indices = paddle.randperm(num_experts)[:count]
        expert_routemap_topk[i, :count] = indices
        prob_value = 1.0 / count
        expert_prob_topk[i, :count] = paddle.full(
            [count], prob_value, dtype=data_type
        )

    # Calculate expert token counts
    valid_indices = expert_routemap_topk.reshape([-1])
    valid_mask = valid_indices >= 0
    valid_experts = valid_indices[valid_mask]
    tokens_per_expert = paddle.histogram(
        valid_experts, bins=num_experts, min=0, max=num_experts - 1
    )
    tokens_per_expert = paddle.cast(tokens_per_expert, "int32")
    tokens_per_expert = list(tokens_per_expert)

    return (
        hidden_states,
        scale,
        expert_routemap_topk,
        expert_prob_topk,
        tokens_per_expert,
    )


def tensor_max_abs_rel_err(a, b, eps=1e-8):
    """Calculate max absolute and relative error between two tensors."""
    max_abs_err = paddle.max(paddle.abs(a - b))
    denom = paddle.maximum(paddle.abs(a), paddle.abs(b))
    denom = paddle.maximum(denom, paddle.to_tensor(eps, dtype=denom.dtype))
    max_rel_err = paddle.max(paddle.abs(a - b) / denom)
    return max_abs_err, max_rel_err


class TestFusedMoePermuteUnpermute(unittest.TestCase):
    """Test cases for moe_permute and moe_unpermute."""

    SEQLEN = 16384
    TOKEN_LEN = 7168
    DTYPES = ["bfloat16"]
    EXPERT_NUMS = [4, 8]
    TOPKS = [4, 8]

    def setUp(self):
        """Initialize test environment."""
        paddle.seed(42)  # For reproducibility

    def test_permute_unpermute_consistency(self):
        """Test that permute + unpermute recovers original tensors."""
        for dt, expert_num, topk in itertools.product(
            self.DTYPES, self.EXPERT_NUMS, self.TOPKS
        ):
            with self.subTest(dtype=dt, expert_num=expert_num, topk=topk):
                (
                    hidden_states,
                    scale,
                    expert_routemap_topk,
                    expert_prob_topk,
                    tokens_per_expert,
                ) = fabricate_dispatch_result(
                    self.SEQLEN,
                    self.TOKEN_LEN,
                    topk,
                    expert_num,
                    data_type=dt,
                    broadcast_ratio=0.5,
                )
                if dt == "bfloat16":
                    scale = None

                # Permute step
                (
                    unzipped_tokens,
                    zipped_expertwise_rowmap,
                    unzipped_probs,
                    unzipped_scales,
                ) = moe_permute(
                    hidden_states,
                    scale,
                    expert_routemap_topk,
                    expert_prob_topk,
                    num_experts=expert_num,
                    tokens_per_expert=tokens_per_expert,
                    padding_alignment=128,
                )

                # Unpermute step
                unzipped_tokens_recovered, expert_prob_topk_recovered = (
                    moe_unpermute(
                        (unzipped_tokens * unzipped_probs.unsqueeze(-1)).astype(
                            "bfloat16"
                        ),
                        zipped_expertwise_rowmap,
                        expert_routemap_topk,
                        unzipped_probs,
                        total_zipped_tokens=self.SEQLEN,
                        num_experts=expert_num,
                    )
                )

                # Check tensor recovery
                max_abs_err, max_rel_err = tensor_max_abs_rel_err(
                    hidden_states, unzipped_tokens_recovered
                )
                self.assertLess(
                    max_rel_err,
                    1e-2,
                    f"Tokens relative error too large, permute-unpermute tokens max relative error: {max_rel_err}",
                )

                np.testing.assert_equal(
                    expert_prob_topk._md5sum(),
                    expert_prob_topk_recovered._md5sum(),
                    err_msg="moe_permute_unpermute probs do not match",
                )


if __name__ == "__main__":
    unittest.main()
