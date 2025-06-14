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
    tokens = paddle.randn([seqlen, token_length], dtype=data_type)

    tokens_scale = paddle.empty([0])
    if data_type == "float8_e4m3fn":
        scale_cols = (token_length + 127) // 128
        tokens_scale = paddle.randn([seqlen, scale_cols], dtype="float32")

    # Calculate expert counts with normal distribution
    expected_experts = max(1, min(broadcast_ratio * num_experts, topk))
    std_dev = max(1, expected_experts / 6)
    experts_count = paddle.normal(expected_experts, std_dev, [seqlen])
    experts_count = paddle.clip(
        paddle.round(experts_count), 1, min(topk, num_experts)
    )
    experts_count = paddle.cast(experts_count, "int32")

    # Preallocate results
    dispatched_indices = paddle.full([seqlen, topk], -1, dtype="int32")
    dispatched_probs = paddle.zeros([seqlen, topk], dtype="float32")

    # Batch generate expert indices and probabilities
    for i in range(seqlen):
        count = experts_count[i].item()
        indices = paddle.randperm(num_experts)[:count]
        dispatched_indices[i, :count] = indices
        prob_value = 1.0 / count
        dispatched_probs[i, :count] = paddle.full(
            [count], prob_value, dtype=data_type
        )

    # Calculate expert token counts
    valid_indices = dispatched_indices.reshape([-1])
    valid_mask = valid_indices >= 0
    valid_experts = valid_indices[valid_mask]
    expert_counts = paddle.histogram(
        valid_experts, bins=num_experts, min=0, max=num_experts - 1
    )
    expert_counts = paddle.cast(expert_counts, "int32")
    expert_counts = list(expert_counts)

    return (
        tokens,
        tokens_scale,
        dispatched_indices,
        dispatched_probs,
        expert_counts,
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
        for dt in self.DTYPES:
            for expert_num in self.EXPERT_NUMS:
                for topk in self.TOPKS:
                    with self.subTest(
                        dtype=dt, expert_num=expert_num, topk=topk
                    ):
                        print(
                            f"Testing with {expert_num} experts, topk {topk}, dtype {dt}"
                        )
                        (
                            tokens,
                            tokens_scale,
                            dispatched_indices,
                            dispatched_probs,
                            expert_tokens_count,
                        ) = fabricate_dispatch_result(
                            self.SEQLEN,
                            self.TOKEN_LEN,
                            topk,
                            expert_num,
                            data_type=dt,
                            broadcast_ratio=0.5,
                        )
                        if dt == "bfloat16":
                            tokens_scale = None

                        # Permute step
                        (
                            unzipped_tokens,
                            zipped_expertwise_rowmap,
                            unzipped_probs,
                            unzipped_scales,
                        ) = moe_permute(
                            tokens,
                            tokens_scale,
                            dispatched_indices,
                            dispatched_probs,
                            topk=topk,
                            num_experts=expert_num,
                            tokens_per_expert=expert_tokens_count,
                            padding_multiplex=128,
                        )

                        # Unpermute step
                        tokens_recovered, probs_recovered = moe_unpermute(
                            (
                                unzipped_tokens * unzipped_probs.unsqueeze(-1)
                            ).astype("bfloat16"),
                            zipped_expertwise_rowmap,
                            dispatched_indices,
                            unzipped_probs,
                            total_zipped_tokens=self.SEQLEN,
                            num_experts=expert_num,
                        )

                        # Check tensor recovery
                        max_abs_err, max_rel_err = tensor_max_abs_rel_err(
                            tokens, tokens_recovered
                        )
                        print(
                            f"permute-unpermute tokens relative error: {max_rel_err}"
                        )
                        self.assertLess(
                            max_rel_err, 1e-2, "Tokens relative error too large"
                        )

                        max_abs_err, max_rel_err = tensor_max_abs_rel_err(
                            dispatched_probs, probs_recovered
                        )
                        print(
                            f"ermute-unpermute probs max absolute error: {max_abs_err}, relative error: {max_rel_err}"
                        )
                        self.assertLess(
                            max_abs_err, 1e-5, "Probs absolute error too large"
                        )
                        self.assertLess(
                            max_rel_err, 1e-5, "Probs relative error too large"
                        )

    def assertLess(self, a, b, msg=None):
        """Custom assert with better error message."""
        if not a < b:
            standard_msg = f"{a} not less than {b}"
            if msg:
                standard_msg = f"{msg}: {standard_msg}"
            self.fail(standard_msg)


if __name__ == "__main__":
    unittest.main()
