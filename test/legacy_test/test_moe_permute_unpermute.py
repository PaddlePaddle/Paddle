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
    using_ue8m0_scale=False,
):
    """Helper function to generate test data."""
    hidden_states = paddle.randn([seqlen, token_length]).astype(data_type)

    scale = paddle.empty([0])
    if data_type == "float8_e4m3fn":
        if using_ue8m0_scale:
            scale_cols = (token_length + 127) // 128
            # if using_ue8m0_scale, four ue8m0 scales will be packed into one int32
            scale_cols = (scale_cols + 3) // 4
            scale = paddle.randn([seqlen, scale_cols], dtype="float32").astype(
                paddle.int32
            )
        else:
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


def gen_golden_expert_indices(
    baseline_compact_routemap,
    permute_rows,  # = baseline_hidden_states_unzipped.shape[0]
):
    num_rows, num_experts = baseline_compact_routemap.shape
    # 初始化为 -1（padding）
    m_indices = paddle.full(
        [permute_rows],
        -1,
        dtype=baseline_compact_routemap.dtype,
    )
    # rowmap[t, e] = r  ==>  m_indices[r] = e
    for e in range(num_experts):
        rows = baseline_compact_routemap[:, e]
        valid_mask = rows >= 0
        valid_rows = rows[valid_mask]
        m_indices[valid_rows] = e
    return m_indices


# Asserting m_indices == expert_indices
def validate_expert_indices(proposed_expert_indices, compact_routemap):
    gold = gen_golden_expert_indices(
        compact_routemap, permute_rows=proposed_expert_indices.shape[0]
    )
    np.testing.assert_array_equal(proposed_expert_indices, gold)
    # check all proposed_expert_indices item is in [0, expert_num) or -1
    expert_num = compact_routemap.shape[1]
    # high performance, parallel assert using paddle vectorized operations
    valid_mask = (proposed_expert_indices == -1) | (
        (proposed_expert_indices >= 0) & (proposed_expert_indices < expert_num)
    )
    assert paddle.all(valid_mask).item(), (
        f"expert_indices contains invalid values outside range [-1, {expert_num})"
    )


class TestFusedMoePermuteUnpermute(unittest.TestCase):
    """Test cases for moe_permute and moe_unpermute."""

    SEQLEN = [5000, 16384]
    TOKEN_LEN = 7168
    DTYPES = ["float8_e4m3fn", "bfloat16"]
    EXPERT_NUMS = [4, 8, 16, 32, 64]
    TOPKS = [4, 8, 16]

    SEQLEN_FOR_INFERENCE = [1024, 1025]
    TOKEN_LEN_FOR_INFERENCE = [1536]
    TOPKS_FOR_INFERENCE = [16]
    EXPERT_NUMS_FOR_INFERENCE = [64, 128, 160, 384]

    def setUp(self):
        """Initialize test environment."""
        paddle.seed(42)  # For reproducibility

    def test_permute_unpermute_consistency(self):
        """Test that permute + unpermute recovers original tensors."""
        for seq_len, dt, expert_num, topk in itertools.product(
            self.SEQLEN, self.DTYPES, self.EXPERT_NUMS, self.TOPKS
        ):
            with self.subTest(
                seq_len=seq_len, dtype=dt, expert_num=expert_num, topk=topk
            ):
                (
                    hidden_states,
                    scale,
                    expert_routemap_topk,
                    expert_prob_topk,
                    tokens_per_expert,
                ) = fabricate_dispatch_result(
                    seq_len,
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
                    permuted_tokens,
                    compact_routemap,
                    permuted_probs,
                    permuted_scales,
                ) = moe_permute(
                    hidden_states,
                    scale,
                    expert_routemap_topk,
                    expert_prob_topk,
                    num_experts=expert_num,
                    tokens_per_expert=tokens_per_expert,
                    padding_alignment=128,
                )
                # do_gather = False
                (
                    _,
                    compact_routemap_no_gather,
                    permuted_probs_no_gather,
                    _,
                ) = moe_permute(
                    hidden_states,
                    scale,
                    expert_routemap_topk,
                    expert_prob_topk,
                    num_experts=expert_num,
                    tokens_per_expert=tokens_per_expert,
                    padding_alignment=128,
                    do_gather=False,
                )

                unpermute_input = (
                    permuted_tokens.astype("float32")
                    * permuted_probs.unsqueeze(-1)
                ).astype("bfloat16")

                permuted_tokens_recovered, expert_prob_topk_recovered = (
                    moe_unpermute(
                        unpermute_input,
                        compact_routemap,
                        expert_routemap_topk,
                        permuted_probs,
                        total_zipped_tokens=seq_len,
                        num_experts=expert_num,
                    )
                )

                # Check tensor recovery
                max_abs_err, max_rel_err = tensor_max_abs_rel_err(
                    hidden_states.astype("float32"),
                    permuted_tokens_recovered.astype("float32"),
                )

                self.assertLess(
                    max_rel_err,
                    1e-1 if dt == "float8_e4m3fn" else 1e-2,
                    f"Tokens relative error too large, permute-unpermute tokens max relative error: {max_rel_err}",
                )

                np.testing.assert_equal(
                    expert_prob_topk._md5sum(),
                    expert_prob_topk_recovered._md5sum(),
                    err_msg="moe_permute_unpermute probs do not match",
                )

                np.testing.assert_equal(
                    compact_routemap_no_gather._md5sum(),
                    compact_routemap._md5sum(),
                    err_msg="no_gather's compact_routemap do not match",
                )
                np.testing.assert_equal(
                    permuted_probs_no_gather._md5sum(),
                    permuted_probs._md5sum(),
                    err_msg="no_gather's permuted_probs do not match",
                )

    def test_inference_specific_functions(self):
        """Test that permute + unpermute recovers original tensors."""
        for seq_len, dt, expert_num, topk in itertools.product(
            self.SEQLEN_FOR_INFERENCE,
            self.DTYPES,
            self.EXPERT_NUMS_FOR_INFERENCE,
            self.TOPKS_FOR_INFERENCE,
        ):
            with self.subTest(
                seq_len=seq_len, dtype=dt, expert_num=expert_num, topk=topk
            ):
                (
                    hidden_states,
                    scale,
                    expert_routemap_topk,
                    expert_prob_topk,
                    tokens_per_expert,
                ) = fabricate_dispatch_result(
                    seq_len,
                    self.TOKEN_LEN,
                    topk,
                    expert_num,
                    data_type=dt,
                    broadcast_ratio=0.5,
                )
                if dt == "bfloat16":
                    scale = None
                #######################################################
                # Normal Permute with weighted & non-weighted unpermute
                #######################################################
                (
                    permuted_tokens,
                    compact_routemap,
                    permuted_probs,
                    permuted_scales,
                    expert_indices,
                ) = moe_permute(
                    hidden_states,
                    scale,
                    expert_routemap_topk,
                    expert_prob_topk,
                    num_experts=expert_num,
                    tokens_per_expert=tokens_per_expert,
                    padding_alignment=128,
                    return_expert_indices=True,
                )
                validate_expert_indices(expert_indices, compact_routemap)
                unpermute_input = (
                    permuted_tokens.astype("float32")
                    * permuted_probs.unsqueeze(-1)
                ).astype("bfloat16")

                permuted_tokens_recovered, expert_prob_topk_recovered = (
                    moe_unpermute(
                        unpermute_input,
                        compact_routemap,
                        expert_routemap_topk,
                        permuted_probs,
                        total_zipped_tokens=seq_len,
                        num_experts=expert_num,
                    )
                )
                (
                    weighted_permuted_tokens_recovered,
                    weighted_expert_prob_topk_recovered,
                ) = moe_unpermute(
                    permuted_tokens.astype("bfloat16"),
                    compact_routemap,
                    expert_routemap_topk,
                    permuted_probs,
                    total_zipped_tokens=seq_len,
                    num_experts=expert_num,
                    using_weighted_combine=True,
                )
                #######################################################
                # Check tensor recovery
                #######################################################
                max_abs_err, max_rel_err = tensor_max_abs_rel_err(
                    hidden_states.astype("float32"),
                    permuted_tokens_recovered.astype("float32"),
                )

                self.assertLess(
                    max_rel_err,
                    1e-1 if dt == "float8_e4m3fn" else 1e-2,
                    f"Tokens relative error too large, permute-unpermute tokens max relative error: {max_rel_err}",
                )

                np.testing.assert_allclose(
                    weighted_permuted_tokens_recovered.numpy(),
                    permuted_tokens_recovered.numpy(),
                    atol=1e-3,
                    rtol=1e-3,
                )
                np.testing.assert_equal(
                    weighted_expert_prob_topk_recovered._md5sum(),
                    expert_prob_topk_recovered._md5sum(),
                    err_msg="moe_permute_unpermute probs do not match",
                )
                np.testing.assert_equal(
                    expert_prob_topk._md5sum(),
                    expert_prob_topk_recovered._md5sum(),
                    err_msg="moe_permute_unpermute probs do not match",
                )
                # Buffer overridden permute with weighted unpermute
                # Actual inference case
                (
                    permuted_tokens_overridden,
                    compact_routemap_overridden,
                    permuted_probs_overridden,
                    permuted_scales_overridden,
                    expert_indices_overridden,
                ) = moe_permute(
                    hidden_states,
                    scale,
                    expert_routemap_topk,
                    expert_prob_topk,
                    num_experts=expert_num,
                    tokens_per_expert=[],
                    padding_alignment=128,
                    return_expert_indices=True,
                    override_buffer_size=permuted_tokens.shape[
                        0
                    ],  # using same buffer size to do functional check
                )
                validate_expert_indices(
                    expert_indices_overridden, compact_routemap
                )
                (
                    permuted_tokens_recovered_overridden,
                    expert_prob_topk_recovered_overridden,
                ) = moe_unpermute(
                    permuted_tokens_overridden.astype("bfloat16"),
                    compact_routemap_overridden,
                    expert_routemap_topk,
                    permuted_probs_overridden,
                    total_zipped_tokens=seq_len,
                    num_experts=expert_num,
                    using_weighted_combine=True,
                )
                np.testing.assert_equal(
                    permuted_tokens_recovered_overridden._md5sum(),
                    weighted_permuted_tokens_recovered._md5sum(),
                    err_msg="weighted recovering do not match with override",
                )

                np.testing.assert_equal(
                    expert_prob_topk_recovered_overridden._md5sum(),
                    expert_prob_topk_recovered._md5sum(),
                    err_msg="moe_permute_unpermute probs do not match",
                )

    def test_permute_unpermute_consistency_for_ue8m0_scale(self):
        """Test that permute + unpermute recovers original tensors for ue8m0 scale."""
        DTYPES = ["float8_e4m3fn"]
        EXPERT_NUMS = [4, 8, 16]
        TOPKS = [4, 8, 16]
        for seq_len, dt, expert_num, topk in itertools.product(
            self.SEQLEN, DTYPES, EXPERT_NUMS, TOPKS
        ):
            with self.subTest(
                seq_len=seq_len, dtype=dt, expert_num=expert_num, topk=topk
            ):
                (
                    hidden_states,
                    scale,
                    expert_routemap_topk,
                    expert_prob_topk,
                    tokens_per_expert,
                ) = fabricate_dispatch_result(
                    seq_len,
                    self.TOKEN_LEN,
                    topk,
                    expert_num,
                    data_type=dt,
                    broadcast_ratio=0.5,
                    using_ue8m0_scale=True,
                )
                if dt == "bfloat16":
                    scale = None

                # Permute step
                (
                    permuted_tokens,
                    compact_routemap,
                    permuted_probs,
                    permuted_scales,
                ) = moe_permute(
                    hidden_states,
                    scale,
                    expert_routemap_topk,
                    expert_prob_topk,
                    num_experts=expert_num,
                    tokens_per_expert=tokens_per_expert,
                    padding_alignment=128,
                    using_ue8m0_scale=True,
                )
                # test the permuted_scales is correct or not
                compact_routemap_np = compact_routemap.numpy()
                scale_np = scale.numpy()
                permuted_scales_np = permuted_scales.numpy()
                assert compact_routemap_np.ndim == 2
                for i in range(compact_routemap_np.shape[0]):
                    valid_indices = compact_routemap_np[i][
                        compact_routemap_np[i] != -1
                    ]
                    for index in valid_indices:
                        np.testing.assert_equal(
                            scale_np[i],
                            permuted_scales_np[index],
                            err_msg="permuted_scales[{i}] is not correct",
                        )


if __name__ == "__main__":
    unittest.main()
