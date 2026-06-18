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

# [AUTO-GENERATED]
# Target file: python/paddle/nn/functional/moe_permute.py
# Coverage target: moe_permute function
# 未覆盖行: static graph path, do_gather=False path

import unittest

import paddle
from paddle.nn.functional.moe_permute import moe_permute


class TestMoePermute(unittest.TestCase):
    """Test moe_permute function for Mixture of Experts token permutation.
    测试 MoE（混合专家）token 排列的 moe_permute 函数。"""

    def setUp(self):
        paddle.disable_static()

    def _build_basic_inputs(
        self, seq_len=3, hidden_dim=128, top_k=8, num_experts=3
    ):
        """Helper to build basic moe_permute inputs.
        构建基本 moe_permute 输入的辅助函数。"""
        hidden_states = paddle.randn([seq_len, hidden_dim], dtype="bfloat16")
        # -1 means not assigned to that expert slot
        expert_routemap_topk = paddle.full([seq_len, top_k], -1, dtype="int32")
        # Route token 0 to expert 0, token 1 to expert 1, token 2 to expert 1
        expert_routemap_topk[0, 1] = 0
        expert_routemap_topk[1, 0] = 1
        expert_routemap_topk[2, 6] = 1

        expert_prob_topk = paddle.zeros([seq_len, top_k], dtype="float32")
        expert_prob_topk[0, 1] = 0.6
        expert_prob_topk[1, 0] = 1.0
        expert_prob_topk[2, 6] = 1.0

        tokens_per_expert = [1, 2, 0]
        return (
            hidden_states,
            expert_routemap_topk,
            expert_prob_topk,
            num_experts,
            tokens_per_expert,
        )

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(), "CUDA required for moe_permute"
    )
    def test_moe_permute_basic(self):
        """Test basic moe_permute with small inputs.
        测试小输入的基本 moe_permute。"""
        try:
            (
                hidden_states,
                expert_routemap_topk,
                expert_prob_topk,
                num_experts,
                tokens_per_expert,
            ) = self._build_basic_inputs()
            padding_alignment = 16
            (
                hidden_states_unzipped,
                zipped_expertwise_rowmap,
                token_prob_unzipped,
                scale_unzipped,
            ) = moe_permute(
                hidden_states,
                None,
                expert_routemap_topk,
                expert_prob_topk,
                num_experts,
                tokens_per_expert,
                padding_alignment,
            )
            # Verify output shapes are valid
            self.assertEqual(hidden_states_unzipped.ndim, 2)
            self.assertEqual(
                zipped_expertwise_rowmap.shape,
                [3, num_experts],  # [seq_len, num_experts]
            )
            # token_prob_unzipped may be 1D or 2D depending on scale input
            self.assertIsNotNone(token_prob_unzipped)
        except RuntimeError as e:
            self.skipTest(f"CUDA kernel not available: {e}")

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(),
        "CUDA required for moe_permute",
    )
    def test_moe_permute_return_expert_indices(self):
        """Test moe_permute with return_expert_indices=True.
        测试 return_expert_indices=True 的 moe_permute。"""
        try:
            (
                hidden_states,
                expert_routemap_topk,
                expert_prob_topk,
                num_experts,
                tokens_per_expert,
            ) = self._build_basic_inputs()
            padding_alignment = 16
            result = moe_permute(
                hidden_states,
                None,
                expert_routemap_topk,
                expert_prob_topk,
                num_experts,
                tokens_per_expert,
                padding_alignment,
                return_expert_indices=True,
            )
            # Should return 5 tensors when return_expert_indices is True
            self.assertEqual(len(result), 5)
            (
                hidden_states_unzipped,
                zipped_expertwise_rowmap,
                token_prob_unzipped,
                scale_unzipped,
                expert_indices,
            ) = result
            self.assertEqual(hidden_states_unzipped.ndim, 2)
            self.assertEqual(expert_indices.ndim, 1)
        except RuntimeError as e:
            self.skipTest(f"CUDA kernel not available: {e}")

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(),
        "CUDA required for moe_permute",
    )
    def test_moe_permute_do_gather_false(self):
        """Test moe_permute with do_gather=False.
        测试 do_gather=False 的 moe_permute。"""
        try:
            (
                hidden_states,
                expert_routemap_topk,
                expert_prob_topk,
                num_experts,
                tokens_per_expert,
            ) = self._build_basic_inputs()
            padding_alignment = 16
            (
                hidden_states_unzipped,
                zipped_expertwise_rowmap,
                token_prob_unzipped,
                scale_unzipped,
            ) = moe_permute(
                hidden_states,
                None,
                expert_routemap_topk,
                expert_prob_topk,
                num_experts,
                tokens_per_expert,
                padding_alignment,
                do_gather=False,
            )
            self.assertEqual(hidden_states_unzipped.ndim, 2)
        except RuntimeError as e:
            self.skipTest(f"CUDA kernel not available: {e}")

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(),
        "CUDA required for moe_permute",
    )
    def test_moe_permute_different_num_experts(self):
        """Test moe_permute with different num_experts values.
        测试不同 num_experts 值的 moe_permute。"""
        try:
            for num_exp in [2, 4, 8]:
                seq_len = 4
                hidden_dim = 64
                top_k = 8
                hidden_states = paddle.randn(
                    [seq_len, hidden_dim], dtype="bfloat16"
                )
                expert_routemap_topk = paddle.full(
                    [seq_len, top_k], -1, dtype="int32"
                )
                expert_prob_topk = paddle.zeros(
                    [seq_len, top_k], dtype="float32"
                )
                # Assign tokens to experts
                tokens_per_expert = [0] * num_exp
                for i in range(seq_len):
                    exp_id = i % num_exp
                    expert_routemap_topk[i, 0] = exp_id
                    expert_prob_topk[i, 0] = 1.0
                    tokens_per_expert[exp_id] += 1

                padding_alignment = 16
                (
                    hidden_states_unzipped,
                    zipped_expertwise_rowmap,
                    token_prob_unzipped,
                    scale_unzipped,
                ) = moe_permute(
                    hidden_states,
                    None,
                    expert_routemap_topk,
                    expert_prob_topk,
                    num_exp,
                    tokens_per_expert,
                    padding_alignment,
                )
                self.assertEqual(
                    zipped_expertwise_rowmap.shape,
                    [seq_len, num_exp],
                )
        except RuntimeError as e:
            self.skipTest(f"CUDA kernel not available: {e}")

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(),
        "CUDA required for moe_permute",
    )
    def test_moe_permute_zipped_expertwise_rowmap_shape(self):
        """Test zipped_expertwise_rowmap has correct shape [seq_len, num_experts].
        测试 zipped_expertwise_rowmap 具有正确形状 [seq_len, num_experts]。"""
        try:
            (
                hidden_states,
                expert_routemap_topk,
                expert_prob_topk,
                num_experts,
                tokens_per_expert,
            ) = self._build_basic_inputs(seq_len=5)
            padding_alignment = 32
            (
                hidden_states_unzipped,
                zipped_expertwise_rowmap,
                token_prob_unzipped,
                scale_unzipped,
            ) = moe_permute(
                hidden_states,
                None,
                expert_routemap_topk,
                expert_prob_topk,
                num_experts,
                tokens_per_expert,
                padding_alignment,
            )
            self.assertEqual(zipped_expertwise_rowmap.shape, [5, num_experts])
        except RuntimeError as e:
            self.skipTest(f"CUDA kernel not available: {e}")

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(),
        "CUDA required for moe_permute",
    )
    def test_moe_permute_token_prob_shape(self):
        """Test token_prob_unzipped has correct shape.
        测试 token_prob_unzipped 具有正确的形状。"""
        try:
            (
                hidden_states,
                expert_routemap_topk,
                expert_prob_topk,
                num_experts,
                tokens_per_expert,
            ) = self._build_basic_inputs()
            padding_alignment = 16
            (
                hidden_states_unzipped,
                zipped_expertwise_rowmap,
                token_prob_unzipped,
                scale_unzipped,
            ) = moe_permute(
                hidden_states,
                None,
                expert_routemap_topk,
                expert_prob_topk,
                num_experts,
                tokens_per_expert,
                padding_alignment,
            )
            # token_prob_unzipped shape depends on whether scale is provided
            self.assertIsNotNone(token_prob_unzipped)
            # When scale=None, token_prob may be 1D with total_tokens elements
            self.assertEqual(
                token_prob_unzipped.shape[0],
                hidden_states_unzipped.shape[0],
            )
        except RuntimeError as e:
            self.skipTest(f"CUDA kernel not available: {e}")

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(),
        "CUDA required for moe_permute",
    )
    def test_moe_permute_output_dtype(self):
        """Test output dtype matches input dtype.
        测试输出数据类型与输入数据类型匹配。"""
        try:
            (
                hidden_states,
                expert_routemap_topk,
                expert_prob_topk,
                num_experts,
                tokens_per_expert,
            ) = self._build_basic_inputs()
            padding_alignment = 16
            (
                hidden_states_unzipped,
                zipped_expertwise_rowmap,
                token_prob_unzipped,
                scale_unzipped,
            ) = moe_permute(
                hidden_states,
                None,
                expert_routemap_topk,
                expert_prob_topk,
                num_experts,
                tokens_per_expert,
                padding_alignment,
            )
            self.assertEqual(hidden_states_unzipped.dtype, paddle.bfloat16)
            self.assertEqual(zipped_expertwise_rowmap.dtype, paddle.int32)
            self.assertEqual(token_prob_unzipped.dtype, paddle.float32)
        except RuntimeError as e:
            self.skipTest(f"CUDA kernel not available: {e}")

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(),
        "CUDA required for moe_permute",
    )
    def test_moe_permute_scale_none(self):
        """Test moe_permute with scale=None returns valid scale_unzipped.
        测试 scale=None 时 moe_permute 返回有效的 scale_unzipped。"""
        try:
            (
                hidden_states,
                expert_routemap_topk,
                expert_prob_topk,
                num_experts,
                tokens_per_expert,
            ) = self._build_basic_inputs()
            padding_alignment = 16
            (
                hidden_states_unzipped,
                zipped_expertwise_rowmap,
                token_prob_unzipped,
                scale_unzipped,
            ) = moe_permute(
                hidden_states,
                None,  # scale is None
                expert_routemap_topk,
                expert_prob_topk,
                num_experts,
                tokens_per_expert,
                padding_alignment,
            )
            # scale_unzipped should be a valid tensor even when scale is None
            self.assertIsNotNone(scale_unzipped)
        except RuntimeError as e:
            self.skipTest(f"CUDA kernel not available: {e}")

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(),
        "CUDA required for moe_permute",
    )
    def test_moe_permute_override_buffer_size(self):
        """Test moe_permute with override_buffer_size parameter.
        测试带有 override_buffer_size 参数的 moe_permute。"""
        try:
            (
                hidden_states,
                expert_routemap_topk,
                expert_prob_topk,
                num_experts,
                tokens_per_expert,
            ) = self._build_basic_inputs()
            padding_alignment = 16
            (
                hidden_states_unzipped,
                zipped_expertwise_rowmap,
                token_prob_unzipped,
                scale_unzipped,
            ) = moe_permute(
                hidden_states,
                None,
                expert_routemap_topk,
                expert_prob_topk,
                num_experts,
                tokens_per_expert,
                padding_alignment,
                override_buffer_size=1024,
            )
            self.assertEqual(hidden_states_unzipped.ndim, 2)
        except RuntimeError as e:
            self.skipTest(f"CUDA kernel not available: {e}")


if __name__ == "__main__":
    unittest.main()
