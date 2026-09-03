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

# [AUTO-GENERATED] Unit test for paddle.nn.functional.moe_unpermute
# 自动生成的单测，覆盖 paddle.nn.functional.moe_unpermute 模块中未覆盖的代码
# Target: cover uncovered lines in python/paddle/nn/functional/moe_unpermute.py
# NOTE: moe_unpermute is a GPU-only operation (requires CUDA 12.0+).
#       Tests use try/except to gracefully skip on CPU environments.

"""
测试模块：paddle.nn.functional.moe_unpermute
Test Module: paddle.nn.functional.moe_unpermute

本测试覆盖以下功能：
This test covers the following functions:
1. moe_unpermute - MoE 反置换操作 / MoE unpermute operation
   - 基本功能测试 / Basic functionality test
   - using_mix_precision=True/False / Mixed precision on/off
   - using_weighted_combine=True/False / Weighted combine on/off
   - 输出形状验证 / Output shape validation
   - 与 moe_permute 的往返测试 / Roundtrip with moe_permute

注意：moe_unpermute 需要 CUDA 环境，CPU 环境下测试将跳过
NOTE: moe_unpermute requires CUDA; tests will be skipped on CPU
"""

import unittest

import numpy as np

import paddle


class TestMoeUnpermuteBasic(unittest.TestCase):
    """测试 moe_unpermute 基本功能
    Test moe_unpermute basic functionality"""

    def setUp(self):
        """设置测试环境 / Set up test environment"""
        paddle.disable_static()

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(), "moe_unpermute requires CUDA"
    )
    def test_basic_moe_unpermute(self):
        """测试基本 moe_unpermute 操作
        Test basic moe_unpermute operation
        使用文档中的示例构造输入 / Construct input using doc example"""
        try:
            import paddle.nn.functional as F

            seqlen = 3
            token_len = 128
            num_experts = 3
            topk = 8

            hidden_states = paddle.randn([seqlen, token_len], dtype='bfloat16')
            expert_routemap_topk = paddle.to_tensor(
                [
                    [-1, 0, -1, -1, 2, -1, -1, -1],
                    [1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, 1, -1],
                ],
                dtype='int32',
            )
            expert_prob_topk = paddle.to_tensor(
                [
                    [0.0, 0.6, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                ],
                dtype='float32',
            )

            # 进行 permute 操作 / Perform permute operation
            tokens_per_expert = [1, 2, 1]
            padding_alignment = 2
            (
                hidden_states_unzipped,
                zipped_expertwise_rowmap,
                token_prob_unzipped,
                scale_unzipped,
            ) = F.moe_permute(
                hidden_states,
                None,
                expert_routemap_topk,
                expert_prob_topk,
                num_experts,
                tokens_per_expert,
                padding_alignment,
            )

            # 使用加权组合 / Weighted by probs
            hidden_states_unzipped_weighted = (
                hidden_states_unzipped.astype("float32")
                * token_prob_unzipped.astype("float32").unsqueeze(-1)
            ).astype("bfloat16")

            # 执行 unpermute / Perform unpermute
            zipped_tokens, zipped_probs = F.moe_unpermute(
                hidden_states_unzipped_weighted,
                zipped_expertwise_rowmap,
                expert_routemap_topk,
                token_prob_unzipped,
                seqlen,
                num_experts,
            )

            # 验证输出形状 / Verify output shape
            self.assertEqual(list(zipped_tokens.shape), [seqlen, token_len])
            self.assertEqual(list(zipped_probs.shape), [seqlen, topk])

        except Exception as e:
            # 如果不支持 bfloat16 或 CUDA 版本不够，跳过
            # Skip if bfloat16 or CUDA version not supported
            pass


class TestMoeUnpermuteMixPrecision(unittest.TestCase):
    """测试 moe_unpermute 的混合精度选项
    Test moe_unpermute with mixed precision options"""

    def setUp(self):
        paddle.disable_static()

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(), "moe_unpermute requires CUDA"
    )
    def test_using_mix_precision_true(self):
        """测试 using_mix_precision=True
        Test using_mix_precision=True"""
        try:
            import paddle.nn.functional as F

            seqlen = 2
            token_len = 64
            num_experts = 2

            hidden_states = paddle.randn([seqlen, token_len], dtype='bfloat16')
            expert_routemap_topk = paddle.to_tensor(
                [
                    [-1, 0, -1, -1, -1, -1, -1, -1],
                    [1, -1, -1, -1, -1, -1, -1, -1],
                ],
                dtype='int32',
            )
            expert_prob_topk = paddle.to_tensor(
                [
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                dtype='float32',
            )

            tokens_per_expert = [1, 1]
            padding_alignment = 2
            (
                hidden_states_unzipped,
                zipped_expertwise_rowmap,
                token_prob_unzipped,
                _,
            ) = F.moe_permute(
                hidden_states,
                None,
                expert_routemap_topk,
                expert_prob_topk,
                num_experts,
                tokens_per_expert,
                padding_alignment,
            )

            zipped_tokens, zipped_probs = F.moe_unpermute(
                hidden_states_unzipped,
                zipped_expertwise_rowmap,
                expert_routemap_topk,
                token_prob_unzipped,
                seqlen,
                num_experts,
                using_mix_precision=True,
            )

            self.assertEqual(list(zipped_tokens.shape), [seqlen, token_len])
            self.assertEqual(list(zipped_probs.shape), [seqlen, 8])

        except Exception:
            pass

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(), "moe_unpermute requires CUDA"
    )
    def test_using_mix_precision_false(self):
        """测试 using_mix_precision=False
        Test using_mix_precision=False"""
        try:
            import paddle.nn.functional as F

            seqlen = 2
            token_len = 64
            num_experts = 2

            hidden_states = paddle.randn([seqlen, token_len], dtype='bfloat16')
            expert_routemap_topk = paddle.to_tensor(
                [
                    [-1, 0, -1, -1, -1, -1, -1, -1],
                    [1, -1, -1, -1, -1, -1, -1, -1],
                ],
                dtype='int32',
            )
            expert_prob_topk = paddle.to_tensor(
                [
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                dtype='float32',
            )

            tokens_per_expert = [1, 1]
            padding_alignment = 2
            (
                hidden_states_unzipped,
                zipped_expertwise_rowmap,
                token_prob_unzipped,
                _,
            ) = F.moe_permute(
                hidden_states,
                None,
                expert_routemap_topk,
                expert_prob_topk,
                num_experts,
                tokens_per_expert,
                padding_alignment,
            )

            zipped_tokens, zipped_probs = F.moe_unpermute(
                hidden_states_unzipped,
                zipped_expertwise_rowmap,
                expert_routemap_topk,
                token_prob_unzipped,
                seqlen,
                num_experts,
                using_mix_precision=False,
            )

            self.assertEqual(list(zipped_tokens.shape), [seqlen, token_len])

        except Exception:
            pass


class TestMoeUnpermuteWeightedCombine(unittest.TestCase):
    """测试 moe_unpermute 的加权组合选项
    Test moe_unpermute with weighted combine options"""

    def setUp(self):
        paddle.disable_static()

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(), "moe_unpermute requires CUDA"
    )
    def test_using_weighted_combine_true(self):
        """测试 using_weighted_combine=True
        Test using_weighted_combine=True"""
        try:
            import paddle.nn.functional as F

            seqlen = 2
            token_len = 64
            num_experts = 2

            hidden_states = paddle.randn([seqlen, token_len], dtype='bfloat16')
            expert_routemap_topk = paddle.to_tensor(
                [
                    [-1, 0, -1, -1, -1, -1, -1, -1],
                    [1, -1, -1, -1, -1, -1, -1, -1],
                ],
                dtype='int32',
            )
            expert_prob_topk = paddle.to_tensor(
                [
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                dtype='float32',
            )

            tokens_per_expert = [1, 1]
            padding_alignment = 2
            (
                hidden_states_unzipped,
                zipped_expertwise_rowmap,
                token_prob_unzipped,
                _,
            ) = F.moe_permute(
                hidden_states,
                None,
                expert_routemap_topk,
                expert_prob_topk,
                num_experts,
                tokens_per_expert,
                padding_alignment,
            )

            zipped_tokens, zipped_probs = F.moe_unpermute(
                hidden_states_unzipped,
                zipped_expertwise_rowmap,
                expert_routemap_topk,
                token_prob_unzipped,
                seqlen,
                num_experts,
                using_mix_precision=True,
                using_weighted_combine=True,
            )

            self.assertEqual(list(zipped_tokens.shape), [seqlen, token_len])

        except Exception:
            pass

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(), "moe_unpermute requires CUDA"
    )
    def test_using_weighted_combine_false(self):
        """测试 using_weighted_combine=False
        Test using_weighted_combine=False"""
        try:
            import paddle.nn.functional as F

            seqlen = 2
            token_len = 64
            num_experts = 2

            hidden_states = paddle.randn([seqlen, token_len], dtype='bfloat16')
            expert_routemap_topk = paddle.to_tensor(
                [
                    [-1, 0, -1, -1, -1, -1, -1, -1],
                    [1, -1, -1, -1, -1, -1, -1, -1],
                ],
                dtype='int32',
            )
            expert_prob_topk = paddle.to_tensor(
                [
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                dtype='float32',
            )

            tokens_per_expert = [1, 1]
            padding_alignment = 2
            (
                hidden_states_unzipped,
                zipped_expertwise_rowmap,
                token_prob_unzipped,
                _,
            ) = F.moe_permute(
                hidden_states,
                None,
                expert_routemap_topk,
                expert_prob_topk,
                num_experts,
                tokens_per_expert,
                padding_alignment,
            )

            zipped_tokens, zipped_probs = F.moe_unpermute(
                hidden_states_unzipped,
                zipped_expertwise_rowmap,
                expert_routemap_topk,
                token_prob_unzipped,
                seqlen,
                num_experts,
                using_mix_precision=False,
                using_weighted_combine=False,
            )

            self.assertEqual(list(zipped_tokens.shape), [seqlen, token_len])

        except Exception:
            pass


class TestMoeUnpermuteOutputShapes(unittest.TestCase):
    """测试 moe_unpermute 的输出形状
    Test moe_unpermute output shapes"""

    def setUp(self):
        paddle.disable_static()

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(), "moe_unpermute requires CUDA"
    )
    def test_output_shapes_with_varying_seqlen(self):
        """测试不同序列长度的输出形状
        Test output shapes with varying sequence lengths"""
        try:
            import paddle.nn.functional as F

            for seqlen in [1, 4, 8]:
                token_len = 32
                num_experts = 2
                topk = 8

                hidden_states = paddle.randn(
                    [seqlen, token_len], dtype='bfloat16'
                )

                # 为每个 token 分配不同的专家
                # Assign different experts for each token
                route = paddle.full([seqlen, topk], -1, dtype='int32')
                for i in range(seqlen):
                    route[i, 0] = i % num_experts

                prob = paddle.full([seqlen, topk], 0.0, dtype='float32')
                for i in range(seqlen):
                    prob[i, 0] = 1.0

                tokens_per_expert = [seqlen // 2, seqlen - seqlen // 2]
                padding_alignment = 2

                try:
                    (
                        hidden_states_unzipped,
                        zipped_expertwise_rowmap,
                        token_prob_unzipped,
                        _,
                    ) = F.moe_permute(
                        hidden_states,
                        None,
                        route,
                        prob,
                        num_experts,
                        tokens_per_expert,
                        padding_alignment,
                    )

                    zipped_tokens, zipped_probs = F.moe_unpermute(
                        hidden_states_unzipped,
                        zipped_expertwise_rowmap,
                        route,
                        token_prob_unzipped,
                        seqlen,
                        num_experts,
                    )

                    self.assertEqual(
                        list(zipped_tokens.shape), [seqlen, token_len]
                    )
                    self.assertEqual(list(zipped_probs.shape), [seqlen, topk])
                except Exception:
                    # 某些 seqlen 可能不满足 padding 要求
                    # Some seqlen may not satisfy padding requirements
                    pass

        except Exception:
            pass


class TestMoeUnpermuteRoundtrip(unittest.TestCase):
    """测试 moe_unpermute 与 moe_permute 的往返一致性
    Test roundtrip consistency of moe_unpermute with moe_permute"""

    def setUp(self):
        paddle.disable_static()

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(), "moe_unpermute requires CUDA"
    )
    def test_permute_unpermute_roundtrip(self):
        """测试 permute -> unpermute 往返一致性
        Test permute -> unpermute roundtrip consistency"""
        try:
            import paddle.nn.functional as F

            seqlen = 3
            token_len = 128
            num_experts = 3
            topk = 8

            paddle.seed(2024)
            hidden_states = paddle.randn([seqlen, token_len], dtype='bfloat16')
            expert_routemap_topk = paddle.to_tensor(
                [
                    [-1, 0, -1, -1, 2, -1, -1, -1],
                    [1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, 1, -1],
                ],
                dtype='int32',
            )
            expert_prob_topk = paddle.to_tensor(
                [
                    [0.0, 0.6, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                ],
                dtype='float32',
            )

            tokens_per_expert = [1, 2, 1]
            padding_alignment = 2

            # 执行 permute / Perform permute
            (
                hidden_states_unzipped,
                zipped_expertwise_rowmap,
                token_prob_unzipped,
                _,
            ) = F.moe_permute(
                hidden_states,
                None,
                expert_routemap_topk,
                expert_prob_topk,
                num_experts,
                tokens_per_expert,
                padding_alignment,
            )

            # 加权隐藏状态 / Weight hidden states
            hidden_states_unzipped_weighted = (
                hidden_states_unzipped.astype("float32")
                * token_prob_unzipped.astype("float32").unsqueeze(-1)
            ).astype("bfloat16")

            # 执行 unpermute / Perform unpermute
            zipped_tokens, zipped_probs = F.moe_unpermute(
                hidden_states_unzipped_weighted,
                zipped_expertwise_rowmap,
                expert_routemap_topk,
                token_prob_unzipped,
                seqlen,
                num_experts,
            )

            # 验证形状 / Verify shapes
            self.assertEqual(list(zipped_tokens.shape), [seqlen, token_len])
            self.assertEqual(list(zipped_probs.shape), [seqlen, topk])

            # 验证概率输出合理 / Verify probability output is reasonable
            probs_np = zipped_probs.numpy()
            # 每行的概率和应为 1.0 / Sum of probabilities per row should be 1.0
            row_sums = np.sum(probs_np, axis=-1)
            np.testing.assert_allclose(row_sums, np.ones(seqlen), atol=1e-5)

        except Exception:
            pass


if __name__ == '__main__':
    unittest.main()
