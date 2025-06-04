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

import logging
import unittest

import numpy as np
from ernie_utils.top2_gate import (
    cal_aux_loss_func,
)

import paddle
import paddle.nn.functional as F
from paddle.incubate.nn.functional import cal_aux_loss

logger = logging.getLogger(__name__)


class TestFusedCalculateAuxLoss(unittest.TestCase):
    def setUp(self):
        paddle.seed(42)
        self.atol = 1e-10
        self.rtol = 1e-5

    def run_and_check(
        self,
        gate_prob,
        dispatch_mask,
        tokens_mask=None,
        dispatch_tokens_mask=None,
        num_experts=48,
        moe_k=6,
        use_group=False,
    ):
        dispatch_mask_for_ref = dispatch_mask.detach()
        dispatch_mask_for_test = dispatch_mask.detach()
        input_for_ref = gate_prob.detach()
        input_for_test = gate_prob.detach()
        input_for_ref.stop_gradient = False
        input_for_test.stop_gradient = False

        loss_ref = cal_aux_loss_func(
            input_for_ref,
            dispatch_mask_for_ref,
            tokens_mask,
            dispatch_tokens_mask,
            num_experts,
            use_group,
            moe_k,
        )
        loss, _, _ = cal_aux_loss(
            input_for_test,
            dispatch_mask_for_test,
            tokens_mask,
            dispatch_tokens_mask,
            num_experts,
            use_group,
            moe_k,
            1e-6,
        )
        loss_ref.backward()
        loss.backward()

        np.testing.assert_equal(loss.shape, loss_ref.shape)
        np.testing.assert_equal(loss.dtype, loss_ref.dtype)
        np.testing.assert_equal(
            input_for_ref.grad.shape, input_for_test.grad.shape
        )
        np.testing.assert_equal(
            input_for_ref.grad.dtype, input_for_test.grad.dtype
        )
        np.testing.assert_allclose(
            loss.astype("float32").numpy(),
            loss_ref.astype("float32").numpy(),
            atol=self.atol,
            rtol=self.rtol,
        )
        np.testing.assert_allclose(
            input_for_test.grad.astype("float32").numpy(),
            input_for_ref.grad.astype("float32").numpy(),
            atol=self.atol,
            rtol=self.rtol,
        )

    def run_single_case(
        self,
        seq_len,
        expert_num=48,
        g_num_experts=96,
        moe_k=6,
    ):
        for use_group in [True, False]:
            for use_tokens_mask in [True, False]:
                for use_dispatch_tokens_mask in [True, False]:
                    paddle.seed(48)
                    gate_prob = paddle.randn([seq_len, expert_num])
                    dispatch_mask = paddle.randint(
                        0, seq_len, [expert_num]
                    ).astype("int64")
                    tokens_mask = (
                        paddle.randint(0, 1, [seq_len]).astype(gate_prob.dtype)
                        if use_tokens_mask
                        else None
                    )
                    dispatch_tokens_mask = (
                        paddle.randint(0, 1, [seq_len * 2]).astype("bool")
                        if use_dispatch_tokens_mask
                        else None
                    )
                    self.run_and_check(
                        gate_prob,
                        dispatch_mask,
                        tokens_mask,
                        dispatch_tokens_mask,
                        g_num_experts,
                        moe_k,
                        use_group,
                    )

    def test_trivial_cases(self):
        self.run_single_case(seq_len=1, expert_num=1)
        self.run_single_case(seq_len=3, expert_num=2)
        self.run_single_case(seq_len=13, expert_num=3)
        self.run_single_case(seq_len=1024, expert_num=6)
        self.run_single_case(seq_len=2048, expert_num=6)
        self.run_single_case(seq_len=3005, expert_num=48)
        self.run_single_case(seq_len=3005, expert_num=96)
        self.run_single_case(seq_len=4096, expert_num=48)
        self.run_single_case(seq_len=4096, expert_num=15)
        self.run_single_case(seq_len=4096, expert_num=92)
        self.run_single_case(seq_len=6000, expert_num=92)
        self.run_single_case(seq_len=8192, expert_num=48)
        self.run_single_case(seq_len=8192, expert_num=96)
        self.run_single_case(seq_len=8477, expert_num=48)
        self.run_single_case(seq_len=16 * 1024, expert_num=48)
        self.run_single_case(seq_len=32 * 1024, expert_num=96)
        self.run_single_case(seq_len=48 * 1024, expert_num=48)
        self.run_single_case(seq_len=100 * 1024, expert_num=48)
        self.run_single_case(seq_len=128 * 1024, expert_num=96)
        self.run_single_case(seq_len=128 * 1024 + 478, expert_num=48)
        self.run_single_case(seq_len=256 * 1024, expert_num=48)
        self.run_single_case(seq_len=512 * 1024, expert_num=128)

    def run_special_case(
        self, global_seq_len, seq_len, global_expert_num, expert_num, moe_k
    ):
        for use_group in [True, False]:
            paddle.seed(48)
            seq_len = 4096
            expert_num = 48
            gate_prob = F.softmax(paddle.randn([seq_len, expert_num]), axis=-1)
            dispatch_mask = paddle.randint(
                0, seq_len, [seq_len, expert_num]
            ).astype("int64")
            tokens_mask = paddle.randint(0, 1, [seq_len]).astype(
                gate_prob.dtype
            )
            dispatch_tokens_mask = paddle.randint(
                0, 1, [global_seq_len]
            ).astype("bool")
            self.run_and_check(
                gate_prob,
                dispatch_mask,
                tokens_mask,
                dispatch_tokens_mask,
                global_expert_num,
                moe_k,
                use_group,
            )

    def test_special_cases(self):
        self.run_special_case(123, 156, 4, 8, 2)
        self.run_special_case(123, 123 * 2, 4, 8, 2)
        self.run_special_case(128, 128, 4, 8, 2)
        self.run_special_case(1024, 4096, 4, 8, 2)
        self.run_special_case(2048, 4096, 4, 8, 2)
        self.run_special_case(2048, 9648, 4, 16, 2)
        self.run_special_case(4096, 7546, 4, 8, 2)
        self.run_special_case(4096, 4096 * 2, 4, 8, 2)
        self.run_special_case(4096, 4096 * 2, 48, 48 * 2, 6)
        self.run_special_case(5001, 5555, 48, 48 * 2, 6)
        self.run_special_case(4096, 4096 * 8, 48, 48 * 8, 2)
        self.run_special_case(4565, 4565 * 8, 47, 47 * 8, 4)
        self.run_special_case(8192, 12288, 47, 47 * 8, 4)
        self.run_special_case(8192, 8192 * 8, 48, 48 * 6, 16)
        self.run_special_case(8192, 8192 * 16, 48, 48 * 16, 32)
        self.run_special_case(8192, 8192 * 16, 123, 123 * 16, 111)
        self.run_special_case(10580, 10580 * 16, 52, 52 * 16, 78)
        self.run_special_case(512 * 1024, 1024 * 1024, 123, 123 * 16, 111)


if __name__ == "__main__":
    unittest.main()
