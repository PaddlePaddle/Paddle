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
from paddle.incubate.nn.functional import cal_aux_loss
from paddle.incubate.nn.functional.cal_aux_loss import math_cal_aux_loss


class TestCalAuxLoss(unittest.TestCase):
    def setUp(self):
        paddle.set_device('gpu')

        self.num_tokens = 6
        self.num_experts = 4
        self.top_k = 2

        self.gate_prob = paddle.to_tensor(
            [
                [0.6, 0.4, 0.0, 0.0],
                [0.7, 0.3, 0.0, 0.0],
                [0.0, 0.0, 0.8, 0.2],
                [0.0, 0.0, 0.5, 0.5],
                [0.3, 0.7, 0.0, 0.0],
                [0.1, 0.9, 0.0, 0.0],
            ],
            dtype='float32',
        )

        self.dispatch_mask = paddle.to_tensor(
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [1, 1, 0, 0],
                [1, 1, 0, 0],
            ],
            dtype='int64',
        )

        self.clip_min = 1e-6

    def check_outputs(
        self, tokens_mask=None, dispatch_tokens_mask=None, use_group=False
    ):
        result_l_aux, result_seqlen_float, result_ce = cal_aux_loss(
            self.gate_prob,
            self.dispatch_mask,
            tokens_mask,
            dispatch_tokens_mask,
            self.num_experts,
            use_group,
            self.top_k,
            self.clip_min,
        )
        expected_l_aux, expected_seqlen_float, expected_ce = math_cal_aux_loss(
            gate_prob=self.gate_prob,
            dispatch_mask=self.dispatch_mask,
            tokens_mask=tokens_mask,
            dispatch_tokens_mask=dispatch_tokens_mask,
            num_experts=self.num_experts,
            use_group=use_group,
            moe_k=self.top_k,
            clip_min=self.clip_min,
        )

        np.testing.assert_allclose(
            result_l_aux.numpy(),
            expected_l_aux.numpy(),
            rtol=1e-5,
            err_msg="aux_loss mismatch",
        )
        np.testing.assert_allclose(
            result_seqlen_float.numpy(),
            expected_seqlen_float.numpy(),
            rtol=1e-5,
            err_msg="seq_len mismatch",
        )
        np.testing.assert_allclose(
            result_ce.numpy(),
            expected_ce.numpy(),
            rtol=1e-5,
            err_msg="ce mismatch",
        )

    def test_aux_loss_consistency(self):
        self.check_outputs()

    def test_tokens_mask(self):
        tokens_mask = paddle.to_tensor([1, 1, 0, 1, 1, 0]).cast(
            self.gate_prob.dtype
        )
        self.check_outputs(tokens_mask=tokens_mask)

    def test_dispatch_tokens_mask(self):
        dispatch_tokens_mask = paddle.to_tensor(
            [1, 1, 0, 1, 1, 0],
        ).cast('bool')
        self.check_outputs(dispatch_tokens_mask=dispatch_tokens_mask)

    def test_use_group(self):
        self.check_outputs(use_group=True)


if __name__ == '__main__':
    unittest.main()
