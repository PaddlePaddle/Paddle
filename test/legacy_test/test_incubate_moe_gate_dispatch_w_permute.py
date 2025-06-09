# !/usr/bin/env python3

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

import os
import unittest

import numpy as np

import paddle
import paddle.nn.functional as F
from paddle.incubate.nn.functional import (
    moe_gate_dispatch,
    moe_gate_dispatch_permute,
)

os.environ["FLAGS_flash_attn_version"] = "v1"
os.environ["FLAGS_cudnn_deterministic"] = "1"
os.environ["FLAGS_embedding_deterministic"] = "1"


class TestFused(unittest.TestCase):

    def test_moe_ops(self):
        """
        test `moe-ops` w/ bias
        """
        S, E, D = 8192, 64, 128
        k = 4
        x = paddle.randn([S, D], dtype="bfloat16")
        gate_logits = paddle.randn([S, E], dtype="float32")
        x_ = x.clone()
        gate_logits_ = gate_logits.clone()
        x.stop_gradient = True
        x_.stop_gradient = True
        gate_logits.stop_gradient = True
        gate_logits_.stop_gradient = True
        bias = paddle.zeros([E], dtype="float32")
        cap = 512

        y, combine_weihgts, scatter_index, expert_offset_, expert_id_ = (
            moe_gate_dispatch(
                x,
                gate_logits,
                None,
                k=k,
                capacity=cap,
                use_pad=True,  # k  # cap
            )
        )

        y_, combine_weihgts_, scatter_index_, expert_offset_, expert_id_ = (
            moe_gate_dispatch(
                x_,
                gate_logits_,
                bias + 1,  # +1也不会破坏路由结果
                k=k,
                capacity=cap,
                use_pad=True,  # k  # cap
            )
        )
        bias_unbalanced = bias.clone()
        bias_unbalanced[0] += 1
        (
            y__,
            combine_weihgts__,
            scatter_index__,
            expert_offset__,
            expert_id__,
        ) = moe_gate_dispatch(
            x_,
            gate_logits_,
            bias_unbalanced,
            k=k,
            capacity=cap,
            use_pad=True,  # k  # cap
        )
        np.testing.assert_equal(
            y.astype("float32").numpy(),
            y_.astype("float32").numpy(),
            err_msg="incubate w bias not match",
        )
        # bias 不影响 prob 概率
        np.testing.assert_equal(
            combine_weihgts.astype("float32").numpy(),
            combine_weihgts_.astype("float32").numpy(),
            err_msg="incubate w bias not match",
        )
        np.testing.assert_(
            (
                y.astype("float32").numpy(0) != y__.astype("float32").numpy()
            ).any(),
        )


class TestDispatchPermute(unittest.TestCase):
    def get_detached_input(self, input, prob):
        ret_input = input.detach()
        ret_prob = prob.detach()
        ret_input.stop_gradient = input.stop_gradient
        ret_prob.stop_gradient = prob.stop_gradient
        return ret_input, ret_prob

    def get_stage_input_list(self, x, world_size, stage):
        print(world_size, stage, x.shape)
        x = x.reshape([world_size * stage, -1, x.shape[-1]])
        stage_input_list = []
        x_list = paddle.split(x, num_or_sections=(world_size * stage), axis=0)
        for stage_id in range(stage):
            stage_input_list.append(
                paddle.unsqueeze(
                    paddle.concat(x_list[stage_id::stage], axis=0), axis=0
                )
            )
        stage_input_list = paddle.concat(stage_input_list, axis=0)
        return stage_input_list

    def test_moe_permute_ops(self):
        paddle.seed(2025)

        test_cases = [
            (8, 4, 2),
            (64, 16, 32),
            (1024, 1024, 1024),
            (8, 2, 4),
            (4096, 4096, 4096),
        ]
        cases = list(zip(*test_cases))
        for _, case in enumerate(cases):
            world_size, num_experts, num_tokens, k, hidden_size = case
            capacity = num_tokens // k
            stages = num_experts // world_size

            input = paddle.randn([num_tokens, hidden_size], dtype="float32")
            prob_logits = paddle.randn(
                [num_tokens, num_experts], dtype="float32"
            )
            prob = F.softmax(prob_logits, axis=-1)
            input.stop_gradient = False
            prob.stop_gradient = False

            compat_args = (None,)

            ref_input, ref_prob = self.get_detached_input(input, prob)
            (
                ref_dispatched_input,
                ref_combine_weights_unnorm,
                ref_scatter_index,
                ref_dispatch_mask,
                _,
            ) = moe_gate_dispatch(
                ref_input,
                ref_prob,
                *compat_args,
                k=k,
                capacity=capacity,
                use_pad=True,
            )

            ref_stage_input_list = self.get_stage_input_list(
                ref_dispatched_input, world_size, stages
            )

            test_input, test_prob = self.get_detached_input(input, prob)
            (
                test_dispatched_input,
                test_combine_weights_unnorm,
                test_scatter_index,
                test_dispatch_mask,
                _,
            ) = moe_gate_dispatch_permute(
                test_input,
                test_prob,
                *compat_args,
                k=k,
                capacity=capacity,
                world_size=world_size,
            )

            np.testing.assert_equal(
                test_dispatched_input.shape,
                ref_stage_input_list.shape,
                err_msg="moe_permute_ops not match",
            )
            np.testing.assert_equal(
                test_dispatched_input._md5sum(),
                ref_stage_input_list._md5sum(),
                err_msg="moe_permute_ops not match",
            )


if __name__ == "__main__":

    unittest.main()
