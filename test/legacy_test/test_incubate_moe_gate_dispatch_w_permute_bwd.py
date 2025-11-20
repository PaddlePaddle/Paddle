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

import unittest

import numpy as np

import paddle
import paddle.nn.functional as F
from paddle.incubate.nn.functional import (
    moe_gate_dispatch,
    moe_gate_dispatch_permute,
)

batch_size = 4
hidden_size = 2
k = 16
capacity = 2
num_experts = 16

world_size = 2


class TestLayer(paddle.nn.Layer):
    def forward(self, x, gate_prob, k, capacity):
        y, combine_weights, scatter_index, expert_offset, expert_id = (
            moe_gate_dispatch(x, gate_prob, None, k, capacity, True)
        )
        return y, combine_weights, scatter_index, expert_offset, expert_id


class TestLayerPermute(paddle.nn.Layer):
    def forward(self, x, gate_prob, k, capacity):
        y, combine_weights, scatter_index, expert_offset, expert_id = (
            moe_gate_dispatch_permute(
                x, gate_prob, None, k, capacity, world_size=world_size
            )
        )
        return y, combine_weights, scatter_index, expert_offset, expert_id


def check_backward_correctness(layer_cls):
    paddle.seed(1024)

    dtype = "bfloat16"
    layer = layer_cls()
    input = paddle.randn([batch_size, hidden_size])

    gate_weight = paddle.randn([hidden_size, num_experts])
    logits = paddle.matmul(input, gate_weight)
    gate_prob = F.softmax(logits, axis=-1)
    print(f"gate_prob: {gate_prob}")

    input = paddle.cast(input, "bfloat16")
    input.stop_gradient = False
    gate_prob.stop_gradient = False

    output, combine_weights, scatter_index, expert_offset, expert_id = layer(
        input, gate_prob, k, capacity
    )

    print(f"output: {output}")
    print(f"combine_weights: {combine_weights}")
    print(f"scatter_index: {scatter_index}")
    print(f"expert_offset: {expert_offset}")
    print(f"expert_id: {expert_id}")

    # output_g = paddle.randn(output.shape).astype(output.dtype)
    # combine_weights_g = paddle.randn(combine_weights.shape).astype(combine_weights.dtype)
    output_g = paddle.ones_like(output)
    combine_weights_g = paddle.ones_like(combine_weights)
    print(f"output_g: {output_g}")
    print(f"combine_weights_g: {combine_weights_g}")

    paddle.autograd.backward(
        tensors=[output, combine_weights],
        grad_tensors=[output_g, combine_weights_g],
    )
    # 数值估算
    epsilon = 0.005
    input_numpy = input.detach().astype("float32").numpy()
    num_grad = paddle.zeros_like(input)
    flattened = num_grad.reshape([-1])

    for i in range(input.numel()):
        input_pos = input_numpy.copy()
        input_neg = input_numpy.copy()
        input_pos.flat[i] += epsilon
        input_neg.flat[i] -= epsilon

        output_pos, _, _, _, _ = layer(
            paddle.to_tensor(input_pos), gate_prob, k, capacity
        )
        output_neg, _, _, _, _ = layer(
            paddle.to_tensor(input_neg), gate_prob, k, capacity
        )

        '''
        flattened[i] = (output_pos.astype("float32").numpy() - output_neg.astype("float32").numpy()).sum() / (
            2 * epsilon
        )
        '''
        grad_value = (output_pos - output_neg).sum() / (2 * epsilon)
        flattened[i] = grad_value

    flattened = flattened.reshape(input.shape)

    print(f"input gradient: {input.grad}")
    print(f"numerical gradient: {flattened}")
    np.testing.assert_allclose(
        input.grad.astype("float32").numpy(),
        flattened.astype("float32").numpy(),
        rtol=1e-5,
        atol=0,
    )

    # 数值估算 gate_prob
    epsilon = 0.0005
    gate_prob_numpy = gate_prob.detach().astype("float32").numpy()
    num_grad = paddle.zeros_like(gate_prob)
    flattened = num_grad.reshape([-1])

    for i in range(gate_prob.numel()):
        input_pos = gate_prob_numpy.copy()
        input_neg = gate_prob_numpy.copy()
        input_pos.flat[i] += epsilon
        input_neg.flat[i] -= epsilon

        _, output_pos, _, _, _ = layer(
            input, paddle.to_tensor(input_pos), k, capacity
        )
        _, output_neg, _, _, _ = layer(
            input, paddle.to_tensor(input_neg), k, capacity
        )

        grad_value = paddle.to_tensor(
            (output_pos.numpy() - output_neg.numpy()).sum() / (2 * epsilon)
        )
        flattened[i] = grad_value

    flattened = flattened.reshape(gate_prob.shape)

    print(f"gate_prob gradient: {gate_prob.grad}")
    print(f"numerical gradient: {flattened}")
    np.testing.assert_allclose(
        gate_prob.grad.astype("float32").numpy(),
        flattened.astype("float32").numpy(),
        rtol=1e-4,
        atol=0,
    )


class TestFused(unittest.TestCase):
    def test_moe_backward(self):
        check_backward_correctness(TestLayer)

    def test_moe_permute_backward(self):
        check_backward_correctness(TestLayerPermute)


if __name__ == "__main__":
    unittest.main()
