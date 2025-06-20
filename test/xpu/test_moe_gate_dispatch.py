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
from paddle.autograd import PyLayer
from paddle.incubate.nn.functional import (
    moe_gate_dispatch,
)

os.environ["FLAGS_flash_attn_version"] = "v1"
os.environ["FLAGS_cudnn_deterministic"] = "1"
os.environ["FLAGS_embedding_deterministic"] = "1"
os.environ["XPU_PADDLE_FC_LOCAL_INT16"] = "1"


def topk_grad(x, dy, indices, w):
    """
    y=gather(topk(x)) 的反向过程
    x:  [s,e]
    dy: [s,k]
    """
    s, e = x.shape
    _, k = dy.shape
    dx = paddle.scatter_nd(
        paddle.stack(
            [
                paddle.arange(s).repeat_interleave(k).cast(indices.dtype),
                indices.reshape([-1]),
            ],
            -1,
        ),
        dy.reshape([-1]),
        shape=[s, e],
    )  # [s,k] -> [s,e]
    # mask
    for i in range(s):
        for j in range(k):
            if w[i, j] <= 0:
                index = indices[i, j]
                dx[i, index] = 0
    return dx  # dx 保持高精度


class GateDispatch(PyLayer):
    """doc"""

    @staticmethod
    def forward(ctx, x, gate_prob, k, capacity, use_pad, eps=1e-12):
        """
        对`gate_prob` 进行 softmax 并根据结果选取 topk 路由expert。 最后根据 expert 号对 `x` 进行重排。
        Args:
            x: [s, d] 输入的 activateion
            gate_prob: [s, e]
        k: int
            capacity: int #no use
        Returns:
            y: [s*k, d] 将所有 `x` 根据其路由的 `expert-id` 升序的排序，融合到 s 维度。
                    当截断发生时 s 会比输入 s 小。
            combine_weights: [s, k], float： 每个 token 第 k 选择的 expert 的权重。
                    当截断发生时 s 会比输入 s 小。
            scatter_index: [k, s] ： 每个 token 第 k 次选择对应到 `y` 中的位置。
            expert_offset: [e]： `y`中每个 expert-id 的分割位置。
            expert_id: [s] `x` 中激活的 expert 号
        """
        ctx.k = k
        ctx.eps = eps
        ctx.capacity = capacity
        ctx.gate_prob = gate_prob
        y, combine_weights, scatter_index, expert_offset, expert_id = (
            moe_gate_dispatch(
                x,
                gate_prob,
                None,
                k=k,
                capacity=capacity,
                use_pad=use_pad,
            )
        )
        ctx.combine_weights = combine_weights
        scatter_index = scatter_index.transpose([1, 0])  # [k,s] ->[s,k]
        ctx.scatter_index = scatter_index
        ctx.expert_id = expert_id
        num_experts = gate_prob.shape[-1]

        ctx.num_experts = num_experts
        ctx.seqlen = gate_prob.shape[0]

        return y, combine_weights, scatter_index, expert_offset, expert_id

    @staticmethod
    def backward(ctx, dy, dw, *_):
        """
        关于 softmax 对 logits 的导数，参考：
        https://stats.stackexchange.com/questions/215521/
        how-to-find-derivative-of-softmax-function-for-the-purpose-of-gradient-descent/328095#328095
        """
        s, k = ctx.combine_weights.shape
        grad = F.embedding(ctx.scatter_index, dy)  # [s, k,d]
        mask = (ctx.combine_weights > 0.0).astype(grad.dtype)  # [s,k]
        print("grad", grad)
        print("mask", mask)
        dx = paddle.matmul(mask.unsqueeze(1), grad).squeeze(
            1
        )  # [s,1,k] @ [s,k,d] -> [s,1,d]
        if ctx.gate_prob.stop_gradient:
            return dx, None

        combine_weights_unnorm = ctx.combine_weights
        dw = dw.astype(combine_weights_unnorm.dtype)
        d_prob = topk_grad(
            ctx.gate_prob, dw, ctx.expert_id, combine_weights_unnorm
        )
        return dx, d_prob


class MoELayer(paddle.nn.Layer):
    def forward(self, x, gate_prob, k, capacity):
        y, combine_weights, scatter_index, expert_offset, expert_id = (
            moe_gate_dispatch(
                x, gate_prob, None, k=k, capacity=capacity, use_pad=True
            )
        )
        scatter_index = scatter_index.transpose([1, 0])  # [k,s] ->[s,k]
        return y, combine_weights, scatter_index, expert_offset, expert_id


class TestFused(unittest.TestCase):
    def test_moe_ops(self):
        """
        test `moe-ops` w/ bias
        """
        # S, E, D = 8192, 64, 128
        S, E, D = 4, 4, 2
        # k = 4
        k = 2
        # cap = 512
        cap = 2
        # x = paddle.randn([S, D], dtype="bfloat16")
        x = paddle.randn([S, D], dtype="float32")
        gate_logits = paddle.randn([S, E], dtype="float32")
        x_ = x.clone()
        gate_logits_ = gate_logits.clone()
        x.stop_gradient = False
        x_.stop_gradient = False
        gate_logits.stop_gradient = False
        gate_logits_.stop_gradient = False
        bias = paddle.zeros([E], dtype="float32")

        layer = MoELayer()
        y, combine_weihgts, scatter_index, expert_offset, expert_id = layer(
            x,
            gate_logits,
            k,
            cap,
        )

        grad_y_numpy = np.random.randn(*y.shape).astype(np.float32)
        grad_w_numpy = np.random.randn(*combine_weihgts.shape).astype(
            np.float32
        )
        grad_y = paddle.to_tensor(grad_y_numpy)
        grad_w = paddle.to_tensor(grad_w_numpy)
        print(grad_y)

        paddle.autograd.backward([y, combine_weihgts], [grad_y, grad_w])

        y_, combine_weihgts_, scatter_index_, expert_offset_, expert_id_ = (
            GateDispatch.apply(x_, gate_logits_, k, cap, True)
        )

        grad_y_ = paddle.to_tensor(grad_y_numpy)
        grad_w_ = paddle.to_tensor(grad_w_numpy)
        print(grad_y_)
        paddle.autograd.backward(
            [y_, combine_weihgts_], [grad_y_, grad_w_], True
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
        np.testing.assert_equal(
            scatter_index.astype("float32").numpy(),
            scatter_index_.astype("float32").numpy(),
            err_msg="incubate w bias not match",
        )
        np.testing.assert_equal(
            expert_offset.astype("float32").numpy(),
            expert_offset_.astype("float32").numpy(),
            err_msg="incubate w bias not match",
        )
        np.testing.assert_equal(
            expert_id.astype("float32").numpy(),
            expert_id_.astype("float32").numpy(),
            err_msg="incubate w bias not match",
        )

        np.testing.assert_allclose(
            x.grad.astype("float32").numpy(),
            x_.grad.astype("float32").numpy(),
            atol=1e-5,
            rtol=1e-5,
        )

        np.testing.assert_allclose(
            gate_logits.grad.astype("float32").numpy(),
            gate_logits_.grad.astype("float32").numpy(),
            atol=1e-5,
            rtol=1e-5,
        )


if __name__ == "__main__":

    unittest.main()
