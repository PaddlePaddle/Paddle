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

"""
moe
"""

import inspect
import logging
from collections import namedtuple

import paddle
from paddle import _C_ops
from paddle.autograd import PyLayer

# from ernie_core.models.moe.moe_layer import _AllToAll
from paddle.incubate.nn.functional import moe_gate_dispatch
from paddle.nn import functional as F

try:
    from src.utils.misc import global_training_logs
except ModuleNotFoundError:
    global_training_logs = {}  # 没有erniebot的环境下无法打印 debug 量

logger = logging.getLogger(__name__)

GateOutput = namedtuple(
    "GateOutput",
    [
        "aux",
        "z",
        "logits",
    ],
)


if False:
    try:
        from paddle_xpu_nn import (
            moe_combine as xpu_moe_combine,
            moe_combine_bwd as xpu_moe_combine_bwd,
        )
    except ImportError:
        xpu_moe_combine = None
        xpu_moe_combine_bwd = None
        logger.warning("`xpu moe combine` not found")
else:
    try:
        from paddle.incubate.nn.functional import moe_combine
    except ImportError:
        moe_combine = None
        logger.warning(
            "`moe-combine` not found, run "
            "`python3  src/ernie_core/ops/moe/setup.py  install` to install"
        )


def average_grad(x, y, dy, eps=1e-12):
    """
    TODO: fuse 这坨 shit
    y=x/x.sum(-1, keepdim=True) 的反向过程
    """
    s, k = x.shape
    xsum = x.sum(axis=-1, keepdim=True)  # [s,1]
    maskpos = (xsum == 0.0).expand_as(x)

    xsum_square = xsum.square()  # [s,1]
    left = paddle.triu(
        paddle.tril((1 / xsum).unsqueeze(-1).expand([s, k, k]))
    )  # aka diag-emb [s,k,k]
    right = (-x / xsum_square).unsqueeze(-1).expand([s, k, k])
    dydx = left + right
    dx = paddle.matmul(dy.unsqueeze(-2).cast(dydx.dtype), dydx).squeeze(
        -2
    )  # [s,1,k] @[s,k,k] -> [s,1,k]
    dx = paddle.where(maskpos, paddle.zeros_like(dx), dx)
    return dx


mask = paddle.to_tensor(
    [
        [1, -1],
        [-1, 1],
    ]
).unsqueeze(0)


def average_grad_bi(x, y, dy, eps=1e-12):
    """
    y=x/x.sum(-1, keepdim=True)
    k=2 下面的反向过程，精度会更准一些:
        dx1 = (y2 *dy1 - y2*dy2)/(y1+y2)**2
        dx2 = (y1 *dy2 - y1*dy1)/(y1+y2)**2
    """
    s, k = x.shape
    assert k == 2, k
    xsum = paddle.clip(x.sum(axis=-1, keepdim=True), min=eps)  # [s,1]
    dydx = (
        x.flip(axis=1).unsqueeze(-2).tile([1, 2, 1])
        * mask.cast(x.dtype)
        / xsum.square().unsqueeze(-1)
    )
    dx = paddle.matmul(dy.unsqueeze(-2).cast(dydx.dtype), dydx).squeeze(
        -2
    )  # [s,1,k] @[s,k,k] -> [s,1,k]
    return dx


def topk_grad(x, dy, indices):
    """
    TODO: fuse 这坨 shit
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
        if "corr_bias" in inspect.signature(moe_gate_dispatch).parameters:
            compat_args = (None,)
        else:
            compat_args = ()
        y, combine_weights, scatter_index, expert_offset, expert_id = (
            moe_gate_dispatch(
                x,
                gate_prob,
                *compat_args,
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
        TODO: 这坨代码可以 fuse 一手。
        关于 softmax 对 logits 的导数，参考：
        https://stats.stackexchange.com/questions/215521/
        how-to-find-derivative-of-softmax-function-for-the-purpose-of-gradient-descent/328095#328095
        """
        s, k = ctx.combine_weights.shape
        grad = F.embedding(ctx.scatter_index, dy)  # [s, k,d]
        mask = (ctx.combine_weights > 0.0).astype(grad.dtype)  # [s,k]
        dx = paddle.matmul(mask.unsqueeze(1), grad).squeeze(
            1
        )  # [s,1,k] @ [s,k,d] -> [s,1,d]
        if ctx.gate_prob.stop_gradient:
            return dx, None

        combine_weights_unnorm = ctx.combine_weights
        dw = dw.astype(combine_weights_unnorm.dtype)
        d_prob = topk_grad(ctx.gate_prob, dw, ctx.expert_id)
        return dx, d_prob


class GateCombine(PyLayer):
    """GateCombine"""

    @staticmethod
    def forward(ctx, x, combine_weights, scatter_index):
        """
        Input:
            x:  [seqlen * k, hidden_size]
            combine_weights: [seqlen, k]
            scatter_index: [seqlen, k]
        Output:
            y: [seqlen, hidden_size]
        """
        ctx.x = x
        ctx.combine_weights = combine_weights
        ctx.scatter_index = scatter_index
        if False:
            assert xpu_moe_combine is not None
            return xpu_moe_combine(x, combine_weights, scatter_index)
        else:
            assert moe_combine is not None
            ret = moe_combine(x, combine_weights, scatter_index)
            return ret

    @staticmethod
    def backward(ctx, grad_y, *_):
        """
        Input:
            grad_y:  [seqlen, hidden_size]
            combine_weights: [seqlen, k]
            scatter_index: [seqlen, k]
        Output:
            grad_x: [seqlen * k, hidden_size]
            grad_combine_weight: [seqlen, k]

        """

        if False:
            assert xpu_moe_combine_bwd is not None
            grad_x, grad_combine_weight_helper = xpu_moe_combine_bwd(
                ctx.x, ctx.combine_weights, ctx.scatter_index, grad_y
            )
        else:
            assert moe_combine is not None
            grad_x, grad_combine_weight_helper = _C_ops.moe_combine_grad(
                ctx.x, ctx.combine_weights, ctx.scatter_index, grad_y
            )
        # grad_combine_weight_helper is the same shape with grad x [seqlen * K, dim]
        # reduce the hidden shape
        # TODO: implement reduce in cuda ops
        grad_combine_weight = grad_combine_weight_helper.sum(-1)
        return (
            grad_x,
            grad_combine_weight.reshape(ctx.combine_weights.shape),
            None,
        )
        # return grad_x, grad_combine_weight_helper


def combining(x, combine_weights, scatter_index, hard_gate=False):
    """
    Args:
        x: Tensor[seq, dim]
        combine_weights: [s, k]
        scatter_index:  ** [k, s] **

    Returns:
        y: Tensor[s, dim]
    """
    if hard_gate:
        x_gatherd = F.embedding(scatter_index, x)  # [s,k,dim]
        return x_gatherd.squeeze(-2)
    ret = GateCombine.apply(x, combine_weights, scatter_index)
    ret.stop_gradient = False
    return ret
